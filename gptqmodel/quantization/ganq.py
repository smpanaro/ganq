import math
import os
import time
import concurrent.futures
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mlx.core as mx
    USE_MLX = torch.mps.is_available()
except ImportError:
    USE_MLX = False

import numpy as np
import kmeans1d
from tqdm import tqdm

from ..utils.logger import setup_logger
from .gptq import GPTQ

log = setup_logger()


def kmeans_fit(row_data):
    weights_np, sample_weight, n_cluster, random_seed = row_data
    _, centroids = kmeans1d.cluster(weights_np, n_cluster, weights=sample_weight)
    return np.array(centroids, dtype=np.float32)

def to_mlx(x: Union[torch.Tensor, mx.array]):
    if isinstance(x, mx.array):
        return x
    return mx.array(x.cpu().numpy())
def from_mlx(x: mx.array):
    return torch.from_numpy(np.array(x, copy=False))

def compute_s(W: mx.array, L: mx.array, C: mx.array):
    """
    Compute one backwards solve along the columns of W.
    """
    num_rows, num_values = C.shape
    _, num_cols = W.shape
    assert L.shape == (W.shape[1], W.shape[1])

    header = """
        constant int ROWS_PER_THREAD = 32;  // must be simd group size to match # of threads launched.
        constant int COL_BLOCK_SIZE = 8;    // flexible - must match TN below

        // MLX GEMV Kernel
        // https://github.com/ml-explore/mlx/blob/5f5770e3a2646a924449125b150ae855486dee72/mlx/backend/metal/kernels/gemv.metal#L13
        #define T float
        #define AccT float

        // TODO: This seems to be worse than manually unrolling. Why?
        #define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

        //constant int BM = 1;    /* Threadgroup rows (in simdgroups) */
        //constant int BN = 1;    /* Threadgroup cols (in simdgroups) */
        constant int SM = 1;    /* Simdgroup rows (in threads) */
        constant int SN = 32;   /* Simdgroup cols (in threads) */
        constant int TM = 1;    /* Thread rows (in elements) */
        constant int TN = 8;    /* Thread cols (in elements) */

        template <typename U = T>
        static METAL_FUNC void
        load_unsafe(const device T* src, thread U dst[TN], const int src_offset = 0) {
            dst[0] = static_cast<U>(src[src_offset + 0]);
            dst[1] = static_cast<U>(src[src_offset + 1]);
            dst[2] = static_cast<U>(src[src_offset + 2]);
            dst[3] = static_cast<U>(src[src_offset + 3]);
            dst[4] = static_cast<U>(src[src_offset + 4]);
            dst[5] = static_cast<U>(src[src_offset + 5]);
            dst[6] = static_cast<U>(src[src_offset + 6]);
            dst[7] = static_cast<U>(src[src_offset + 7]);
        }

        template <typename U = T>
        static METAL_FUNC void load_safe(
            const device T* src,
            thread U dst[TN],
            const int src_offset = 0,
            const int src_size = TN) {
                if (src_offset + TN <= src_size) {
                    for (int tn = 0; tn < TN; tn++) {
                        dst[tn] = static_cast<U>(src[src_offset + tn]);
                    }
                } else { // Edgecase
                    for (int tn = 0; tn < TN; tn++) {
                        dst[tn] = src_offset + tn < src_size
                            ? static_cast<U>(src[src_offset + tn])
                            : U(0);
                    }
                }
            }
    """
    source = """
        int num_rows =  W_shape[0];
        int num_cols = W_shape[1]; // Also L.shape[0] and L.shape[1]
        int num_values = C_shape[1];

        short row_idx = thread_position_in_grid.x;
        int codebook_start_idx = row_idx * num_values;
        int W_start_idx = row_idx * num_cols;

        float R = 0;
        for (short j = num_cols-1; j>= 0; j--) {
            float wr_value = W[W_start_idx + j] + (R / L[j * num_cols + j]);

            float min_distance = INFINITY;
            short min_idx = 0;
            float min_value = INFINITY;

            for (short v_idx = 0; v_idx < num_values; v_idx++) {
                float codebook_value = C[codebook_start_idx + v_idx];
                float distance = metal::abs(wr_value - codebook_value);

                if (distance < min_distance) {
                    min_distance = distance;
                    min_idx = v_idx;
                    min_value = codebook_value;
                }
                // min_distance = fmin(min_distance, distance);
                // min_idx = select(min_idx, v_idx, min_distance < distance);
                // min_value = select(min_value, codebook_value, min_distance < distance);
            }

            float row_err = W[W_start_idx + j] - min_value;
            Werr[W_start_idx + j] = row_err;
            Q[W_start_idx + j] = min_idx;
            uint S_idx = (row_idx * num_values * num_cols) + (min_idx * num_cols) + j;
            S[S_idx] = 1; // (row,val,col)

            if (j == 0) break; // No need to compute the last residual. Avoids out of bounds.

            // Compute the residual. Werr[row_idx, j:] @ L[j:, j-1]

            ///////////////////////////////////////
            ////// Approach 3 -  Modified MLX GEMV
            ///////////////////////////////////////

            // >>> L must be transposed before this kernel. <<<

            // Lightly modified version of MLX's gemv. Instead of sharing L across threads
            // each thread owns a part of L and processes a chunk of Werr columns for all
            // rows in the threadgroup.

            // Werr[tg_start_row:tg_end_row, j:] @ L[j:, j-1]
            // mat @ vec = [M,K] @ [K,N], N = 1


            //                 □─Chunk 1□─Chunk 2□─Chunk 3□─Chunk 4□
            //                 ┌────────┬────────┬────────┬────────┬──┐
            //                 │Vector                                │
            //                 └────────┴────────┴────────┴────────┴──┘
            //
            //         □       ┌────────┬────────┬────────┬────────┬──┐
            //         │       │thread 1 thread 2 thread 1 thread 2│  │
            //         │       │ iter 1 │ iter 1 │ iter 2 │ iter 2 │  │
            //         │       │                                   │  │
            //     32 Rows     │ rows * │ rows * │ rows * │ rows * │  │
            //   32 Threads    │ Chunk1   Chunk2   Chunk3   Chunk4 │  │
            //  1 threadgroup  │        │        │        │        │  │
            //   1 simdgroup   │                                   │  │
            //         │       │        │        │        │        │  │
            //         │       │                                   │  │
            //         □       ├────────┴────────┴────────┴────────┘  │
            //                 │                                      │
            //                 │                                      │
            //                 │                                      │
            //                 │                                      │
            //                 │Matrix                                │
            //                 └──────────────────────────────────────┘

            thread float resM[ROWS_PER_THREAD] = {0};
            thread float vec_chunk[COL_BLOCK_SIZE] = {0};
            thread float row_chunk[COL_BLOCK_SIZE] = {0}; // tmp holding for row columns

            short K = num_cols - j;

            // Assumes: ROWS_PER_THREAD >= threads_per_threadgroup == threads_per_simdgroup
            short num_chunks = K / threads_per_threadgroup.x / COL_BLOCK_SIZE;
            short leftover = K - (num_chunks * threads_per_threadgroup.x * COL_BLOCK_SIZE);

            short K_offset = thread_position_in_threadgroup.x * COL_BLOCK_SIZE;
            const device float* mat = &Werr[threads_per_threadgroup.x * threadgroup_position_in_grid.x * num_cols];
            const device float* vec = &L[(j-1)*num_cols]; // Requires L to be transposed before kernel.

            // Each thread process one chunk at a time.
            // Chunks assignments are interleaved across threads,
            // so all chunks that are being processed are adjacent.
            for (short i = 0; i < num_chunks; i++) {
                // Unsafe is ok, because we know the chunk is complete.
                load_unsafe<AccT>(vec, vec_chunk, j + K_offset);

                int mat_offset = 0;
                //MLX_MTL_PRAGMA_UNROLL
                for (short m = 0; m < ROWS_PER_THREAD; m++) {
                    // Unsafe is ok, because we know the chunk is complete.
                    load_unsafe(mat, row_chunk, mat_offset + j + K_offset);

                    // Accumulate results
                    //MLX_MTL_PRAGMA_UNROLL
                    //for (short k = 0; k < COL_BLOCK_SIZE; k++) {
                    //    resM[m] += row_chunk[k] * vec_chunk[k];
                    //}
                    resM[m] += row_chunk[0] * vec_chunk[0];
                    resM[m] += row_chunk[1] * vec_chunk[1];
                    resM[m] += row_chunk[2] * vec_chunk[2];
                    resM[m] += row_chunk[3] * vec_chunk[3];
                    resM[m] += row_chunk[4] * vec_chunk[4];
                    resM[m] += row_chunk[5] * vec_chunk[5];
                    resM[m] += row_chunk[6] * vec_chunk[6];
                    resM[m] += row_chunk[7] * vec_chunk[7];

                    mat_offset += num_cols;
                }

                K_offset += threads_per_threadgroup.x * COL_BLOCK_SIZE;
            }

            // If column count isn't divisible into a full block, handle the leftovers.
            if (leftover > 0) {
                // Use safe because this can overflow past the last column.
                load_safe<AccT>(vec, vec_chunk, j+K_offset, num_cols);

                //MLX_MTL_PRAGMA_UNROLL
                for (short m = 0; m < ROWS_PER_THREAD; m++) {
                    // Use safe because this can overflow past the last column.
                    load_safe(&mat[m*num_cols], row_chunk, j+K_offset, num_cols);

                    //MLX_MTL_PRAGMA_UNROLL
                    for (short k = 0; k < COL_BLOCK_SIZE; k++) {
                        resM[m] += row_chunk[k] * vec_chunk[k];
                    }
                    // Manually unrolling both in load_safe and here is a lot slower.
                    // Either or neither is ~same.
                    //resM[m] += row_chunk[0] * vec_chunk[0];
                    //resM[m] += row_chunk[1] * vec_chunk[1];
                    //resM[m] += row_chunk[2] * vec_chunk[2];
                    //resM[m] += row_chunk[3] * vec_chunk[3];
                    //resM[m] += row_chunk[4] * vec_chunk[4];
                    //resM[m] += row_chunk[5] * vec_chunk[5];
                    //resM[m] += row_chunk[6] * vec_chunk[6];
                    //resM[m] += row_chunk[7] * vec_chunk[7];
                }
            }

            // Thread 0 gets the sum of all resM[0], Thread 1 all resM[1], ... and so forth.
            //MLX_MTL_PRAGMA_UNROLL
            for (short i = 0; i < SN; ++i) {
                resM[i] = simd_sum(resM[i]); // technically only thread i needs this.
            }

            simdgroup_barrier(mem_flags::mem_none);
            R = resM[thread_position_in_threadgroup.x];
        }
    """
    # threads_per_simdgroup: 32
    # thread_execution_width: 32
    # simdgroups_per_threadgroup: 24
    # quadgroups_per_threadgroup: 192 # ceil(threads_per_threadgroup/4)
    # threads_per_grid: [768,1,1] # as set below
    kernel = mx.fast.metal_kernel(
        name="compute_s",
        input_names=["W", "L", "C"],
        output_names=["Q", "S", "Werr"],
        header=header,
        source=source,
    )

    outputs = kernel(
        inputs=[W,L.T,C],
        grid=(num_rows, 1, 1), # threads_per_grid
        threadgroup=(32, 1, 1), # threads_per_threadgroup (1 simdgroup)
        output_shapes=[(num_rows, num_cols), (num_rows, num_values, num_cols), (num_rows, num_cols)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32],
        init_value=0, # Lot of outputs, better to let MLX do it.
    )

    return outputs[0].astype(mx.int64), outputs[1]

def find_nearest_codebook_indices(W: mx.array, R: mx.array, C: mx.array):
    """
    Compute argmin_s |W + R - C_:,s| for each row of W.

    Args:
        W: Shape [num_rows], the weight column
        R: Shape [num_rows], the residual
        C: Shape [num_rows, num_values], the codebook per-column

    Returns:
        indices: Shape [num_rows], index of closest value in C for each row
        S: Shape [num_rows, num_values], one-hot encoding of the closest codebook index
        Werr: Shape [num_rows], the error of the closest codebook index
    """
    num_rows, num_values = C.shape

    source = """
        uint row_idx = thread_position_in_grid.x;
        uint num_values = C_shape[1];

        float wr_value = W[row_idx] + R[row_idx];
        uint codebook_start_idx = row_idx * num_values;

        float min_distance = INFINITY;
        uint min_idx = 0;

        for (uint col = 0; col < num_values; col++) {
            float codebook_value = C[codebook_start_idx + col];
            float distance = metal::abs(wr_value - codebook_value);

            if (distance < min_distance) {
                min_distance = distance;
                min_idx = col;
            }
        }

        indices[row_idx] = min_idx;
        S[codebook_start_idx + min_idx] = 1;
        Werr[row_idx] = W[row_idx] - C[codebook_start_idx + min_idx];
    """
    kernel = mx.fast.metal_kernel(
        name="find_nearest_codebook_indices",
        input_names=["W", "R", "C"],
        output_names=["indices", "S", "Werr"],
        source=source,
    )

    outputs = kernel(
        inputs=[W,R,C],
        grid=(num_rows, 1, 1), # threads_per_grid
        threadgroup=(min(256, num_rows), 1, 1), # threads_per_threadgroup
        output_shapes=[(num_rows,), (num_rows, num_values), (num_rows,)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32],
        init_value=0,  # Manually zero-ing S in the kernel is faster for small problems.
    )

    return outputs[0], outputs[1], outputs[2]

@mx.compile
def solve_for_s(W, L, Q, S, T, r, num_rows, num_cols):
    """1:1 MLX version of the torch algorithm."""
    row_indices = mx.arange(num_rows)

    # Paper: for j ← n-1 to 0 do
    for j in range(num_cols-1, -1, -1):
        # Paper: idx = argmin_s |W_:,j + r/L_j,j - T^k_:,s| # row-wise
        effective_w = W[:, [j]] + r /  L[j, j] # [num_rows, 1]

        # Find the closest codebook value for each row
        distances = mx.abs(effective_w - T)     # [num_rows, num_values]
        indices = mx.argmin(distances, axis=1)  # [num_rows]

        # Paper: Q^(k+1)_:,j = idx
        Q[:, j] = indices

        # Paper: Update S^(k+1)_:,:,j using idx # one-hot encoding
        S[:, :, j] = 0
        S[row_indices, indices, j] = 1

        # T: (m, 2^N), S: (m, 2^N, n), W: (m, n), L: (n, n)
        # Paper: r = (W_:,j: - T^k S^(k+1)_:,:,j:)L_j:,j # update residual
        Wq = mx.take_along_axis(T, Q[:, j:], axis=1)
        r = mx.matmul(W[:, j:] - Wq, L[j:, [j-1]]) # j-1, not j, see equation 20.

    return Q, S

@mx.compile
def solve_for_s_fast(W, L, Q, S, T, r, num_rows, num_cols):
    LT = L.T

    # Paper: for j ← n-1 to 0 do
    Werr = mx.zeros_like(W)
    WerrT = mx.zeros_like(W).T
    for j in range(num_cols-1, -1, -1):
        # Paper: idx = argmin_s |W_:,j + r/L_j,j - T^k_:,s| # row-wise
        indices, Sj, Werrj = find_nearest_codebook_indices(W[:, j], r / L[j,j], T) # [num_rows], [num_rows, num_values], [num_rows]
        S[:,:,j] = Sj # Paper: Update S^(k+1)_:,:,j using idx (one-hot)
        Q[:, j] = indices # Paper: Q^(k+1)_:,j = idx

        # T: (m, 2^N), S: (m, 2^N, n), W: (m, n), L: (n, n)
        # Paper: r = (W_:,j: - T^k S^(k+1)_:,:,j:)L_j:,j # update residual
        # Werr[:, j] = Werrj
        # r = mx.matmul(Werr[:, j:], L[j:, [j-1]]) # j-1, not j, see equation 20.

        # Row-major for better caching (? I think. Either way small boost but easy).
        WerrT[j, :] = Werrj
        r = mx.matmul(LT[[j-1], j:], WerrT[j:, :]).T

    return Q, S

def solve_for_t(W, H, S):
    """
    Solve for T using MLX
    T^(k+1) = WH(S^(k+1))^T((S^(k+1))H^T(S^(k+1))^T)^†
    """
    assert False, "incorrect results for this method, check before using."
    damp =  mx.eye(S.shape[1]) * 1e-4
    return mx.linalg.solve(
        S @ H @ S.swapaxes(1,2) + damp, S @ (W@H)[:, :, None], stream=mx.cpu).squeeze(-1)

def quad_loss_2(W, Q, G):
    """from gptqv"""
    Werr = W - Q
    return (Werr.mm(G) * Werr).sum()

class GANQ(GPTQ):
    """
    Quantize following "GANQ: GPU-Adaptive Layer-Wise LUT-Based Non-Uniform Quantization".
    """

    def __init__(self, module, qcfg=None):
        super().__init__(module, qcfg)
        self.iterations = getattr(self.qcfg, "ganq_iterations", 5)

    def _initialize_codebook_linear(self, num_rows, num_bits, device):
        """Initialize with uniform values between -1 and 1"""
        num_values = 2**num_bits
        return torch.linspace(-1, 1, num_values, device=device).unsqueeze(0).expand(num_rows, -1) # [num_rows, 2^num_bits]

    def _initialize_codebook_normal(self, num_rows, num_bits, device, mean=0.0, std=1.0):
        """Initialize with values representing evenly spaced quantiles of a normal distribution"""
        num_values = 2**num_bits
        probs = torch.linspace(0, 1, num_values + 2, device=device)[1:-1]  # exclude 0 and 1 to avoid inf
        quantiles = torch.erfinv(2 * probs - 1) * math.sqrt(2)
        quantiles = torch.nan_to_num(quantiles, nan=0.0, posinf=4.0, neginf=-4.0)

        codebook = quantiles.unsqueeze(0).expand(num_rows, -1)
        codebook = codebook * std + mean

        return codebook

    def _initialize_codebook_kmeans(self, W, Hinv, num_bits, device):
        """Initialize using k-means weighted by H^-1 diagonal.
        See LeanQuant (https://arxiv.org/abs/2407.10032)"""
        kmeans_tasks = []
        exp = 4 # per paper 3 or 4 is good.
        W_np = W.cpu().numpy()
        Hinv_diagonal_np = (torch.diagonal(Hinv) ** (-exp)).cpu().numpy()
        for j in range(W_np.shape[0]):
            kmeans_tasks.append((W_np[j, :, None], Hinv_diagonal_np, 2 ** num_bits, 42))

        num_threads = min(os.cpu_count() or 1, 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            kmeans_results = list(tqdm(executor.map(kmeans_fit, kmeans_tasks), total=len(kmeans_tasks)))

        centroids = torch.from_numpy(np.stack(kmeans_results)).reshape(W.shape[0], 2 ** num_bits).to(W.device)
        return centroids

    def solve_via_mlx(self, W, H, L, Q, S, T):
        W = to_mlx(W)
        # H = to_mlx(H)
        L = to_mlx(L)
        # Q = to_mlx(Q)
        # S = to_mlx(S)
        T = to_mlx(T)
        # r = mx.zeros((W.shape[0], 1))
        # These are compiled, so easiest if they are not methods on the class.
        # Q, S = solve_for_s(W, L, Q, S, T, r, W.shape[0], W.shape[1]) # just compiled MLX (better than torch MPS)
        # Q, S = solve_for_s_fast(W, L, Q, S, T, r, W.shape[0], W.shape[1]) # kernel for just codebook lookup (helps a lot)
        Q,S = compute_s(W,L,T) # fused kernel (helps a ton)
        # T = solve_for_t(W, H, S) # Sadly, this is not numerically stable or maybe buggy.
        return [from_mlx(o) for o in [Q,S]]

    @torch.no_grad()
    def _perform_quantization_loop(self, W, Hinv, blocksize, perm=None, invperm=None):
        """
        Algorithm 1 from "GANQ: GPU-Adaptive Layer-Wise LUT-Based Non-Uniform Quantization"

        Input:  W in R^(m x n), X in R^(n x p), initial codebook T0 in R^(m x 2^N),
                number of iterations K
        Output: Updated T^K and query matrix Q^K in {0, 2^N - 1}^(m x n)

        Initialize S0 = 0^(m x 2^N x n)                           # tensor format
        Compute H = X * X^T
        Compute L = Cholesky(H)                                   # Cholesky decomposition
        for k = 0 to K - 1 do
            Initialize r = 0^(m x 1)                              # previous residual vector
            for j = n - 1 to 0 do
                idx = argmin_s ||W[:,j] + r/L[j,j] - T[:,s]^k||   # row-wise
                Q[:,j]^(k+1) = idx
                Update S[:,j,s]^(k+1) using idx                   # one-hot encoding
                r = (W[:,j:] - T^k * S[:,:,j:]^(k+1)) * L[j:,j-1] # update residual
            end for
            T^(k+1) = W * H * (S^(k+1))^T * ((S^(k+1) * H^T * (S^(k+1))^T)^+) # batch update pseudo-inverse
        end for
        Return T^K, Q^K
        """

        start = time.perf_counter()
        # Input: W ∈ ℝᵐ×ⁿ, X ∈ ℝⁿ×ᵖ, initial codebook T⁰ ∈ ℝᵐ×²ᴺ, number of iterations K
        num_rows, num_cols = W.shape   # W ∈ ℝᵐ×ⁿ
        num_bits = self.qcfg.bits      # N bits for quantization
        num_values = 2**num_bits       # 2^N possible values in codebook

        # Q^K ∈ {0, 2^N - 1}^(m×n)
        Q = torch.zeros_like(W, dtype=torch.long)  # Stores indices into codebook (T)

        # Not supported, here for compatibility.
        scale = []
        zero = []
        if self.qcfg.group_size != -1:
            self.quantizer.find_params(W, weight=True)
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        # Paper: Initialize T⁰ ∈ ℝᵐ×²ᴺ (initial codebook)
        # T = self._initialize_codebook_normal(num_rows, num_bits, W.device) # Aligning std/mean to weight doesn't seem to help.
        # T = self._initialize_codebook_linear(num_rows, num_bits, W.device)
        # T = T * W.max(dim=1).values.unsqueeze(1) # Align initial codebooks to row maximums. TODO: Why is this ok for non-linear initialization?
        T = self._initialize_codebook_kmeans(W, Hinv, num_bits, W.device) # LeanQuant-style.
        assert T.shape == (num_rows, num_values), f"Expected shape ({num_rows=}, {num_values=}), got {T.shape=}"

        # Paper: Initialize S⁰ = 0^(m×2^N×n) # tensor format (one-hot encoding)
        S = torch.zeros(num_rows, num_values, num_cols, device=W.device)

        # Paper: Compute H = XX^T
        H = self.Xxt_damped  # computed in superclass
        assert H.shape == (num_cols, num_cols), f"Expected shape ({num_cols=}, {num_cols=}), got {H.shape=}"

        # Paper: Compute L = Cholesky(H) # Cholesky decomposition
        L = self.L  # computed in superclass
        assert L.shape == (num_cols, num_cols), f"Expected shape ({num_cols=}, {num_cols=}), got {L.shape=}"

        Wadj = torch.empty_like(W) if not USE_MLX else None
        best = (float('inf'), None, None) # dist, T, Q

        if USE_MLX:
            # These are constant throughout, convert once.
            W_mlx = to_mlx(W)
            L_mlx = to_mlx(L)

        # Paper: for k ← 0 to K-1 do
        print("init", time.perf_counter() - start, "sec")
        for k in range(self.iterations):
            start = time.perf_counter()

            if USE_MLX:
                Q, S = self.solve_via_mlx(W_mlx, H, L_mlx, Q, S, T)
                Q, S = Q.to(W.device), S.to(W.device)
                # print("mlx loop time: ", time.perf_counter() - start, "sec")
            else:
                # Paper: Initialize residual r = 0^(m×1)
                r = torch.zeros(num_rows, 1, device=W.device)
                # Paper: for j ← n-1 to 0 do
                for j in range(num_cols-1, -1, -1):
                    w_j = W[:, j].unsqueeze(-1) # unsqueeze faster than W[:, [j]] on MPS

                    # Paper: idx = argmin_s |W_:,j + r/L_j,j - T^k_:,s| # row-wise
                    L_jj = L[j, j]
                    # assert L_jj > 1e-8 # Not sure if eps needed. Slow on MPS.
                    effective_w = w_j + r / L_jj # [num_rows]
                    Wadj[:, j] = effective_w.squeeze()

                    # Find the closest codebook value for each row
                    distances = torch.abs(effective_w - T)    # [num_rows, num_values]
                    indices = torch.argmin(distances, dim=1)  # [num_rows]

                    # Paper: Q^(k+1)_:,j = idx
                    Q[:, j] = indices

                    # Paper: Update S^(k+1)_:,:,j using idx # one-hot encoding
                    S[:, :, j] = 0
                    # scatter_ does not work on MPS, nothing is updated.
                    # https://github.com/pytorch/pytorch/issues/115152
                    # ones = torch.ones(num_rows, device=W.device)
                    # S[:, :, j].scatter_(1, indices.unsqueeze(1), ones.unsqueeze(1))
                    S[torch.arange(num_rows), indices, j] = 1

                    # T: (m, 2^N), S: (m, 2^N, n)
                    # Paper: r = (W_:,j: - T^k S^(k+1)_:,:,j:)L_j:,j # update residual
                    # S is one hot, so sum selects the non-zero entry.
                    # Wq = torch.sum(T.unsqueeze(2) * S[:, :, j:], dim=1) # << leaks memory on MPS and very slow.
                    Wq = T.gather(1, Q[:, j:]) # wicked fast on CPU
                    r = (W[:, j:] - Wq)@L[j:, j-1].unsqueeze(-1) # j-1, not j, See equation 20.
                    # assert r.shape == (num_rows, 1), f"Expected shape ({num_rows=}, 1), got {r.shape=}"
                    # print("r", r[0], L[j,j])
                # print("torch loop time", time.perf_counter() - start, "sec")

            # Paper: T^(k+1) = WH(S^(k+1))^T((S^(k+1))H^T(S^(k+1))^T)^† # batch update
            assert S.shape == (num_rows, num_values, num_cols), f"Expected shape ({num_rows=}, {num_values=}, {num_cols=}), got {S.shape=}"

            # Per docs lstsq is both faster and more numerically stable.
            # However, our matrix is ill-conditioned so we can only use
            # it on CPU where gelsd is available.
            cpu_fallback_enabled = W.device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "") == "1"
            mode = "least_squares" if cpu_fallback_enabled or W.device.type == "cpu" else "manual"
            if mode == "least_squares":
                # Derivation of lstsq rearrangement:
                # T_new = W @ H @ S.T @ (S @ H.T @ S.T)†
                # T_new = A @ (B)†       # say A = (W @ H @ S.T), B = (S @ H.T @ S.T)
                # T_new.T = (B)†.T @ A.T
                # T_new.T = (B.T)† @ A.T # pseudoinverse commutes with transposition
                # torch.linalg.lstsq(A, B).solution == A.pinv() @ B
                # T_new.T = lstsq(B.T, A.T)
                # T_new = lstsq(B.T, A.T).T
                # T_new = lstsq(S @ H @ S.T, S @ H.T @ W.T).T
                # T_new = lstsq(S @ H @ S.T, S @ (W @ H).T).T
                T_new = torch.linalg.lstsq(
                    S @ H @ S.mT, S @ (W@H).unsqueeze(1).mT,
                    driver="gelsd").solution.mT.squeeze(-2) # gelsd for ill-conditioned matrices
            elif mode == "descent":
                T_new = self.optimize_t(W, H, Q, T)
            else:
                # (S^(k+1))H^T(S^(k+1))^T
                ST = S.mT      # [num_rows, num_cols, num_values]
                SHST = torch.bmm(
                    S @ H.mT,  # [num_rows, num_values, num_cols]
                    ST         # [num_rows, num_cols, num_values]
                )              # [num_rows, num_values, num_values]

                # ((S^(k+1))H^T(S^(k+1))^T)^†
                SHST_pinv = torch.pinverse(SHST)

                # WH(S^(k+1))^T
                WHST = torch.bmm(
                    # TODO: Is this unsqueeze right?
                    (W@H).unsqueeze(1),  # [num_rows, 1, num_cols]
                    ST                   # [num_rows, num_cols, num_values]
                ).squeeze(1)             # [num_rows, num_values]

                # Paper: T^(k+1) = WH(S^(k+1))^T((S^(k+1))H^T(S^(k+1))^T)^†
                T_new = torch.bmm(
                    WHST.unsqueeze(1),  # [num_rows, 1, num_values]
                    SHST_pinv           # [num_rows, num_values, num_values]
                ).squeeze(1)            # [num_rows, num_values]

            assert T_new.shape == (num_rows, num_values), f"Expected shape ({num_rows}, {num_values}), got {T_new.shape}"
            T = T_new

            Wq = T.gather(1, Q)
            curr_dist = quad_loss_2(W, Wq, H)
            print("loop dist", curr_dist, time.perf_counter() - start, "sec via", "mlx" if USE_MLX else "torch")

            if curr_dist.item() < best[0]:
                best = (curr_dist.item(), T, Q)

        # Paper: Return T^K, Q^K

        # Convert to quantized weight matrix, Wq
        # Wq[i,j] = T[i, Q[i,j]]

        (_, T, Q) = best
        Wq = T.gather(1, Q)

        # Match the GPTQ loss almost (W in GPTQ is the adjusted weight).
        d = Hinv.diag()
        Losses = ((W - Wq) ** 2) / d**2 / 2

        # Unused, compatibility with interface.
        if not scale:
            self.quantizer.find_params(W, weight=True)
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        return Wq, Losses, scale, zero

    def make_quantized_weight(self, Q, T):
        return T.gather(1, Q)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def optimize_t(self, W, H, Q, T):
        """ optimize T via gradient descent, a la gptqv"""
        # inference mode :eyeroll:
        W = W.clone()
        H = H.clone()
        Q = Q.clone()
        T = T.clone()

        tick = time.perf_counter()
        with torch.no_grad():
            offset = (W.mm(H) * W).sum()
            Wq = self.make_quantized_weight(Q,T)
            orig_loss = quad_loss_2(W, Wq, H)
            snr_before = 10 * np.log10(offset.item() / orig_loss.item())

        must_restart = True
        lr = 1e-3

        all_centroids = T
        while must_restart:
            orig_centroids = all_centroids.clone()
            all_centroids.requires_grad_()
            o = torch.optim.Adam([all_centroids], lr=lr)
            for _ in range(25):
                must_restart = False
                o.zero_grad()
                Wq = self.make_quantized_weight(Q, all_centroids)
                loss = quad_loss_2(W, Wq, H)
                if loss > orig_loss or torch.isnan(loss):
                    lr *= 1e-1
                    print(f"Inner loop: Restarting M-step with lr={lr:.2e}")
                    must_restart = True
                    all_centroids = orig_centroids
                    break
                loss.backward()
                o.step()

            if not must_restart:
                new_all_centroids = all_centroids
                Wq = self.make_quantized_weight(Q, new_all_centroids)
                loss = quad_loss_2(W, Wq, H)
                if torch.isnan(loss):
                    lr *= 1e-1
                    print(f"Outer loop: Restarting M-step with lr={lr:.2e}")
                    must_restart = True
                    all_centroids = orig_centroids
                    continue

                del orig_centroids
                # print(
                #     f"time M-step SGD {(time.perf_counter() - tick):.2f}; final loss: {loss.item():.4f}"
                # )
                orig_loss = quad_loss_2(W, Wq, H)
                snr_after = 10 * np.log10(offset.item() / orig_loss.item())

                # print(f"improvement: {snr_before:.2f} -> {snr_after:.2f}")

        return all_centroids.detach()

__all__ = ["GANQ"]
