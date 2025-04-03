import math
import os
import time
import concurrent.futures

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mlx.core as mx
    USE_MLX = True
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

def to_mlx(x: torch.Tensor):
    return mx.array(x.cpu().numpy())
def from_mlx(x: mx.array):
    return torch.from_numpy(np.array(x))

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
        init_value=0, # Avoid manually zero-ing one-hot outputs.
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
    row_indices = mx.arange(num_rows)
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
        H = to_mlx(H)
        L = to_mlx(L)
        Q = to_mlx(Q)
        S = to_mlx(S)
        T = to_mlx(T)
        r = mx.zeros((W.shape[0], 1))
        # These are compiled, so easiest if they are not methods on the class.
        # Q, S = solve_for_s(W, L, Q, S, T, r, W.shape[0], W.shape[1])
        Q, S = solve_for_s_fast(W, L, Q, S, T, r, W.shape[0], W.shape[1])
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

        start = time.time()
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

        # Paper: for k ← 0 to K-1 do
        print("init", time.time() - start, "sec")
        for k in range(self.iterations):
            start = time.time()

            if USE_MLX:
                Q, S = self.solve_via_mlx(W, H, L, Q, S, T)
                Q, S = Q.to(W.device), S.to(W.device)
                # print("mlx loop time: ", time.time() - start, "sec")
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
                # print("torch loop time", time.time() - start, "sec")

            # Paper: T^(k+1) = WH(S^(k+1))^T((S^(k+1))H^T(S^(k+1))^T)^† # batch update
            assert S.shape == (num_rows, num_values, num_cols), f"Expected shape ({num_rows=}, {num_values=}, {num_cols=}), got {S.shape=}"

            # Per docs lstsq is both faster and more numerically stable.
            # However, our matrix is ill-conditioned so we can only use
            # it on CPU where gelsd is available.
            mode = "least_squares" if True and W.device == torch.device("cpu") else "manual"
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
            print("loop dist", curr_dist, time.time() - start, "sec via", "mlx" if USE_MLX else "torch")

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

        tick = time.time()
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
                #     f"time M-step SGD {(time.time() - tick):.2f}; final loss: {loss.item():.4f}"
                # )
                orig_loss = quad_loss_2(W, Wq, H)
                snr_after = 10 * np.log10(offset.item() / orig_loss.item())

                # print(f"improvement: {snr_before:.2f} -> {snr_after:.2f}")

        return all_centroids.detach()

__all__ = ["GANQ"]
