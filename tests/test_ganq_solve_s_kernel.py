import unittest
import mlx.core as mx
import time
import os

class TestGANQSolveKernel(unittest.TestCase):
    def setUp(self):
        mx.random.seed(42)
        self.m, self.v, self.n = 768, 16, 768*3
        self.W = mx.random.normal((self.m, self.n))
        self.L = mx.tril(mx.random.normal((self.n, self.n)))
        self.C = mx.random.normal((self.m, self.v))
        mx.eval(self.W, self.L, self.C)

    def test_compute_s(self):
        base_results = looper_base(self.W, self.L, self.C)
        base_Q, base_S, base_Werr = base_results

        # custom kernel
        kernel_results = looper_compute_s(self.W, self.L, self.C)
        kernel_Q, kernel_S, kernel_Werr = kernel_results

        self.assertTrue(mx.allclose(base_Werr, kernel_Werr).all())
        self.assertTrue(mx.equal(base_Q, kernel_Q).all())
        self.assertTrue(mx.equal(base_S, kernel_S).all())

    def test_timing_comparison(self):
        # Warmup
        for _ in range(5):
            mx.eval(looper_base(self.W, self.L, self.C))
            mx.eval(looper_compute_s(self.W, self.L, self.C))

        # base implementation
        num_iters = 10
        tic = time.perf_counter()
        for _ in range(num_iters):
            mx.eval(looper_base(self.W, self.L, self.C))
        toc = time.perf_counter()
        base_time = 1e3 * (toc - tic) / num_iters

        # kernel implementation
        tic = time.perf_counter()
        for _ in range(num_iters):
            mx.eval(looper_compute_s(self.W, self.L, self.C))
        toc = time.perf_counter()
        kernel_time = 1e3 * (toc - tic) / num_iters

        # no need to assert
        print(f"\nPerformance:")
        print(f"Base implementation: {base_time:.5f} msec")
        print(f"Kernel implementation: {kernel_time:.5f} msec")

    def test_trace_kernel(self):
        mx.eval(self.W, self.L, self.C)
        traceName = f"solve-s.gputrace"
        mx.metal.start_capture(traceName)
        for _ in range(3):
            mx.eval(*looper_compute_s(self.W, self.L, self.C))
        mx.metal.stop_capture()

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
        constant int THREADGROUP_CACHE_SIZE = 32;

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
            //for (int tn = 0; tn < TN; tn++) {
            //    dst[tn] = static_cast<U>(src[src_offset + tn]);
            //}
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
                    //dst[0] = static_cast<U>(src[src_offset + 0]);
                    //dst[1] = static_cast<U>(src[src_offset + 1]);
                    //dst[2] = static_cast<U>(src[src_offset + 2]);
                    //dst[3] = static_cast<U>(src[src_offset + 3]);
                    //dst[4] = static_cast<U>(src[src_offset + 4]);
                    //dst[5] = static_cast<U>(src[src_offset + 5]);
                    //dst[6] = static_cast<U>(src[src_offset + 6]);
                    //dst[7] = static_cast<U>(src[src_offset + 7]);
                } else { // Edgecase
                    for (int tn = 0; tn < TN; tn++) {
                        dst[tn] = src_offset + tn < src_size
                            ? static_cast<U>(src[src_offset + tn])
                            : U(0);
                    }
                    //dst[0] = src_offset + 0 < src_size ? static_cast<U>(src[src_offset + 0]) : U(0);
                    //dst[1] = src_offset + 1 < src_size ? static_cast<U>(src[src_offset + 1]) : U(0);
                    //dst[2] = src_offset + 2 < src_size ? static_cast<U>(src[src_offset + 2]) : U(0);
                    //dst[3] = src_offset + 3 < src_size ? static_cast<U>(src[src_offset + 3]) : U(0);
                    //dst[4] = src_offset + 4 < src_size ? static_cast<U>(src[src_offset + 4]) : U(0);
                    //dst[5] = src_offset + 5 < src_size ? static_cast<U>(src[src_offset + 5]) : U(0);
                    //dst[6] = src_offset + 6 < src_size ? static_cast<U>(src[src_offset + 6]) : U(0);
                    //dst[7] = src_offset + 7 < src_size ? static_cast<U>(src[src_offset + 7]) : U(0);
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

            // TODO: Pre-computing the W*L terms?
            // W[0, 4:] @ L[4:, 3] = W[4]*L[4,3] = (W[4]-C[4])*L[4,3] = W[4]*L[4,3]-C[4]*L[4,3]
            // W[0, 3:] @ L[3:, 2] = W[3]*L[3,2] + W[4]*L[4,2] = W[3]*L[3,2] - C[3]*L[3,2] + W[4]*L[4,2] - C[4]*L[4,2]
            //                     = (W[3]*L[3,2] + W[4]*L[4,2]) - C[4]*L[4,2] - C[3]*L[3,2]
            // W[0, 2:] @ L[2:, 1] = W[2]*L[2,1] + W[3]*L[3,1] + W[4]*L[4,1] = W[2]*L[2,1] + W[3]*L[3,1] + W[4]*L[4,1] - C[4]*L[4,1]
            // W[0, 1:] @ L[1:, 0] = W[1]*L[1,0] + W[2]*L[2,0] + W[3]*L[3,0] + W[4]*L[4,0] = W[1]*L[1,0] + W[2]*L[2,0] + W[3]*L[3,0] + W[4]*L[4,0] - C[4]*L[4,0]

            // Compute the residual. Werr[row_idx, j:] @ L[j:, j-1]

            ///////////////////////////////////////
            ////// Approach 1 - Naive Loop - Slow
            ///////////////////////////////////////

            /*
            R = row_err * L[j*L_shape[0] + j-1]; // L[j,j-1]
            for (uint k = j+1; k < L_shape[0]; k++) {
                float Lkj1 = L[k*L_shape[0] + j-1];
                R += Werr[W_start_idx + k] * Lkj1;
            }
            */

            ///////////////////////////////////////
            ////// Approach 2 - ThreadGroup L Cache - Fast for small num_cols, slow for large.
            ///////////////////////////////////////
            /*
            threadgroup float Lcol[THREADGROUP_CACHE_SIZE];

            // Loop from row j in L to end.
            R = 0;
            uint chunks = ceil((float)(L_shape[0] - j) / THREADGROUP_CACHE_SIZE);
            for (int chunk_idx = 0; chunk_idx < chunks; chunk_idx ++) {
                // Load this chunk of L into threadgroup memory.
                for (uint stride = 0; stride < ceil(1.*THREADGROUP_CACHE_SIZE / threads_per_threadgroup.x); stride++) {
                    uint group_idx = thread_position_in_threadgroup.x + (stride*threads_per_threadgroup.x);
                    uint Lrow_idx = (chunk_idx * THREADGROUP_CACHE_SIZE) + j + group_idx;
                    if (group_idx < THREADGROUP_CACHE_SIZE && Lrow_idx < num_cols) {
                        Lcol[group_idx] = L[Lrow_idx*num_cols + j-1];
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Sum in each thread.
                uint Wcol_start = (chunk_idx * THREADGROUP_CACHE_SIZE) + j;
                for (uint l = 0; l < THREADGROUP_CACHE_SIZE && (Wcol_start + l) < num_cols; l++) {
                    float err = chunk_idx == 0 && l == 0 ? row_err : Werr[W_start_idx + Wcol_start + l];
                    R += err * Lcol[l];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            */


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

            ///////////////////////////////////////
            ////// Approach 4 -  Modified MLX GEMV with row splits
            ///////////////////////////////////////

            /*
            // Approach 3, but rows are processed in chunks instead of 32 at once.
            // Slower.

            thread float resM[ROWS_PER_THREAD] = {0};
            thread float vec_chunk[COL_BLOCK_SIZE] = {0};
            thread float row_chunk[COL_BLOCK_SIZE] = {0}; // tmp holding for row columns

            int K = num_cols - j;

            int ROW_DIV = 1; // 32 / this

            // assert ROWS_PER_THREAD >= threads_per_threadgroup == threads_per_simdgroup // assuming this.
            int num_chunks = K / threads_per_threadgroup.x / COL_BLOCK_SIZE;
            int leftover = K - (num_chunks * threads_per_threadgroup.x * COL_BLOCK_SIZE);

            const device float* mat = &Werr[threads_per_threadgroup.x * threadgroup_position_in_grid.x * num_cols];
            const device float* vec = &L[(j-1)*num_cols]; // Requires L to be transposed before kernel.

            // Each thread process one chunk at a time.
            // Chunks assignments are interleaved across threads,
            // so all chunks that are being processed are adjacent.
            for (int row_offset = 0; row_offset < threads_per_threadgroup.x; row_offset += threads_per_threadgroup.x/ROW_DIV) {
                int K_offset = thread_position_in_threadgroup.x * COL_BLOCK_SIZE;
                for (int i = 0; i < num_chunks; i++) {
                    // Unsafe is ok, because we know the chunk is complete.
                    load_unsafe<AccT>(vec, vec_chunk, j + K_offset);

                    int mat_offset = num_cols*row_offset;
                    MLX_MTL_PRAGMA_UNROLL
                    for (int m = row_offset; m < row_offset+ROWS_PER_THREAD/ROW_DIV; m++) {
                        // Unsafe is ok, because we know the chunk is complete.
                        load_unsafe(mat, row_chunk, mat_offset + j + K_offset);

                        // Accumulate results
                        MLX_MTL_PRAGMA_UNROLL
                        for (int k = 0; k < COL_BLOCK_SIZE; k++) {
                            resM[m] += row_chunk[k] * vec_chunk[k];
                        }

                        mat_offset += num_cols;
                    }

                    K_offset += threads_per_threadgroup.x * COL_BLOCK_SIZE;
                }

                // If column count isn't divisible into a full block, handle the leftovers.
                if (leftover > 0) {
                    // Use safe because this can overflow past the last column.
                    load_safe<AccT>(vec, vec_chunk, j+K_offset, num_cols);

                    MLX_MTL_PRAGMA_UNROLL
                    for (int m = row_offset; m < row_offset+ROWS_PER_THREAD/ROW_DIV; m++) {
                        // Use safe because this can overflow past the last column.
                        load_safe(&mat[m*num_cols], row_chunk, j+K_offset, num_cols);

                        MLX_MTL_PRAGMA_UNROLL
                        for (int k = 0; k < COL_BLOCK_SIZE; k++) {
                            resM[m] += row_chunk[k] * vec_chunk[k];
                        }
                    }
                }
            }

            // Thread 0 gets the sum of all resM[0], Thread 1 all resM[1], ... and so forth.
            MLX_MTL_PRAGMA_UNROLL
            for (int i = 0; i < SN; ++i) {
                resM[i] = simd_sum(resM[i]); // technically only thread i needs this.
            }

            simdgroup_barrier(mem_flags::mem_none);
            R = resM[thread_position_in_threadgroup.x];
            */
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

    return outputs[0], outputs[1], outputs[2]

@mx.compile
def base(W, R, C):
    wr = W + R
    distances = mx.abs(wr - C)
    indices = mx.argmin(distances, axis=1)
    S = mx.zeros((C.shape[0], C.shape[1]))
    S[mx.arange(S.shape[0]), indices] = 1
    Wq = mx.take_along_axis(C, indices[:, None], axis=1)
    Werr = W - Wq
    return indices, S, Werr.squeeze()

@mx.compile
def looper_base(W, L, C):
    Q = mx.zeros_like(W).astype(mx.uint32)
    S = mx.zeros((C.shape[0], C.shape[1], W.shape[1])).astype(mx.uint32)
    R = mx.zeros((W.shape[0], 1))
    Werr = mx.zeros_like(W)
    for j in range(W.shape[1]-1, -1, -1):
        indices, Sj, Werrj = base(W[:, [j]], R / L[j,j], C)
        S[:,:,j] = Sj
        Q[:, j] = indices
        Werr[:, j] = Werrj
        # R = Werrj[:, None]
        R = mx.matmul(Werr[:, j:], L[j:, [j-1]])

        # Different way of computing the same thing (closer to how the kernel does it).
        # R2 = (Werr[:, j:] * L[j:, [j-1]].T).sum(axis=-1, keepdims=True)
        # if not mx.equal(R, R2).all():
        #     print("iter", j, "diff", (R-R2).abs().max())
        #     print(Werr, L)

    return Q, S, Werr

@mx.compile
def looper_compute_s(W, L, C):
    Q, S, Werr = compute_s(W, L, C)
    return Q, S, Werr
