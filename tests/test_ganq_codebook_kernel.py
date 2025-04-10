import unittest
import mlx.core as mx
import time
import os

class TestGANQCodebookKernel(unittest.TestCase):
    def setUp(self):
        mx.random.seed(42)
        self.m, self.v, self.n = 768, 16, 768
        self.W = mx.random.normal((self.m, self.n))
        self.R = mx.random.normal((self.m,))
        self.C = mx.random.normal((self.m, self.v))
        mx.eval(self.W, self.R, self.C)

    def test_find_nearest_codebook_indices(self):
        base_results = looper_base(self.W, self.R, self.C)
        base_indices, base_S, base_Werr = base_results

        # custom kernel
        kernel_results = looper_find_nearest_codebook_indices(self.W, self.R, self.C)
        kernel_indices, kernel_S, kernel_Werr = kernel_results

        self.assertTrue(mx.equal(base_indices, kernel_indices).all())
        self.assertTrue(mx.equal(base_S, kernel_S).all())
        self.assertTrue(mx.allclose(base_Werr, kernel_Werr).all())

    def test_timing_comparison(self):
        # Warmup
        for _ in range(5):
            mx.eval(looper_base(self.W, self.R, self.C))
            mx.eval(looper_find_nearest_codebook_indices(self.W, self.R, self.C))

        # base implementation
        num_iters = 1
        tic = time.perf_counter()
        for _ in range(num_iters):
            mx.eval(looper_base(self.W, self.R, self.C))
        toc = time.perf_counter()
        base_time = 1e3 * (toc - tic) / num_iters

        # kernel implementation
        tic = time.perf_counter()
        for _ in range(num_iters):
            mx.eval(looper_find_nearest_codebook_indices(self.W, self.R, self.C))
        toc = time.perf_counter()
        kernel_time = 1e3 * (toc - tic) / num_iters

        # no need to assert
        print(f"\nPerformance:")
        print(f"Base implementation: {base_time:.5f} msec")
        print(f"Kernel implementation: {kernel_time:.5f} msec")

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
        inputs=[W, R, C],
        grid=(num_rows, 1, 1),  # threads_per_grid
        threadgroup=(min(256, num_rows), 1, 1),  # threads_per_threadgroup
        output_shapes=[(num_rows,), (num_rows, num_values), (num_rows,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.float32],
        init_value=0,  # Manually zero-ing S in the kernel is faster for small problems.
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
def looper_base(W, R, C):
    Q = mx.zeros_like(W).astype(mx.uint32)
    S = mx.zeros((C.shape[0], C.shape[1], W.shape[1])).astype(mx.uint32)
    for j in range(W.shape[1]-1, -1, -1):
        indices, Sj, Werrj = base(W[:, [j]], R[:, None], C)
        S[:,:,j] = Sj
        Q[:, j] = indices
        R = Werrj # Omit the matrix-vector multiply since that dominates the latency.
    return indices, S, Q

@mx.compile
def looper_find_nearest_codebook_indices(W, R, C):
    Q = mx.zeros_like(W).astype(mx.uint32)
    S = mx.zeros((C.shape[0], C.shape[1], W.shape[1])).astype(mx.uint32)
    for j in range(W.shape[1]-1, -1, -1):
        indices, Sj, Werrj = find_nearest_codebook_indices(W[:, [j]], R[:, None], C)
        S[:,:,j] = Sj
        Q[:, j] = indices
        R = Werrj # Omit the matrix-vector multiply since that dominates the latency.
    return indices, S, Q
