import mlx.core as mx
import time
import os
import math

# TODO: Move this to a test.

def find_nearest_codebook_indices(W: mx.array, Werr: mx.array, R: mx.array, C: mx.array, L: mx.array, j: int):

    """
    Compute argmin_s |W + R - C_:,s| for each row of W and return a one-hot S,
    and Werr, the W - Cs.

    Args:
        W: Shape [num_rows]
        R: Shape [num_rows]
        C: Shape [num_rows, num_values]

    Returns:
        indices: Shape [num_rows] - index of closest value in C for each row
    """
    num_rows, num_values = C.shape
    assert W.shape == Werr.shape
    assert L.shape == (W.shape[1], W.shape[1])

    header = """
        constant uint THREADGROUP_CACHE_SIZE = 32;
    """
    source = """
        uint num_rows = W_shape[0];
        uint num_cols = W_shape[1]; // Also L.shape[0] and L.shape[1]
        uint num_values = C_shape[1];

        uint row_idx = thread_position_in_grid.x;
        uint W_start_idx = row_idx * num_cols;

        float wr_value = W[W_start_idx + j] + (R[row_idx] / L[j * num_cols + j]);
        uint codebook_start_idx = row_idx * num_values;

        float min_distance = INFINITY;
        uint min_idx = 0;

        for (uint v_idx = 0; v_idx < num_values; v_idx++) {
            float codebook_value = C[codebook_start_idx + v_idx];
            float distance = metal::abs(wr_value - codebook_value);

            if (distance < min_distance) {
                min_distance = distance;
                min_idx = v_idx;
            }
            // min_distance = fmin(min_distance, distance);
            // min_idx = select(min_idx, col, min_distance < distance);
        }

        float row_err = W[W_start_idx + j] - C[codebook_start_idx + min_idx];

        indices[row_idx] = min_idx;
        Sj[codebook_start_idx + min_idx] = 1;
        Werrj[row_idx] = row_err;

        // Compute the residual. Werr[row_idx, j:] @ L[j:, j-1]
        // Naive (slow).
        // float residual = 0;
        // residual += row_err * L[j*L_shape[0] + j-1]; // L[j,j-1]
        // for (uint k = j+1; k < L_shape[0]; k++) {
        //     float Lkj1 = L[k*L_shape[0] + j-1];
        //     residual += Werr[W_start_idx + k] * Lkj1;
        // }
        // Rj[row_idx] = select(residual, 0.0, j == 0); // No residual for j = 0.

        // TODO: maybe it's better to transpose L (row major but we're picking a column).
        float residual = 0;
        threadgroup float Lcol[THREADGROUP_CACHE_SIZE];

        // Loop from row j in L to end.
        uint chunks = ceil((float)(L_shape[0] - j) / THREADGROUP_CACHE_SIZE);
        for (uint chunk_idx = 0; chunk_idx < chunks; chunk_idx ++) {
            // Load this chunk into threadgroup memory.
            uint group_idx = thread_position_in_threadgroup.x;
            uint Lrow_idx = (chunk_idx * THREADGROUP_CACHE_SIZE) + j + thread_position_in_threadgroup.x;
            if (group_idx < THREADGROUP_CACHE_SIZE && Lrow_idx < num_cols) {
                Lcol[group_idx] = L[Lrow_idx*num_cols + j-1];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Sum in each thread.
            uint Wcol_start = (chunk_idx * THREADGROUP_CACHE_SIZE) + j;
            for (uint l = 0; l < THREADGROUP_CACHE_SIZE && (Wcol_start + l) < num_cols; l++) {
                residual += (chunk_idx == 0 && l == 0 ? row_err : Werr[W_start_idx + Wcol_start + l]) * Lcol[l];
            }
        }
        Rj[row_idx] = residual;
    """
    # threads_per_simdgroup: 32
    # thread_execution_width: 32
    # simdgroups_per_threadgroup: 24
    # quadgroups_per_threadgroup: 192 # ceil(threads_per_threadgroup/4)
    # threads_per_grid: [768,1,1] # as set below
    kernel = mx.fast.metal_kernel(
        name="find_nearest_codebook_indices",
        input_names=["W", "Werr", "R", "C", "L", "j"],
        output_names=["indices", "Sj", "Werrj", "Rj"],
        header=header,
        source=source,
    )

    outputs = kernel(
        inputs=[W,Werr,R,C,L,j],
        grid=(num_rows, 1, 1), # threads_per_grid
        threadgroup=(min(32, num_rows), 1, 1), # threads_per_threadgroup
        output_shapes=[(num_rows,), (num_rows, num_values), (num_rows,), (num_rows, 1)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.float32],
        init_value=0, # No need to initialize one hot outputs.
    )

    return outputs[0], outputs[1], outputs[2], outputs[3]


@mx.compile
def base(W,Werr,R,T,L,j):
    wj = W[:, [j]] + R /  L[j, j]
    distances = mx.abs(wj - T)               # [num_rows, num_values]
    indices = mx.argmin(distances, axis=1)  # [num_rows]
    Sj = mx.zeros(T.shape) # coincidentally the same shape
    Sj[mx.arange(Sj.shape[0]), indices] = 1
    Wq = mx.take_along_axis(T, indices[:, None], axis=1)
    Werr[:, j] = W[:, j] - Wq.squeeze(-1)
    Rnext = mx.matmul(Werr[:, j:], L[j:, [j-1]])
    return indices, Sj, Werr[:, j], Rnext
    # return indices, S, (W - Wq).squeeze()

def bench(fn, *args, **kwargs):
    # msg = kwargs.pop("msg", None)
    # if msg:
    #     print(f"Timing {msg} ...", end=" ")
    # else:
    #     print(f"Timing {fn.__name__} ...", end=" ")

    # warmup
    for _ in range(5):
        mx.eval(fn(*args, **kwargs))

    num_iters = 1000
    tic = time.perf_counter()
    for _ in range(num_iters):
        x = mx.eval(fn(*args, **kwargs))
    toc = time.perf_counter()

    msec = 1e3 * (toc - tic) / num_iters
    # print(f"{msec:.5f} msec")
    return fn(*args, **kwargs), msec

def kernel_loop(iters, W,R,T):
    for _ in range(iters):
        _, _, R = find_nearest_codebook_indices(W,R,T)
    return R

mx.random.seed(42)
m,n,v = 768,768*3,16
j = 1
W = mx.random.normal((m,n))
Werr = mx.random.normal((m,n))
R = mx.random.normal((m,1))
L = mx.tril(mx.random.normal((n,n)))
T = mx.random.normal((m,v))
mx.eval(W,R,T)


if os.environ.get("MTL_CAPTURE_ENABLED", None) == "1":
    mx.eval(W,Werr,R,T,L,j)
    traceName = f"codebook.gputrace"
    mx.metal.start_capture(traceName)
    for _ in range(10):
        # mx.eval(*find_nearest_codebook_indices(W,Werr,R,T,L,j))
        mx.eval(*base(W,Werr,R,T,L,j))
    mx.metal.stop_capture()
    import sys; sys.exit(0)

(base_indices, base_S, base_Werrj, base_R), base_time = bench(base, W,Werr,R,T,L,j)
(kernel_indices, kernel_S, kernel_Werrj, kernel_R), kernel_time = bench(find_nearest_codebook_indices,W,Werr,R,T,L,j)

kernel_Werr = mx.array(Werr)
kernel_Werr[:, j] = kernel_Werrj
manual_r = mx.matmul(Werr[:, j:], L[j:, [j-1]])

print(mx.equal(base_indices, kernel_indices).all())
print(mx.equal(base_S, kernel_S).all())
print(mx.equal(base_Werrj, kernel_Werrj).all())
print(mx.equal(base_R, kernel_R).all(), mx.allclose(base_R, kernel_R).all())
print(mx.equal(base_R, manual_r).all())
print((base_R-kernel_R).abs().max())
print(kernel_R)
print("probably better to leave the mat vec to MLX kernels")
# print(base_R)
# print(kernel_R)

print(f"{base_time:.5f} vs. {kernel_time:.5f}")

# for mul in range(1, 4):
#     _, loop_time = bench(kernel_loop, 768*mul, W, R, T)
#     print(f"{768*mul}: {loop_time:.5f}")
