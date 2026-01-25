from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from testing import assert_almost_equal
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import min_finite
from random import shuffle
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

comptime SEQ_LEN = 64
comptime BATCH = 4
comptime TRANSPOSE_BLOCK_DIM_XY = 16  # Square blocks for input and output
comptime MATMUL_BLOCK_DIM_XY = 16  # Square blocks for a, b and output
comptime SOFTMAX_BLOCK_DIM_X = 1 << log2_ceil(SEQ_LEN)


# Tiled matrix multiplication (from p16), updated to:
# 1) Support different layouts for input (a, b) and output LayoutTensors.
# 2) Handle cases where the inner dimension is not a multiple of MATMUL_BLOCK_DIM_XY.
# 3) Explicitly check for out-of-bounds elements.
# The approach still tiles all three LayoutTensors (a, b, and output) into identical square tiles
# of size (MATMUL_BLOCK_DIM_XY x MATMUL_BLOCK_DIM_XY) with each thread loading one element
# from a and b, and writing one element to output.
fn matmul_idiomatic_tiled[
    a_layout: Layout,
    b_layout: Layout,
    out_layout: Layout,
    batch: Int,
    rows: Int,
    cols: Int,
    inner: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    b: LayoutTensor[dtype, b_layout, MutAnyOrigin],
):
    """Updated idiomatic tiled matrix multiplication from p16."""
    batch_idx = Int(block_idx.z)
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    tiled_row = Int(block_idx.y) * MATMUL_BLOCK_DIM_XY + local_row
    tiled_col = Int(block_idx.x) * MATMUL_BLOCK_DIM_XY + local_col
    a_shared = LayoutTensor[
        dtype,
        Layout.row_major(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    b_shared = LayoutTensor[
        dtype,
        Layout.row_major(MATMUL_BLOCK_DIM_XY, MATMUL_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var acc: output.element_type = 0

    @parameter
    for idx in range((inner + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY):
        a_col = idx * MATMUL_BLOCK_DIM_XY + local_col
        b_row = idx * MATMUL_BLOCK_DIM_XY + local_row

        var a_val: output.element_type = 0
        if batch_idx < batch and tiled_row < rows and a_col < inner:
            a_val = a[batch_idx, tiled_row, a_col]
        var b_val: output.element_type = 0
        if batch_idx < batch and b_row < inner and tiled_col < cols:
            b_val = b[batch_idx, b_row, tiled_col]

        a_shared[local_row, local_col] = a_val
        b_shared[local_row, local_col] = b_val

        barrier()

        @parameter
        for k in range(MATMUL_BLOCK_DIM_XY):
            if tiled_row < rows and tiled_col < cols:
                acc += a_shared[local_row, k] * b_shared[k, local_col]

        barrier()

    if batch_idx < batch and tiled_row < rows and tiled_col < cols:
        output[batch_idx, tiled_row, tiled_col] = acc


# ANCHOR: transpose_kernel
fn transpose_kernel[
    layout_in: Layout,  # Layout for input matrix (seq_len, d)
    layout_out: Layout,  # Layout for output matrix (d, seq_len)
    batch: Int,
    rows: Int,
    cols: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout_out, MutAnyOrigin],
    inp: LayoutTensor[dtype, layout_in, ImmutAnyOrigin],
):
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TRANSPOSE_BLOCK_DIM_XY, TRANSPOSE_BLOCK_DIM_XY),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    batch_idx = Int(block_idx.z)
    local_row = Int(thread_idx.y)
    local_col = Int(thread_idx.x)
    global_row = Int(block_idx.y) * TRANSPOSE_BLOCK_DIM_XY + local_row
    global_col = Int(block_idx.x) * TRANSPOSE_BLOCK_DIM_XY + local_col

    if batch_idx < batch and global_row < rows and global_col < cols:
        shared[local_row, local_col] = inp[batch_idx, global_row, global_col]
    barrier()

    out_row = Int(block_idx.x) * TRANSPOSE_BLOCK_DIM_XY + local_row
    out_col = Int(block_idx.y) * TRANSPOSE_BLOCK_DIM_XY + local_col
    if batch_idx < batch and out_col < rows and out_row < cols:
        output[batch_idx, out_row, out_col] = shared[local_col, local_row]


# ANCHOR_END: transpose_kernel


# Apply softmax to attention scores taken from p16
fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, MutAnyOrigin],
):
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(SOFTMAX_BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(SOFTMAX_BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = Int(thread_idx.x)
    batch_idx = Int(block_idx.y)

    # Initialize out-of-bounds (shared_max[local_i], global_i >= input_size) shared memory addresses to the minimum
    # finite value for dtype, ensuring that if these elements are accessed in the parallel max reduction below they
    # do not influence the result (max(min_finite, x) == x for any x).
    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input[batch_idx, global_i])
    shared_max[global_i] = val

    barrier()

    # Parallel reduction to find max similar to reduction we saw before
    stride = SOFTMAX_BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )
        barrier()
        stride = stride // 2

    block_max = shared_max[0]

    # Initialize out-of-bounds (shared_max[global_i], global_i >= input_size) shared memory addresses to 0.0,
    # ensuring that if these elements are accessed in the parallel sum reduction below they
    # do not influence the result (adding 0.0 does not change the sum).
    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(val - block_max))
    shared_sum[global_i] = exp_val
    barrier()

    # Parallel reduction for sum similar to reduction we saw before
    stride = SOFTMAX_BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]
        barrier()
        stride = stride // 2

    block_sum = shared_sum[0]

    # Normalize by sum
    if global_i < input_size:
        output[batch_idx, global_i] = exp_val / block_sum


fn attention_cpu_kernel[
    layout_q: Layout,
    layout_k: Layout,
    layout_v: Layout,
    layout_out: Layout,
    batch: Int,
    seq_len: Int,
    d: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout_out, MutAnyOrigin],
    q: LayoutTensor[dtype, layout_q, MutAnyOrigin],
    k: LayoutTensor[dtype, layout_k, ImmutAnyOrigin],
    v: LayoutTensor[dtype, layout_v, MutAnyOrigin],
):
    for b in range(batch):
        var scores = List[Float32]()
        var weights = List[Float32]()
        for _ in range(seq_len):
            scores.append(0.0)
            weights.append(0.0)

        for i in range(seq_len):
            var score: Float32 = 0.0
            for dim in range(d):
                score += rebind[Float32](q[b, dim]) * rebind[
                    Float32
                ](k[b, i, dim])
            scores[i] = score

        var max_score: Float32 = scores[0]
        for i in range(1, seq_len):
            if scores[i] > max_score:
                max_score = scores[i]

        var sum_exp: Float32 = 0.0
        for i in range(seq_len):
            weights[i] = exp(scores[i] - max_score)
            sum_exp = sum_exp + weights[i]

        for i in range(seq_len):
            weights[i] = weights[i] / sum_exp

        for dim in range(d):
            var weighted_sum: Float32 = 0.0
            for i in range(seq_len):
                weighted_sum = weighted_sum + weights[i] * rebind[
                    Float32
                ](v[b, i, dim])
            output[b, dim] = rebind[Scalar[dtype]](weighted_sum)


@compiler.register("attention")
struct AttentionCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        batch: Int,
        seq_len: Int,
        d: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=2],  # Output matrix (batch, d)
        q: InputTensor[rank=2],  # Query matrix (batch, d)
        k: InputTensor[rank=3],  # Key tensor (batch, seq_len, d)
        v: InputTensor[rank=3],  # Value tensor (batch, seq_len, d)
        ctx: DeviceContextPtr,
    ) raises:
        comptime layout_q = Layout.row_major(batch, d)
        comptime layout_k = Layout.row_major(batch, seq_len, d)
        comptime layout_v = Layout.row_major(batch, seq_len, d)
        comptime layout_out = Layout.row_major(batch, d)
        comptime layout_scores = Layout.row_major(batch, seq_len)

        var output_tensor = rebind[
            LayoutTensor[dtype, layout_out, MutAnyOrigin]
        ](output.to_layout_tensor())
        var q_tensor = rebind[LayoutTensor[dtype, layout_q, MutAnyOrigin]](
            q.to_layout_tensor()
        )
        var k_tensor = rebind[LayoutTensor[dtype, layout_k, ImmutAnyOrigin]](
            k.to_layout_tensor()
        )
        var v_tensor = rebind[LayoutTensor[dtype, layout_v, MutAnyOrigin]](
            v.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            var gpu_ctx = rebind[DeviceContext](ctx[])

            comptime layout_q_2d = Layout.row_major(batch, 1, d)
            comptime layout_k_t = Layout.row_major(batch, d, seq_len)
            comptime layout_scores_2d = Layout.row_major(batch, 1, seq_len)
            comptime layout_weights_2d = Layout.row_major(batch, 1, seq_len)
            comptime layout_result_2d = Layout.row_major(batch, 1, d)

            comptime transpose_threads_per_block = (
                TRANSPOSE_BLOCK_DIM_XY,
                TRANSPOSE_BLOCK_DIM_XY,
            )
            comptime transpose_blocks_per_grid = (
                (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
                (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1)
                // TRANSPOSE_BLOCK_DIM_XY,
                batch,
            )
            comptime matmul_threads_per_block = (
                MATMUL_BLOCK_DIM_XY,
                MATMUL_BLOCK_DIM_XY,
            )
            comptime scores_blocks_per_grid = (
                (seq_len + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
                1,
                batch,
            )
            comptime softmax_threads = SOFTMAX_BLOCK_DIM_X
            comptime softmax_blocks_per_grid = (1, batch)
            comptime result_blocks_per_grid = (
                (d + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
                1,
                batch,
            )

            k_t_buf = gpu_ctx.enqueue_create_buffer[dtype](batch * seq_len * d)
            scores_weights_buf = gpu_ctx.enqueue_create_buffer[dtype](
                batch * seq_len
            )

            k_t = LayoutTensor[dtype, layout_k_t, MutAnyOrigin](k_t_buf)

            q_2d = q_tensor.reshape[layout_q_2d]()

            comptime kernel = transpose_kernel[
                layout_k, layout_k_t, batch, seq_len, d, dtype
            ]
            gpu_ctx.enqueue_function[kernel, kernel](
                k_t,
                k_tensor,
                grid_dim=transpose_blocks_per_grid,
                block_dim=transpose_threads_per_block,
            )

            attention_scores = LayoutTensor[
                dtype, layout_scores_2d, MutAnyOrigin
            ](scores_weights_buf)
            comptime kernel_scores = matmul_idiomatic_tiled[
                layout_q_2d,
                layout_k_t,
                layout_scores_2d,
                batch,
                1,
                seq_len,
                d,
                dtype,
            ]
            gpu_ctx.enqueue_function[kernel_scores, kernel_scores](
                attention_scores,
                q_2d,
                k_t,
                grid_dim=scores_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

            weights = attention_scores.reshape[layout_scores]()

            comptime kernel_softmax = softmax_gpu_kernel[
                layout_scores, seq_len, dtype
            ]
            gpu_ctx.enqueue_function[kernel_softmax, kernel_softmax](
                weights,
                weights,
                grid_dim=softmax_blocks_per_grid,
                block_dim=softmax_threads,
            )

            weights_2d = weights.reshape[layout_weights_2d]()
            result_2d = output_tensor.reshape[layout_result_2d]()
            comptime kernel_attention = matmul_idiomatic_tiled[
                layout_weights_2d,
                layout_v,
                layout_result_2d,
                batch,
                1,
                d,
                seq_len,
                dtype,
            ]
            gpu_ctx.enqueue_function[kernel_attention, kernel_attention](
                result_2d,
                weights_2d,
                v_tensor,
                grid_dim=result_blocks_per_grid,
                block_dim=matmul_threads_per_block,
            )

        elif target == "cpu":
            attention_cpu_kernel[
                layout_q,
                layout_k,
                layout_v,
                layout_out,
                batch,
                seq_len,
                d,
                dtype,
            ](output_tensor, q_tensor, k_tensor, v_tensor)

        else:
            raise Error("Unsupported target: " + target)

def main():
    comptime dtype = DType.float32
    comptime batch = BATCH
    comptime seq_len = SEQ_LEN
    comptime d = 64
    comptime layout_q = Layout.row_major(batch, d)
    comptime layout_k = Layout.row_major(batch, seq_len, d)
    comptime layout_v = Layout.row_major(batch, seq_len, d)
    comptime layout_out = Layout.row_major(batch, d)
    comptime layout_scores = Layout.row_major(batch, seq_len)

    # Define layouts for matrix multiplication
    comptime layout_q_2d = Layout.row_major(batch, 1, d)
    comptime layout_k_t = Layout.row_major(batch, d, seq_len)
    comptime layout_scores_2d = Layout.row_major(batch, 1, seq_len)
    comptime layout_weights_2d = Layout.row_major(batch, 1, seq_len)
    comptime layout_result_2d = Layout.row_major(batch, 1, d)

    comptime transpose_threads_per_block = (
        TRANSPOSE_BLOCK_DIM_XY,
        TRANSPOSE_BLOCK_DIM_XY,
    )
    comptime transpose_blocks_per_grid = (
        (d + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
        (seq_len + TRANSPOSE_BLOCK_DIM_XY - 1) // TRANSPOSE_BLOCK_DIM_XY,
        batch,
    )
    comptime matmul_threads_per_block = (
        MATMUL_BLOCK_DIM_XY,
        MATMUL_BLOCK_DIM_XY,
    )
    comptime scores_blocks_per_grid = (
        (seq_len + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
        1,
        batch,
    )
    comptime softmax_threads = SOFTMAX_BLOCK_DIM_X
    comptime softmax_blocks_per_grid = (1, batch)
    comptime result_blocks_per_grid = (
        (d + MATMUL_BLOCK_DIM_XY - 1) // MATMUL_BLOCK_DIM_XY,
        1,
        batch,
    )

    with DeviceContext() as ctx:
        q_buf = ctx.enqueue_create_buffer[dtype](batch * d)
        k_buf = ctx.enqueue_create_buffer[dtype](batch * seq_len * d)
        v_buf = ctx.enqueue_create_buffer[dtype](batch * seq_len * d)
        output_buf = ctx.enqueue_create_buffer[dtype](batch * d)
        output_buf.enqueue_fill(0)

        expected_output = ctx.enqueue_create_host_buffer[dtype](batch * d)
        expected_output.enqueue_fill(0)

        buf_q = List[Int](range(batch * d))
        buf_k = List[Int](range(batch * seq_len * d))
        buf_v = List[Int](range(batch * seq_len * d))
        shuffle(buf_q)
        shuffle(buf_k)
        shuffle(buf_v)

        with q_buf.map_to_host() as q_host:
            with k_buf.map_to_host() as k_host:
                with v_buf.map_to_host() as v_host:
                    scale: Float32 = 0.01
                    for b in range(batch):
                        for i in range(d):
                            q_host[b * d + i] = scale * buf_q[b * d + i]

                        for row in range(seq_len):
                            for col in range(d):
                                idx = (b * seq_len + row) * d + col
                                k_host[idx] = scale * buf_k[idx]
                                v_host[idx] = scale * buf_v[idx]

                    for b in range(batch):
                        var scores = List[Float32]()
                        var weights = List[Float32]()
                        for _ in range(seq_len):
                            scores.append(0.0)
                            weights.append(0.0)

                        for row in range(seq_len):
                            var score: Float32 = 0.0
                            for col in range(d):
                                score += rebind[Float32](
                                    q_host[b * d + col]
                                ) * rebind[Float32](
                                    k_host[(b * seq_len + row) * d + col]
                                )
                            scores[row] = score

                        var max_score: Float32 = scores[0]
                        for row in range(1, seq_len):
                            if scores[row] > max_score:
                                max_score = scores[row]

                        var sum_exp: Float32 = 0.0
                        for row in range(seq_len):
                            weights[row] = exp(scores[row] - max_score)
                            sum_exp += weights[row]

                        for row in range(seq_len):
                            weights[row] = weights[row] / sum_exp

                        for col in range(d):
                            var weighted_sum: Float32 = 0.0
                            for row in range(seq_len):
                                weighted_sum += weights[row] * rebind[
                                    Float32
                                ](v_host[(b * seq_len + row) * d + col])
                            expected_output[b * d + col] = rebind[
                                Scalar[dtype]
                            ](weighted_sum)

        q_tensor = LayoutTensor[dtype, layout_q, MutAnyOrigin](q_buf)
        k_tensor = LayoutTensor[dtype, layout_k, ImmutAnyOrigin](k_buf)
        v_tensor = LayoutTensor[dtype, layout_v, MutAnyOrigin](v_buf)
        output_tensor = LayoutTensor[dtype, layout_out, MutAnyOrigin](
            output_buf
        )

        k_t_buf = ctx.enqueue_create_buffer[dtype](batch * seq_len * d)
        scores_weights_buf = ctx.enqueue_create_buffer[dtype](batch * seq_len)

        k_t = LayoutTensor[dtype, layout_k_t, MutAnyOrigin](k_t_buf)

        q_2d = q_tensor.reshape[layout_q_2d]()
        comptime kernel_transpose = transpose_kernel[
            layout_k, layout_k_t, batch, seq_len, d, dtype
        ]
        ctx.enqueue_function[kernel_transpose, kernel_transpose](
            k_t,
            k_tensor,
            grid_dim=transpose_blocks_per_grid,
            block_dim=transpose_threads_per_block,
        )

        attention_scores = LayoutTensor[dtype, layout_scores_2d, MutAnyOrigin](
            scores_weights_buf
        )
        comptime kernel_scores = matmul_idiomatic_tiled[
            layout_q_2d,
            layout_k_t,
            layout_scores_2d,
            batch,
            1,
            seq_len,
            d,
            dtype,
        ]
        ctx.enqueue_function[kernel_scores, kernel_scores](
            attention_scores,
            q_2d,
            k_t,
            grid_dim=scores_blocks_per_grid,
            block_dim=matmul_threads_per_block,
        )

        weights = attention_scores.reshape[layout_scores]()
        comptime kernel_softmax = softmax_gpu_kernel[layout_scores, seq_len, dtype]
        ctx.enqueue_function[kernel_softmax, kernel_softmax](
            weights,
            weights,
            grid_dim=softmax_blocks_per_grid,
            block_dim=softmax_threads,
        )

        weights_2d = weights.reshape[layout_weights_2d]()
        result_2d = output_tensor.reshape[layout_result_2d]()
        comptime kernel_attention = matmul_idiomatic_tiled[
            layout_weights_2d,
            layout_v,
            layout_result_2d,
            batch,
            1,
            d,
            seq_len,
            dtype,
        ]
        ctx.enqueue_function[kernel_attention, kernel_attention](
            result_2d,
            weights_2d,
            v_tensor,
            grid_dim=result_blocks_per_grid,
            block_dim=matmul_threads_per_block,
        )

        ctx.synchronize()

        with output_buf.map_to_host() as output_host:
            for i in range(batch * d):
                assert_almost_equal(
                    output_host[i], expected_output[i], rtol=1e-4, atol=1e-4
                )
            print("✅ Batched attention GPU kernel passed")
