from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from testing import assert_equal, assert_almost_equal
from gpu.memory import AddressSpace
from gpu.primitives import block
from layout import Layout, LayoutTensor
from math import exp, log2
from bit import log2_ceil
from utils.numerics import max_finite, min_finite
from random import shuffle

comptime SIZE = 100028 # Size >> TPB to simulate "large-scale" softmax
comptime GRID_DIM_X = ()
comptime TPB_MAX = 256
comptime BLOCK_DIM_X = min(1 << log2_ceil(SIZE), TPB_MAX) # ]1;256]
comptime BLOCK_PER_GRID_X = (SIZE + BLOCK_DIM_X - 1) // BLOCK_DIM_X # [1; ANY]
comptime layout = Layout.row_major(SIZE)
comptime inter_layout = Layout.row_major(BLOCK_PER_GRID_X)

# Take tensor of size N and return X local maximums per block
fn local_max_gpu_kernel[
    layout: Layout,
    out_layout: Layout,
    input_size: Int,
    dtype: DType,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    global_i = Int(thread_idx.x + block_idx.x * UInt(BLOCK_DIM_X))
    local_i = Int(thread_idx.x)

    val: output.element_type = min_finite[dtype]()
    if global_i < input_size:
        val = input[global_i]

    block_max = block.max[block_size = BLOCK_DIM_X, broadcast = False](val)
    if local_i == 0:
        output[block_idx.x] = block_max

# Input is local maximums per block, output is local sums per block
fn local_sum_gpu_kernel[
    layout: Layout,
    inter_layout: Layout,
    dtype: DType,
](
    output: LayoutTensor[dtype, inter_layout, MutAnyOrigin],
    local_max: LayoutTensor[dtype, inter_layout, ImmutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    max_output: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin]
):
    global_i = Int(thread_idx.x + block_idx.x * block_dim.x)
    local_i = Int(thread_idx.x)

    # Simplified version, assume BLOCK_PER_GRID_X < TPB..
    val: output.element_type = min_finite[dtype]()
    if local_i < local_max.shape[0]():
        val = local_max[local_i]
    global_max = block.max[block_size = BLOCK_DIM_X, broadcast = True](val)
    if global_i == 0:
        max_output[0] = global_max

    # get local_sum
    val = min_finite[dtype]()
    if global_i < input.shape[0]():
        val = input[global_i]

    exp_val: output.element_type = 0.0
    if global_i < input.shape[0]():
        exp_val = exp(val - global_max)

    block_sum = block.sum[block_size = BLOCK_DIM_X, broadcast = False](exp_val)
    if local_i == 0:
        output[block_idx.x] = block_sum

fn softmax_gpu_kernel[
    layout: Layout,
    inter_layout: Layout,
    dtype: DType,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    local_sum: LayoutTensor[dtype, inter_layout, ImmutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    global_max: LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin]
):
    global_i = Int(thread_idx.x + block_idx.x * block_dim.x)
    local_i = Int(thread_idx.x)

    # Simplified version, assume BLOCK_PER_GRID_X < TPB..
    val: output.element_type = 0.0
    if local_i < local_sum.shape[0]():
        val = local_sum[local_i]
    global_sum = block.sum[block_size = BLOCK_DIM_X, broadcast = True](val)

    # get local_sum
    val = min_finite[dtype]()
    if global_i < input.shape[0]():
        val = input[global_i]

    exp_val: output.element_type = 0.0
    if global_i < input.shape[0]():
        exp_val = exp(val - global_max[0])

    if global_i < output.shape[0]():
        output[global_i] = exp_val / global_sum


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    max_val = min_finite[dtype]()
    for i in range(input_size):
        if input[i] > max_val:
            max_val = rebind[Scalar[dtype]](input[i])

    exp_sum = Scalar[dtype](0)
    for i in range(input_size):
        exp_sum += rebind[Scalar[dtype]](exp(input[i] - max_val))

    for i in range(input_size):
        output[i] = exp(input[i] - max_val) / exp_sum


# ANCHOR_END: softmax_cpu_kernel

def main():
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        buf = List[Int](range(SIZE))
        shuffle(buf)

        input_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf = ctx.enqueue_create_buffer[dtype](BLOCK_PER_GRID_X)
        output_buf.enqueue_fill(0)
        output_buf2 = ctx.enqueue_create_buffer[dtype](BLOCK_PER_GRID_X)
        output_buf2.enqueue_fill(0)
        output_buf3 = ctx.enqueue_create_buffer[dtype](SIZE)
        output_buf3.enqueue_fill(0)
        global_max_buf = ctx.enqueue_create_buffer[dtype](1)
        global_max_buf.enqueue_fill(0)

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, inter_layout, MutAnyOrigin](output_buf)
        output_tensor2 = LayoutTensor[dtype, inter_layout, MutAnyOrigin](output_buf2)
        output_tensor3 = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf3)
        global_max_tensor = LayoutTensor[dtype, Layout.row_major(1), MutAnyOrigin](global_max_buf)

        expected = ctx.enqueue_create_host_buffer[dtype](BLOCK_PER_GRID_X)
        expected.enqueue_fill(0)
        expected_sum = ctx.enqueue_create_host_buffer[dtype](BLOCK_PER_GRID_X)
        expected_sum.enqueue_fill(0)
        expected_softmax = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected_softmax.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for i in range(SIZE):
                input_host[i] = buf[i]

            for block_i in range(BLOCK_PER_GRID_X):
                start = block_i * BLOCK_DIM_X
                end = min(start + BLOCK_DIM_X, SIZE)
                max_val = min_finite[dtype]()
                for i in range(start, end):
                    if input_host[i] > max_val:
                        max_val = input_host[i]
                expected[block_i] = max_val

            global_max = min_finite[dtype]()
            for i in range(BLOCK_PER_GRID_X):
                if expected[i] > global_max:
                    global_max = expected[i]

            for block_i in range(BLOCK_PER_GRID_X):
                start = block_i * BLOCK_DIM_X
                end = min(start + BLOCK_DIM_X, SIZE)
                exp_sum = Scalar[dtype](0)
                for i in range(start, end):
                    exp_sum += exp(input_host[i] - global_max)
                expected_sum[block_i] = exp_sum

            global_sum = Scalar[dtype](0)
            for i in range(BLOCK_PER_GRID_X):
                global_sum += expected_sum[i]

            for i in range(SIZE):
                expected_softmax[i] = exp(input_host[i] - global_max) / global_sum

        comptime kernel = local_max_gpu_kernel[layout, inter_layout, SIZE, dtype]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCK_PER_GRID_X,
            block_dim=BLOCK_DIM_X,
        )

        comptime kernel2 = local_sum_gpu_kernel[layout, inter_layout, dtype]
        ctx.enqueue_function[kernel2, kernel2](
            output_tensor2,
            output_tensor,
            input_tensor,
            global_max_tensor,
            grid_dim=BLOCK_PER_GRID_X,
            block_dim=BLOCK_DIM_X,
        )

        comptime kernel3 = softmax_gpu_kernel[layout, inter_layout, dtype]
        ctx.enqueue_function[kernel3, kernel3](
            output_tensor3,
            output_tensor2,
            input_tensor,
            global_max_tensor,
            grid_dim=BLOCK_PER_GRID_X,
            block_dim=BLOCK_DIM_X,
        )

        ctx.synchronize()

        with output_buf.map_to_host() as output_host:
            for i in range(BLOCK_PER_GRID_X):
                assert_equal(output_host[i], expected[i])
            print("✅ Local max passed")

        with output_buf2.map_to_host() as output_host:
            for i in range(BLOCK_PER_GRID_X):
                assert_almost_equal(output_host[i], expected_sum[i], rtol=1e-5, atol=1e-5)
            print("✅ Local sum passed")

        with output_buf3.map_to_host() as output_host:
            for i in range(SIZE):
                assert_almost_equal(output_host[i], expected_softmax[i], rtol=1e-5, atol=1e-5)
            print("✅ Softmax passed")
