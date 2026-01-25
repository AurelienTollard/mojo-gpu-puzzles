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

comptime BATCH = 10
comptime SIZE = 1028 # Size >> TPB to simulate "large-scale" softmax
comptime TPB = 512
comptime BLOCK_DIM = TPB
comptime BLOCK_PER_GRID = (
    1,
    BATCH
)
comptime layout = Layout.row_major(SIZE, BATCH)

# Each block process one row, so one batch for simplicity (MAX kernel actually do the same..)
fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    comptime row_length = input.shape[0]()

    local_i = Int(thread_idx.x)
    batch_i = Int(block_idx.y)

    thread_max: output.element_type = min_finite[dtype]()
    for idx in range(local_i, row_length, BLOCK_DIM):
        val = input[idx, batch_i]
        if val > thread_max:
            thread_max = val
    row_max = block.max[block_size = BLOCK_DIM, broadcast = True](thread_max)

    thread_sum: output.element_type = 0.0
    for idx in range(local_i, row_length, BLOCK_DIM):
        val = input[idx, batch_i]
        exp_val = exp(val - row_max)
        thread_sum += exp_val
    row_sum = block.sum[block_size = BLOCK_DIM, broadcast = True](thread_sum)

    for idx in range(local_i, row_length, BLOCK_DIM):
        val = input[idx, batch_i]
        exp_val = exp(val - row_max)
        output[idx, batch_i] = exp_val / row_sum


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

        input_buf = ctx.enqueue_create_buffer[dtype](SIZE * BATCH)
        output_buf = ctx.enqueue_create_buffer[dtype](SIZE * BATCH)
        output_buf.enqueue_fill(0)

        input_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](input_buf)
        output_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](output_buf)

        expected_softmax = ctx.enqueue_create_host_buffer[dtype](SIZE * BATCH)
        expected_softmax.enqueue_fill(0)

        with input_buf.map_to_host() as input_host:
            for b in range(BATCH):
                for i in range(SIZE):
                    input_host[i * BATCH + b] = buf[i] + b

            for b in range(BATCH):
                max_val = min_finite[dtype]()
                for i in range(SIZE):
                    val = input_host[i * BATCH + b]
                    if val > max_val:
                        max_val = val

                exp_sum = Scalar[dtype](0)
                for i in range(SIZE):
                    exp_sum += exp(input_host[i * BATCH + b] - max_val)

                for i in range(SIZE):
                    expected_softmax[i * BATCH + b] = exp(input_host[i * BATCH + b] - max_val) / exp_sum

        comptime kernel = softmax_gpu_kernel[layout, SIZE, dtype]
        ctx.enqueue_function[kernel, kernel](
            output_tensor,
            input_tensor,
            grid_dim=BLOCK_PER_GRID,
            block_dim=BLOCK_DIM,
        )

        ctx.synchronize()

        with output_buf.map_to_host() as output_host:
            for i in range(SIZE * BATCH):
                assert_almost_equal(output_host[i], expected_softmax[i], rtol=1e-5, atol=1e-5)
            print("✅ Batched softmax passed")
