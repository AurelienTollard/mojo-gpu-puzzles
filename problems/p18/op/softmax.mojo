from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp, log2
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = Layout.row_major(SIZE)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    global_i = Int(thread_idx.x)

    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()

    var val: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        val = rebind[Scalar[dtype]](input[global_i])
    shared_max[global_i] = val
    barrier()

    # 1. Find max using reduction. max will be stored in shared_max[0]
    stride = BLOCK_DIM_X // 2
    @parameter
    for i in range(Int(log2(Scalar[dtype](input_size)))):
        if global_i < stride:
           val_l = shared_max[global_i]
           val_r = shared_max[global_i + stride]
           # left side of the buffer store high value
           if val_r > val_l:
               shared_max[global_i] = val_r
        barrier()
        stride //=2
    global_max = shared_max[0]


    # 2. prepare sum using reduction again..
    var exp_val: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_val = rebind[Scalar[dtype]](exp(val - global_max))
    shared_sum[global_i] = exp_val
    barrier()

    stride = BLOCK_DIM_X // 2
    @parameter
    for i in range(Int(log2(Scalar[dtype](input_size)))):
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]
        barrier()
        stride //= 2
    global_sum = shared_sum[0]

    # 3. softmax normalization
    if global_i < input_size:
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

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
            output.to_layout_tensor()
        )
        var input_tensor = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
            input.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    output_tensor.ptr,
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[layout, input_size, dtype]
            gpu_ctx.enqueue_function[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
