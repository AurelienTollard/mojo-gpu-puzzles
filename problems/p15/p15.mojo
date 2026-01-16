from sys import size_of
from testing import assert_equal
from gpu.host import DeviceContext

# ANCHOR: axis_sum
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import log2


comptime TPB = 8
comptime BATCH = 4
comptime SIZE = 6
comptime BLOCKS_PER_GRID = (1, BATCH)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime in_layout = Layout.row_major(BATCH, SIZE)
comptime out_layout = Layout.row_major(BATCH, 1)


fn axis_sum[
    in_layout: Layout, out_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: UInt,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = Int(thread_idx.x)
    batch = block_idx.y
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB), # TBD
        MutAnyOrigin,
        address_space = AddressSpace.SHARED
    ].stack_allocation()

    if global_i < size * BATCH:
        shared[local_i] = a[batch, local_i]
    barrier()

    offset = TPB // 2
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        temp: output.element_type = 0
        if local_i - offset >= 0 and global_i < size:
            temp = shared[local_i]
        barrier()

        if local_i - offset >= 0 and global_i < size:
            shared[local_i - offset] += temp
        barrier()

        offset //= 2

    if local_i == 0:
        output[batch, 0] = shared[0]

# ANCHOR_END: axis_sum


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](BATCH)
        out.enqueue_fill(0)
        inp = ctx.enqueue_create_buffer[dtype](BATCH * SIZE)
        inp.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    inp_host[row * SIZE + col] = row * SIZE + col

        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out)
        inp_tensor = LayoutTensor[dtype, in_layout, ImmutAnyOrigin](inp)

        comptime kernel = axis_sum[in_layout, out_layout]
        ctx.enqueue_function[kernel, kernel](
            out_tensor,
            inp_tensor,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](BATCH)
        expected.enqueue_fill(0)
        with inp.map_to_host() as inp_host:
            for row in range(BATCH):
                for col in range(SIZE):
                    expected[row] += inp_host[row * SIZE + col]

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out)
            print("expected:", expected)
            for i in range(BATCH):
                assert_equal(out_host[i], expected[i])
