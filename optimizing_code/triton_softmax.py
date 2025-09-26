import torch
from torch.nn import functional as F

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
import pdb

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # x.shape = (M, N)
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(
    x_ptr, 
    output_ptr, 
    M_elements,
    N_elements,
    M_stride,
    N_stride,
    BLOCK_SIZE: tl.constexpr
):
    # input is an M x N tensor
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    pdb.set_trace()

    row_start = pid * M_stride
    row_offsets = N_stride * tl.arange(0, BLOCK_SIZE)

    mask = row_offsets < N_stride * N_elements

    row_ptrs = x_ptr + row_start + row_offsets
    x = tl.load(row_ptrs, mask=mask, other=-float('inf'))

    x_max = tl.max(x, axis=0)

    x = x - x_max
    
    numerator = tl.exp(x)

    denominator = tl.sum(numerator)

    output = numerator / denominator
    pdb.set_trace()

    tl.store(output_ptr + row_start + row_offsets, output, mask=mask)


def fused_softmax(x):
    output = torch.empty_like(x)
    M, N = x.shape
    M_stride, N_stride = x.stride()
    grid = (2,)

    softmax_kernel[grid](x, output, M, N, M_stride, N_stride, BLOCK_SIZE=1024)

    return output
    


if __name__ == "__main__":
    x = [
        [0, 1, 2, 3, 4],
        [1, 1, 1, 1, 1]
    ]
    x = torch.tensor(x).to(dtype=torch.float32, device="cuda")
    y = fused_softmax(x)
    print(y)
    print(y.sum(dim=1))

    print(F.softmax(x))
