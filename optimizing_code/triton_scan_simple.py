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

@triton.jit
def scan_sample(
    V_ptr, # [batch_size, seq_len, window_size]
    a_ptr, # [batch_size, seq_len + window_size]
    U_ptr, # [batch_size, seq_len]
    batch_size,
    seq_len,
    window_size,
    stride_Vb, stride_Vs, stride_Vw,
    stride_ab, stride_as,
    stride_Ub, stride_Us,
    SEQ_BLOCK_SIZE: tl.constexpr,
    WINDOW_BLOCK_SIZE: tl.constexpr
):    
    # Each program processes one sample in the batch
    pid = tl.program_id(axis=0)
    
    V_row_ptr = V_ptr + pid * stride_Vb
    a_row_ptr = a_ptr + pid * stride_ab
    U_row_ptr = U_ptr + pid * stride_Ub

    # V_row_offsets = stride_Vs * tl.arange(0, SEQ_BLOCK_SIZE)
    a_row_offsets = stride_as * tl.arange(0, SEQ_BLOCK_SIZE)
    # U_row_offsets = stride_Us * tl.arange(0, SEQ_BLOCK_SIZE)

    V_window_offsets = stride_Vw * tl.arange(0, WINDOW_BLOCK_SIZE)

    a_ptrs = a_row_ptr + a_row_offsets
    # U_ptrs = U_row_ptr + U_row_offsets
    # V_ptrs = V_row_ptr + V_row_offsets[:, None] + V_window_offsets[None, :]

    a_mask = a_row_offsets < stride_as * (seq_len + window_size)
    # U_mask = U_row_offsets < stride_Us * seq_len
    # V_mask = V_row_offsets[:, None] + V_window_offsets[None, :] < stride_Vw * window_size + stride_Vs * seq_len

    pdb.set_trace()

    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    for s in range(SEQ_BLOCK_SIZE):

        # look make a mask equivalent to a[b s-window_size:s+1]
        gather_idxs = tl.arange(0, WINDOW_BLOCK_SIZE) + s - window_size
        a_window = tl.gather(a, gather_idxs, axis=0)

        pdb.set_trace()

        V_ptrs = V_row_ptr + s * stride_Vs + V_window_offsets
        V_s = tl.load(V_ptrs, mask=V_ptrs, other=0.0)
        
        pdb.set_trace()

        logit = tl.sum(a_window*V_s)
        prob = logit.sigmoid()

        U_s = tl.load(U_row_ptr + s * stride_Us)
        a_new = prob > U_s

        pdb.set_trace()


    # for s in range(SEQ_BLOCK_SIZE):
    #     l = 0.0

    #     for w in range(WINDOW_BLOCK_SIZE):
    #         a_w = a[s - w:s]
    #         V_w = V[s, w]
    #         l = l + tl.dot(a_w, V_w)

    #     p = l.sigmoid()
    #     a[s] = p > U[s]
        

    




def fused_scan_sample(V):
    batch_size, seq_len, window_size = V.shape
    batch_stride, seq_stride, window_stride = V.stride()
    
    U = torch.rand(batch_size, seq_len).to(dtype=V.dtype, device=V.device)
    a = torch.ones(batch_size, seq_len + window_size).to(dtype=V.dtype, device=V.device)
    a[:, :window_size] = 0

    SEQ_BLOCK_SIZE = triton.next_power_of_2(seq_len + window_size)
    WINDOW_BLOCK_SIZE = triton.next_power_of_2(window_size)
    grid = (batch_size,)

    scan_sample[grid](
        V, a, U, 
        batch_size, seq_len, window_size, 
        V.stride(0), V.stride(1), V.stride(2),
        a.stride(0), a.stride(1),
        U.stride(0), U.stride(1),
        SEQ_BLOCK_SIZE=SEQ_BLOCK_SIZE,
        WINDOW_BLOCK_SIZE=WINDOW_BLOCK_SIZE
    )
    
    print(a)

def logit_soft_cap(logits, soft_cap=6.0):
    logits = F.tanh(logits / soft_cap) * soft_cap
    return logits


@torch.no_grad()
def torch_scan_simple(V):
    batch_size, seq_len, window_size = V.shape
    V_init_device = V.device
    V = V.to("cpu") # For now, run this on CPU (faster than naive torch on GPU)
    
    U = torch.rand(batch_size, seq_len).to(dtype=V.dtype, device=V.device)

    # We set a to be 1 by default, such that V[:, :, -1] is always applied in the sum.
    a = torch.ones(batch_size, seq_len + window_size - 1).to(dtype=V.dtype, device=V.device)
    a[:, :window_size-1] = 0

    probs = torch.zeros(batch_size, seq_len)
    logits = torch.zeros(batch_size, seq_len)

    for s in range(seq_len):
        a_window = a[:, s:s+window_size]
        V_window = V[:, s, :]
        logit = torch.sum(a_window * V_window, dim=1)
        logit = logit_soft_cap(logit)
        prob = F.sigmoid(logit)
        probs[:, s] = prob
        logits[:, s] = logit
        a[:, s+window_size-1] = prob > U[:, s]

    a = a[:, window_size-1:]
    a = a.to(device=V_init_device)
    
    return logits, probs, a



def compute_probs(a, V):
    batch_size, seq_len, window_size = V.shape
    a_extended = torch.cat([torch.zeros(batch_size, window_size-1, device=a.device, dtype=a.dtype), a], dim=1)

    a_idxs = torch.arange(0, seq_len, device=a.device).reshape(1, seq_len, 1) + torch.arange(0, window_size, device=a.device).reshape(1, 1, window_size)
    a_idxs = a_idxs.repeat(batch_size, 1, 1)

    a_extended = a_extended.unsqueeze(-1).repeat(1, 1, window_size)
    a_window = torch.gather(a_extended, dim=1, index=a_idxs)

    # V[:, :, -1] should always be applied in the sum. This ensures this.
    a_window[:, :, -1] = 1. 
    logits = torch.sum(a_window * V, dim=2)
    logits = logit_soft_cap(logits)
    probs = F.sigmoid(logits)
    return logits, probs


@torch.no_grad()
def torch_scan_simple_scaled(V, scale_factor=1.0, bias=0.0):
    batch_size, seq_len, window_size = V.shape
    V_init_device = V.device
    V = V.to("cpu") # For now, run this on CPU (faster than naive torch on GPU)
    
    U = torch.rand(batch_size, seq_len)

    # We set a to be 1 by default, such that V[:, :, -1] is always applied in the sum.
    a = torch.ones(batch_size, seq_len + window_size - 1).to(dtype=V.dtype, device=V.device)
    a[:, :window_size-1] = 0

    probs = torch.zeros(batch_size, seq_len)
    logits = torch.zeros(batch_size, seq_len)

    for s in range(seq_len):
        a_window = a[:, s:s+window_size]
        V_window = V[:, s, :]
        logit = torch.sum(a_window * V_window, dim=1)
        logit = logit * scale_factor + bias
        logit = logit_soft_cap(logit)
        prob = F.sigmoid(logit)
        probs[:, s] = prob
        logits[:, s] = logit
        a[:, s+window_size-1] = prob > U[:, s]

    a = a[:, window_size-1:]
    a = a.to(device=V_init_device)
    
    return logits, probs, a



def compute_probs_scaled(a, V, scale_factor=1.0, bias=0.0):
    batch_size, seq_len, window_size = V.shape
    a_extended = torch.cat([torch.zeros(batch_size, window_size-1, device=a.device, dtype=a.dtype), a], dim=1)

    a_idxs = torch.arange(0, seq_len, device=a.device).reshape(1, seq_len, 1) + torch.arange(0, window_size, device=a.device).reshape(1, 1, window_size)
    a_idxs = a_idxs.repeat(batch_size, 1, 1)

    a_extended = a_extended.unsqueeze(-1).repeat(1, 1, window_size)
    a_window = torch.gather(a_extended, dim=1, index=a_idxs)

    # V[:, :, -1] should always be applied in the sum. This ensures this.
    a_window[:, :, -1] = 1. 
    logits = torch.sum(a_window * V, dim=2)
    logits = logits * scale_factor + bias
    logits = logit_soft_cap(logits)
    probs = F.sigmoid(logits)
    return logits, probs


if __name__ == "__main__":

    V = torch.ones((2, 10, 4)).to(dtype=torch.bfloat16, device="cuda")
    V = V * torch.tensor([
        [-1, -1, -1, 2],
        [-1, 1, -1, 1]
    ]).reshape(2, 1, 4).to(dtype=torch.bfloat16, device="cuda")

    logits_scan, probs_scan, a = torch_scan_simple(V)


    print(f"{a.shape=} {a=} {a.dtype=}")
    print(f"{V.dtype=}")

    logits, probs = compute_probs(a, V)
    print(f"{probs.shape=} {probs=} {probs.dtype=}")
    print(f"{probs_scan.shape=} {probs_scan=} {probs_scan.dtype=}")
    print(f"{logits.shape=} {logits=} {logits.dtype=}")
    print(f"{logits_scan.shape=} {logits_scan=} {logits_scan.dtype=}")
