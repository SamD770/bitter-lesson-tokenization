# The same as experiment_6.py, but with a different discount rate (0.9) with half the number of batches and a smaller print every to speed up training.
from transformers import AutoTokenizer
from torch import nn
from typing import List

import random

from copy import deepcopy
import torch

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

import datasets
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer, Gemma2Config, Gemma2Attention, Gemma2Model
from torch.profiler import profile, record_function, ProfilerActivity

from torch import nn
import torch.nn.functional as F

from scipy.signal import lfilter

# Helper functions to count the number of parameters in a torch.nn.Module
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def display_gpu_memory():
    # torch can give a more accurate memory usage than nvidia-smi
    for i in range(torch.cuda.device_count()):
        total_memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        allocated_memory_gb = torch.cuda.memory_allocated(i) / (1024**3)
        free_memory_gb = torch.cuda.mem_get_info(i)[0] / (1024**3)
        print(f"GPU {i}:")
        print(f"  Total GPU memory: {total_memory_gb:.1f} GB")
        print(f"  Free GPU memory: {free_memory_gb:.1f} GB")
        print(f"  Allocated GPU memory: {allocated_memory_gb:.1f} GB")


def parameter_count_string(module):
    n_params = count_parameters(module)
    if n_params > 10**6:
        return f"{n_params/10**6:.1f}M"
    elif n_params > 10**3:
        return f"{n_params/10**3:.1f}k"
    else:
        return f"{n_params}" 
    

class TokenDownsampler(nn.Module):
    def __init__(self, downsample_rate: float):
        super().__init__()
        self.downsample_rate = downsample_rate


class TokenUpsampler(nn.Module):
    def __init__(self, upsample_rate: float):
        super().__init__()
        self.upsample_rate = upsample_rate



def get_merge_dst(gate_samples: torch.Tensor) -> torch.Tensor:
    """
    Returns (merge_dst, dst_idx) the merge destination for each token in the sequence and the number of unique merge destinations.
    Input is a tensor of shape (batch_size, sequence_length) with 0 tokens are merged into the next 1 token.
    mapping:
    1 0 1 1 0 1 1 0 0 0 1
    |   | |   | |       |
    0 1 1 2 3 3 4 5 5 5 5
    """
    # "cycle" the gate samples, appending a zero to the beginning of each batch and ignoring the last gate.
    preceding_gate_samples = torch.cat([torch.zeros_like(gate_samples[:, -1]).unsqueeze(1), gate_samples[:, :-1]], dim=1).to(dtype=torch.long)
    # Trick: use cumsum to do this in a vectorized way.
    merge_dst = preceding_gate_samples.cumsum(dim=1)
    n_dst = merge_dst[:, -1] + 1 # The number of unique merge destinations is the last token's merge destination + 1.
    return merge_dst, n_dst


class AverageTokenDownsampler(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, down_merge_dst: torch.Tensor, n_dst: torch.Tensor) -> torch.Tensor:
        """
        1 2 3 4 5
        1 0 0 1 1
        ->
        1 3 5
        inputs:
        x.shape = (batch_size, seq_len, embedding_dim)
        gate_samples.shape = (batch_size, seq_len)
        position_ids.shape = (batch_size, seq_len)
        returns:
        x_downsampled.shape = (batch_size, n_dst, embedding_dim)
        position_ids_downsampled.shape = (batch_size, n_dst)
        """
        batch_size, _, embedding_dim = x.shape

        # Merge the tokens into the next token where the gate is 1.)
        max_n_dst = n_dst.max().item()

        # Also merge the position ids.
        position_ids_downsampled = torch.zeros(batch_size, max_n_dst, dtype=position_ids.dtype).to(x.device)
        position_ids_downsampled = torch.scatter_reduce(position_ids_downsampled, dim=1, index=down_merge_dst, src=position_ids, reduce="mean", include_self=False)

        # Merge the downsampled tokens.
        down_merge_dst = down_merge_dst.unsqueeze(-1).expand(-1, -1, embedding_dim)

        x_downsampled = torch.zeros(batch_size, max_n_dst, embedding_dim, dtype=x.dtype).to(x.device)
        x_downsampled = torch.scatter_reduce(x_downsampled, dim=1, index=down_merge_dst, src=x, reduce="mean", include_self=False)

        return x_downsampled, position_ids_downsampled


class DistributeTokenUpsampler():
    def forward(self, x: torch.Tensor, gate_samples: torch.Tensor) -> torch.Tensor:
        """
        1 2 3
        1 0 0 1 1
        ->
        1 1 1 2 3
        inputs:
        x.shape = (batch_size, shortened_seq_len, embedding_dim)
        gate_samples.shape = (batch_size, seq_len)
        returns:
        x_upsampled.shape = (batch_size, seq_len, embedding_dim)
        """
        _, _, embedding_dim = x.shape

        # Get the merge destination for each token
        up_merge_dst, _ = get_merge_dst(gate_samples)
        up_merge_dst = up_merge_dst.unsqueeze(-1).expand(-1, -1, embedding_dim)

        # Add the upsampled deviation to the input to the middle layers
        x_upsampled = torch.gather(x, dim=1, index=up_merge_dst)

        return x_upsampled



def compute_discounted_rewards(rewards, discount):
    """
    Assumes that rewards is a numpy array of shape (n_episodes, n_timesteps). Returns tensor of same shape.
    credit to: https://stackoverflow.com/questions/47970683/vectorize-a-numpy-discount-calculation/47971187#47971187,
    minor modifications made to vectorise computation.
    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    # This can probably be sped up by using a 1d convolution.
    r = rewards[:, ::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    return y[:, ::-1]


def discounted_rewards_torch(rewards, discount):
    """torch wrapper for compute_discounted_rewards. Warning: does _not_ allow for backprop through the rewards, which is fine for policy gradients."""
    # This can probably be sped up by using a 1d convolution.
    rewards_device = rewards.device
    rewards = rewards.detach().cpu().numpy()
    discounted_rewards = compute_discounted_rewards(rewards, discount)
    discounted_rewards = torch.tensor(discounted_rewards.copy(), device=rewards_device) # Copy as torch doesn't like converting negatively strided arrays
    return discounted_rewards


class DownGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate

    def gate_samples(self, down_gate_probs: torch.Tensor) -> torch.Tensor:
        gate_samples = torch.bernoulli(down_gate_probs)
        return gate_samples


class LinearGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, 1)
        self.downsample_rate = downsample_rate
        self.downsample_rate_scale = 5.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_gate_logits = self.linear(x)
        down_gate_probs = F.sigmoid(down_gate_logits)
        return down_gate_logits, down_gate_probs # We need to return the logits for stable backprop
    

class RandomGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        gate_probs = torch.ones(batch_size, seq_len, 1, dtype=x.dtype, device=x.device) * self.downsample_rate
        gate_logits = torch.log(gate_probs / (1 - gate_probs))
        return gate_logits, gate_probs


class EquidistantGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.gate_every = round(1 / downsample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        gate_probs = torch.zeros(batch_size, seq_len, 1, dtype=x.dtype, device=x.device) 
        gate_probs[:, ::self.gate_every] = 1.
        gate_logits = gate_probs * 40. - 20. # Avoid Nans
        return gate_logits, gate_probs


def create_gemma2DecoderLayer(config: Gemma2Config, layer_idx: int, compile: bool = False):
    # Gemma2Attention.__init__ overrides config.sliding_window with None if layer_idx % 2 == 0.
    # This is a hack to get the sliding window for even layers indices.
    layer = Gemma2DecoderLayer(config, layer_idx)
    if compile:
        layer = torch.compile(layer)
    layer.self_attn.sliding_window = config.sliding_window
    layer.is_sliding = config.sliding_window is not None
    return layer


def get_gemma2_attention_mask(batch_size, seq_len, device, dtype):
    

    cache_position = torch.arange(seq_len, dtype=torch.long, device=device)

    my_attention_mask = Gemma2Model._prepare_4d_causal_attention_mask_with_cache_position(
        None,
        seq_len,
        seq_len,
        dtype,
        device,
        cache_position,
        batch_size=batch_size,
    )

    return cache_position, my_attention_mask


def display_gating(tokens_ids, merge_dst, tokenizer):
    """Display how a SmallBitterLLM merges a sequence. token_ids and merge_dst are tensors of shape (sequence_length,)."""
    previous_merge_dst = 0
    for t_id, merge_destinantion in zip(tokens_ids, merge_dst):
        merge_destinantion = merge_destinantion.item()
        
        if merge_destinantion != previous_merge_dst:
            print(f"|", end="")
            previous_merge_dst = merge_destinantion
        
        t_txt = tokenizer.decode(t_id)
        print(t_txt.replace('\n', '\\n'), end="")

    print()

