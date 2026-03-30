# The same as experiment_6.py, but with a different discount rate (0.9) with half the number of batches and a smaller print every to speed up training.
from transformers import AutoTokenizer
from torch import nn
from typing import List, Tuple

import math
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
    

def get_merge_dst(gate_samples: torch.Tensor) -> torch.Tensor:
    """
    Returns (merge_dst, dst_idx) the merge destination for each token in the sequence and the number of unique merge destinations.
    Input is a tensor of shape (batch_size, sequence_length) with 0 tokens are merged into the next 1 token. 
    An implicit final merge destination is used irrespective of whether the last sequence index is 0 or 1.
    mapping:
    1 0 1 1 0 1 1 0 0 0 1
    |   | |   | |       |
    0 1 1 2 3 3 4 5 5 5 5
    input:
        gate_samples B S
    returns:
        merge_dst B S
        n_dst B
    """
    # "cycle" the gate samples, appending a zero to the beginning of each batch and ignoring the last gate.
    preceding_gate_samples = torch.cat([torch.zeros_like(gate_samples[:, -1]).unsqueeze(1), gate_samples[:, :-1]], dim=1).to(dtype=torch.long)
    # Trick: use cumsum to do this in a vectorized way.
    merge_dst = preceding_gate_samples.cumsum(dim=1)
    n_dst = merge_dst[:, -1] + 1 # The number of unique merge destinations is the last token's merge destination + 1.
    return merge_dst, n_dst


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


class Gater(nn.Module):
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x B S D
        Returns:
            down_gate_logits B S
            down_gate_probs B S
            gate_samples B S
        """
        raise NotImplementedError()


class Donwsampler(nn.Module):
    def forward() -> torch.Tensor:
        """
        Arguments:
            x B S D
            position_ids B S
            gate_samples B S
        Returns:
            x_downsampled B S' D
            position_ids_downsampled B S'
        """
        raise NotImplementedError()


class Upsampler(nn.Module):
    def forward(self, x, x_downsampled, y_downsampled, down_gate_samples, down_gate_probs) -> torch.Tensor:
        """
        Arguments:
            x B S D
            x_downsampled B S' D
            y_downsampled B S' D
            gate_samples B S
            down_gate_probs B S
        Returns:
            y B S D
        """
        raise NotImplementedError()


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


class ScaledLinearGater(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float, scale_factor: float = 1/16.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.scale_factor = scale_factor
        self.linear = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, downsample_rate: float = None) -> torch.Tensor:
        if downsample_rate is None:
            downsample_rate = self.downsample_rate

        downsample_rate = min(downsample_rate, 0.999)
        bias = math.log(downsample_rate / (1 - downsample_rate))

        gate_logits = self.linear(x) * self.scale_factor + bias
        gate_probs = torch.sigmoid(gate_logits)
        gate_samples = torch.bernoulli(gate_probs)
        return gate_logits, gate_probs, gate_samples


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


class IndependentWrapperGater(nn.Module):
    def __init__(self, gater: nn.Module):
        super().__init__()
        self.gater = gater

    def forward(self, x: torch.Tensor, downsample_rate=None) -> torch.Tensor:
        """
        Values are defined as so:
        gate_probs[b, s] = sigmoid(gate_logits[b, s])
        gate_samples[b, s] ~ bernoulli(gate_probs[b, s]) i.i.d. for all b, s.
        """

        # Sample gating binary variables for each token.
        gate_logits, gate_probs = self.gater(x)

        # Sample from the gate_probs independently for each token.
        gate_samples = torch.bernoulli(gate_probs)

        return gate_logits, gate_probs, gate_samples


class ExactRandomGater(nn.Module):
    """
    This gater samples from the distribution generated by resampling b_1 ... b_{4096} iid bernoulli until exactly 1024 of them are 1. 
    This is the similar to RandomGater, but with some serial dependency that makes it more GPU-efficient and reliable.
    """

    def __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate

    def forward(self, x: torch.Tensor, downsample_rate=None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        if downsample_rate is None:
            downsample_rate = self.downsample_rate

        num_ones = round(seq_len * downsample_rate)

        # Generate random values for each position
        latents = torch.rand(batch_size, seq_len, 1, device=x.device)

        # Set the first and last tokens to infinity to ensure it is selected.
        latents[:, 0] = torch.tensor(float('inf'), dtype=latents.dtype, device=x.device)
        latents[:, -1] = torch.tensor(float('inf'), dtype=latents.dtype, device=x.device)

        # Use topk to find the indices of the k largest values
        # This gives us exactly num_ones indices per batch
        _, top_indices = torch.topk(latents, k=num_ones, dim=1)
        gate_samples = torch.zeros(batch_size, seq_len, 1, dtype=x.dtype, device=x.device)
        gate_samples.scatter_(1, top_indices, 1)

        gate_probs = torch.ones(batch_size, seq_len, 1, dtype=x.dtype, device=x.device) * downsample_rate
        # take the inverse sigmoid of the probability to get the logits
        gate_logits = torch.log(gate_probs / (1 - gate_probs)) 
        return gate_logits, gate_probs, gate_samples


def get_boundary_indices(gate_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the indices of the boundary for each batch. Pads with -1.
    1 0 0 1 1
    1 1 0 0 0
    ->
    0 3 4
    0 1 -1
    inputs:
        gate_samples B S
    returns:
        boundary_token_indices B S'
        merge_dst B S'
    """
    batch_size, seq_len = gate_samples.shape
    merge_dst, n_dst = get_merge_dst(gate_samples)
    n_dst_max = n_dst.max().item()
    # We use the trick that:
    # if: gate_samples = 1  0  1  1  0  1  1  0  0
    # then:        src = 0 -1  2  3 -1  5  6 -1 -1
    # and so reducing with max and index:
    #        merge_dst = 0  1  1  2  3  3  4  5  5
    # gives: boundary_indices = 0 2 3 5 6 -1    
    boundary_indices = torch.ones(batch_size, n_dst_max, dtype=torch.long).to(gate_samples.device) * -1
    src = torch.arange(seq_len, device=gate_samples.device).unsqueeze(0).expand(batch_size, -1)
    # For long sequences, running this with float gate_samples will lead to rounding errors.
    gate_samples_int = gate_samples.to(dtype=torch.long)
    src = (src * gate_samples_int - 1 + gate_samples_int) 
    boundary_indices = torch.scatter_reduce(boundary_indices, dim=1, index=merge_dst, src=src, reduce="max", include_self=False)
    return boundary_indices, merge_dst


def select(x: torch.Tensor, boundary_indices: torch.Tensor, pad_value=0.0) -> torch.Tensor:
    """
    Selects the tokens according to the boundary token indices.
    1 2 3 4 5
    0 3 4
    ->
    1 4 5
    inputs:
        x B S [D]
        boundary_token_indices B S'
    returns:
        x_selected B S' D
    """
    # Add an implicit embedding dimension if not there already 
    if len(x.shape) == 2:
        return select(x.unsqueeze(-1), boundary_indices, pad_value=pad_value).squeeze(-1)

    batch_size, seq_len, embedding_dim = x.shape
    _, new_seq_len = boundary_indices.shape

    boundary_indices = boundary_indices.unsqueeze(-1).expand(-1, -1, embedding_dim)

    # boundary_indices is padded with -1, which we need to wrap to avoid errors.
    pad_mask = (boundary_indices == -1)

    boundary_indices = boundary_indices.masked_fill(pad_mask, 0)
    x_selected = torch.gather(x, dim=1, index=boundary_indices)
    x_selected = x_selected.masked_fill(pad_mask, pad_value)

    return x_selected


class SelectTokenDownsampler(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, gate_samples: torch.Tensor) -> torch.Tensor:
        """
        Selects the tokens where the gate is 1. accordnig to:
        1 2 3 4 5
        1 0 0 1 1
        ->
        1 4 5
        inputs:
            x = (batch_size, seq_len, embedding_dim)
            position_ids = (batch_size, seq_len)
            gate_samples = (batch_size, seq_len)
        returns:
            x_downsampled = (batch_size, n_dst, embedding_dim)
            position_ids_downsampled = (batch_size, n_dst)
        """
        boundary_indices, merge_dst = get_boundary_indices(gate_samples)

        x_downsampled = select(x, boundary_indices)
        position_ids_downsampled = select(position_ids, boundary_indices)

        return x_downsampled, position_ids_downsampled, merge_dst.unsqueeze(-1)


class AverageTokenDownsampler(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, down_gate_samples: torch.Tensor) -> torch.Tensor:
        """
        1 2 3 4 5
        1 0 0 1 1
        ->
        1 3 5
        inputs:
        x.shape = (batch_size, seq_len, embedding_dim)
        position_ids.shape = (batch_size, seq_len)
        down_gate_samples.shape = (batch_size, seq_len)
        returns:
        x_downsampled.shape = (batch_size, n_dst, embedding_dim)
        position_ids_downsampled.shape = (batch_size, n_dst)

        Warning: this is nondeterministic due to race conditions in scatter_reduce and rounding errors.
        """
        batch_size, _, embedding_dim = x.shape
        down_merge_dst, n_dst = get_merge_dst(down_gate_samples)

        # Merge the tokens into the next token where the gate is 1.)
        max_n_dst = n_dst.max().item()

        # Also merge the position ids.
        position_ids_downsampled = torch.zeros(batch_size, max_n_dst, dtype=position_ids.dtype).to(x.device)
        position_ids_downsampled = torch.scatter_reduce(position_ids_downsampled, dim=1, index=down_merge_dst, src=position_ids, reduce="mean", include_self=False)

        # Merge the downsampled tokens.
        down_merge_dst = down_merge_dst.unsqueeze(-1).expand(-1, -1, embedding_dim)

        x_downsampled = torch.zeros(batch_size, max_n_dst, embedding_dim, dtype=x.dtype).to(x.device)
        x_downsampled = torch.scatter_reduce(x_downsampled, dim=1, index=down_merge_dst, src=x, reduce="mean", include_self=False)

        return x_downsampled, position_ids_downsampled, down_merge_dst


def distribute(x, gate_samples) -> torch.Tensor:
    """
    Distributes the values of x to the next token where the gate is 1.
    1 2 3
    1 0 0 1 1
    ->
    1 1 1 2 3
    """
    batch_size, _, embedding_dim = x.shape
    # Upsample by removing the first token merge group, shifting all token groups down and adding another one token group at the end.
    up_gate_samples = gate_samples[:, 1:]
    up_gate_samples = torch.cat([up_gate_samples, torch.ones(batch_size, 1, dtype=up_gate_samples.dtype).to(up_gate_samples.device)], dim=1)
    up_merge_dst, _ = get_merge_dst(up_gate_samples)
    up_merge_dst = up_merge_dst.unsqueeze(-1).expand(-1, -1, embedding_dim)

    x_upsampled = torch.gather(x, dim=1, index=up_merge_dst)

    return x_upsampled, up_merge_dst


class DistributeAddUpsampler(nn.Module):
    def forward(self, x, x_downsampled, y_downsampled, down_gate_samples, down_gate_probs) -> torch.Tensor:
        y_upsampled, up_merge_dst = distribute(y_downsampled, down_gate_samples)
        y = x + y_upsampled
        return y, up_merge_dst


class DistributeDeviationUpsampler(nn.Module):
    def forward(self, x, x_downsampled, y_downsampled, down_gate_samples, down_gate_probs) -> torch.Tensor:
        deviation = y_downsampled - x_downsampled
        upsampled_deviation, up_merge_dst = distribute(deviation, down_gate_samples)
        y = x + upsampled_deviation

        return y, up_merge_dst


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

