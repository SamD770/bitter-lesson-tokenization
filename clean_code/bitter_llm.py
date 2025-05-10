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
    r = rewards[:, ::-1]
    a = [1, -discount]
    b = [1]
    y = lfilter(b, a, x=r)
    return y[:, ::-1]


def discounted_rewards_torch(rewards, discount):
    """torch wrapper for compute_discounted_rewards. Warning: does _not_ allow for backprop through the rewards, which is fine for policy gradients."""
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


def create_gemma2DecoderLayer(config: Gemma2Config, layer_idx: int):
    # Gemma2Attention.__init__ overrides config.sliding_window with None if layer_idx % 2 == 0.
    # This is a hack to get the sliding window for even layers indices.
    layer = Gemma2DecoderLayer(config, layer_idx)
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



class CausalGemmaMiniBitterLLM(nn.Module):
    # Use Gemma2DecoderLayer as a drop in replacement for the TransformerEncoderLayer, with RoPE and sliding window pre-implemented.
    # Also uses a causal mask.
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, downsample_rate: float = 0.25, sliding_window = 64, 
                 GaterClass=LinearGater, DownSamplerClass=AverageTokenDownsampler, UpsamplerClass=DistributeTokenUpsampler,
                 n_down_layers=2, n_mid_layers=6, n_up_layers=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        head_dim = embedding_dim // num_heads

        self.byte_layer_config = Gemma2Config(
            head_dim=head_dim,
            query_pre_attn_scalar=head_dim, 
            sliding_window=sliding_window,
            intermediate_size=embedding_dim,
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
        )

        self.deep_layer_config = Gemma2Config(
            head_dim=head_dim,
            query_pre_attn_scalar=head_dim, 
            sliding_window=None,
            intermediate_size=embedding_dim * 4, # dim_feedforward should scale inversely with the number of tokens in the sequence.
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads
        )

        print(f"byte_layer_config: {self.byte_layer_config._attn_implementation=}")

        # Layer idx=0 is necessary for the sliding window to be applied.
        self.down_layers = nn.ModuleList([
            torch.compile(create_gemma2DecoderLayer(self.byte_layer_config, layer_idx=i)) for i in range(n_down_layers)
        ])

        self.mid_layers = nn.ModuleList([
            torch.compile(create_gemma2DecoderLayer(self.deep_layer_config, layer_idx=i+n_down_layers)) for i in range(n_mid_layers) 
        ])

        self.up_layers = nn.ModuleList([
            torch.compile(create_gemma2DecoderLayer(self.byte_layer_config, layer_idx=i+n_down_layers+n_mid_layers)) for i in range(n_up_layers)
        ])

        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        self.down_layer_gate = GaterClass(embedding_dim, downsample_rate)
        self.downsample_rate = downsample_rate
        self.downsampler = DownSamplerClass()


    def forward(
            self, 
            input_ids: torch.Tensor, 
            position_ids: torch.Tensor=None        
        ) -> torch.Tensor:

        batch_size, max_seq_len = input_ids.shape

        x = self.embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(max_seq_len, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1).to(x.device)      
        
        # Position_ids are used for RoPE
        # cache_position is used for the cache.update() function which retrieves relevant kvs
        byte_cache_position, byte_attention_mask = get_gemma2_attention_mask(batch_size, max_seq_len, x.device, x.dtype)

        # Apply down layers to byte tokens        
        for layer in self.down_layers:
            x = layer(x, 
                attention_mask=byte_attention_mask,
                position_ids=position_ids,
                cache_position=byte_cache_position,
            )[0]

        # Sample gating binary variables for each token.
        down_gate_logits, down_gate_probs = self.down_layer_gate(x)
        down_gate_samples = torch.bernoulli(down_gate_probs)

        # Hack: ensure that we always gate on the first token:
        # We need to also set the corresponding logits to 100. to avoid the gradient exploding based on this.
        down_gate_samples[:, 0] = 1.
        down_gate_probs = torch.cat([torch.ones(batch_size, 1, 1, dtype=down_gate_probs.dtype).to(down_gate_probs.device), down_gate_probs[:, 1:]], dim=1)
        down_gate_logits = torch.cat([100*torch.ones(batch_size, 1, 1, dtype=down_gate_logits.dtype).to(down_gate_logits.device), down_gate_logits[:, 1:]], dim=1)

        # Merge the tokens into the next token where the gate is 1.
        down_gate_samples = down_gate_samples.squeeze(-1)
        down_merge_dst, n_dst = get_merge_dst(down_gate_samples)

        x_downsampled, position_ids_downsampled = self.downsampler(x, position_ids, down_merge_dst, n_dst)
        max_n_dst = x_downsampled.shape[1]

        # Apply mid layers to merged tokens and compute the deviation
        downsampled_cache_position, downsampled_attention_mask = get_gemma2_attention_mask(batch_size, max_n_dst, x.device, x.dtype)

        y_downsampled = x_downsampled

        for layer in self.mid_layers:
            y_downsampled = layer(
                y_downsampled, 
                attention_mask=downsampled_attention_mask,
                position_ids=position_ids_downsampled,
                cache_position=downsampled_cache_position,
            )[0]
        
        deviation = y_downsampled - x_downsampled        

        # Upsample by removing the first token merge group, shifting all token groups down and adding another one token group at the end.
        up_gate_samples = down_gate_samples[:, 1:]
        up_gate_samples = torch.cat([up_gate_samples, torch.ones(batch_size, 1, dtype=up_gate_samples.dtype).to(up_gate_samples.device)], dim=1)
        up_merge_dst, _ = get_merge_dst(up_gate_samples)
        up_merge_dst = up_merge_dst.unsqueeze(-1).expand(-1, -1, self.embedding_dim)

        # Add the upsampled deviation to the input to the middle layers
        upsampled_deviation = torch.gather(deviation, dim=1, index=up_merge_dst)
        y = x + upsampled_deviation

        # Apply up layers to byte tokens
        for layer in self.up_layers:
            y = layer(
                y, 
                attention_mask=byte_attention_mask,
                position_ids=position_ids,
                cache_position=byte_cache_position,
            )[0]

        # Map residual stream to logits
        logits = self.output_layer(y)
        logits = F.log_softmax(logits, dim=-1)

        out = {
            "logits": logits,
            "down_gate_probs": down_gate_probs.squeeze(-1),
            "down_gate_logits": down_gate_logits.squeeze(-1),
            "down_gate_samples": down_gate_samples.to(dtype=torch.long),
            "down_merge_dst": down_merge_dst, 
            "up_merge_dst": up_merge_dst[:, :, 0], # This dimension is repeated.
            "n_dst": n_dst,
            "position_ids": position_ids,
            "key_values": None
        }

        return out
    

def bitter_tokenizer_training_step(model, batch, optimizer, learn_gating=True, downsample_rate_target=0.25, consistency_loss_weight=2., discount_rate = 0.9, relative_gating_loss_weight=1.):
    """
    Assume that batch is torch.tensor of token ids of shape (batch, sequence_length). returns a dict of floats of the training losses for the batch.
    """
    batch_size, _ = batch.shape

    optimizer.zero_grad()

    out = model(batch)
    logits = out["logits"]
    down_gate_samples = out["down_gate_samples"]
    down_gate_probs = out["down_gate_probs"]
    down_gate_logits = out["down_gate_logits"]
    
    # Compute autoregressive loss: log probability of next token.
    next_token_ids = batch[:, 1:]
    current_token_logits = logits[:, :-1]
    next_token_logits = F.cross_entropy(current_token_logits.transpose(1, 2), next_token_ids, reduction="none") # Transpose as F.cross_entropy wants shape [batch, classes, ...]
    ar_loss = next_token_logits.mean()
    true_downsample_rate = down_gate_probs.mean()

    if learn_gating:
        # Compute gating loss: discounted log probabilities of following token(s).
        next_token_logits_padded = torch.cat([next_token_logits, torch.zeros(batch_size, 1, device=next_token_logits.device)], dim=-1) # Pad the last reward as zero
        discounted_rewards = discounted_rewards_torch(next_token_logits_padded, discount_rate)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=0)) # Simple estimate of the advantage

        # action 0 = continue, action 1 = gate
        action_log_probs = torch.stack([torch.zeros_like(down_gate_logits), down_gate_logits], dim=1) # As a sigmoid is equivalent to having one logit as 0.
        selected_action_log_probs = F.cross_entropy(action_log_probs, down_gate_samples, reduction="none")
        gating_loss = - (discounted_rewards * selected_action_log_probs).mean() # Negative as we want to maximise the reward.
        gating_loss = relative_gating_loss_weight * gating_loss
        # Hacky additional consistency loss: make the downsampling rate match the training gating.
        down_gate_rate_loss = consistency_loss_weight*(downsample_rate_target - true_downsample_rate) **2

        total_loss = ar_loss + gating_loss + down_gate_rate_loss
    else:
        selected_action_log_probs = torch.tensor(0.0)
        gating_loss = torch.tensor(0.0)
        down_gate_rate_loss = torch.tensor(0.0) # For logging purposes.
        total_loss = ar_loss

    # Optimizer step
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    out = {
        "ar_loss": ar_loss.item(),
        "gating_loss": gating_loss.item(),
        "true_downsample_rate": true_downsample_rate.item(),
        "rate_consistency_loss": down_gate_rate_loss.item(),
        "total_loss": total_loss.item(),
        "selected_action_ce": selected_action_log_probs.mean().item()
    }

    return out


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


def bitter_tokenizer_training_loop(model, train_dataset, tokenizer, learn_gating=True, 
                                   num_epochs=1, batch_size=128, batch_limit=None, max_seq_length=1024, 
                                   batch_print_every=10, print_example_gating=True, discount_rate=0.9, relative_gating_loss_weight=1.,
                                   downsample_rate_target=0.25, consistency_loss_weight=2.):

    # Create data loaders
    # Create distributed sampler and data loader    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    # See how the model merges a sequence.
    test_string = train_dataset[-1]["text"][:200]
    test_batch = tokenizer.encode(test_string, return_tensors="pt", padding=True).cuda()

    # Initialize model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        model = model.cuda()

        print(f"Epoch {epoch+1}/{num_epochs}, GPU usage:")
        display_gpu_memory()

        for batch_count, batch in enumerate(train_loader):

            batch = batch["text"]
            batch = tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
            batch = batch[:, :max_seq_length]  # Truncate to maximum length of 4096 to save GPU memory.
            batch = batch.cuda()

            loss_dict = bitter_tokenizer_training_step(
                model, batch, optimizer, 
                learn_gating=learn_gating, 
                discount_rate=discount_rate, 
                relative_gating_loss_weight=relative_gating_loss_weight,
                consistency_loss_weight=consistency_loss_weight,
                downsample_rate_target=downsample_rate_target
            )
            train_losses.append(loss_dict)

            # See if this fixes the OOMing issue.
            optimizer.zero_grad()

            # Memory tracking for each batch
            if batch_count % batch_print_every == 0:
                print(f"Batch {batch_count} ar train loss: {loss_dict['ar_loss']} nats/token selected action ce: {loss_dict['selected_action_ce']}")
                if print_example_gating:
                    with torch.no_grad():
                        out = model(test_batch)

                        gate_samples = out["down_gate_samples"]
                        merge_dst = out["down_merge_dst"]
                        true_rate = gate_samples.float().mean().item()
                        implied_iid_ce = -true_rate * np.log(true_rate) - (1 - true_rate) * np.log(1 - true_rate)

                        print(f"Downsample rate: {true_rate:4f} implied iid ce: {implied_iid_ce:4f}")
                        display_gating(test_batch[0], merge_dst[0], tokenizer)

            if batch_limit is not None and batch_count > batch_limit:
                break

        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train loss: {np.mean([l['total_loss'] for l in train_losses]):.4f}")
    
    train_losses = pd.DataFrame(train_losses)

    return train_losses


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

