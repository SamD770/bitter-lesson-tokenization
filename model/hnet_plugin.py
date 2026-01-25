import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F

from .hnet_downsampler import RoutingModule, ChunkLayer, DeChunkLayer, ste_func, load_balancing_loss, RoutingModuleOutput
from .modules import get_merge_dst


def hnet_consistency_loss(
    down_gate_probs: torch.Tensor,
    down_gate_samples: torch.Tensor,
    downsample_rate_target: float,
) -> torch.Tensor:
    """
    Compute the consistency loss for HNet. Just copy the load_balancing_loss formula.
    """

    N = 1 / downsample_rate_target

    true_ratio = down_gate_samples.float().mean()
    average_prob = down_gate_probs.float().mean()

    return (
        (1 - true_ratio) * (1 - average_prob) +
        (true_ratio) * (average_prob) * (N-1)
    ) * N / (N-1)




class HNetGater(nn.Module):
    """
    Wraps HNet's RoutingModule to produce gating outputs compatible with the repository interface.
    
    The RoutingModule computes boundary probabilities based on cosine similarity between
    adjacent token embeddings. Unlike learned gaters, it doesn't use a downsample_rate parameter.
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        downsample_rate: float = 0.25,  # Ignored, kept for interface compatibility
        random_init: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate  # Not used by RoutingModule
        self.routing_module = RoutingModule(d_model=embedding_dim, random_init=random_init)

    def forward(self, x: torch.Tensor, downsample_rate=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: (B, S, D) input embeddings
            downsample_rate: ignored (kept for interface compatibility)
        Returns:
            gate_logits: (B, S, 1) inverse sigmoid of gate probabilities
            gate_probs: (B, S, 1) probability of boundary at each position
            gate_samples: (B, S, 1) binary boundary decisions (0 or 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create mask of all True (no padding, batched mode)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Call RoutingModule (batched mode, not sequence packed)
        routing_output = self.routing_module(
            hidden_states=x,
            cu_seqlens=None,
            mask=mask,
            inference_params=None
        )
        
        # Extract outputs:
        # boundary_prob has shape (B, S, 2) where [..., 1] is the boundary probability
        # boundary_mask has shape (B, S) and is boolean
        gate_probs = routing_output.boundary_prob[..., 1]  # (B, S)
        gate_samples = routing_output.boundary_mask.float()  # (B, S)
        
        # Compute logits as inverse sigmoid, with clamping to avoid numerical issues
        gate_probs_clamped = torch.clamp(gate_probs, min=1e-6, max=1 - 1e-6)
        gate_logits = torch.log(gate_probs_clamped / (1 - gate_probs_clamped))
        
        # Add trailing dimension for compatibility with repository interface
        gate_logits = gate_logits.unsqueeze(-1)  # (B, S, 1)
        gate_probs = gate_probs.unsqueeze(-1)  # (B, S, 1)
        gate_samples = gate_samples.unsqueeze(-1)  # (B, S, 1)
        
        return gate_logits, gate_probs, gate_samples


class HNetDownsampler(nn.Module):
    """
    Wraps HNet's ChunkLayer to downsample embeddings at boundary positions.
    
    Selects tokens where gate_samples == 1 (boundary positions).
    """
    
    def __init__(self):
        super().__init__()
        self.chunk_layer = ChunkLayer()
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: torch.Tensor, 
        gate_samples: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: (B, S, D) input embeddings
            position_ids: (B, S) position indices
            gate_samples: (B, S) binary boundary decisions (0 or 1)
        Returns:
            x_downsampled: (B, S', D) downsampled embeddings
            position_ids_downsampled: (B, S') downsampled position indices
            down_merge_dst: (B, S, 1) merge destination for each token
        """
        batch_size, seq_len, embedding_dim = x.shape
        
        # Convert gate_samples to boolean boundary_mask
        boundary_mask = gate_samples.bool()  # (B, S)
        
        # Create mask of all True (batched mode, no padding)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Call ChunkLayer (returns next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask)
        x_downsampled, _, _, next_mask = self.chunk_layer(
            hidden_states=x,
            boundary_mask=boundary_mask,
            cu_seqlens=None,
            mask=mask
        )
        
        # Downsample position_ids using the same logic as ChunkLayer
        # Count number of boundary tokens per batch
        num_tokens = boundary_mask.sum(dim=-1)  # (B,)
        next_max_seqlen = int(num_tokens.max())
        
        # Create indices for sorting (push non-boundary tokens to the end)
        token_idx = (
            torch.arange(seq_len, device=x.device)[None, :] 
            + (~boundary_mask).long() * seq_len
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)
        
        # Gather position_ids using sorted indices
        position_ids_downsampled = torch.gather(
            position_ids,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen]
        )
        
        # Compute merge_dst for compatibility with repository interface
        down_merge_dst, _ = get_merge_dst(gate_samples)
        down_merge_dst = down_merge_dst.unsqueeze(-1)  # (B, S, 1)
        
        return x_downsampled, position_ids_downsampled, down_merge_dst


class HNetUpsampler(nn.Module):
    """
    Wraps HNet's DeChunkLayer to upsample embeddings back to full sequence length.
    
    Uses EMA-based deaggregation (via Mamba2 kernel) to spread information from
    boundary positions to all positions, then adds a residual connection.
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        block_size: int = 256,
        headdim: int = 32,
        detach_probs: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dechunk_layer = DeChunkLayer(
            d_model=embedding_dim,
            dtype=dtype,
            block_size=block_size,
            headdim=headdim,
        )
        self.detach_probs = detach_probs
        
    
    def forward(
        self, 
        x: torch.Tensor, 
        x_downsampled: torch.Tensor, 
        y_downsampled: torch.Tensor, 
        down_gate_samples: torch.Tensor, 
        down_gate_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x: (B, S, D) original byte-level embeddings (before downsampling)
            x_downsampled: (B, S', D) downsampled input embeddings (not used by HNet)
            y_downsampled: (B, S', D) processed downsampled embeddings
            down_gate_samples: (B, S) or (B, S, 1) binary boundary decisions
            down_gate_probs: (B, S) or (B, S, 1) boundary probabilities
        Returns:
            y: (B, S, D) upsampled output embeddings
            up_merge_dst: (B, S, 1) merge destination for each token
        """
        batch_size, seq_len, _ = x.shape

        # Interesting ablation: What if we don't allow gradients to flow through the down_gate_probs but keep the initialization?
        if self.detach_probs:
            down_gate_probs = down_gate_probs.detach()

        # Handle potential trailing dimension
        if down_gate_samples.dim() == 3:
            down_gate_samples = down_gate_samples.squeeze(-1)  # (B, S)
        if down_gate_probs.dim() == 3:
            down_gate_probs = down_gate_probs.squeeze(-1)  # (B, S)
        
        # Convert gate_samples to boolean boundary_mask
        boundary_mask = down_gate_samples.bool()  # (B, S)
        
        # Reconstruct boundary_prob tensor: (B, S, 2)
        # boundary_prob[..., 0] = 1 - gate_probs (no boundary)
        # boundary_prob[..., 1] = gate_probs (boundary)
        boundary_prob = torch.stack([1 - down_gate_probs, down_gate_probs], dim=-1)
        
        # Create mask of all True (batched mode, no padding)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Call DeChunkLayer to spread y_downsampled back to full sequence
        y_dechunked = self.dechunk_layer(
            hidden_states=y_downsampled,
            boundary_mask=boundary_mask,
            boundary_prob=boundary_prob,
            cu_seqlens=None,
            mask=mask,
            inference_params=None
        )
        
        # Recompute the selected_probs from the Routing module's output.
        selected_idx = down_gate_samples.unsqueeze(-1).to(torch.long)
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx)

        # Apply residual connection and ste function (following HNet's pattern)
        y = y_dechunked * ste_func(selected_probs) + x
        
        # Compute up_merge_dst for compatibility
        # For upsampling, shift gate_samples to align with the "distribute" pattern
        up_gate_samples = down_gate_samples[:, 1:]
        up_gate_samples = torch.cat([
            up_gate_samples, 
            torch.ones(batch_size, 1, dtype=up_gate_samples.dtype, device=up_gate_samples.device)
        ], dim=1)
        up_merge_dst, _ = get_merge_dst(up_gate_samples)
        up_merge_dst = up_merge_dst.unsqueeze(-1)  # (B, S, 1)
        
        return y, up_merge_dst
