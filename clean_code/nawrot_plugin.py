
import torch
from torch import nn

import nawrot_downsampler
from .modules import get_merge_dst


class NawrotDownsampler(nn.Module):
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, gate_samples: torch.Tensor) -> torch.Tensor:

        # Nawrot's downsampler is numerically unstable, so we need to upcast to float32 in order to avoid shape mismatches.

        x_dtype_store = x.dtype
        position_ids_dtype_store = position_ids.dtype
        x = x.to(torch.float32)
        position_ids = position_ids.to(torch.float32)

        x_downsampled = self.downsample_x(x, gate_samples)
        position_ids_downsampled = self.downsample_position_ids(position_ids, gate_samples)

        down_merge_dst, _ = get_merge_dst(gate_samples)
        down_merge_dst = down_merge_dst.unsqueeze(-1)

        x_downsampled = x_downsampled.to(x_dtype_store)
        position_ids_downsampled = position_ids_downsampled.to(position_ids_dtype_store)


        return x_downsampled, position_ids_downsampled, down_merge_dst
    
    
    def downsample_x(self, x: torch.Tensor, hard_boundaries: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor x using the Nawrot et al. 2023 method."""
        # x is of shape [bs, seq_len, emb_dim], but nawrot_downsampler expects [seq_len, bs, emb_dim]
        x = x.transpose(0, 1)

        # Downsample the input
        x = downsample_without_null_group(
            hard_boundaries, 
            x
        )

        x = x.transpose(0, 1)
        return x

    def downsample_position_ids(self, position_ids: torch.Tensor, hard_boundaries: torch.Tensor) -> torch.Tensor:
        """Downsamples the position ids using the Nawrot et al. 2023 method."""
        # position_ids is of shape [bs, seq_len], but nawrot_downsampler expects [seq_len, bs, d]
        position_ids = position_ids.transpose(0, 1)
        position_ids = position_ids.unsqueeze(-1)

        position_ids = downsample_without_null_group(
            hard_boundaries, 
            position_ids
        )

        position_ids = position_ids.squeeze(-1)
        position_ids = position_ids.transpose(0, 1)
        return position_ids
    
    
class NawrotUpsampler(nn.Module):

    def forward(self, x: torch.Tensor, gate_samples: torch.Tensor) -> torch.Tensor:

        
        x_dtype_store = x.dtype
        x = x.to(torch.float32)

        x_upsampled = self.upsample_x(x, gate_samples)
        up_merge_dst, _ = get_merge_dst(gate_samples)
        up_merge_dst = up_merge_dst.unsqueeze(-1)

        x_upsampled = x_upsampled.to(x_dtype_store)

        return x_upsampled, up_merge_dst

    def upsample_x(self, x: torch.Tensor, hard_boundaries: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor x using the Nawrot et al. 2023 method."""
        # x is of shape [bs, seq_len, emb_dim], but nawrot_downsampler expects [seq_len, bs, emb_dim]
        x = x.transpose(0, 1)

        # Upsample the input
        x_upsampled = upsample_without_null_group(
            hard_boundaries, 
            x
        )

        # Return to the original shape
        x_upsampled = x_upsampled.transpose(0, 1)

        return x_upsampled



def downsample_without_null_group(boundaries, hidden):
    """
    nawrot_downsampler.downsample uses a "null group", a learned vector prepended as the first chunk. As we already do this with a <bos> token and forcing gating at the first byte, we don't want it.
    This is a minimal modification of their function to do so.
    """

    boundaries = boundaries.to(hidden.dtype) 
    foo = nawrot_downsampler.common(boundaries, upsample=False)  # B x L x S

    bar = nawrot_downsampler.final(foo=foo, upsample=False)  # B x L x S


    foo = foo.to(hidden.dtype)
    bar = bar.to(hidden.dtype)

    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)

    return shortened_hidden



def upsample_without_null_group(boundaries, shortened_hidden):
    """
    nawrot_downsampler.usample  expects the "null group", and so makes the first group one too large.
    This is a minimal modification of their function to do so.
    """

    foo = common_without_null_group(boundaries, upsample=True)  # B x L x S
    bar = nawrot_downsampler.final(foo, upsample=True)  # B x L x S

    return torch.einsum('sbd,bls->lbd', shortened_hidden, bar)



def common_without_null_group(boundaries, upsample=False):
    """
    Copied from nawrot_downsampler.common, but modified for the upsampling case without a null group.
    """
    boundaries = boundaries.clone()

    n_segments = boundaries.sum(dim=-1).max().item()

    if n_segments == 0:
        return None

    tmp = torch.zeros_like(
        boundaries
    ).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device
    )

    hh1 = boundaries.cumsum(1)

    if not upsample:
        hh1 -= boundaries

    foo = tmp - hh1.unsqueeze(-1)
    foo = foo + 1 # This is the correction necessary to make the model causal.

    return foo



class NawrotGater(nn.Module):
    def  __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.boundary_predictor = nawrot_downsampler.BoundaryPredictor(embedding_dim, embedding_dim, "relu", 
                                                                       temp=1.0, prior=self.downsample_rate, bp_type="gumbel")

        
    def compute_boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the boundaries for the input tensor x using the Nawrot et al. 2023 method."""
        # x is of shape [bs, seq_len, emb_dim], but nawrot_downsampler expects [seq_len, bs, emb_dim]
        x = x.transpose(0, 1)

        # Get the boundary predictions
        soft_boundaries, hard_boundaries = self.boundary_predictor(x)
        
        return soft_boundaries, hard_boundaries


    def forward(self, x: torch.Tensor, downsample_rate=None) -> torch.Tensor:

        if downsample_rate is not None:
            raise NotImplementedError("passed downsample_rate is not implemented for NawrotGater")
            
        soft_boundaries, hard_boundaries = self.compute_boundaries(x)
        gate_probs = soft_boundaries.unsqueeze(-1)
        gate_samples = hard_boundaries.unsqueeze(-1)

        # take the inverse sigmoid of the probability to get the logits
        gate_logits = torch.log(gate_probs / (1 - gate_probs))
        
        return gate_logits, gate_probs, gate_samples
