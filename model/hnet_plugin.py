import torch
from torch import nn
from typing import Tuple
from copy import deepcopy
# Problem: different methods require different things for up/downsampling beyond probs, logits and samples.
# We can abstract this later.
import torch.nn.functional as F


class HNetGater(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        downsample_rate: float, 
        qk_identity_init: float = True
    ):
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # In the H-net code, this initialization is used for the Gater.
        # https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py
        if qk_identity_init:
            with torch.no_grad():
                self.q_proj_layer.weight.copy_(torch.eye(embedding_dim))
                self.k_proj_layer.weight.copy_(torch.eye(embedding_dim))
                self.q_proj_layer.weight._no_reinit = True
                self.k_proj_layer.weight._no_reinit = True


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            x B S D
        Returns:
            gate_logits B S
            gate_probs B S
            gate_samples B S
        """
        batch_size, _, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)

        p = (1 - F.cosine_similarity(q[:, 1:], k[:, :-1], dim=-1)) / 2
        p = torch.cat([torch.ones(batch_size,), p])
        b = (p > 0.5)

        gate_probs = p
        gate_samples = b
        gate_logits = torch.log(gate_probs / (1 - gate_probs))

        return gate_logits, gate_probs, gate_samples,


# problem 2: need to compute the ema in a stateful way:

# z_t = p_t x_t + (1 - p_t) x

class HNetUpsampler(nn.Module):
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
        z_hat = x_downsampled
        gate_probs_downsampled = ...

        # Todo: compute the ema z_t = p_t z_hat_t + (1 - p_t) z_{t-1}
        
        
        raise NotImplementedError()

