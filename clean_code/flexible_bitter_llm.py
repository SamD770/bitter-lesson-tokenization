"""
Generalise the on and off-policy Bitter LLM by simply allowing for the gate samples to be passed in as an argument.

This will allow us to use the same code for both on- and off-policy training (the off-policy is not dependent on the samples).

Also: compatible with the new Gemma2 model implementation in transformers.
"""

from torch import nn
import torch
import torch.nn.functional as F

from .bitter_llm import LinearGater, AverageTokenDownsampler, DistributeTokenUpsampler, get_gemma2_attention_mask, get_merge_dst, create_gemma2DecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2RotaryEmbedding


def flash_aware_get_attention_mask(attn_implementation, batch_size, max_seq_len, device, dtype):
    if attn_implementation == "flash_attention_2":
        # Flash attention 2 does not need the attention mask to be causal
        # Source: trust me bro (testing_flash_attn_causality.py)
        return None, None
    elif attn_implementation == "eager":
        # Eager attention needs the attention mask to be causal
        return get_gemma2_attention_mask(batch_size, max_seq_len, device, dtype)
    else:
        raise NotImplementedError(f"Attention implementation {attn_implementation} not implemented")


class FlexibleBitterLLM(nn.Module):
    # Use Gemma2DecoderLayer as a drop in replacement for the TransformerEncoderLayer, with RoPE and sliding window pre-implemented.
    # Also uses a causal mask.
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, downsample_rate: float = 0.25, sliding_window = 64, 
                 GaterClass=LinearGater, DownSamplerClass=AverageTokenDownsampler, UpsamplerClass=DistributeTokenUpsampler,
                 n_down_layers=2, n_mid_layers=6, n_up_layers=2, flash_attn=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        head_dim = embedding_dim // num_heads

        self.attn_implementation = "flash_attention_2" if flash_attn else "eager"
        print(f"using {self.attn_implementation=}")

        self.byte_layer_config = Gemma2Config(
            head_dim=head_dim,
            query_pre_attn_scalar=head_dim, 
            sliding_window=sliding_window,
            intermediate_size=embedding_dim,
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            attn_implementation=self.attn_implementation
        )

        self.deep_layer_config = Gemma2Config(
            head_dim=head_dim,
            query_pre_attn_scalar=head_dim, 
            sliding_window=None,
            intermediate_size=embedding_dim * 4, # dim_feedforward should scale inversely with the number of tokens in the sequence.
            hidden_size=embedding_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            attn_implementation=self.attn_implementation
        )

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
        self.rotary_emb = Gemma2RotaryEmbedding(config=self.byte_layer_config)


    def forward(
            self, 
            input_ids: torch.Tensor, 
            position_ids: torch.Tensor=None,
            prescribed_down_gate_samples: torch.Tensor=None,
            down_gate_mask: torch.Tensor=None
        ) -> torch.Tensor:
        """
        prescribed_down_gate_samples: if provided, use these to gate the down layers.
        down_gate_mask: if provided, use this to mask the down layers.
        """

        batch_size, max_seq_len = input_ids.shape

        x = self.embedding(input_ids)

        if position_ids is None:
            position_ids = torch.arange(max_seq_len, dtype=x.dtype).unsqueeze(0).expand(batch_size, -1).to(x.device)      
        
        # Position_ids are used for RoPE
        # cache_position is used for the cache.update() function which retrieves relevant kvs
        byte_cache_position, byte_attention_mask = flash_aware_get_attention_mask(self.attn_implementation, batch_size, max_seq_len, x.device, x.dtype)

        position_embeddings = self.rotary_emb(x, position_ids)

        # Apply down layers to byte tokens        
        for layer in self.down_layers:
            x = layer(
                x,
                position_embeddings,
                attention_mask=byte_attention_mask,
                position_ids=position_ids,
                cache_position=byte_cache_position,
            )[0]

        # Sample gating binary variables for each token.
        down_gate_logits, down_gate_probs = self.down_layer_gate(x)

        # If no down_gate_samples are provided, sample from the down_gate_probs.
        model_down_gate_samples = torch.bernoulli(down_gate_probs)

        if prescribed_down_gate_samples is None:
            down_gate_samples = model_down_gate_samples
        elif down_gate_mask is None:
            down_gate_samples = prescribed_down_gate_samples.unsqueeze(-1)
        else:
            # Where down_gate_mask is True, use the prescribed down_gate_samples
            down_gate_samples = torch.where(down_gate_mask.unsqueeze(-1), prescribed_down_gate_samples.unsqueeze(-1), model_down_gate_samples)

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
        downsampled_cache_position, downsampled_attention_mask = flash_aware_get_attention_mask(self.attn_implementation, batch_size, max_n_dst, x.device, x.dtype)

        y_downsampled = x_downsampled

        position_embeddings_downsampled = self.rotary_emb(x_downsampled, position_ids_downsampled)
        for layer in self.mid_layers:
            y_downsampled = layer(
                y_downsampled,
                position_embeddings_downsampled,
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
                position_embeddings,
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
            "model_down_gate_samples": model_down_gate_samples.to(dtype=torch.long),
            "down_gate_samples": down_gate_samples.to(dtype=torch.long),
            "down_merge_dst": down_merge_dst, 
            "up_merge_dst": up_merge_dst[:, :, 0], # This dimension is repeated.
            "n_dst": n_dst,
            "position_ids": position_ids,
            "key_values": None
        }

        return out
    