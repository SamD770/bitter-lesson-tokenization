import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from clean_code.bitter_llm import Gemma2Config, create_gemma2DecoderLayer, get_gemma2_attention_mask, display_gpu_memory
import nawrot_downsampler


class NawrotDownsampler(nn.Module):
    def __init__(self, embedding_dim: int, downsample_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.downsample_rate = downsample_rate
        self.boundary_predictor = nawrot_downsampler.BoundaryPredictor(embedding_dim, embedding_dim, "relu", 
                                                                       temp=1.0, prior=self.downsample_rate, bp_type="gumbel")
        
        self.null_group = nn.Parameter(torch.Tensor(1, 1, embedding_dim).zero_())
        nn.init.normal_(self.null_group)

    def compute_boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the boundaries for the input tensor x using the Nawrot et al. 2023 method."""
        # x is of shape [bs, seq_len, emb_dim], but nawrot_downsampler expects [seq_len, bs, emb_dim]
        x = x.transpose(0, 1)

        # Get the boundary predictions
        soft_boundaries, hard_boundaries = self.boundary_predictor(x)
        
        return soft_boundaries, hard_boundaries

    def downsample(self, x: torch.Tensor, hard_boundaries: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor x using the Nawrot et al. 2023 method."""
        # x is of shape [bs, seq_len, emb_dim], but nawrot_downsampler expects [seq_len, bs, emb_dim]
        x = x.transpose(0, 1)

        # Downsample the input
        x = nawrot_downsampler.downsample(
            hard_boundaries, 
            x, 
            self.null_group
        )

        # Return to the original shape
        x = x.transpose(0, 1)
        return x
    
    def downsample_position_ids(self, position_ids: torch.Tensor, hard_boundaries: torch.Tensor) -> torch.Tensor:
        """Downsamples the position ids using the Nawrot et al. 2023 method."""
        # position_ids is of shape [bs, seq_len], but nawrot_downsampler expects [seq_len, bs, d]
        position_ids = position_ids.transpose(0, 1)
        position_ids = position_ids.unsqueeze(-1)

        position_ids = nawrot_downsampler.downsample(
            hard_boundaries, 
            position_ids, 
            torch.Tensor(1, 1, 1).zero_().to(position_ids.device)
        )

        position_ids = position_ids.squeeze(-1)
        position_ids = position_ids.transpose(0, 1)
        return position_ids
    
    def upsample(self, x: torch.Tensor, hard_boundaries: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor x using the Nawrot et al. 2023 method."""
        # x is of shape [bs, seq_len, emb_dim], but nawrot_downsampler expects [seq_len, bs, emb_dim]
        x = x.transpose(0, 1)

        # Upsample the input
        x = nawrot_downsampler.upsample(
            hard_boundaries, 
            x
        )

        # Return to the original shape
        x = x.transpose(0, 1)

        return x
    
    def consistency_loss(self, hard_boundaries: torch.Tensor) -> torch.Tensor:
        return self.boundary_predictor.calc_loss(
            preds=hard_boundaries, gt=None
        )


# TODO: make NawrotGemmaMiniBitterLLM and CausalGemmaMiniBitterLLM inherit from a common class.

class NawrotGemmaMiniBitterLLM(nn.Module):
    # Use Gemma2DecoderLayer as a drop in replacement for the TransformerEncoderLayer, with RoPE and sliding window pre-implemented.
    # Also uses a causal mask.
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, downsample_rate: float = 0.25, sliding_window = 64, n_down_layers=2, n_mid_layers=6, n_up_layers=2):
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
        
        self.downsample_rate = downsample_rate
        self.downsampler = NawrotDownsampler(embedding_dim, downsample_rate)


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

        soft_boundaries, hard_boundaries = self.downsampler.compute_boundaries(x)
        x_downsampled = self.downsampler.downsample(x, hard_boundaries)

        position_ids_downsampled = self.downsampler.downsample_position_ids(position_ids, hard_boundaries)
        # Apply mid layers to merged tokens and compute the deviation
        downsampled_cache_position, downsampled_attention_mask = get_gemma2_attention_mask(batch_size, x_downsampled.shape[1], x.device, x.dtype)

        y_downsampled = x_downsampled

        for layer in self.mid_layers:
            y_downsampled = layer(
                y_downsampled, 
                attention_mask=downsampled_attention_mask,
                position_ids=position_ids_downsampled,
                cache_position=downsampled_cache_position,
            )[0]
        
        deviation = y_downsampled - x_downsampled        

        # Add the upsampled deviation to the input to the middle layers
        upsampled_deviation = self.downsampler.upsample(deviation, hard_boundaries)

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
            "down_gate_samples": hard_boundaries,
            "soft_boundaries": soft_boundaries,
            "position_ids": position_ids,
            "key_values": None
        }

        return out


def NawrotBitterLLM_training_step(model, batch, optimizer, learn_gating=True):
    """
    Assume that batch is torch.tensor of token ids of shape (batch, sequence_length). returns a dict of floats of the training losses for the batch.
    """
    batch_size, _ = batch.shape

    optimizer.zero_grad()

    out = model(batch)
    logits = out["logits"]
    hard_boundaries = out["down_gate_samples"]
    
    # Compute autoregressive loss: log probability of next token.
    next_token_ids = batch[:, 1:]
    current_token_logits = logits[:, :-1]
    next_token_logits = F.cross_entropy(current_token_logits.transpose(1, 2), next_token_ids, reduction="none") # Transpose as F.cross_entropy wants shape [batch, classes, ...]
    ar_loss = next_token_logits.mean()
    consistency_loss = model.downsampler.consistency_loss(hard_boundaries)
    total_loss = ar_loss + consistency_loss

    # Optimizer step
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    out = {
        "ar_loss": ar_loss.item(),
        "consistency_loss": consistency_loss.item(),
        "total_loss": total_loss.item(),
    }

    return out


def nawrot_training_loop(model, train_dataset, tokenizer, num_epochs=1, batch_size=128, batch_limit=None, max_seq_length=1024, 
                                   batch_print_every=10):

    # TODO: validation dataset
    # Create data loaders
    # Create distributed sampler and data loader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

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

            loss_dict = NawrotBitterLLM_training_step(model, batch, optimizer)
            train_losses.append(loss_dict)

            # See if this fixes the OOMing issue.
            optimizer.zero_grad()

            # Memory tracking for each batch
            if batch_count % batch_print_every == 0:
                print(f"Batch {batch_count} ar train loss: {loss_dict['ar_loss']} nats/token consistency loss: {loss_dict['consistency_loss']}")

            if batch_limit is not None and batch_count > batch_limit:
                break

        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train loss: {np.mean([l['total_loss'] for l in train_losses]):.4f}")
    
    train_losses = pd.DataFrame(train_losses)

    return train_losses