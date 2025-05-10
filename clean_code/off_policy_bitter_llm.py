from clean_code.bitter_llm import CausalGemmaMiniBitterLLM, RandomGater, get_gemma2_attention_mask, get_merge_dst, discounted_rewards_torch, display_gating, display_gpu_memory

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class OffPolicyBitterLLM(CausalGemmaMiniBitterLLM):
    def __init__(self, *args, OffPolicyGaterClass=RandomGater, use_off_policy=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_policy_gater = OffPolicyGaterClass(self.embedding_dim, self.downsample_rate)
        self.use_off_policy = use_off_policy

    def forward(
            self, 
            input_ids: torch.Tensor, 
            position_ids: torch.Tensor=None      
        ) -> torch.Tensor:
        """
        if not self.use_off_policy, this should be identical to the parent class.
        """

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
        on_policy_logits, on_policy_probs = self.down_layer_gate(x)

        if self.use_off_policy:
            down_gate_logits, down_gate_probs = self.off_policy_gater(x)
        else:
            down_gate_logits = on_policy_logits
            down_gate_probs = on_policy_probs
        
        down_gate_samples = torch.bernoulli(down_gate_probs)

        # Hack: ensure that we always gate on the first token:
        # We need to also set the corresponding logits to 100. to avoid the gradient exploding based on this.
        down_gate_samples[:, 0] = 1.
        on_policy_probs = torch.cat([torch.ones(batch_size, 1, 1, dtype=on_policy_probs.dtype).to(on_policy_probs.device), on_policy_probs[:, 1:]], dim=1)
        on_policy_logits = torch.cat([100*torch.ones(batch_size, 1, 1, dtype=on_policy_logits.dtype).to(on_policy_logits.device), on_policy_logits[:, 1:]], dim=1)

        if self.use_off_policy: # For the on-policy case this has already been done.
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
            "on_policy_probs": on_policy_probs.squeeze(-1),
            "on_policy_logits": on_policy_logits.squeeze(-1),
            "down_gate_samples": down_gate_samples.to(dtype=torch.long),
            "down_merge_dst": down_merge_dst, 
            "up_merge_dst": up_merge_dst[:, :, 0], # This dimension is repeated.
            "n_dst": n_dst,
            "position_ids": position_ids,
            "key_values": None
        }

        return out
    


def off_policy_bitter_tokenizer_training_step(model, batch, optimizer, learn_gating=True, downsample_rate_target=0.25, rate_loss_weight=2., discount_rate = 0.9):
    """
    Assume that batch is torch.tensor of token ids of shape (batch, sequence_length). returns a dict of floats of the training losses for the batch.
    """
    batch_size, _ = batch.shape

    optimizer.zero_grad()

    out = model(batch)
    logits = out["logits"]
    down_gate_samples = out["down_gate_samples"]
    off_policy_gate_probs = out["down_gate_probs"]
    on_policy_probs = out["on_policy_probs"]
    on_policy_logits = out["on_policy_logits"]
    # Compute autoregressive loss: log probability of next token.
    next_token_ids = batch[:, 1:]
    current_token_logits = logits[:, :-1]
    next_token_logits = F.cross_entropy(current_token_logits.transpose(1, 2), next_token_ids, reduction="none") # Transpose as F.cross_entropy wants shape [batch, classes, ...]
    ar_loss = next_token_logits.mean()

    true_downsample_rate = on_policy_probs.mean()

    if learn_gating:
        # Compute gating loss: discounted log probabilities of following token(s).
        next_token_logits_padded = torch.cat([next_token_logits, torch.zeros(batch_size, 1, device=next_token_logits.device)], dim=-1) # Pad the last reward as zero
        discounted_rewards = discounted_rewards_torch(next_token_logits_padded, discount_rate)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean(dim=0)) # Simple estimate of the advantage

        # action 0 = continue, action 1 = gate
        action_log_probs = torch.stack([torch.zeros_like(on_policy_logits), on_policy_logits], dim=1) # As a sigmoid is equivalent to having one logit as 0.
        selected_action_log_probs = F.cross_entropy(action_log_probs, down_gate_samples, reduction="none")

        # likelihood_ratios [:, :, 1] gives the likelihood ratio for the action of gating.
        likelihood_ratios = torch.stack([
            (1 - on_policy_probs) / (1 - off_policy_gate_probs),
            on_policy_probs / off_policy_gate_probs
        ], dim=-1)
        likelihood_ratios = likelihood_ratios.detach() # Detach as we don't want to backpropagate through this.

        # Get the likelihood ratios for the selected actions for importance sampling.
        selected_action_likelihood_ratios = likelihood_ratios.gather(dim=-1, index=down_gate_samples.unsqueeze(-1))
        selected_action_likelihood_ratios = selected_action_likelihood_ratios.squeeze(-1)

        gating_loss = - (selected_action_likelihood_ratios * discounted_rewards * selected_action_log_probs).mean() # Negative as we want to maximise the reward.

        # Hacky additional consistency loss: make the downsampling rate match the training gating.
        down_gate_rate_loss = rate_loss_weight*(downsample_rate_target - true_downsample_rate) **2

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



def off_policy_bitter_tokenizer_training_loop(model, train_dataset, tokenizer, learn_gating=True, 
                                   num_epochs=1, batch_size=128, batch_limit=None, max_seq_length=1024, 
                                   batch_print_every=10, print_example_gating=True, discount_rate=0.9):

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

            loss_dict = off_policy_bitter_tokenizer_training_step(model, batch, optimizer, learn_gating=learn_gating, discount_rate=discount_rate)
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
