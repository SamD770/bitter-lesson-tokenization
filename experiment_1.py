from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
from typing import List

from copy import deepcopy
import torch

from scipy.signal import lfilter
import numpy as np

import pandas as pd

from torch.utils.data import DataLoader

import datasets


device = "cuda"
assert torch.cuda.is_available()

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
    
byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

import os

username = "sdauncey"
scratch_dir = f"/scratch/{username}/tokenizer_training"

if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)


# Download a portion of OpenWebText dataset
# This will download a subset of the OpenWebText corpus
print("Downloading OpenWebText dataset...")

# Load a small portion of OpenWebText (1% of the dataset)
openwebtext_8k = datasets.load_dataset(
    "openwebtext",
    split="train[:1%]",  # Using only 1% samples of the dataset for now.
    cache_dir=os.path.join(scratch_dir, "openwebtext_8k_cache"),
    trust_remote_code=True
)

print(f"Downloaded {len(openwebtext_8k)} examples from OpenWebText")

# Display a sample
print("\nSample text from OpenWebText:")
print(openwebtext_8k[0]['text'][:500] + "...")

def get_merge_dst(gate_samples: torch.Tensor) -> torch.Tensor:
    """
    Returns (merge_dst, dst_idx) the merge destination for each token in the sequence and the number of unique merge destinations.
    For now, has a janky python for-loop implementation.
    Input is a tensor of shape (batch_size, sequence_length) with 0 tokens are merged into the next 1 token.
    """
    batch_size, seq_len = gate_samples.shape
    merge_dst = torch.zeros_like(gate_samples, dtype=torch.long)
    n_dst = torch.zeros(batch_size, dtype=torch.long)

    # Process each batch separately
    for b in range(batch_size):
        dst_idx = 0
        for i in range(seq_len):
            merge_dst[b, i] = dst_idx
            if gate_samples[b, i] == 1 and i < seq_len - 1:
                # If previous position had gate=1, keep the same destination
                dst_idx += 1

        n_dst[b] = dst_idx + 1

    return merge_dst, n_dst


class MiniBitterLLM(nn.Module):
    # A mini BitterLLM with 2 down, 4 mid, and 2 up layers. As a vibe check on the idea.
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, dropout: float=0.01, downsample_rate: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.down_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_dim, num_heads, dropout=dropout, batch_first=True, dim_feedforward=embedding_dim) for _ in range(2)
        ])

        # dim_feedforward should scale inversely with the number of tokens in the sequence.
        self.mid_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_dim, num_heads, dropout=dropout, batch_first=True, dim_feedforward=embedding_dim*4) for _ in range(4) 
        ])

        self.up_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_dim, num_heads, dropout=dropout, batch_first=True, dim_feedforward=embedding_dim) for _ in range(2)
        ])

        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        # Initialize a gate for each layer.
        layer_gate_init = nn.Linear(embedding_dim, 1)

        # Copy the gate for each layer. 
        # Initializing by copying inductively biases the model to tokenize in a later layer if the gate is high but the model chose not to.
        self.down_layer_gate = deepcopy(layer_gate_init)
        self.up_layer_gate = deepcopy(layer_gate_init)
        self.downsample_rate = downsample_rate

    def apply_local_layers(self, layers, x: torch.Tensor, context_window_length) -> torch.Tensor:
        """Again a janky python for-loop implementation that re-constructs the causal mask for each layer."""
        _, seq_len, _ = x.shape

        # Create causal mask for context length of 64
        mask = torch.ones(seq_len, seq_len) * float('-inf')

        for i in range(seq_len):
            # Allow attention to self and previous window_size tokens
            start_idx = max(0, i - context_window_length + 1)
            mask[i, start_idx:i+1] = 0.0
        
        # Process through down layers with the specified context length
        for layer in layers:
            x = layer(x, src_mask=mask, is_causal=True)

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, _ = x.shape

        x = self.embedding(x)

        # Apply down layers  byte tokens        
        x = self.apply_local_layers(self.down_layers, x, 64)

        # Sample gating binary variables for each token.
        down_gate_logits = self.down_layer_gate(x)
        down_gate_probs = F.sigmoid(down_gate_logits)

        # Re-scale the gate probabilities to control the downsampling rate.
        down_gate_probs = down_gate_probs * (self.downsample_rate / down_gate_probs.mean())
        down_gate_samples = torch.bernoulli(down_gate_probs)

        # Hack: ensure for now that we always gate on the first token:
        down_gate_samples[:, 0] = 1.

        # Merge the tokens into the next token where the gate is 1.
        down_gate_samples = down_gate_samples.squeeze(-1)
        down_merge_dst, n_dst = get_merge_dst(down_gate_samples)
        down_merge_dst = down_merge_dst.unsqueeze(-1).expand(-1, -1, self.embedding_dim)

        x_downsampled = torch.zeros(batch_size, n_dst.max(), self.embedding_dim, dtype=torch.float32).to(x.device)
        x_downsampled = torch.scatter_reduce(x_downsampled, dim=1, index=down_merge_dst, src=x, reduce="mean", include_self=False)

        # Apply mid layers to merged tokens and compute the deviation
        y_downsampled = self.apply_local_layers(self.mid_layers, x_downsampled, 64*4)
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
        y = self.apply_local_layers(self.up_layers, y, 64)

        # Apply second gating to the downsampled output, for use in inference and a consistency loss in training.
        up_gate_logits = self.up_layer_gate(y)
        up_gate_probs = F.sigmoid(up_gate_logits)
        up_gate_probs = up_gate_probs * (self.downsample_rate / up_gate_probs.mean())

        # Map residual stream to logits
        logits = self.output_layer(y)
        logits = F.log_softmax(logits, dim=-1)

        out = {
            "logits": logits,
            "down_gate_probs": down_gate_probs.squeeze(-1),
            "up_gate_probs": up_gate_probs.squeeze(-1),
            "down_gate_samples": down_gate_samples.to(dtype=torch.long),
            "up_gate_samples": up_gate_samples.to(dtype=torch.long),
            "down_merge_dst": down_merge_dst[:, :, 0], # This dimension is repeated.
            "up_merge_dst": up_merge_dst[:, :, 0],
            "n_dst": n_dst,
        }

        return out



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


def bitter_tokenizer_training_step(model, batch, optimizer):
    """
    Assume that batch is torch.tensor of token ids of shape (batch, sequence_length). returns a dict of floats of the training losses for the batch.
    """
    batch_size, _ = batch.shape

    optimizer.zero_grad()

    out = model(batch)
    logits = out["logits"]
    down_gate_samples = out["down_gate_samples"]
    down_gate_probs = out["down_gate_probs"]
    up_gate_samples = out["up_gate_samples"]
    up_gate_probs = out["up_gate_probs"]

    # Compute autoregressive loss: log probability of next token.
    next_token_ids = batch[:, 1:]
    current_token_logits = logits[:, :-1]
    next_token_logits = F.cross_entropy(current_token_logits.transpose(1, 2), next_token_ids, reduction="none") # Transpose as F.cross_entropy wants shape [batch, classes, ...]
    ar_loss = next_token_logits.mean()

    # Compute gating loss: discounted log probabilities of following token(s).
    discount_rate = 0.95
    next_token_logits_padded = torch.cat([next_token_logits, torch.zeros(batch_size, 1, device=next_token_logits.device)], dim=-1) # Pad the last reward as zero
    discounted_rewards = discounted_rewards_torch(next_token_logits_padded, discount_rate)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) # Simple estimate of the advantage

    # action 0 = continue, action 1 = gate
    action_log_probs = torch.stack([(1 - down_gate_probs).log() , down_gate_probs.log()], dim=1)
    selected_action_log_probs = F.cross_entropy(action_log_probs, down_gate_samples, reduction="none")
    gating_loss =  - (discounted_rewards * selected_action_log_probs).mean() # Negative as we want to maximise the reward.

    # Compute consistency loss: minimize difference between training gating and inference gating
    up_gating_log_probs = torch.stack([(1 - up_gate_probs).log() , up_gate_probs.log()], dim=1)
    consistency_loss = F.cross_entropy(up_gating_log_probs, up_gate_samples, reduction="mean")

    # Hacky additional consistency loss: make the downsampling rate match the training gating.
    down_gate_rate_loss = (model.downsample_rate - down_gate_probs.mean()) **2
    up_gate_rate_loss = (model.downsample_rate - up_gate_probs.mean()) **2
    rate_consistency_loss = 5.*(down_gate_rate_loss + up_gate_rate_loss)

    # Optimizer step
    total_loss = ar_loss + gating_loss + consistency_loss + rate_consistency_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    out = {
        "ar_loss": ar_loss.item(),
        "gating_loss": gating_loss.item(),
        "consistency_loss": consistency_loss.item(),
        "total_loss": total_loss.item(),
        "selected_action_ce": selected_action_log_probs.mean().item()
    }

    return out

import psutil
import gc
import tracemalloc

def display_gating(tokens_ids, merge_dst):
    """Display how a SmallBitterLLM merges a sequence. token_ids and merge_dst are tensors of shape (sequence_length,)."""
    previous_merge_dst = 0
    for t_id, merge_destinantion in zip(tokens_ids, merge_dst):
        merge_destinantion = merge_destinantion.item()
        
        if merge_destinantion != previous_merge_dst:
            print(f"|", end="")
            previous_merge_dst = merge_destinantion
        
        t_txt = byte5_tokenizer.decode(t_id)
        print(f"{t_txt}", end="")


def bitter_tokenizer_training_loop(model, train_dataset):
    # TODO: validation dataset
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # See how the model merges a sequence.
    test_string = openwebtext_8k[-1]["text"][:200]
    test_batch = byte5_tokenizer.encode(test_string, return_tensors="pt", padding=True)

    # Initialize model and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        model = model.to(device)
        train_losses = []

        print(f"Epoch {epoch+1}/{num_epochs}, GPU usage:")
        display_gpu_memory()
        
        # CPU memory tracking
        process = psutil.Process()
        print(f"CPU Memory before epoch: {process.memory_info().rss / (1024 * 1024):.2f} MB")

        batch_count = 0
        for batch in train_loader:

            batch = batch["text"]
            batch = byte5_tokenizer(batch, return_tensors="pt", padding=True)["input_ids"]
            batch = batch[:, :4096]  # Truncate to maximum length of 4096 to save GPU memory.
            batch = batch.to(device)

            loss_dict = bitter_tokenizer_training_step(model, batch, optimizer)
            train_losses.append(loss_dict)

            # Memory tracking for each batch
            if batch_count % 10 == 0:
                # print(f"CPU Memory at batch {batch_count}: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                # current, peak = tracemalloc.get_traced_memory()
                # print(f"Current memory usage: {current / 10**6:.2f} MB; Peak: {peak / 10**6:.2f} MB")
                
                # # Force garbage collection to see if memory is being properly released
                # gc.collect()
                # print(f"After GC: {process.memory_info().rss / (1024 * 1024):.2f} MB")
                print(f"Batch {batch_count} ar train loss: {loss_dict['ar_loss']}")
                with torch.no_grad():
                    out = model(test_batch)
                display_gating(test_batch[0], out["down_merge_dst"][0])

            batch_count += 1

        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train loss: {np.mean([l['total_loss'] for l in train_losses]):.4f}")
        
        # Memory snapshot at end of epoch
        print(f"CPU Memory after epoch: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("[ Top 10 memory consumers ]")
        for stat in top_stats[:10]:
            print(stat)
    
    # Stop memory tracking
    tracemalloc.stop()
    
    train_losses = pd.DataFrame(train_losses)

    return train_losses


my_model = MiniBitterLLM(vocab_size=byte5_tokenizer.vocab_size, embedding_dim=128, num_heads=4, downsample_rate=0.25)
print(f"my_model has {parameter_count_string(my_model)} parameters")

train_losses = bitter_tokenizer_training_loop(my_model, openwebtext_8k)

model_file_name = "bitter-llm-v2.pt"
net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")

# Save the model to the specified directory
os.makedirs(net_scratch_dir, exist_ok=True)
model_save_file = os.path.join(net_scratch_dir, model_file_name)
torch.save(my_model, model_save_file)
print(f"Model saved to {model_save_file}")

# Save the train losses to the specified directory
train_losses_file = os.path.join(net_scratch_dir, "train_losses.csv")
train_losses.to_csv(train_losses_file, index=False)
print(f"Train losses saved to {train_losses_file}")
