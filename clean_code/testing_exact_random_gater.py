import torch

from clean_code.flexible_bitter_llm import ExactRandomGater, get_merge_dst, SelectTokenDownsampler

gater = ExactRandomGater(embedding_dim=768, downsample_rate=0.25)
downsampler = SelectTokenDownsampler()

batch_size = 20
seq_len = 4096
embedding_dim = 768
x = torch.randn(batch_size, seq_len, embedding_dim, device="cuda", dtype=torch.float32) # We need to use float32 for testing this as bfloat16 will lead to rounding errors in computing the below sums:
position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

# Performed in forward():

gate_logits, gate_probs, gate_samples = gater(x)

print(f"{gate_samples.shape=} {gate_samples[:, 0, 0]=}")
print(f"{gate_samples.sum(dim=1)[:, 0]=}    {gate_samples.sum()=}")

gate_samples[:, 0] = 1.
gate_probs = torch.cat([torch.ones(batch_size, 1, 1, dtype=gate_probs.dtype).to(gate_probs.device), gate_probs[:, 1:]], dim=1)
gate_logits = torch.cat([100*torch.ones(batch_size, 1, 1, dtype=gate_logits.dtype).to(gate_logits.device), gate_logits[:, 1:]], dim=1)

gate_samples = gate_samples.squeeze(-1)

x_downsampled, position_ids_downsampled, down_merge_dst = downsampler(x, position_ids, gate_samples)

# display results:

print()
print()

print(f"{gate_samples[:, 0]=}")

down_merge_dst, n_dst = get_merge_dst(gate_samples)
print(f"{n_dst=}")

print(f"{gate_samples.sum(dim=1)=}    {gate_samples.sum()=}")
print(f"{(gate_samples.sum(dim=1)==n_dst).all().item()=}")

print(f"{gate_logits.shape=} {gate_probs.shape=} {gate_samples.shape=}")

print(f"{x_downsampled.shape=} {position_ids_downsampled.shape=} {down_merge_dst.shape=}")

# # Performed in SelectTokenDownsampler:


# # Merge the tokens into the next token where the gate is 1.)
# max_n_dst = n_dst.max().item()