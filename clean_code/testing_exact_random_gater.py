import torch

from clean_code.flexible_bitter_llm import ExactRandomGater

gater = ExactRandomGater(embedding_dim=768, downsample_rate=0.25)

x = torch.randn(25, 4096, 768, device="cuda", dtype=torch.bfloat16)

gate_logits, gate_probs, gate_samples = gater(x)

print(f"{gate_logits.shape=} {gate_probs.shape=} {gate_samples.shape=}")

print(f"{gate_samples.sum(dim=1)=}")
print(f"{gate_samples}")