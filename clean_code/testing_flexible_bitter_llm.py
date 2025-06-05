from .flexible_bitter_llm import FlexibleBitterLLM
from .bitter_llm import get_gemma2_attention_mask
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2DecoderLayer, Gemma2Attention, Gemma2RotaryEmbedding
import torch

my_model = FlexibleBitterLLM(
    vocab_size=256,
    embedding_dim=768,
    num_heads=12,
    downsample_rate=0.25,
    sliding_window=64,
    flash_attn=True
).to("cuda", dtype=torch.bfloat16)

my_model.eval()

sequence_length = 4096
batch_size = 32

my_x = torch.randint(0, 256, (batch_size, sequence_length)).to("cuda")

out = my_model(my_x)

logits = out["logits"]

logits.mean().backward()

print("successfully did a backward pass without OOMing")