from clean_code.flexible_bitter_llm import FlexibleBitterLLM
import torch
from clean_code.utils import parameter_count_string

my_model = FlexibleBitterLLM(
    vocab_size=256,
    embedding_dim=768,
    num_heads=12,
    downsample_rate=0.25,
    sliding_window=64,
    flash_attn=True
).to("cuda", dtype=torch.bfloat16)

print(f"{parameter_count_string(my_model)=}")

print(f"{my_model.byte_level_parameters_count /10**6 = }")
print(f"{my_model.mid_layer_parameters_count /10**6 = }")

