from .flexible_bitter_llm import FlexibleBitterLLM
from .bitter_llm import get_gemma2_attention_mask
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2DecoderLayer, Gemma2Attention, Gemma2RotaryEmbedding
import torch


my_config = Gemma2Config(attn_implementation="flash_attention_2", sliding_window=64)
print(f"{my_config._attn_implementation=}")
my_model = Gemma2DecoderLayer(my_config, 0).to("cuda", dtype=torch.float16)
my_rotary_emb = Gemma2RotaryEmbedding(my_config, 0).to("cuda", dtype=torch.float16)
print(f"{my_model.is_sliding=} {my_model.sliding_window=}")

# TODO: test flash attention with an uneven sequence length (it appears to be broken for this case)
my_x = torch.randn(32, 1033, my_config.hidden_size).to("cuda", dtype=torch.float16)
my_position_ids = torch.arange(1033).unsqueeze(0).expand(32, -1).to("cuda")
my_position_embeddings = my_rotary_emb(my_x, my_position_ids)

# Enable autograd to test causality
my_x.requires_grad_(True)

# Run the model
out, = my_model(
    my_x, 
    my_position_embeddings,
    position_ids=my_position_ids,
)

# Test causality by checking if future tokens affect past tokens
# Pick a position in the middle to check
mid_pos = 512
loss = out[:, mid_pos, :].sum()
loss.backward()

# In a causal model, gradients for positions > mid_pos should be zero
future_grad_sum = my_x.grad[:, mid_pos+1:, :].abs().sum().item()
past_grad_sum = my_x.grad[:, :mid_pos+1, :].abs().sum().item()

print(f"Causality test - future gradient sum: {future_grad_sum}")
print(f"Causality test - past gradient sum: {past_grad_sum}")
print(f"Model is causal: {future_grad_sum == 0}")

# Test sliding window by checking if tokens beyond the window affect the current position
# The sliding window is set to 64, so positions before (mid_pos - 64) should have zero gradient
if my_model.sliding_window is not None:
    window_size = my_model.sliding_window
    outside_window_grad_sum = my_x.grad[:, :mid_pos-window_size, :].abs().sum().item()
    inside_window_grad_sum = my_x.grad[:, mid_pos-window_size:mid_pos+1, :].abs().sum().item()
    
    print(f"\nSliding window test - window size: {window_size}")
    print(f"Gradient sum outside window: {outside_window_grad_sum}")
    print(f"Gradient sum inside window: {inside_window_grad_sum}")
    print(f"Sliding window working correctly: {outside_window_grad_sum == 0 and inside_window_grad_sum > 0}")

# Reset gradients for any further operations
my_x.grad = None
my_x.requires_grad_(False)