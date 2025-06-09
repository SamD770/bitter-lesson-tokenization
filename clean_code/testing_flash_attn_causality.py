from .flexible_bitter_llm import FlexibleBitterLLM, ExactRandomGater
from .bitter_llm import get_gemma2_attention_mask
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2DecoderLayer, Gemma2Attention, Gemma2RotaryEmbedding
import torch


def test_causality(x, out, window_size=64):

    # Test causality by checking if future tokens affect past tokens
    # Pick a position in the middle to check
    mid_pos = 512
    loss = out[:, mid_pos, :].mean()
    loss.backward()

    # In a causal model, gradients for positions > mid_pos should be zero
    future_grad_sum = x.grad[:, mid_pos+1:, :].abs().sum().item()
    past_grad_sum = x.grad[:, :mid_pos+1, :].abs().sum().item()

    print(f"Causality test - future gradient sum: {future_grad_sum}")
    print(f"Causality test - past gradient sum: {past_grad_sum}")
    print(f"Model is causal: {future_grad_sum == 0}")

    # Test sliding window by checking if tokens beyond the window affect the current position
    # The sliding window is set to 64, so positions before (mid_pos - 64) should have zero gradient
    if window_size is not None:
        outside_window_grad_sum = x.grad[:, :mid_pos-window_size, :].abs().sum().item()
        inside_window_grad_sum = x.grad[:, mid_pos-window_size:mid_pos+1, :].abs().sum().item()
        
        print(f"\nSliding window test - window size: {window_size}")
        print(f"Gradient sum outside window: {outside_window_grad_sum}")
        print(f"Gradient sum inside window: {inside_window_grad_sum}")
        print(f"Sliding window working correctly: {outside_window_grad_sum == 0 and inside_window_grad_sum > 0}")

    # Reset gradients for any further operations
    x.grad = None
    x.requires_grad_(False)
    


if __name__ == "__main__":

    print("#"*100)
    print("single layer test:")
    print("#"*100)

    dtype = torch.bfloat16

    my_config = Gemma2Config(attn_implementation="flash_attention_2", sliding_window=64)
    print(f"{my_config._attn_implementation=}")
    my_model = Gemma2DecoderLayer(my_config, 0).to("cuda", dtype=dtype)
    my_rotary_emb = Gemma2RotaryEmbedding(my_config, 0).to("cuda", dtype=dtype)
    print(f"{my_model.is_sliding=} {my_model.sliding_window=}")

    # TODO: test flash attention with an uneven sequence length (it appears to be broken for this case)
    my_x = torch.randn(32, 1033, my_config.hidden_size).to("cuda", dtype=dtype)
    my_position_ids = torch.arange(1033).unsqueeze(0).expand(32, -1).to("cuda")
    my_position_embeddings = my_rotary_emb(my_x, my_position_ids)

    # Enable autograd to test causality
    my_x.requires_grad_(True)

    # Run the model
    my_out, = my_model(
        my_x, 
        my_position_embeddings,
        position_ids=my_position_ids,
    )

    test_causality(my_x, my_out, window_size=my_model.sliding_window)

    print("#"*100)
    print("model test:")
    print("#"*100)

    my_model = FlexibleBitterLLM(
        vocab_size=256, 
        embedding_dim=512, 
        num_heads=8, 
        downsample_rate=0.25, 
        sliding_window=64,
        flash_attn=True,
        down_layer_gate=ExactRandomGater(embedding_dim=512, downsample_rate=0.25)
    ).to(device="cuda", dtype=dtype)

    my_x = torch.randn(32, 1033, 512).to("cuda", dtype=dtype)

    my_x.requires_grad_(True)

    my_out, _, _ = my_model.forward_backbone(my_x)

    test_causality(my_x, my_out, window_size=None)