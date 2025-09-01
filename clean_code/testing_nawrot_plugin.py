"""
Test two things: 
1. that forward/backward passes through the model with NawrotGater, NawrotDownsampler and NawrotUpsampler work
2. that the gater gradient is populated during the backward pass
"""
from .flexible_bitter_llm import FlexibleBitterLLM
from .nawrot_plugin import NawrotDownsampler, NawrotUpsampler, NawrotGater
import torch

if __name__ == "__main__":

    dtype = torch.bfloat16
    device = "cuda"

    my_model = FlexibleBitterLLM(
        vocab_size=256, 
        embedding_dim=128, 
        num_heads=2, 
        downsample_rate=0.25, 
        sliding_window=64,
        flash_attn=False,
        DownSamplerClass=NawrotDownsampler,
        UpsamplerClass=NawrotUpsampler,
        GaterClass=NawrotGater,
    ).to(device="cuda", dtype=dtype)

    print(f"{my_model.down_layer_gate.boundary_predictor=}")

    my_x = torch.randn(32, 2048, 128).to("cuda", dtype=dtype)

    my_out, _, _ = my_model.forward_backbone(my_x)

    print(f"{my_out.shape=}")

    my_out.sum().backward()

    for my_grad in my_model.down_layer_gate.boundary_predictor.parameters():

        print(f"{my_grad.shape=}")
        print(f"{my_grad.sum().item()=}")