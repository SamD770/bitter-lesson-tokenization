from clean_code.flexible_bitter_llm import FlexibleBitterLLM, DownsampleRateEmbedding, ExactRandomGater, SelectTokenDownsampler
import torch

def test_equivalence():
    """
    Test that the output of the model will only differ if both the downsample rate is different and a downsample rate embedding is used.
    (we prescribe the same down_gate_samples for both models)
    """

    embedding_dim = 256
    device = "cuda"
    dtype = torch.bfloat16

    model_with_embedding = FlexibleBitterLLM(
        vocab_size=256, 
        embedding_dim=embedding_dim, 
        num_heads=8, 
        GaterClass=ExactRandomGater, # will not be used as we prescribe down_gate_samples
        DownsampleRateEmbeddingClass=DownsampleRateEmbedding,
        DownSamplerClass=SelectTokenDownsampler, # Can _Not_ use AverageTokenDownsampler as scatter_reduce is nondeterministic.
    ).to(device=device, dtype=dtype)

    model_without_embedding = FlexibleBitterLLM(
        vocab_size=256, 
        embedding_dim=embedding_dim, 
        num_heads=8, 
        GaterClass=ExactRandomGater, # will not be used as we prescribe down_gate_samples
        DownsampleRateEmbeddingClass=None,
        DownSamplerClass=SelectTokenDownsampler, # Can _Not_ use AverageTokenDownsampler as scatter_reduce is nondeterministic.
    ).to(device=device, dtype=dtype)

    batch_size = 8
    sequence_length = 4096

    x = torch.randn(batch_size, sequence_length, embedding_dim).to(device=device, dtype=dtype)

    down_gate_samples = torch.randint(0, 2, (batch_size, sequence_length)).to(device=device, dtype=dtype)

    with torch.no_grad():
        y1, _, out1 = model_with_embedding.forward_backbone(x, downsample_rate=0.5, prescribed_down_gate_samples=down_gate_samples)
        y2, _, out2 = model_with_embedding.forward_backbone(x, downsample_rate=0.25, prescribed_down_gate_samples=down_gate_samples)

        y3, _, out3 = model_without_embedding.forward_backbone(x, downsample_rate=0.5, prescribed_down_gate_samples=down_gate_samples)
        y4, _, out4 = model_without_embedding.forward_backbone(x, downsample_rate=0.25, prescribed_down_gate_samples=down_gate_samples)

        print(f"difference between downsample rates with embedding: {(y1 - y2).abs().max().item()=}")
        print(f"difference between downsample rates without embedding: {(y3 - y4).abs().max().item()=}")

        assert not torch.allclose(y1, y2)
        assert torch.allclose(y3, y4)

        print("tests passed")


if __name__ == "__main__":
    test_equivalence()
