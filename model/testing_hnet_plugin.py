from model.hnet_downsampler import DeChunkLayer
from model.hnet_plugin import HNetDownsampler, HNetUpsampler, HNetGater
import torch
from model.testing_flash_attn_causality import test_causality
from model.model import AutoregressiveUnet


def test_dechunk_layer():
    """
    Tests that the container runs the DeChunkLayer correctly.
    """
    device = torch.device("cuda")
    dtype = torch.float16

    batch_size, seq_len, embedding_dim = 4, 2048, 128

    my_dechunk_layer = DeChunkLayer(embedding_dim)

    boundary_probs = torch.rand(batch_size, seq_len).to(device, dtype)
    boundary_mask = torch.bernoulli(boundary_probs).to(torch.bool)

    boundary_probs = torch.stack([(1 - boundary_probs), boundary_probs], dim=-1)

    downsampled_seqlen = boundary_mask.sum(dim=-1).max().item()
    hidden_states = torch.randn(batch_size, downsampled_seqlen, embedding_dim).to(device, dtype)

    print(f"{hidden_states.shape=}")
    print(f"{boundary_mask.shape=}")
    print(f"{boundary_probs.shape=}")

    mask = torch.ones(batch_size, downsampled_seqlen, device=device)

    out = my_dechunk_layer(hidden_states, boundary_mask, boundary_probs, mask=mask)
    print(f"{out.shape=}")


def test_forward_pass():
    """
    Tests that the forward pass of the HNetDownsampler works. and is causal.
    """
    dtype = torch.bfloat16
    device = "cuda"

    my_model = AutoregressiveUnet(
        vocab_size=256, 
        embedding_dim=128, 
        num_heads=2, 
        downsample_rate=0.25, 
        sliding_window=64,
        flash_attn=True,
        DownSamplerClass=HNetDownsampler,
        UpsamplerClass=HNetUpsampler,
        GaterClass=HNetGater,
        upsampler_kwargs={"embedding_dim": 128},
    ).to(device="cuda", dtype=dtype)

    # print(f"{my_model.down_layer_gate.boundary_predictor=}")

    my_x = torch.randn(32, 2048, 128).to("cuda", dtype=dtype)
    my_x.requires_grad_(True)

    my_out, _, _ = my_model.forward_backbone(my_x)

    test_causality(my_x, my_out, window_size=None)


if __name__ == "__main__":
    test_forward_pass()