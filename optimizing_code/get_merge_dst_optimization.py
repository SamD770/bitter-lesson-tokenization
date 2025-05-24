import torch
import cProfile
import pstats
from pstats import SortKey

def get_merge_dst_old(gate_samples: torch.Tensor) -> torch.Tensor:
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


def get_merge_dst(gate_samples: torch.Tensor) -> torch.Tensor:
    """
    Returns tuple (merge_dst, dst_idx) the merge destination for each token in the sequence and the number of unique merge destinations.
    Input is a tensor of shape (batch_size, sequence_length) with 0 tokens that are merged into the next 1 token.
    mapping:
    1 0 1 1 0 1 1 0 0 0 1
    |   | |   | |       |
    0 1 1 2 3 3 4 5 5 5 5
    """
    # "cycle" the gate samples, appending a zero to the beginning of each batch and ignoring the last gate.
    preceding_gate_samples = torch.cat([torch.zeros_like(gate_samples[:, -1]).unsqueeze(1), gate_samples[:, :-1]], dim=1).to(dtype=torch.long)
    # Trick: use cumsum to do this in a vectorized way.
    merge_dst = preceding_gate_samples.cumsum(dim=1)
    n_dst = merge_dst[:, -1] + 1 # The number of unique merge destinations is the last token's merge destination + 1.
    return merge_dst, n_dst


def test_get_merge_dst():
    gate_samples = torch.randint(0, 2, (2, 10)).to("cuda")
    gate_samples[:, 0] = 1
    print(gate_samples)
    merge_dst_old, n_dst_old = get_merge_dst_old(gate_samples)
    merge_dst, n_dst = get_merge_dst(gate_samples)
    print(f"{merge_dst_old=}")
    print(f"{merge_dst=}"   )
    print(f"{n_dst_old=} {n_dst_old.max()=}")
    print(f"{n_dst=} {n_dst.max()=}")
    max_n_dst=  n_dst.max()
    max_n_dst_old = n_dst_old.max()
    print(f"{max_n_dst=} {max_n_dst_old=}")
    torch.zeros(2, max_n_dst)
    torch.zeros(2, max_n_dst_old)
    print("tensors created with no problems")


def profile_get_merge_dst():
    for _ in range(20):
        gate_samples = torch.randint(0, 2, (32, 1024)).to("cuda")
        gate_samples[:, 0] = 1
        get_merge_dst(gate_samples)
        get_merge_dst_old(gate_samples)


if __name__ == "__main__":
    # test_get_merge_dst()
    fname = "get_merge_dst_profile.prof"
    cProfile.run("profile_get_merge_dst()", filename=fname)
    p = pstats.Stats(fname)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(10)
    # Should see tottime 30.11s -> 0.03s, a 1000x speedup of the bottlenecking part of the model. Not bad!
