
import torch
from .modules import SelectTokenDownsampler, get_boundary_indices, select

if __name__ == "__main__":
    batch_size = 8
    seq_len = 4096
    embedding_dim = 768
    my_x = torch.randn(batch_size, seq_len, embedding_dim).to(device="cuda", dtype=torch.bfloat16)
    my_position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device="cuda", dtype=torch.long)
    my_gate_samples = torch.randint(0, 2, (batch_size, seq_len), device="cuda", dtype=torch.bfloat16)

    my_downsampler = SelectTokenDownsampler()
    my_x_downsampled, my_position_ids_downsampled, _ = my_downsampler(my_x, my_position_ids, my_gate_samples)
    
    print(f"{my_x=}")
    print(f"{my_position_ids=}")
    print(f"{my_gate_samples=}")
    print("="*20)
    print(f"{my_x_downsampled=}")
    print(f"{my_position_ids_downsampled=}")
    print("="*20)

    print("tests passed")