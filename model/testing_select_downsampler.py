
import torch
from .modules import SelectTokenDownsampler, get_boundary_indices, select

if __name__ == "__main__":
    my_x = torch.randn(2, 5, 1)
    my_position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
    my_gate_samples = torch.randint(0, 2, (2, 5))

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