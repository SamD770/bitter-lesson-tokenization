import torch
from .flexible_bitter_llm import DistributeTokenUpsampler

# should output:
# x_upsampled[0, :, 0]=tensor([0, 1, 2, 2, 3, 3, 4]) 
# up_merge_dst[0, :, 0]=tensor([0, 1, 2, 2, 3, 3, 4])

g = torch.tensor([[1, 1, 1, 0, 1, 0, 1]])
x = torch.arange(5).reshape(1, 5, 1)

upsampler = DistributeTokenUpsampler()
x_upsampled, up_merge_dst = upsampler(x, g)

print(f"{x_upsampled[0, :, 0]=} \n {up_merge_dst[0, :, 0]=}")