import torch
from .flexible_bitter_llm import DistributeDeviationUpsampler, DistributeAddUpsampler

# should output:
# y[0, :, 0]=tensor([-1, 0, 1, 1, 2, 2, 3]) 
# up_merge_dst[0, :, 0]=tensor([0, 1, 2, 2, 3, 3, 4])

# y[0, :, 0]=tensor([0, 1, 2, 2, 3, 3, 4]) 
# up_merge_dst[0, :, 0]=tensor([0, 1, 2, 2, 3, 3, 4])


down_gate_samples = torch.tensor([[1, 1, 1, 0, 1, 0, 1]])
down_gate_probs = down_gate_samples

y_downsampled = torch.arange(5).reshape(1, 5, 1)

x = torch.zeros(1, 7, 1)
x_downsampled = torch.ones_like(y_downsampled)

print("-"*30, "inputs", "-"*30)
print(f"{x[0, :, 0]=} \n {x_downsampled[0, :, 0]=} \n {y_downsampled[0, :, 0]=} \n {down_gate_samples[0, :]=} \n {down_gate_probs[0, :]=}")

for upsampler in [DistributeDeviationUpsampler(), DistributeAddUpsampler()]:
    print("-"*30, "upsampler", upsampler.__class__.__name__, "-"*30)
    y, up_merge_dst = upsampler(
        x, 
        x_downsampled, 
        y_downsampled, 
        down_gate_samples, 
        down_gate_probs
    )
    print(f"{y[0, :, 0]=} \n {up_merge_dst[0, :, 0]=}")