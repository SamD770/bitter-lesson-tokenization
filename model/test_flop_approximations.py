from clean_code.flexible_bitter_llm import FlexibleBitterLLM, ExactRandomGater
import torch
from clean_code.utils import parameter_count_string, count_parameters

my_model = FlexibleBitterLLM(
    vocab_size=256,
    embedding_dim=1024,
    num_heads=16,
    downsample_rate=0.25,
    sliding_window=64,
    n_down_layers=2,
    n_mid_layers=20,
    n_up_layers=2,
    flash_attn=True,
    GaterClass=ExactRandomGater
).to("cuda", dtype=torch.bfloat16)

print(f"{parameter_count_string(my_model)=}")
embedding_parameters_count = count_parameters(my_model.embedding)

print("\n\nTest 1: are we accounting for all the parameters?")

print(f"{my_model.byte_level_parameters_count /10**6 = }")
print(f"{my_model.mid_layers_parameters_count /10**6 = }")
print(f"{(my_model.mid_layers_parameters_count + my_model.byte_level_parameters_count + embedding_parameters_count) /10**6 = }")

print("\n\nTest 2: are we accounting for all the flops?")

sequence_length = 4096
batch_size = 32
my_batch = torch.randint(0, 256, (batch_size, sequence_length)).to("cuda", dtype=torch.int32)

_, _, my_down_gate_samples = my_model.down_layer_gate(my_batch.unsqueeze(-1))
max_n_patches = my_down_gate_samples.sum(dim=1).max()
print(f"{max_n_patches.item() = }")

dummy_out = {
    "down_gate_samples": my_down_gate_samples,
}

total_flops = my_model.get_num_flops(my_batch, dummy_out)

print(f"{total_flops.item() / 10**12 = }")

byte_level_parameter_counts = {
    "down_layer_gate": my_model.down_layer_gate_parameters_count,
    "down_layer": my_model.down_layers_parameters_count,
    "up_layer": my_model.up_layers_parameters_count,
    "output_layer": my_model.unembedding_parameters_count,
    "early_output_layer": my_model.early_exit_parameters_count,
}

flop_ratios = []

for layer, n_params in byte_level_parameter_counts.items():
    layer_flops = 6 * n_params * batch_size * sequence_length
    flops_ratio = layer_flops / total_flops.item()
    print(f"{layer}: {flops_ratio = :.2%}")
    flop_ratios.append(flops_ratio)

mid_layer_flops = 6 * my_model.mid_layers_parameters_count * batch_size * max_n_patches
mid_layer_flops_ratio = mid_layer_flops / total_flops.item()
print(f"mid_layer: {mid_layer_flops_ratio = :.2%}")
flop_ratios.append(mid_layer_flops_ratio)

print(f"{sum(flop_ratios) = :.2%}")