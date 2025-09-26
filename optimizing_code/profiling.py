from clean_code.flexible_bitter_llm import (
    FlexibleBitterLLM, 
    off_policy_flexible_training_step, 
    ExactRandomGater,
    LinearGater
)
from clean_code.conditional_sequential import (
    SequentiallyDependentLinearGater,
    SequentiallyDependentRandomGater,
    OptimizedSequentialyDependentLinearGater
)

from torch.profiler import record_function, profile, ProfilerActivity, schedule
import torch
from accelerate import Accelerator
import time

from clean_code.utils import parameter_count_string

# Tomer Porian∗ Mitchell Wortsman† Jenia Jitsev‡ Ludwig Schmidt† Yair Carmon∗
# Resolving Discrepancies in Compute-Optimal Scaling of Language Models
# https://arxiv.org/pdf/2406.19146


accelerator = Accelerator()

# Quite similar to GPT-2-small
my_model = FlexibleBitterLLM(
    vocab_size=256,
    embedding_dim=1024,
    num_heads=16,
    downsample_rate=0.5,
    sliding_window=64,
    n_down_layers=3,
    n_mid_layers=18,
    n_up_layers=3,
    GaterClass=LinearGater,
    flash_attn=True
).to(accelerator.device, dtype=torch.bfloat16)

print(f"{my_model.down_layer_gate=}")

print(f"{parameter_count_string(my_model)=}")

batch_size = 8
sequence_length = 4096

# According to Yair Carmon, we should use a learning rate of 5e-3 and a warmup of 800M bytes for a ~100M parameter model. 
my_optimizer = torch.optim.AdamW(my_model.parameters(), lr=5e-3)
warmup_steps = 8 * 10**8 / (batch_size * sequence_length)
print(f"{warmup_steps=}")
my_scheduler = torch.optim.lr_scheduler.LinearLR(my_optimizer, start_factor=0.1, total_iters=warmup_steps)

my_profile_schedule = schedule(
    skip_first=5,
    wait=1,
    warmup=1,
    active=1
)

my_model, my_optimizer, my_scheduler = accelerator.prepare(my_model, my_optimizer, my_scheduler)

# my_model = torch.compile(my_model)

# Profile the models
print("Profiling....")

peak_memory = 0

start_time = time.time()

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=my_profile_schedule
# ) as prof:

for i in range(20):
    print(f"Batch {i}/{20}")
    # According to Yair Carmon, we should use a total batch size between 128 and 256 for a ~100M parameter model and a sequence length of 2048 "tokens". This would correspond to a batch size of 256-512 for our shorter character sequences pf 4096.
    input_ids = torch.randint(0, 256, (batch_size, sequence_length)).to("cuda")
    loss_mask = torch.ones(batch_size, sequence_length).to("cuda")

    with record_function("training_step"):
        off_policy_flexible_training_step(
            my_model, 
            my_optimizer, 
            input_ids, 
            loss_mask, 
            my_scheduler, 
            accelerator, 
            learn_gating=True
        )

    # Track peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    print(f"Peak CUDA memory usage: {peak_memory / 1024**3:.2f} GB")

        # prof.step()

end_time = time.time()
print(f"Wallclock time per step: {(end_time - start_time)/20:.2f} seconds")

# print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
