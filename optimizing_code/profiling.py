from training_step_optimization import CausalGemmaMiniBitterLLM, bitter_tokenizer_training_step
from experiment_9_profile import NawrotGemmaMiniBitterLLM, NawrotBitterLLM_training_step
from torch.profiler import record_function, profile, ProfilerActivity, schedule
import torch


my_model = CausalGemmaMiniBitterLLM(
    vocab_size=256,
    embedding_dim=512,
    num_heads=8,
    downsample_rate=0.25,
    sliding_window=64
).to("cuda")

my_optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4)

nawrot_model = NawrotGemmaMiniBitterLLM(
    vocab_size=256,
    embedding_dim=512,
    num_heads=8,
    downsample_rate=0.25,
    sliding_window=64
).to("cuda", dtype=torch.bfloat16)

nawrot_optimizer = torch.optim.AdamW(nawrot_model.parameters(), lr=1e-4)


    # for _ in range(10):
    #     my_model(input_ids)
    #     nawrot_model(input_ids)

my_schedule = schedule(
    skip_first=5,
    wait=1,
    warmup=1,
    active=1
)
# Profile the models
print("Profiling....")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    schedule=my_schedule
) as prof:
    
    for i in range(20):
        print(f"Batch {i}/{20}")
        input_ids = torch.randint(0, 256, (32, 2048)).to("cuda")
        
        with record_function("my_model"):
            # bitter_tokenizer_training_step(my_model, input_ids, my_optimizer, learn_gating=False)
            my_model(input_ids)

        # with record_function("nawrot_model"):
        #     NawrotBitterLLM_training_step(nawrot_model, input_ids, nawrot_optimizer)
        #     # nawrot_model(input_ids)

        prof.step()

print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))