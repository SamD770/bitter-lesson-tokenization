import os
from .flexible_bitter_llm import per_token_losses_backbone

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
from transformers import AutoTokenizer

byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

# my_samples = ["ABC", "Hello world", "This is a test"]
my_samples = ["AB", "CDEF", "GHI"]

pad_token_id = byte5_tokenizer.pad_token_id

batch = byte5_tokenizer(my_samples, return_tensors="pt", padding=True)

# print(f"{batch['input_ids'].shape=}")

# print(f"{batch['attention_mask'].shape=}")
# print(f"{batch['attention_mask'].sum(dim=1)=}")

print(f"{byte5_tokenizer.batch_decode(batch['input_ids'])=}")

loss_mask = batch['attention_mask']

input_ids = batch['input_ids']
next_token_ids = input_ids[:, 1:]

print(f"batch shape: {input_ids.shape=} number of bytes in the batch: {loss_mask.sum().item()=}")

# Test 1: if all the logits are 0, but we interfere with the early model uniformly across the sequence, rewards all should be positive but the discounted reward should still be 0. 

print("-"*100)
print("TEST 1:")
print("-"*100, "\n")

logits = torch.zeros(*input_ids.shape, 256)
logits = F.log_softmax(logits, dim=-1)

early_logits = torch.zeros(*input_ids.shape, 256)
early_logits[:, :, 224:256] = 2.0
early_logits = F.log_softmax(early_logits, dim=-1)

down_gate_probs = torch.ones(*input_ids.shape) * 0.3
off_policy_gate_probs = torch.ones(*input_ids.shape) * 0.2
down_gate_samples = torch.bernoulli(off_policy_gate_probs).to(dtype=torch.long)

dummy_out = {
    "logits": logits,
    "early_logits": early_logits,
    "down_gate_samples": down_gate_samples,
    "down_gate_probs": down_gate_probs,
    "down_gate_logits": torch.log(down_gate_probs)
}

per_token_losses = per_token_losses_backbone(batch["input_ids"], loss_mask, dummy_out, off_policy_gate_probs)

for k, v in per_token_losses.items():
    print(f"{k}: \n{v}\n")

# Test 2: if the logits are all 0, and we interfere with the early model but only in the first element of the batch, the discounted reward should be positive for this sequence and negative for the others.
print("-"*100)
print("TEST 2:")
print("-"*100, "\n")

logits = torch.zeros(*input_ids.shape, 256)
logits = F.log_softmax(logits, dim=-1)

early_logits = torch.zeros(*input_ids.shape, 256)
early_logits[0, :, 224:256] = 2.0
early_logits[1, :, 224:256] = 1.0

dummy_out = {
    "logits": logits,
    "early_logits": early_logits,
    "down_gate_samples": down_gate_samples,
    "down_gate_probs": down_gate_probs,
    "down_gate_logits": torch.log(down_gate_probs)
}

per_token_losses = per_token_losses_backbone(batch["input_ids"], loss_mask, dummy_out, off_policy_gate_probs)

for k, v in per_token_losses.items():
    print(f"{k}: \n{v}\n")




# my_path = os.path.join(os.environ.get("PROJECT_DIR", os.getcwd()), "test_openwebtext_samples.pkl")

# with open(my_path, 'rb') as f:
#     my_samples = pickle.load(f)
#     my_samples = my_samples["text"]

# sample_lengths = [len(s) for s in my_samples]

# print(f"{sum(l > 1024 for l in sample_lengths)=}")

# is_padding = batch['input_ids'] == pad_token_id
# print(f"{(is_padding != batch['attention_mask']).all().item()=}")

# print(f"{sample_lengths=}")

# print(f"{batch.keys()=}")

# truncated_batch = {k: v[:, :4096] for k, v in batch.items()}
# truncated_input_ids = truncated_batch['input_ids']
# truncated_pad_mask = truncated_batch['attention_mask']

# print(f"{truncated_pad_mask.sum()/truncated_pad_mask.numel()=}")