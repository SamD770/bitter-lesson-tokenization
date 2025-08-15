import datasets
import os
from pathlib import Path

username = "sdauncey"
scratch_dir = f"/scratch/{username}/tokenizer_training"

print("loading dataset...")
openwebtext_full = datasets.load_dataset(
    "openwebtext",
    split="train",  # Using the full dataset.
    cache_dir=os.path.join(scratch_dir, "openwebtext_full_cache"),
    trust_remote_code=True
)

print(f"loaded {len(openwebtext_full['text'])} examples")

length_clip = 4096  # 4096 utf-8 bytes is the truncation length for our batching.

def clipped_length(example):
    return min(len(example.encode('utf-8')), length_clip)

total_bytes = sum(clipped_length(example) for example in openwebtext_full["text"])
print(f"total bytes after clipping: {total_bytes/10**9=:.2f} Billion Bytes")

min_length = 3072

# Filter dataset for sequences longer than 4096 utf-8 bytes
print("filtering dataset...")
filtered_dataset = openwebtext_full.filter(lambda example: len(example["text"].encode('utf-8')) > min_length)

# Save the filtered dataset to the net_scratch directory
print("saving filtered dataset...")
net_scratch_dir = Path("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/")
data_dir = os.path.join(net_scratch_dir, "data")
filtered_dataset.save_to_disk(os.path.join(data_dir, "openwebtext_full_filtered"))

print(f"saved {len(filtered_dataset['text'])} examples")

# Count the size of the dataset if turned into batches of length filtered_dataset 
remaining_training_bytes = sum(clipped_length(example) for example in filtered_dataset["text"])
print(f"{remaining_training_bytes/10**9=:.2f} Billion Bytes")
