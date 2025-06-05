"""
Experiment 19: trying to train a model sans warm start.
"""
from clean_code.bitter_llm import set_seed, parameter_count_string
from clean_code.bitter_llm import bitter_tokenizer_training_loop, CausalGemmaMiniBitterLLM, LinearGater, set_seed, parameter_count_string
import datasets
from transformers import AutoTokenizer
import os
import torch

byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

username = "sdauncey"
scratch_dir = f"/scratch/{username}/tokenizer_training"

if not os.path.exists(scratch_dir):
    os.makedirs(scratch_dir)

# Download a portion of OpenWebText dataset
# This will download a subset of the OpenWebText corpus
print("Downloading OpenWebText dataset...")

# Load a small portion of OpenWebText (25% of the dataset)
openwebtext_25p = datasets.load_dataset(
    "openwebtext",
    split="train[:25%]",  # Using only 25% samples of the dataset for now.
    cache_dir=os.path.join(scratch_dir, "openwebtext_25p_cache"),
    trust_remote_code=True
)

my_samples = openwebtext_25p[-512:-256]
import pickle
net_scratch_dir = f"/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/"

# Define the path for the pickle file
pickle_path = os.path.join(net_scratch_dir, "test_openwebtext_samples.pkl")

# Dump the samples using pickle
with open(pickle_path, 'wb') as f:
    pickle.dump(my_samples, f)

print(f"Samples dumped to {pickle_path}")
