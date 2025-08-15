import datasets
import os
from tqdm import trange

username = "sdauncey"
scratch_dir = f"/scratch/{username}/tokenizer_training"

shards_dir = os.path.join(scratch_dir, "fineweb_100B_shards")
os.makedirs(shards_dir, exist_ok=True)

min_length = 4096

def above_min_length(example):
    return len(example["text"].encode('utf-8')) > min_length

n_shards = 1024
n_documents = 0

if __name__ == "__main__":
    
    print("loading dataset...")
    fineweb_100B = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-100BT",
        split="train",  # Using the full dataset.
        cache_dir=os.path.join(scratch_dir, "fineweb_100B_cache")
    )

    print("downloading and filtering fineweb 100B...")
    for shard_idx in range(n_shards):
        shard_idx_dir = os.path.join(shards_dir, f"shard_{shard_idx}")
        os.makedirs(shard_idx_dir, exist_ok=True)

        fineweb_100B_shard = fineweb_100B.shard(num_shards=n_shards, index=shard_idx)

        filtered_shard = fineweb_100B_shard.filter(above_min_length)

        filtered_shard.save_to_disk(os.path.join(shard_idx_dir))

        n_documents += len(filtered_shard)
        print(f"shard {shard_idx} done, {n_documents} documents found")

    print(f"found {n_documents/1e6} million documents of length > {min_length} after filtering")
    print(f"saved {n_shards} shards to {shards_dir}")
    print("done")