import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_billion_tokens", type=float, default=100.0,
        help="Approximate number of billion tokens to download (default: 100). "
             "The 100BT dataset has 1024 shards, so N billion tokens ≈ round(N/100*1024) shards."
    )
    args = parser.parse_args()

    n_billion = args.n_billion_tokens
    n_shards = max(1, min(round(n_billion / 100 * 1024), 1024))
    n_label = int(n_billion) if n_billion == int(n_billion) else n_billion
    output_shards_dir = os.path.join(scratch_dir, f"fineweb_{n_label}B_shards")
    os.makedirs(output_shards_dir, exist_ok=True)

    print(f"downloading ~{n_billion}B tokens ({n_shards}/1024 shards) to {output_shards_dir}...")
    fineweb_100B = datasets.load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-100BT",
        split="train",  # Using the full dataset.
        cache_dir=os.path.join(scratch_dir, "fineweb_100B_cache")
    )

    print("downloading and filtering fineweb...")
    n_documents = 0
    for shard_idx in range(n_shards):
        shard_idx_dir = os.path.join(output_shards_dir, f"shard_{shard_idx}")
        os.makedirs(shard_idx_dir, exist_ok=True)

        fineweb_100B_shard = fineweb_100B.shard(num_shards=1024, index=shard_idx)

        filtered_shard = fineweb_100B_shard.filter(above_min_length)

        filtered_shard.save_to_disk(os.path.join(shard_idx_dir))

        n_documents += len(filtered_shard)
        print(f"shard {shard_idx} done, {n_documents} documents found")

    print(f"found {n_documents/1e6} million documents of length > {min_length} after filtering")
    print(f"saved {n_shards} shards to {output_shards_dir}")
    print("done")