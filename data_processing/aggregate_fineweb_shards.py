import datasets
from .download_and_filter_fineweb_100B import shards_dir, scratch_dir
import os
from tqdm import trange

n_shards = 1024

shards = []

aggregated_dir = os.path.join(scratch_dir, f"fineweb_100B_filtered")


if __name__ == "__main__":
    for shard_idx in trange(n_shards):
        shard_idx_dir = os.path.join(shards_dir, f"shard_{shard_idx}")
        shard = datasets.load_from_disk(shard_idx_dir)

        shards.append(shard)

    aggregated = datasets.concatenate_datasets(shards)

    print(f"aggregated {len(aggregated)} documents")

    os.makedirs(aggregated_dir, exist_ok=True)

    aggregated.save_to_disk(aggregated_dir)

    print(f"saved aggregated shard to {aggregated_dir}")

    print("done")