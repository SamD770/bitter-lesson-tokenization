import datasets
import os
from tqdm import trange

from .download_and_filter_fineweb_100B import scratch_dir
from .aggregate_fineweb_shards import aggregated_dir

seed = 42

if __name__ == "__main__":
    aggregated = datasets.load_from_disk(aggregated_dir)

    print("splitting dataset...")

    # Make a test set of 32,768 documents (= 1e8 bytes when truncated to 4096)
    train_test_split = aggregated.train_test_split(test_size=32768, seed=seed)
    test_set = train_test_split['test']
    remaining = train_test_split['train']

    # Make a validation set of 256 documents (= 1e6 bytes when truncated to 4096)
    train_val_split = remaining.train_test_split(test_size=256, seed=seed)
    val_set = train_val_split['test']
    train_set = train_val_split['train']

    print(f"{train_set[0]['text'][:200]=}")
    print(f"{val_set[0]['text'][:200]=}")