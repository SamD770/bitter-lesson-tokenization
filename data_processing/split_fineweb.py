import datasets
import os
from tqdm import trange

from .download_and_filter_fineweb_100B import scratch_dir
from .aggregate_fineweb_shards import aggregated_dir

from torch.utils.data import DataLoader

seed = 42

def get_splits():
    aggregated = datasets.load_from_disk(aggregated_dir)

    # Make a test set of 32,768 documents (= 1e8 bytes when truncated to 4096)
    train_test_split = aggregated.train_test_split(test_size=32768, seed=seed)
    test_set = train_test_split['test']
    remaining = train_test_split['train']

    # Make a validation set of 256 documents (= 1e6 bytes when truncated to 4096)
    train_val_split = remaining.train_test_split(test_size=256, seed=seed)
    val_set = train_val_split['test']
    train_set = train_val_split['train']
    return train_set, val_set, test_set


def get_split(split_name: str):
    train_set, val_set, test_set = get_splits()
    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }[split_name]



if __name__ == "__main__":
    train_set, val_set, test_set = get_splits()
    print("splitting dataset...")
    print(f"{len(train_set)=}")
    print(f"{len(val_set)=}")
    print(f"{len(test_set)=}")

    print(f"{train_set[0]['text'][:200]=}")
    print(f"{val_set[0]['text'][:200]=}")
    print(f"{test_set[0]['text'][:200]=}")

    train_dl = DataLoader(train_set, batch_size=16, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_set, batch_size=16, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_set, batch_size=16, num_workers=4, pin_memory=True)

    for i, train_batch in enumerate(train_dl):
        x = train_batch["text"]
        if i > 10:
            break

    for j, val_batch in enumerate(val_dl):
        y = val_batch["text"]
        if j > 10:
            break