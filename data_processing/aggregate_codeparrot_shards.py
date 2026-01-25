import datasets
import os
from glob import glob
from tqdm import tqdm

from .stream_and_filter_codeparrot import output_dir, scratch_dir

aggregated_dir = os.path.join(scratch_dir, "codeparrot_filtered_aggregated")


if __name__ == "__main__":
    # Find all JSONL shard files
    shard_files = sorted(glob(os.path.join(output_dir, "shard_*.jsonl")))
    print(f"Found {len(shard_files)} shard files")
    
    # Load all shards as a single dataset from JSONL files
    print("Loading JSONL shards...")
    aggregated = datasets.load_dataset(
        'json',
        data_files=shard_files,
        split='train'
    )
    
    print(f"Aggregated {len(aggregated)} documents")
    
    os.makedirs(aggregated_dir, exist_ok=True)
    
    print(f"Saving to {aggregated_dir}...")
    aggregated.save_to_disk(aggregated_dir)
    
    print(f"Saved aggregated dataset to {aggregated_dir}")
    print("Done")
