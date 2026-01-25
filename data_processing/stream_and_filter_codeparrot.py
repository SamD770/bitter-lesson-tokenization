from huggingface_hub import HfApi, hf_hub_download
import gzip
import json
import os

huggingface_slug = "transformersbook/codeparrot"

total_num_utf8_bytes = 20e9  # 20GB

min_num_bytes = 4096

username = "sdauncey"
scratch_dir = f"/scratch/{username}/tokenizer_training"
cache_dir = os.path.join(scratch_dir, "codeparrot_cache")
output_dir = os.path.join(scratch_dir, "codeparrot_filtered")


def write_jsonl_shard(documents: list, shard_idx: int):
    """Write a list of documents to a JSONL shard file."""
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.jsonl")
    with open(shard_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"Wrote {len(documents)} documents to {shard_path}")


def download_and_filter_codeparrot():
    """
    Download CodeParrot dataset .json.gz files one at a time to minimize HTTP requests,
    filter for documents >= 4096 bytes, and stop at 20GB total.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    api = HfApi()
    
    # List all .json.gz files in the dataset
    print(f"Listing files in {huggingface_slug}...")
    files = api.list_repo_files(huggingface_slug, repo_type="dataset")
    json_gz_files = sorted([f for f in files if f.endswith('.json.gz')])
    print(f"Found {len(json_gz_files)} .json.gz files")
    
    total_bytes = 0
    total_docs = 0
    shard_idx = 0
    
    for file_idx, gz_file in enumerate(json_gz_files):
        print(f"\n[{file_idx + 1}/{len(json_gz_files)}] Downloading {gz_file}...")
        
        # Download entire file (one HTTP request)
        local_path = hf_hub_download(
            huggingface_slug,
            gz_file,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        
        print(f"Processing {gz_file}...")
        
        # Read and decompress the gzipped JSON file
        buffer = []
        with gzip.open(local_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # CodeParrot uses "content" field for the code
                text = record.get("content", "")
                if not text:
                    continue
                    
                text_bytes = len(text.encode('utf-8'))
                if text_bytes >= min_num_bytes:
                    buffer.append({"text": text})
                    total_bytes += text_bytes
                    total_docs += 1
        
        # Write shard if we have documents
        if buffer:
            write_jsonl_shard(buffer, shard_idx)
            shard_idx += 1
        
        print(f"Progress: {total_bytes / 1e9:.2f} GB collected, {total_docs} documents")
        
        if total_bytes >= total_num_utf8_bytes:
            print(f"\nReached target of {total_num_utf8_bytes / 1e9:.0f} GB!")
            break
    
    print(f"\n=== Complete ===")
    print(f"Total bytes: {total_bytes / 1e9:.2f} GB")
    print(f"Total documents: {total_docs}")
    print(f"Total shards: {shard_idx}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    download_and_filter_codeparrot()
