
We use the $100B$ subset of [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), filtered for sequences of $>4096$ utf-8 bytes. To download the dataset, split it into $1024$ shards, filter each shard and save, set the `SCRATCH_DIR` environment variable to indicate where you want to save the dataset to and run:

```bash
python -m data_processing.download_and_filter_fineweb_100B
```

This takes ~$6$ hours on my academic setup.

Once this has completed, you can aggregate these filtered shards into one dataset, containing all $29,285,683$ documents each with $>4096$ utf-8 bytes:

```bash
python -m data_processing.aggregate_fineweb_shards
```

This takes ~$15$ minutes on my setup.