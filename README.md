# You Can Learn Tokenization End-to-End with Reinforcement Learning

This is the codebase used for our ICML 2026 paper [You Can Learn Tokenization End-to-End with Reinforcement Learning](https://openreview.net/forum?id=0rXBwsTWDB))

The code is a research framework for **dynamic tokenization** — learning to compress
variable-length byte sequences into variable-length token sequences *during*
training, rather than relying on a fixed, pre-trained tokenizer.
It directly plugs into the code for straight-throug estimators such as [HNet](https://openreview.net/forum?id=ZbfLR9NbNF) and [Dynamic Token Pooling](https://aclanthology.org/2023.acl-long.353/). 

The core model is an autoregressive U-Net over bytes. Each forward pass runs a
**Gater → Downsampler → Mid-layers → Upsampler** pipeline that learns where to
place token boundaries instead of having them imposed by a BPE vocabulary.

```text
Bytes → down_layers → Gater → Downsampler → mid_layers → Upsampler → up_layers → predictions
```

## Installation

Requires Python 3.11+. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # or: pip install -e .
```

Copy `.env.template` to `.env` and fill in your credentials:

```bash
cp .env.template .env
# WANDB_API_KEY=...   (optional, for Weights & Biases logging)
# HF_TOKEN=...        (for gated/large HuggingFace datasets)
```

### Environment variables

Scripts read a few paths from the environment, falling back to local defaults
so the repo runs out-of-the-box:

| Variable       | Purpose                                    | Default                              |
| -------------- | ------------------------------------------ | ------------------------------------ |
| `SCRATCH_DIR`  | Dataset cache / scratch space              | `/tmp/$USER/tokenizer_training`      |
| `PROJECT_DIR`  | Project root (for saving data/checkpoints) | current working directory            |
| `WANDB_ENTITY` | W&B entity for logging (optional)          | unset (W&B uses your default entity) |

## Running with Apptainer / Singularity

For HPC clusters we run inside an [Apptainer](https://apptainer.org/) (formerly
Singularity) container. The image is defined by
[`pytorch_flashattn_container.def`](pytorch_flashattn_container.def), which is
built on `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel` and bakes in
`flash-attn`, `mamba-ssm`, `transformers`, and `datasets`.

Build the image (needs root or `--fakeroot`):

```bash
apptainer build pytorch_flashattn_container.sif pytorch_flashattn_container.def
```

Then run any command inside it, binding your scratch space so datasets and
checkpoints persist:

```bash
apptainer exec --nv \
    --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" \
    pytorch_flashattn_container.sif \
    python -m training_random_base_model.run --size 18M --architecture random --batch_size 32
```

`--nv` exposes the host GPUs. The example SLURM script
`training_random_base_model/73M_job.sh` shows how to wrap a distributed launch
in `apptainer exec`.

## Data preprocessing

See [data_processing/dataset_README.md](data_processing/dataset_README.md) for
full details. Datasets are written under `$SCRATCH_DIR` (see the table above).

**FineWeb** (the 100B-token subset, filtered to sequences of >4096 UTF-8 bytes).
Download + per-shard filter, then aggregate the shards into a single dataset:

```bash
# Download & filter (~6h for the full 100B; use --n_billion_tokens for a subset)
python -m data_processing.download_and_filter_fineweb_100B --n_billion_tokens 100

# Aggregate the 1024 filtered shards into one dataset (~15 min)
python -m data_processing.aggregate_fineweb_shards
```

**CodeParrot** (streamed, filtered, sharded, then aggregated):

```bash
python -m data_processing.stream_and_filter_codeparrot
python -m data_processing.aggregate_codeparrot_shards
```

## Training

```bash
# Single command (uses 🤗 accelerate under the hood for multi-GPU/multi-node)
python -m training_random_base_model.run --size 18M --architecture random --batch_size 32

# Nawrot-style dynamic pooling baseline
python -m training_random_base_model.run_nawrot --size 18M --batch_size 32

# Resume from a checkpoint
python -m training_random_base_model.run --size 18M --architecture hnet --run_type hnet --resume_checkpoint <dir>
```

Example launch wrappers live in `training_random_base_model/`
(`launch_distributed.sh`, `73M_job.sh` as a SLURM template).

**CLI args:**

- `--size`: `18M | 32M | 73M | 130M | 346M`
- `--architecture`: `random | linear | sequential | nawrot | hnet`
- `--run_type`: `default | random | nawrot | hnet | linear | sequential`
- `--dataset`: `fineweb | codeparrot`
- `--aspect_ratio`: `1–8` (controls sequence length)

## Evaluation

```bash
bash eval/run_all_evals.sh <checkpoint_dir>
```

## Repository layout

| Path                          | Contents                                                              |
| ----------------------------- | --------------------------------------------------------------------- |
| `model/`                      | `AutoregressiveUnet`, Gater/Downsampler/Upsampler modules, plugins    |
| `training_random_base_model/` | Training loop, layered JSON config system, launch scripts             |
| `data_processing/`            | FineWeb / OpenWebText / CodeParrot download & filtering               |
| `eval/`                       | BPB tables, LAMBADA/PIQA, evaluation harness                          |
| `flexify_training/`           | Aspect-ratio / flexible-sequence-length training experiments          |
| `plots/`, `optimizing_code/`  | Analysis notebooks and profiling experiments                          |

## Architecture notes

- **Gater** ([model/modules.py](model/modules.py)) produces gate logits/probs/samples per byte, deciding token boundaries. Variants: `LinearGater`, `RandomGater`, `EquidistantGater`, `ExactRandomGater`, `NawrotGater`, `HNetGater`.
- **Downsampler** merges bytes into tokens: `SelectTokenDownsampler`, `AverageTokenDownsampler`, `NawrotDownsampler`, `HNetDownsampler`.
- **Upsampler** reconstructs byte-level features: `DistributeAddUpsampler`, `DistributeDeviationUpsampler`, `NawrotUpsampler`, `HNetUpsampler`.
- Gating is non-differentiable by nature; gradients flow via straight-through / score-function estimators (see `off_policy_flexible_training_step()` in [model/model.py](model/model.py)).
- The downsampling rate can be annealed during training by `DefaultDownsampleRateScheduler` ([model/downsample_rate_scheduler.py](model/downsample_rate_scheduler.py)). See `flexify_training/` for experiments (didn't make it into paper).
- Configs are layered JSON files merged at runtime by [training_random_base_model/config_loader.py](training_random_base_model/config_loader.py); string class names are resolved via [training_random_base_model/class_registry.py](training_random_base_model/class_registry.py).

## License

[MIT](LICENSE)

## BibTeX

```bibtex
@inproceedings{
    dauncey2026tokenizationrl,
    title={You Can Learn Tokenization End-to-End with Reinforcement Learning},
    author={Sam Dauncey and Roger Wattenhofer},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=0rXBwsTWDB}
}
```
