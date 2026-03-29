# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research framework for **dynamic tokenization** — learning to compress variable-length byte sequences into variable-length token sequences during training, rather than using a fixed tokenizer. The architecture follows a **Gater → Downsampler → Mid-layers → Upsampler** pipeline.

## Setup & Installation

```bash
# Install dependencies (Python 3.11+)
uv sync   # or pip install -e .

# Environment variables required (copy from .env.template)
# WANDB_API_KEY and HF_TOKEN
```

## Common Commands

```bash
python -m training_random_base_model.run --size 18M --architecture random --batch_size 32
python -m training_random_base_model.run_nawrot --size 18M --batch_size 32
python -m training_random_base_model.run --size 18M --architecture hnet --run_type hnet --resume_checkpoint <dir>
```

## Architecture

The core model is `AutoregressiveUnet` in [model/model.py](model/model.py). Data flows as:

```
Bytes → down_layers → Gater → Downsampler → mid_layers → Upsampler → up_layers → predictions
```

**Gater** ([model/modules.py](model/modules.py)): Produces gate logits/probs/samples at each byte position, deciding token boundaries. Variants: `LinearGater`, `RandomGater`, `EquidistantGater`, `ExactRandomGater`, `NawrotGater`, `HNetGater`, Sequential variants.

**Downsampler**: Merges bytes into tokens based on gate samples. Variants: `SelectTokenDownsampler`, `AverageTokenDownsampler`, `NawrotDownsampler`, `HNetDownsampler`.

**Upsampler**: Reconstructs byte-level features from token-level output. Variants: `DistributeAddUpsampler`, `DistributeDeviationUpsampler`, `NawrotUpsampler`, `HNetUpsampler`.

Plugin architectures (Nawrot, HNet) each have their own `*_plugin.py` file with matching Gater+Downsampler+Upsampler classes.

## Configuration System

Configs are layered JSON files merged at runtime by [training_random_base_model/config_loader.py](training_random_base_model/config_loader.py):

- `configs/base/sizes.json` — model sizes (18M, 32M, 73M, 130M, 346M): embedding_dim, num_heads, layer counts
- `configs/base/architectures.json` — which Gater/Downsampler/Upsampler classes to use per architecture name
- `configs/base/optimization.json` — size-specific LR, batch size, warmup/training bytes
- `configs/runs/*.json` — training loop behavior (loss terms, consistency losses)

[training_random_base_model/class_registry.py](training_random_base_model/class_registry.py) maps string names in JSON configs to actual Python classes for dynamic instantiation.

**CLI args:**
- `--size`: 18M | 32M | 73M | 130M | 346M
- `--architecture`: random | linear | sequential | nawrot | hnet
- `--run_type`: default | random | nawrot | hnet | linear | sequential
- `--dataset`: fineweb | codeparrot
- `--aspect_ratio`: 1–8 (controls sequence length)

## Training Details

- Uses HuggingFace `accelerate` for multi-GPU/multi-node distributed training
- Validates every 100 effective batches; checkpoints at 10%, 20%, ..., 100% of `training_bytes`
- Checkpoint state tracked in `elapsed_vals.json` inside checkpoint dir
- W&B logging (online or offline via `wandb/` dir)
- Gradient estimation for gating uses straight-through estimators (STE)
- Downsampling rate is gradually increased during training via `DefaultDownsampleRateScheduler` ([model/downsample_rate_scheduler.py](model/downsample_rate_scheduler.py))
- Key metrics: `ar_loss` (nats/token), `true_downsample_rate`, FLOPs, MFU

## Key Implementation Notes

- The gating mechanism is non-differentiable by nature; gradients flow through STE or score function estimators
- `off_policy_flexible_training_step()` in [model/model.py](model/model.py) handles the gradient computation
- Nawrot and HNet plugins inject additional consistency losses alongside the standard autoregressive loss
- Container for HPC: `mamba_container_sfcompute.sif` (Apptainer/Singularity)
