#!/bin/bash


# Set environment variables for running accelerate
WANDB_DIR="/scratch/${USER}/tokenizer_training/wandb_logs"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
WANDB_API_KEY="0cc09cfe82acf2f43a1521089ae7e096795e5ce1"

export WANDB_DIR WANDB_CACHE_DIR WANDB_API_KEY

WANDB_MODE="offline"

export WANDB_MODE

python debug_wandb_accelerate.py