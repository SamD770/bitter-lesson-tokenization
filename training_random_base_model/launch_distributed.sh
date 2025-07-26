#!/bin/bash

# SEED=$1
# MODEL_SIZE=$2

# Set environment variables for running accelerate
source .env
WANDB_DIR="/scratch/${USER}/tokenizer_training/wandb_logs"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
WANDB_MODE="offline"

export WANDB_DIR WANDB_CACHE_DIR WANDB_API_KEY WANDB_MODE
mkdir -vp "${WANDB_CACHE_DIR}"

export PATH="/home/sdauncey/.local/bin:$PATH"

accelerate launch --main_process_port 0 -m training_random_base_model.run # --seed $SEED --model_size $MODEL_SIZE

TAR_FILE="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/training_random_base_model/logs/wandb_${SLURM_JOB_ID}.tar.gz"
RUN_DIR="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/training_random_base_model/logs/"

# Move the wandb logs to the net_scratch directory in a compressed tar file.
# rm -r "${WANDB_CACHE_DIR}"
tar -czf "${TAR_FILE}" -C "${WANDB_DIR}" .
# rm -r "${WANDB_DIR}"
tar -xzf "${TAR_FILE}" -C "${RUN_DIR}"

wandb sync "${RUN_DIR}wandb/latest-run"
