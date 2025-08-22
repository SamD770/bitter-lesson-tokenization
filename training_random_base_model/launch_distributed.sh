#!/bin/bash

# for rtx_3090: model size 18M, batch size 32
# for A100: model size 130M, batch size 32

SEED=42
MODEL_SIZE=$1
BATCH_SIZE=$2

unset TMPDIR # to fix OSError: Device or resource busy https://discuss.pytorch.org/t/num-workers-in-dataloader-always-gives-this-error/64718/8

# Set environment variables for running accelerate
source .env
WANDB_DIR="/scratch/${USER}/tokenizer_training/wandb_logs"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
WANDB_MODE="offline"

export WANDB_DIR WANDB_CACHE_DIR WANDB_API_KEY WANDB_MODE
mkdir -vp "${WANDB_CACHE_DIR}"

export PATH="/home/sdauncey/.local/bin:$PATH"

accelerate launch --main_process_port 0 -m training_random_base_model.run --seed $SEED --model_size $MODEL_SIZE --batch_size $BATCH_SIZE

TAR_FILE="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/training_random_base_model/logs/wandb_${SLURM_JOB_ID}.tar.gz"
RUN_DIR="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/training_random_base_model/logs/"

# # Move the wandb logs to the net_scratch directory in a compressed tar file.
# # rm -r "${WANDB_CACHE_DIR}"
# tar -czf "${TAR_FILE}" -C "${WANDB_DIR}" .
# # # rm -r "${WANDB_DIR}"
# tar -xzf "${TAR_FILE}" -C "${RUN_DIR}"

# wandb sync "${RUN_DIR}wandb/latest-run"
