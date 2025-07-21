#!/bin/bash

SEED=$1

Set environment variables for running accelerate
WANDB_DIR="/scratch/${USER}/tokenizer_training/wandb_logs"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
WANDB_API_KEY="0cc09cfe82acf2f43a1521089ae7e096795e5ce1"
WANDB_MODE="offline"

export WANDB_DIR WANDB_CACHE_DIR WANDB_API_KEY WANDB_MODE
mkdir -vp "${WANDB_CACHE_DIR}"

export PATH="/home/sdauncey/.local/bin:$PATH"

accelerate launch --main_process_port 0 -m experiment_33.run --seed $SEED 

TAR_FILE="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/experiment_33/logs/wandb_${SLURM_JOB_ID}.tar.gz"
RUN_DIR="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/experiment_33/logs/"

# Move the wandb logs to the net_scratch directory in a compressed tar file.
# rm -r "${WANDB_CACHE_DIR}"
tar -czf "${TAR_FILE}" -C "${WANDB_DIR}" .
# rm -r "${WANDB_DIR}"
tar -xzf "${TAR_FILE}" -C "${RUN_DIR}"

wandb sync "${RUN_DIR}wandb/latest-run"
