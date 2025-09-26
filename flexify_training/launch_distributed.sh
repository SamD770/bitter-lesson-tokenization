#!/bin/bash

# for rtx_3090: model size 18M, batch size 32
# for A100: model size 130M, batch size 32

SEED=42
MODEL_SIZE=$1
BATCH_SIZE=$2

unset TMPDIR # to fix OSError: Device or resource busy https://discuss.pytorch.org/t/num-workers-in-dataloader-always-gives-this-error/64718/8

source wandb_offline_integration/setup.sh
OUTPUT_FILE=flexify_training/${MODEL_SIZE}_output.log

CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --main_process_port 0 \
    -m flexify_training.run \
    --seed $SEED \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE 2>&1 # | tee ${OUTPUT_FILE}

source wandb_offline_integration/parse_wandb_path.sh # ${OUTPUT_FILE}

wandb sync $WANDB_SYNC_PATH


