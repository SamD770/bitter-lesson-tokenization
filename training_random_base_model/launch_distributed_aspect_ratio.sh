#!/bin/bash

# for rtx_3090: model size 18M, batch size 32
# for A100: model size 130M, batch size 32

MODEL_SIZE=$1
BATCH_SIZE=$2
ASPECT_RATIO=$3

SEED=${4:-42}

echo "Starting training random base model with seed $SEED at $(date)"

unset TMPDIR # to fix OSError: Device or resource busy https://discuss.pytorch.org/t/num-workers-in-dataloader-always-gives-this-error/64718/8

source wandb_offline_integration/setup.sh
OUTPUT_FILE=training_random_base_model/${MODEL_SIZE}_aspect_ratio_${ASPECT_RATIO}_${SEED}_output.log

accelerate launch \
    --main_process_port 0 \
    -m training_random_base_model.run \
    --seed $SEED \
    --model_size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --aspect_ratio $ASPECT_RATIO 2>&1 | tee ${OUTPUT_FILE}

source wandb_offline_integration/parse_wandb_path.sh ${OUTPUT_FILE}

wandb sync $WANDB_SYNC_PATH
echo "Finished training random base model with seed $SEED at $(date)"


