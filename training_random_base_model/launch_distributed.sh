#!/bin/bash

# for rtx_3090: model size 18M, batch size 32
# for A100: model size 130M, batch size 32


MODEL_SIZE=$1
BATCH_SIZE=$2
UPDOWN_SAMPLER=${3:-"random"}

SEED=${4:-42}

echo "Starting training random base model with seed $SEED at $(date)"

unset TMPDIR # to fix OSError: Device or resource busy https://discuss.pytorch.org/t/num-workers-in-dataloader-always-gives-this-error/64718/8

source wandb_offline_integration/setup.sh
OUTPUT_FILE=training_random_base_model/${MODEL_SIZE}_${SEED}_output.log
bash run_apptainer.sh

CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --main_process_port 0 \
    -m training_random_base_model.run \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --run_type $UPDOWN_SAMPLER \
    --architecture $UPDOWN_SAMPLER \
    --seed $SEED \
    2>&1 | tee ${OUTPUT_FILE}

source wandb_offline_integration/parse_wandb_path.sh ${OUTPUT_FILE}

wandb sync $WANDB_SYNC_PATH
echo "Finished training random base model with seed $SEED at $(date)"


