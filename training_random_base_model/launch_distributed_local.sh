#!/bin/bash

# for rtx_3090: model size 18M, batch size 32
# for A100: model size 130M, batch size 32


MODEL_SIZE=$1
BATCH_SIZE=$2
RUN_TYPE=${3:-"random"}
ARCHITECTURE=${4:-"random"}
DATASET=${5:-"fineweb"}
SEED=${6:-42}


echo "Starting training random base model with seed $SEED at $(date)"

unset TMPDIR # to fix OSError: Device or resource busy https://discuss.pytorch.org/t/num-workers-in-dataloader-always-gives-this-error/64718/8∂

accelerate launch \
    --main_process_port 0 \
    -m training_random_base_model.run \
    --size $MODEL_SIZE \
    --batch_size $BATCH_SIZE \
    --run_type $RUN_TYPE \
    --architecture $ARCHITECTURE \
    --dataset $DATASET \
    --seed $SEED \
    2>&1 | tee ${OUTPUT_FILE}

echo "Finished training random base model with seed $SEED at $(date)"


