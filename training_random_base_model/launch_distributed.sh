#!/bin/bash
# Example multi-GPU launch via 🤗 accelerate.
#
# Usage:
#   bash training_random_base_model/launch_distributed.sh <MODEL_SIZE> <BATCH_SIZE> [RUN_TYPE] [ARCHITECTURE] [DATASET] [SEED]
#
# Rough guidance:
#   - RTX 3090: model size 18M, batch size 32
#   - A100:     model size 130M, batch size 32

MODEL_SIZE=$1
BATCH_SIZE=$2
RUN_TYPE=${3:-"random"}
ARCHITECTURE=${4:-"random"}
DATASET=${5:-"fineweb"}
SEED=${6:-42}

echo "Starting training run (size=$MODEL_SIZE, seed=$SEED) at $(date)"

unset TMPDIR  # workaround for "OSError: Device or resource busy" with DataLoader workers

OUTPUT_FILE=training_random_base_model/${MODEL_SIZE}_${SEED}_output.log

accelerate launch \
    --main_process_port 0 \
    -m training_random_base_model.run \
    --size "$MODEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --run_type "$RUN_TYPE" \
    --architecture "$ARCHITECTURE" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    2>&1 | tee "${OUTPUT_FILE}"

echo "Finished training run (size=$MODEL_SIZE, seed=$SEED) at $(date)"
