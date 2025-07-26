#!/bin/bash

# SEED=$1
# MODEL_SIZE=$2

uv run accelerate launch --main_process_port 0 -m training_random_base_model.run # --seed $SEED --model_size $MODEL_SIZE
