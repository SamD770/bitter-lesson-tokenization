#!/bin/bash

# Prompt user for node name
read -p "Enter node name: " NODE

# Prompt user for number of GPUs
read -p "Enter number of GPUs: " N_GPU

# Prompt user for memory amount
read -p "Enter memory amount (e.g., 32G, 64G): " MEM

srun  --mem=$MEM --gres=gpu:$N_GPU --nodelist=$NODE --pty bash -i
