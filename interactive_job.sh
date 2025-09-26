#!/bin/bash

# Prompt user for node name
read -p "Enter node name: " NODE

# Prompt user for number of GPUs
read -p "Enter number of GPUs [1]: " N_GPU
N_GPU=${N_GPU:-1}

# Prompt user for memory amount
read -p "Enter memory amount [32G]: " MEM
MEM=${MEM:-32G}

# Prompt user for number of CPUs
read -p "Enter number of CPUs [8]: " N_CPU
N_CPU=${N_CPU:-8}

srun  --mem=$MEM --gres=gpu:$N_GPU --nodelist=$NODE --cpus-per-task=$N_CPU --pty bash -i
