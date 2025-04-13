#!/bin/bash
#SBATCH --job-name=bitter_tokenizer
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=tikgpu09
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=bitter_tokenizer_%j.out
#SBATCH --error=bitter_tokenizer_%j.err

conda init
conda activate geometric_diffusers

# Run the experiment with time tracking
echo "Starting experiment at $(date)"
CUDA_LAUNCH_BLOCKING=1 python experiment_1.py
EXIT_CODE=$?
echo "Finished experiment at $(date)"

# Check if the script exited due to an error (like OOM)
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error occurred (possibly OOM) at $(date)"
    echo "Exit code: $EXIT_CODE"
fi
