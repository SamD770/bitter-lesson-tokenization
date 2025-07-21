#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=bitter_tokenizer
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=tikgpu10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/bitter_tokenizer_%j.out
#SBATCH --error=logs/bitter_tokenizer_%j.err

conda init
conda activate geometric_diffusers

# Run the experiment with time tracking
echo "Starting experiment 22 at $(date)"

# Set environment variables for CUDA debugging
python -m experiment_22.run

EXIT_CODE=$?
echo "Finished experiment 22 at $(date)"

# Check if the script exited due to an error (like OOM)
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error occurred (possibly OOM) at $(date)"
    echo "Exit code: $EXIT_CODE"
fi
