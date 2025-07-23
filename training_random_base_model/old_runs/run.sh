#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=bitter_tokenizer
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=tikgpu10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=training_random_base_model/logs/log_%A_%a.out
#SBATCH --error=training_random_base_model/logs/errors_%A_%a.err
#SBATCH --array=42

# Get the seed from the array job ID
SEED=$SLURM_ARRAY_TASK_ID

# Run the experiment with time tracking
echo "Starting training random base model with seed $SEED at $(date)"

# Set environment variables for CUDA debugging
apptainer exec --nv --bind /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey /scratch/sdauncey/sams_favorite_build.sif python -m training_random_base_model.run_select_early_output --seed $SEED

EXIT_CODE=$?
echo "Finished training random base model with seed $SEED at $(date)"

# Check if the script exited due to an error (like OOM)
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error occurred (possibly OOM) at $(date)"
    echo "Exit code: $EXIT_CODE"
fi
