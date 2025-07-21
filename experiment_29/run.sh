#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=bitter_tokenizer
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=tikgpu10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_29/logs/log_%A_%a.out
#SBATCH --error=experiment_29/logs/errors_%A_%a.err
#SBATCH --array=1-3

# Get the seed from the array job ID
SEED=$SLURM_ARRAY_TASK_ID

# Run the experiment with time tracking
echo "Starting experiment 29 with seed $SEED at $(date)"

# Set environment variables for CUDA debugging
apptainer exec --nv --bind /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey /scratch/sdauncey/sams_favorite_build.sif python -m experiment_29.run --seed $SEED

EXIT_CODE=$?
echo "Finished experiment 29 with seed $SEED at $(date)"

# Check if the script exited due to an error (like OOM)
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error occurred (possibly OOM) at $(date)"
    echo "Exit code: $EXIT_CODE"
fi
