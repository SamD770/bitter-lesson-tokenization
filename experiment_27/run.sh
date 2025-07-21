#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=bitter_tokenizer
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodelist=tikgpu10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_27/logs/log_%A_%a.out
#SBATCH --error=experiment_27/logs/errors_%A_%a.err
#SBATCH --array=47

# Get the seed from the array job ID
SEED=$SLURM_ARRAY_TASK_ID

# Run the experiment with time tracking
echo "Starting experiment 27 with seed $SEED at $(date)"

# Set environment variables for CUDA debugging
apptainer exec --nv --bind /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey /scratch/sdauncey/sams_favorite_build.sif python -m experiment_27.run --seed $SEED

EXIT_CODE=$?
echo "Finished experiment 27 with seed $SEED at $(date)"

# Check if the script exited due to an error (like OOM)
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error occurred (possibly OOM) at $(date)"
    echo "Exit code: $EXIT_CODE"
fi
