#!/bin/bash
#SBATCH --job-name=18M_run
#SBATCH --output=training_random_base_model/logs/18M_%j.out
#SBATCH --error=training_random_base_model/logs/18M_%j.err
#SBATCH --time=15:00:00
#SBATCH --nodelist=tikgpu10
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB

# Run the experiment with time tracking
echo "Starting training random base model with seed $SEED at $(date)"
apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed.sh 18M 32