#!/bin/bash
#SBATCH --job-name=130M_run
#SBATCH --output=training_random_base_model/logs/130M_%j.out
#SBATCH --error=training_random_base_model/logs/130M_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodelist=tikgpu08
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB

# Run the experiment with time tracking
echo "Starting training random base model with seed $SEED at $(date)"
apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed.sh 130M 4