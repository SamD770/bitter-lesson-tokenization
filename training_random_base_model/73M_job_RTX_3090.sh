#!/bin/bash
#SBATCH --job-name=73M_run
#SBATCH --output=training_random_base_model/logs/73M_%j.out
#SBATCH --error=training_random_base_model/logs/73M_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodelist=tikgpu07
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

# Run the experiment with time tracking
echo "Starting training random base model at $(date)"
apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed.sh 73M 4

# batch size 16 for flash attention, batch size 8 for non-flash attention

