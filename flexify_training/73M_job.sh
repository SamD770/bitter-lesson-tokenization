#!/bin/bash
#SBATCH --job-name=Flexi_73M_run
#SBATCH --output=flexify_training/logs/73M_%j.out
#SBATCH --error=flexify_training/logs/73M_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodelist=tikgpu09
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB

# Run the experiment with time tracking
echo "Starting flexify training with seed $SEED at $(date)"
apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash flexify_training/launch_distributed.sh 73M 4