#!/bin/bash
#SBATCH --job-name=32M_aspect_ratio_run
#SBATCH --output=training_random_base_model/logs/32M_aspect_ratio_%A_%a.out
#SBATCH --error=training_random_base_model/logs/32M_aspect_ratio_%A_%a.err
#SBATCH --time=15:00:00
#SBATCH --nodelist=tikgpu10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --array=1,2,4,6%4

# Run the experiment with time tracking
apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed_aspect_ratio.sh 32M 16 $SLURM_ARRAY_TASK_ID