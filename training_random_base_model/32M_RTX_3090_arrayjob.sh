#!/bin/bash
#SBATCH --job-name=32M_run
#SBATCH --output=training_random_base_model/logs/32M_%j.out
#SBATCH --error=training_random_base_model/logs/32M_%j.err
#SBATCH --time=15:00:00
#SBATCH --nodelist=tikgpu09
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --array=42-44%1

apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed.sh 32M 16 $SLURM_ARRAY_TASK_ID