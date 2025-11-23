#!/bin/bash
#SBATCH --job-name=18M_run
#SBATCH --output=training_random_base_model/logs/18M_%A_%a.out
#SBATCH --error=training_random_base_model/logs/18M_%A_%a.err
#SBATCH --time=15:00:00
#SBATCH --nodelist=tikgpu07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --array=42-47%2

apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed.sh 18M 8 $SLURM_ARRAY_TASK_ID