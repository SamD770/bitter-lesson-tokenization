#!/bin/bash
#SBATCH --job-name=eval_ours
#SBATCH --output=eval/logs/ours_%j.out
#SBATCH --error=eval/logs/ours_%j.err
#SBATCH --time=36:00:00
#SBATCH --nodelist=tikgpu09
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/mamba_container2.sif \
 eval/run_lambada_fineweb.sh training_random_base_model/checkpoints/147M_sequential_sequential___2026.01.26_15.56