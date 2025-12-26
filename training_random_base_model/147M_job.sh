#!/bin/bash
#SBATCH --job-name=147M_run
#SBATCH --output=training_random_base_model/logs/147M_%A_%a.out
#SBATCH --error=training_random_base_model/logs/147M_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --nodelist=tikgpu10
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --array=3

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    UPDOWN_SAMPLER="sequential"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    UPDOWN_SAMPLER="nawrot"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    UPDOWN_SAMPLER="random"
fi

apptainer exec --nv --bind \
 /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey \
 /scratch/sdauncey/sams_favorite_build.sif \
 bash training_random_base_model/launch_distributed.sh 147M 8 $UPDOWN_SAMPLER 43

