#!/bin/bash

#SBATCH --cpus-per-task=96
#SBATCH --mem=128GB
#SBATCH --nodelist=tikgpu07
#SBATCH --time=12:00:00
#SBATCH --output=data_processing/fineweb_job.out
#SBATCH --error=data_processing/fineweb_job.err

apptainer exec --nv --bind /itet-stor/sdauncey/net_scratch:/itet-stor/sdauncey/net_scratch,/scratch/sdauncey:/scratch/sdauncey /scratch/sdauncey/sams_favorite_build.sif \
    python -m data_processing.download_and_filter_fineweb_100B