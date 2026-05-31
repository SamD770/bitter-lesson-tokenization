#!/bin/bash
# Example SLURM batch script for a 73M-parameter training run.
# Adjust partition / nodelist / resources for your cluster.
#
# If you run inside a container (Apptainer/Singularity), wrap the launch, e.g.:
#   apptainer exec --nv --bind "${SCRATCH_DIR}:${SCRATCH_DIR}" "${CONTAINER_IMAGE}" \
#       bash training_random_base_model/launch_distributed.sh 73M 8
#
#SBATCH --job-name=73M_run
#SBATCH --output=training_random_base_model/logs/73M_%j.out
#SBATCH --error=training_random_base_model/logs/73M_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

echo "Starting 73M training run at $(date)"

# batch size 32 is possible when using flash attention
bash training_random_base_model/launch_distributed.sh 73M 8
