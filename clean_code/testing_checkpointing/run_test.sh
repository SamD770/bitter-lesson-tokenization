# for i in {1..10}; do
#     python -m clean_code.testing_checkpointing --run_name test_run_$i --n_steps 10
# done

# Set environment variables for running accelerate
source .env
WANDB_DIR="/scratch/${USER}/tokenizer_training/wandb_logs/testing_checkpointing"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
WANDB_MODE="offline"

export WANDB_DIR WANDB_CACHE_DIR WANDB_API_KEY WANDB_MODE
mkdir -vp "${WANDB_CACHE_DIR}"

export PATH="/home/sdauncey/.local/bin:$PATH"


upload_wandb() {
    TAR_FILE="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/clean_code/testing_checkpointing/logs/wandb_${SLURM_JOB_ID}.tar.gz"
    RUN_DIR="/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/clean_code/testing_checkpointing/logs/"


    # Move the wandb logs to the net_scratch directory in a compressed tar file.

    tar -czf "${TAR_FILE}" -C "${WANDB_DIR}" .
    tar -xzf "${TAR_FILE}" -C "${RUN_DIR}"

    wandb sync "${RUN_DIR}wandb/latest-run"
}

accelerate launch --main_process_port 0 -m clean_code.testing_checkpointing --run_name test_run_A --n_steps 13
upload_wandb

accelerate launch --main_process_port 0 -m clean_code.testing_checkpointing --run_name test_run_A --n_steps 29
upload_wandb

accelerate launch --main_process_port 0 -m clean_code.testing_checkpointing --run_name test_run_B --n_steps 29
upload_wandb