# Set environment variables for running accelerate
source /itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization/.env
WANDB_DIR="/scratch/${USER}/tokenizer_training/wandb_logs"
WANDB_CACHE_DIR="${WANDB_DIR}/.cache"
WANDB_MODE="offline"

export WANDB_DIR WANDB_CACHE_DIR WANDB_API_KEY WANDB_MODE
mkdir -vp "${WANDB_CACHE_DIR}"

export PATH="/home/sdauncey/.local/bin:$PATH"

