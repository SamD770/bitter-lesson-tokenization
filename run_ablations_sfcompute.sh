#!/bin/bash
# Launch all 6 sequential 40M ablation runs in parallel, one per GPU in separate screen sessions.
# Usage: bash run_ablations_sfcompute.sh
# Requires: screen, apptainer

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_PATH="$REPO_DIR/mamba_container_mithril2.sif"
SCRATCH_DIR="/data/scratch"
LOG_DIR="$REPO_DIR/training_random_base_model/logs/ablations"
mkdir -p "$LOG_DIR"

# Load env vars (WANDB_API_KEY, HF_TOKEN) from .env
if [ -f "$REPO_DIR/.env" ]; then
    source "$REPO_DIR/.env"
fi

# Experiments: (screen_name, GPU_id, architecture, run_type)
declare -a EXPERIMENTS=(
    "abl_baseline        0  sequential          sequential"
    "abl_discount1       1  sequential          ablations/sequential_discount1"
    "abl_no_early_exit   2  sequential          ablations/sequential_no_early_exit"
    "abl_no_batch_rel    3  sequential          ablations/sequential_no_batch_relative"
    "abl_filter1         4  sequential_filter1  sequential"
    "abl_no_consistency  5  sequential          ablations/sequential_no_consistency"
)

for entry in "${EXPERIMENTS[@]}"; do
    read -r SESSION GPU ARCH RUN_TYPE <<< "$entry"

    LOG_FILE="$LOG_DIR/${SESSION}.log"

    APPTAINER_CMD="apptainer exec --nv \
--bind $SCRATCH_DIR:/scratch/sdauncey \
--env WANDB_API_KEY=$WANDB_API_KEY \
--env CUDA_VISIBLE_DEVICES=$GPU \
$CONTAINER_PATH \
bash -c \"cd $REPO_DIR && python -m accelerate.commands.launch --num_processes=1 --main_process_port 0 -m training_random_base_model.run --size 40M --architecture $ARCH --run_type $RUN_TYPE --batch_size 32 2>&1 | tee $LOG_FILE\""

    echo "Launching screen session: $SESSION (GPU $GPU, arch=$ARCH, run_type=$RUN_TYPE)"
    screen -dmS "$SESSION" bash -c "$APPTAINER_CMD"
done

echo ""
echo "All sessions launched. Monitor with:"
echo "  screen -ls                          # list sessions"
echo "  screen -r <session_name>            # attach to a session"
echo "  tail -f $LOG_DIR/<session>.log      # follow a log"
