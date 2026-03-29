#!/bin/bash
# Kill all ablation screen sessions and their child processes.

# Kill the underlying training processes first
pkill -f "training_random_base_model.run.*--size 40M" && echo "Training processes terminated." || echo "No matching training processes found."

# Then close the screen sessions
screen -ls | grep 'abl_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit
echo "All ablation screen sessions terminated."
