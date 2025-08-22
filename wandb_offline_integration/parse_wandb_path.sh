#!/bin/bash

# Parse the wandb sync path from the output
WANDB_SYNC_PATH=$(grep "wandb sync" $1 | sed 's/.*wandb sync //')

if [ -n "$WANDB_SYNC_PATH" ]; then
    echo "Found wandb sync path: $WANDB_SYNC_PATH"
    echo "To sync this run to the cloud, run:"
    echo "wandb sync $WANDB_SYNC_PATH"
else
    WANDB_SYNC_PATH="/scratch/sdauncey/tokenizer_training/wandb_logs/wandb/latest-run"
    echo "Could not find wandb sync path in output, defaulting to $WANDB_SYNC_PATH"
fi

