#!/bin/bash
source wandb_offline_integration/setup.sh

python -m wandb_offline_integration.testing 2>&1 | tee output.log

source wandb_offline_integration/parse_wandb_path.sh output.log

wandb sync $WANDB_SYNC_PATH