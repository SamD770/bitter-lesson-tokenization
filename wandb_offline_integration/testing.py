from accelerate import Accelerator
import random
from math import log
from datetime import datetime

def main():

    print("Starting accelerator")
    accelerator = Accelerator(log_with="wandb")
    
    config = {
        "test_hparam": 123,
    }

    accelerator.init_trackers(
        "testing_wandb", 
        config=config, 
        init_kwargs={
            "wandb": {
                "config": config,
                "entity": "samdauncey-eth-z-rich",
                "id": f"test_run_id_{datetime.now().strftime('%Y_%m%d_%H%M')}"
            }
        },
    )

    for i in range(1, 20):
        accelerator.log({"value": log(i) + random.random()})

    wandb_tracker = accelerator.get_tracker("wandb")
    print(f"Wandb run id: {wandb_tracker.run.id=}")

    accelerator.end_training()


if __name__ == "__main__":
    main()