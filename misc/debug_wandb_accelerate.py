from accelerate import Accelerator

# Tell the Accelerator object to log with wandb
accelerator = Accelerator(log_with="wandb")

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
        project_name="debug_wandb",
        config={"dropout": 0.1, "learning_rate": 1e-2},
        init_kwargs={"wandb": {"config": {"dropout": "abc"}}}
    )

for i in range(10):
    # Log to wandb by calling `accelerator.log`, `step` is optional
    accelerator.log({"train_loss": 1.12**(-i), "valid_loss": 0.8**(-i)}, step=i)


# Make sure that the wandb tracker finishes correctly
accelerator.end_training()