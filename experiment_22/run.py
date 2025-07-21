"""
Experiment 22: like experiment 19 but off policy training.
"""
from clean_code.bitter_llm import set_seed, parameter_count_string
from clean_code.bitter_llm import LinearGater, RandomGater, set_seed, parameter_count_string
from clean_code.off_policy_bitter_llm import OffPolicyBitterLLM, off_policy_bitter_tokenizer_training_loop
import datasets
from transformers import AutoTokenizer
import os
import torch

if __name__ == "__main__":
    byte5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")

    username = "sdauncey"
    scratch_dir = f"/scratch/{username}/tokenizer_training"

    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)

    # Download a portion of OpenWebText dataset
    # This will download a subset of the OpenWebText corpus
    print("Downloading OpenWebText dataset...")

    # Load a small portion of OpenWebText (25% of the dataset)
    openwebtext_25p = datasets.load_dataset(
        "openwebtext",
        split="train[:25%]",  # Using only 25% samples of the dataset for now.
        cache_dir=os.path.join(scratch_dir, "openwebtext_25p_cache"),
        trust_remote_code=True
    )

    print(f"Downloaded {len(openwebtext_25p)} examples from OpenWebText")

    set_seed(42)

    model = OffPolicyBitterLLM(
        vocab_size=byte5_tokenizer.vocab_size, 
        embedding_dim=512, 
        num_heads=8, 
        downsample_rate=0.25, 
        sliding_window=64, 
        GaterClass=LinearGater,
        OffPolicyGaterClass=RandomGater,
        use_off_policy=True
    ).cuda()

    print(f"my_model has {parameter_count_string(model)} parameters")

    train_losses = off_policy_bitter_tokenizer_training_loop(
        model, 
        openwebtext_25p, 
        tokenizer=byte5_tokenizer,
        num_epochs=1, 
        batch_size=32, 
        batch_print_every=30, 
        batch_limit=10*10**3,
        learn_gating=True,
        discount_rate=0.97
    )

    model.use_off_policy = False
    # Messed up: forgot to save the train losses for the off policy training.

    train_losses = off_policy_bitter_tokenizer_training_loop(
        model, 
        openwebtext_25p, 
        tokenizer=byte5_tokenizer,
        num_epochs=1, 
        batch_size=32, 
        batch_print_every=30, 
        batch_limit=10*10**3,
        learn_gating=True,
        discount_rate=0.97
    )

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")

    # Save the train losses to the specified directory
    train_losses_file = os.path.join(net_scratch_dir, f"experiment_22/train_losses_exp22_learned.csv")
    train_losses.to_csv(train_losses_file, index=False)
    print(f"Train losses saved to {train_losses_file}")

    model_file_name = "experiment_22/bitter-llm-exp22.pt"

    # Save the model to the specified directory
    os.makedirs(net_scratch_dir, exist_ok=True)
    model_save_file = os.path.join(net_scratch_dir, model_file_name)
    torch.save(model, model_save_file)
    print(f"Model saved to {model_save_file}")

