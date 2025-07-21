"""
Experiment 25: like experiment_24 but with 8x the data and half the tokenization constants (now that flash attention is online)
"""
from clean_code.bitter_llm import set_seed, parameter_count_string
from clean_code.bitter_llm import LinearGater, set_seed, parameter_count_string, CausalGemmaMiniBitterLLM, bitter_tokenizer_training_loop
import datasets
from transformers import AutoTokenizer
import os
import torch
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Experiment 23 for bitter tokenization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
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

    set_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    model = CausalGemmaMiniBitterLLM(
        vocab_size=byte5_tokenizer.vocab_size, 
        embedding_dim=512, 
        num_heads=8, 
        downsample_rate=0.25, 
        sliding_window=64, 
        GaterClass=LinearGater,
    ).cuda()

    print(f"my_model has {parameter_count_string(model)} parameters")

    train_losses = bitter_tokenizer_training_loop(
        model, 
        openwebtext_25p, 
        tokenizer=byte5_tokenizer,
        num_epochs=1, 
        batch_size=128, 
        max_seq_length=2048,
        batch_print_every=30, 
        batch_limit=20*10**3,
        learn_gating=True,
        discount_rate=0.9,
        relative_gating_loss_weight=0.005,
        downsample_rate_target=0.25,
        consistency_loss_weight=0.01
    )

    net_scratch_dir = os.path.join("/itet-stor/sdauncey/net_scratch/VScodeProjects/bitter-lesson-tokenization")

    # Save the train losses to the specified directory
    train_losses_file = os.path.join(net_scratch_dir, f"experiment_25/train_losses_exp25_{args.seed}.csv")
    train_losses.to_csv(train_losses_file, index=False)
    print(f"Train losses saved to {train_losses_file}")

    model_file_name = f"experiment_25/bitter-llm-exp25_{args.seed}.pt"

    # Save the model to the specified directory
    os.makedirs(net_scratch_dir, exist_ok=True)
    model_save_file = os.path.join(net_scratch_dir, model_file_name)
    torch.save(model, model_save_file)
    print(f"Model saved to {model_save_file}")

