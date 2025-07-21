"""
Define hyperparameters for a series of models with sizes 16M -> 346M. For a standardised pretraining run.
"""
from clean_code.flexible_bitter_llm import SelectTokenDownsampler, ExactRandomGater, FlexibleBitterLLM
from clean_code.utils import parameter_count_string

model_sizes = ["18M", "32M", "73M", "130M", "346M"]

fixed_model_hparams = {
    "vocab_size": 256,
    "downsample_rate": 0.25,
    "sliding_window": 64,
    "GaterClass": ExactRandomGater,
    "DownSamplerClass": SelectTokenDownsampler,
}

variable_model_hparam_dict  = {
    "18M": {
        "embedding_dim": 384,
        "num_heads": 6,
        "n_down_layers": 2,
        "n_mid_layers": 6,
        "n_up_layers": 2,
    },
    "32M": {
        "embedding_dim": 512,
        "num_heads": 8,
        "n_down_layers": 2,
        "n_mid_layers": 6,
        "n_up_layers": 2,
    },
    "73M": {
        "embedding_dim": 768,
        "num_heads": 12,
        "n_down_layers": 2,
        "n_mid_layers": 6,
        "n_up_layers": 2,
    },
    "130M":{
        "embedding_dim": 768,
        "num_heads": 12,
        "n_down_layers": 2,
        "n_mid_layers": 12,
        "n_up_layers": 2,
    },
    "346M": {
        "embedding_dim": 1024,
        "num_heads": 16,
        "n_down_layers": 3,
        "n_mid_layers": 18,
        "n_up_layers": 3,
    },
}

# According to https://arxiv.org/pdf/2406.19146, we want to use:
# - learning rate that decreases 0.5x per 10x increase in model size, I'll use an anchor point of 1.5e-3 for 346M
# - batch size that increases 4x per 3x increase in model size, I'll use an anchor point of 256 for 346M
# - total training and warmup_bytes that increases in proportion to model size, I'll use an anchor point of 1.2B and 16B for 346M (this is undertraining, compute optimal would be more like 5.7B tokens = 25B bytes but the model will be finetuned etc.)
optimization_hparam_dict = {
    "18M": {
        "learning_rate": 3.6e-3,
        "true_batch_size": 32,
        "warmup_bytes": 6.2e7,
        "training_bytes": 8.3e8,
    },
    "32M": {
        "learning_rate": 3e-3,
        "true_batch_size": 64,
        "warmup_bytes": 1.1e8,
        "training_bytes": 1.5e9,
    },
    "73M": {
        "learning_rate": 2.4e-3,
        "true_batch_size": 128,
        "warmup_bytes": 2.5e8,
        "training_bytes": 3.4e9,
    },
    "130M": {
        "learning_rate": 2e-3,
        "true_batch_size": 128,
        "warmup_bytes": 4.5e8,
        "training_bytes": 6e9,
    },
    "346M": {
        "learning_rate": 1.5e-3,
        "true_batch_size": 256,
        "warmup_bytes": 1.2e9,
        "training_bytes": 16e9
    }
}


def bytes_to_steps(n_bytes, sequence_length, batch_size):
    """
    Converts a number of bytes to a number of steps, given a sequence length and batch size.
    """
    return n_bytes / (sequence_length * batch_size)


def print_model_hparams(model_size):
    model_kwargs = {**fixed_model_hparams, **variable_model_hparam_dict[model_size]}
    model = FlexibleBitterLLM(**model_kwargs)
    print(f"Model {model_size} has {parameter_count_string(model)} parameters")

if __name__ == "__main__":
    for model_size in model_sizes:
        print_model_hparams(model_size)

