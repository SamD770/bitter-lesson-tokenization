from training_random_base_model.config_loader import load_model_config_from_file
from model.model import AutoregressiveUnet
import torch
import os
from safetensors.torch import load_file


def load_model(checkpoint_directory):
    model_state = load_file(os.path.join(checkpoint_directory, "model.safetensors"))

    if any(k.startswith("model.") for k in model_state):
        from bpe_tokenizer.bpe_tokenizer import BPEAutoRegressiveUnet
        return BPEAutoRegressiveUnet.from_checkpoint(checkpoint_directory)

    model_config = load_model_config_from_file(os.path.join(checkpoint_directory, "model_config.json"))
    model = AutoregressiveUnet(**model_config)
    missing, unexpected = model.load_state_dict(model_state, strict=False)

    if len(missing) > 0 or len(unexpected) > 0:
        print("WARNING: missing or unexpected keys in model state dict")
        print("missing:", missing)
        print("unexpected:", unexpected)

    return model