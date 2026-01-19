"""
Configuration loader for training runs.
Loads JSON configs and resolves class references to actual Python classes.

Model configs are generated on-the-fly by combining:
  - configs/base/model_defaults.json (vocab_size, sliding_window, etc.)
  - configs/base/sizes.json (embedding_dim, num_heads, layers per size)
  - configs/base/architectures.json (GaterClass, DownSamplerClass, UpsamplerClass per architecture)

Run configs only contain training loop parameters in configs/runs/{run_type}.json
"""

import json
import os
from typing import Tuple, Dict, Any, Optional, List

from .class_registry import resolve_class, CLASS_REGISTRY


# Get the directory where this file is located
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


def _load_json(filepath: str) -> dict:
    """Load a JSON file and return its contents."""
    with open(filepath, 'r') as f:
        return json.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Values in override take precedence.
    Nested dicts are merged recursively.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_classes_in_dict(config: dict) -> dict:
    """
    Resolve any class references (string names) to actual Python classes.
    Modifies the dict in place and returns it.
    """
    for class_type in CLASS_REGISTRY.keys():
        if class_type in config and isinstance(config[class_type], str):
            config[class_type] = resolve_class(class_type, config[class_type])
    return config


def get_available_sizes() -> List[str]:
    """Get list of available model sizes from base/sizes.json."""
    sizes_file = os.path.join(CONFIG_DIR, "base", "sizes.json")
    sizes = _load_json(sizes_file)
    return sorted(sizes.keys())


def get_available_architectures() -> List[str]:
    """Get list of available architectures from base/architectures.json."""
    arch_file = os.path.join(CONFIG_DIR, "base", "architectures.json")
    architectures = _load_json(arch_file)
    return sorted(architectures.keys())


def get_run_types() -> List[str]:
    """Get list of available run types from the runs/ config directory."""
    runs_dir = os.path.join(CONFIG_DIR, "runs")
    types = []
    for filename in os.listdir(runs_dir):
        if filename.endswith('.json'):
            types.append(filename[:-5])  # Remove .json extension
    return sorted(types)


def _build_model_config(size: str, architecture: str) -> dict:
    """
    Build a complete model config by combining base defaults, size config, and architecture config.
    
    Args:
        size: Model size (e.g., "18M", "32M")
        architecture: Architecture name (e.g., "random", "nawrot")
        
    Returns:
        Dict of model kwargs (with class names as strings, not yet resolved)
    """
    # Load base configs
    model_defaults = _load_json(os.path.join(CONFIG_DIR, "base", "model_defaults.json"))
    sizes = _load_json(os.path.join(CONFIG_DIR, "base", "sizes.json"))
    architectures = _load_json(os.path.join(CONFIG_DIR, "base", "architectures.json"))
    
    # Validate inputs
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}. Available: {list(sizes.keys())}")
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(architectures.keys())}")
    
    # Remove class references from model_defaults (they come from architectures)
    base_config = {k: v for k, v in model_defaults.items() 
                   if k not in ["GaterClass", "DownSamplerClass", "UpsamplerClass"]}
    
    size_config = sizes[size]
    arch_config = architectures[architecture].copy()
    
    # Combine configs
    model_config = {
        **base_config,
        **size_config,
        **arch_config,
    }
    
    # Handle upsampler_kwargs_from_embedding_dim
    if model_config.pop("upsampler_kwargs_from_embedding_dim", False):
        model_config["upsampler_kwargs"] = {
            "embedding_dim": model_config["embedding_dim"]
        }
    
    return model_config


def load_model_config(size: str, architecture: str) -> dict:
    """
    Load a complete model configuration by combining size and architecture.
    
    Args:
        size: Model size (e.g., "18M", "32M")
        architecture: Architecture name (e.g., "random", "nawrot")
        
    Returns:
        Dict of model kwargs ready for AutoregressiveUnet (with classes resolved)
    """
    model_kwargs = _build_model_config(size, architecture)
    model_kwargs = _resolve_classes_in_dict(model_kwargs)
    return model_kwargs


# Alias for backward compatibility
load_model_config_by_parts = load_model_config


def load_training_config() -> dict:
    """
    Load base training loop configuration.
    
    Returns:
        Dict of training loop kwargs
    """
    return _load_json(os.path.join(CONFIG_DIR, "base", "training_defaults.json"))


def load_run_config(run_type: str) -> dict:
    """
    Load run-specific training configuration overrides.
    
    Args:
        run_type: Run type identifier (e.g., "default", "nawrot", "hnet")
        
    Returns:
        Dict of training loop parameter overrides
    """
    run_config_path = os.path.join(CONFIG_DIR, "runs", f"{run_type}.json")
    if not os.path.exists(run_config_path):
        raise ValueError(f"Unknown run type: {run_type}. Available: {get_run_types()}")
    
    return _load_json(run_config_path)


def load_optimization_config(model_size: str) -> dict:
    """
    Load optimization configuration for a given model size.
    
    Args:
        model_size: Model size identifier (e.g., "18M", "32M", "346M")
        
    Returns:
        Dict of optimization kwargs (learning_rate, effective_batch_size, etc.)
    """
    all_optimization = _load_json(os.path.join(CONFIG_DIR, "base", "optimization.json"))
    
    if model_size not in all_optimization:
        raise ValueError(f"No optimization config for model size: {model_size}. Available: {list(all_optimization.keys())}")
    
    return all_optimization[model_size]


def load_config(
    size: str,
    architecture: str,
    run_type: str = "default",
    aspect_ratio: Optional[int] = None,
    optimization_overrides: Optional[dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load complete configuration for a training run.
    
    Args:
        size: Model size (e.g., "18M", "32M")
        architecture: Architecture name (e.g., "random", "nawrot")
        run_type: Run type for training loop params (e.g., "default", "nawrot")
        aspect_ratio: Optional aspect ratio override (1-8)
        optimization_overrides: Optional dict of optimization param overrides
        
    Returns:
        Tuple of (model_kwargs, training_loop_kwargs, optimization_kwargs)
    """
    # Load model config (generated on-the-fly)
    model_kwargs = load_model_config(size, architecture)
    
    # Load training config with run-specific overrides
    training_loop_kwargs = load_training_config()
    run_config = load_run_config(run_type)
    # Remove any comment fields
    run_config = {k: v for k, v in run_config.items() if k != "comment"}
    training_loop_kwargs = _deep_merge(training_loop_kwargs, run_config)
    
    # Load optimization config
    optimization_kwargs = load_optimization_config(size)
    
    # Apply aspect ratio if specified
    if aspect_ratio is not None:
        model_kwargs["n_mid_layers"] = aspect_ratio
        model_kwargs["n_down_layers"] = 4
        model_kwargs["n_up_layers"] = 4
        model_kwargs["downsample_rate"] = 1 / aspect_ratio
        training_loop_kwargs["downsample_rate_target"] = 1 / aspect_ratio
        optimization_kwargs["training_bytes"] = 3e9
    
    # Apply any additional optimization overrides
    if optimization_overrides:
        optimization_kwargs = _deep_merge(optimization_kwargs, optimization_overrides)
    
    return model_kwargs, training_loop_kwargs, optimization_kwargs


def get_model_name(size: str, architecture: str) -> str:
    """Get the full model name from size and architecture."""
    return f"{size}_{architecture}"


def config_to_wandb(model_kwargs: dict, training_loop_kwargs: dict, optimization_kwargs: dict, stop_condition) -> dict:
    """
    Convert configuration dictionaries to a wandb-compatible config.
    Converts class objects to string representations.
    
    Args:
        model_kwargs: Model configuration
        training_loop_kwargs: Training loop configuration
        optimization_kwargs: Optimization configuration
        stop_condition: The stop condition object
        
    Returns:
        Dict ready for wandb config
    """
    config = {**training_loop_kwargs, **optimization_kwargs, **model_kwargs}
    
    # Convert class objects to string representations
    for class_type in CLASS_REGISTRY.keys():
        if class_type in config and hasattr(config[class_type], '__name__'):
            config[class_type] = config[class_type].__name__
    
    config["stop_condition"] = stop_condition.__class__.__name__
    config["bytes_limit"] = stop_condition.bytes_limit
    
    return config


def model_kwargs_to_json(model_kwargs: dict) -> dict:
    """
    Convert model kwargs to JSON-serializable dict.
    Converts class objects to string names.
    
    Args:
        model_kwargs: Model configuration dict with resolved classes
        
    Returns:
        Dict that can be serialized to JSON
    """
    config_to_save = model_kwargs.copy()
    for class_type in CLASS_REGISTRY.keys():
        if class_type in config_to_save and hasattr(config_to_save[class_type], '__name__'):
            config_to_save[class_type] = config_to_save[class_type].__name__
    return config_to_save


def save_model_config(model_kwargs: dict, filepath: str):
    """
    Save model configuration to a JSON file.
    Useful for saving alongside checkpoints.
    
    Args:
        model_kwargs: Model configuration dict
        filepath: Path to save JSON file
    """
    config_to_save = model_kwargs_to_json(model_kwargs)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config_to_save, f, indent=2)


def load_model_config_from_file(filepath: str) -> dict:
    """
    Load model configuration from any JSON file path.
    Useful for loading configs saved alongside checkpoints.
    
    Args:
        filepath: Path to JSON config file
        
    Returns:
        Dict of model kwargs with classes resolved
    """
    model_kwargs = _load_json(filepath)
    model_kwargs = _resolve_classes_in_dict(model_kwargs)
    return model_kwargs
