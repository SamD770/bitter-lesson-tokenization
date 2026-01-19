"""
DEPRECATED: This module is kept for backward compatibility.
New code should use config_loader.py instead.

For model configurations, see configs/models/{size}_{architecture}.json
For run configurations, see configs/runs/{run_type}.json
For optimization configs, see configs/base/optimization.json
"""

from .config_loader import (
    load_model_config_by_parts,
    load_optimization_config,
    load_training_config,
    get_available_sizes,
)

# Re-export model_sizes for backward compatibility
model_sizes = get_available_sizes()


def get_model_kwargs(model_size, architecture="random"):
    """
    DEPRECATED: Use config_loader.load_model_config_by_parts() instead.
    
    Args:
        model_size: Size like "18M", "32M"
        architecture: Architecture like "random", "nawrot" (default: "random")
    """
    return load_model_config_by_parts(model_size, architecture)


def get_optimization_kwargs(model_size):
    """
    DEPRECATED: Use config_loader.load_optimization_config() instead.
    """
    return load_optimization_config(model_size)


# Keep training_loop_hparam_defaults for backward compatibility
training_loop_hparam_defaults = load_training_config()
