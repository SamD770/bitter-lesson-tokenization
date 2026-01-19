"""
Registry mapping string names to actual Python classes for Gaters, Downsamplers, and Upsamplers.
This allows JSON configs to reference classes by string names.
"""

from model.modules import (
    LinearGater,
    RandomGater,
    EquidistantGater,
    ExactRandomGater,
    SelectTokenDownsampler,
    AverageTokenDownsampler,
    DistributeAddUpsampler,
    DistributeDeviationUpsampler,
)

from model.nawrot_plugin import NawrotDownsampler, NawrotUpsampler, NawrotGater
from model.hnet_plugin import HNetDownsampler, HNetUpsampler, HNetGater
from model.conditional_sequential import (
    SequentiallyDependentRandomGater,
    SequentiallyDependentLinearGater,
    OptimizedSequentialyDependentLinearGater,
    ScaledSequentialyDependentLinearGater,
)


GATER_REGISTRY = {
    "LinearGater": LinearGater,
    "RandomGater": RandomGater,
    "EquidistantGater": EquidistantGater,
    "ExactRandomGater": ExactRandomGater,
    "NawrotGater": NawrotGater,
    "HNetGater": HNetGater,
    "SequentiallyDependentRandomGater": SequentiallyDependentRandomGater,
    "SequentiallyDependentLinearGater": SequentiallyDependentLinearGater,
    "OptimizedSequentialyDependentLinearGater": OptimizedSequentialyDependentLinearGater,
    "ScaledSequentialyDependentLinearGater": ScaledSequentialyDependentLinearGater,
}

DOWNSAMPLER_REGISTRY = {
    "SelectTokenDownsampler": SelectTokenDownsampler,
    "AverageTokenDownsampler": AverageTokenDownsampler,
    "NawrotDownsampler": NawrotDownsampler,
    "HNetDownsampler": HNetDownsampler,
}

UPSAMPLER_REGISTRY = {
    "DistributeAddUpsampler": DistributeAddUpsampler,
    "DistributeDeviationUpsampler": DistributeDeviationUpsampler,
    "NawrotUpsampler": NawrotUpsampler,
    "HNetUpsampler": HNetUpsampler,
}


CLASS_REGISTRY = {
    "GaterClass": GATER_REGISTRY,
    "DownSamplerClass": DOWNSAMPLER_REGISTRY,
    "UpsamplerClass": UPSAMPLER_REGISTRY,
}


def resolve_class(class_type: str, class_name: str):
    """
    Resolve a class name string to the actual class.
    
    Args:
        class_type: One of "GaterClass", "DownSamplerClass", "UpsamplerClass"
        class_name: The string name of the class
        
    Returns:
        The actual Python class
        
    Raises:
        ValueError: If class_type or class_name is not found in registry
    """
    if class_type not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class type: {class_type}. Must be one of {list(CLASS_REGISTRY.keys())}")
    
    registry = CLASS_REGISTRY[class_type]
    if class_name not in registry:
        raise ValueError(f"Unknown {class_type}: {class_name}. Must be one of {list(registry.keys())}")
    
    return registry[class_name]


def get_available_classes(class_type: str) -> list:
    """Get list of available class names for a given class type."""
    if class_type not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class type: {class_type}. Must be one of {list(CLASS_REGISTRY.keys())}")
    return list(CLASS_REGISTRY[class_type].keys())
