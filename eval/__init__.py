"""
Evaluation module for byte-level language models using lm_eval.
"""

from .wrapper import ByteLevelLMWrapper
from .piqa import evaluate_piqa

__all__ = ["ByteLevelLMWrapper", "evaluate_piqa"]
