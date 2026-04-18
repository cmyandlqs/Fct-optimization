"""Vulture whitelist for framework-discovered and public API symbols."""

from scripts.model import SubModel
from scripts.predict_optimal_threshold import find_optimal_threshold

# PyTorch discovers and calls `forward` implicitly.
SubModel.forward

# Public inference API used by external callers.
find_optimal_threshold
