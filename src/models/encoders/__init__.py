"""Encoders module for feature encoding components."""

from .numerical import NumericalEncoder
from .categorical import CategoricalEncoder
from .text import TextEncoder

# GraphEncoder requires torch-geometric, provide fallback if not available
try:
    from .graph import GraphEncoder, HAS_TORCH_GEOMETRIC
except ImportError:
    from .graph import GraphEncoderFallback as GraphEncoder
    HAS_TORCH_GEOMETRIC = False

__all__ = ['NumericalEncoder', 'CategoricalEncoder', 'TextEncoder', 'GraphEncoder', 'HAS_TORCH_GEOMETRIC']
