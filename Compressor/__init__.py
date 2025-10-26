"""
Compressor module for space compression functionality.

This module provides various compression strategies for configuration spaces:
- Dimension compression: Reduce the number of hyperparameters
- Range compression: Reduce the range of hyperparameter values
- Expert compression: Use expert knowledge to select important parameters
- Compressor: Unified compressor that coordinates dimension compression and range compression
"""

from .base import BaseCompressor
from .dimension import DimensionCompressor
from .range import RangeCompressor
from .expert import ExpertCompressor
from .compressor import Compressor

__all__ = [
    'BaseCompressor',
    'DimensionCompressor', 
    'RangeCompressor',
    'ExpertCompressor',
    'Compressor'
]
