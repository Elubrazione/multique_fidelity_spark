"""
Compressor module for space compression functionality.

This module provides various compression strategies for configuration spaces:
- Dimension compression: Reduce the number of hyperparameters
- Range compression: Reduce the range of hyperparameter values
- Expert compression: Use expert knowledge to select important parameters
"""

from .base import BaseCompressor
from .dimension import DimensionCompressor
from .range import RangeCompressor
from .expert import ExpertCompressor
from .space_analyzer import SpaceAnalyzer
from .compression_manager import CompressionManager

__all__ = [
    'BaseCompressor',
    'DimensionCompressor', 
    'RangeCompressor',
    'ExpertCompressor',
]
