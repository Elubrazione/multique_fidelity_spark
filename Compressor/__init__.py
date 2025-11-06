"""
Compressor module for space compression functionality.

This module provides various compression strategies for configuration spaces:
- Dimension compression: Reduce the number of hyperparameters
- Range compression: Reduce the range of hyperparameter values
- Expert compression: Use expert knowledge to select important parameters
- Compressor: Unified compressor that coordinates dimension compression and range compression
- LlamaTune methods: Low-dimensional embeddings (REMBO/HesBO), quantization
"""

from typing import Type, Optional
from ConfigSpace import ConfigurationSpace

from .base import BaseCompressor
from .dimension import DimensionCompressor
from .range import RangeCompressor
from .expert import ExpertCompressor
from .compressor import Compressor
from .shap_compressor import SHAPCompressor
from .llamatune import LlamaTuneCompressor
from .low_embeddings import LinearEmbeddingConfigSpace, REMBOConfigSpace, HesBOConfigSpace
from .quantization import Quantization

_COMPRESSOR_REGISTRY = {
    'shap': SHAPCompressor,
    'llamatune': LlamaTuneCompressor,
    'expert': SHAPCompressor,
    'none': None,
}

def get_compressor(compressor_type: Optional[str] = None, 
                   config_space: ConfigurationSpace = None,
                   **kwargs):
    if compressor_type is None:
        if 'adapter_alias' in kwargs or 'le_low_dim' in kwargs:
            compressor_type = 'llamatune'
        else:
            compressor_type = kwargs.get('strategy', 'shap')
            if compressor_type == 'none':
                compressor_type = 'none'
            else:
                compressor_type = 'shap'
    
    if compressor_type == 'none':
        class NoCompressor(Compressor):
            def compress_space(self, space_history=None):
                return config_space, config_space
            def unproject_point(self, point):
                return point.get_dictionary() if hasattr(point, 'get_dictionary') else dict(point)
        return NoCompressor(config_space=config_space, **kwargs)
    
    if compressor_type not in _COMPRESSOR_REGISTRY:
        raise ValueError(f"Unknown compressor type: {compressor_type}. "
                        f"Available types: {list(_COMPRESSOR_REGISTRY.keys())}")
    
    compressor_class = _COMPRESSOR_REGISTRY[compressor_type]
    if compressor_class is None:
        raise ValueError(f"Compressor type '{compressor_type}' is not available")
    
    return compressor_class(config_space=config_space, **kwargs)

__all__ = [
    'BaseCompressor',
    'DimensionCompressor', 
    'RangeCompressor',
    'ExpertCompressor',
    'Compressor',
    'SHAPCompressor',
    'LlamaTuneCompressor',
    'LinearEmbeddingConfigSpace',
    'REMBOConfigSpace',
    'HesBOConfigSpace',
    'Quantization',
    'get_compressor',
]
