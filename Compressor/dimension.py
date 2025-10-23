"""
Dimension-based compression for reducing the number of hyperparameters.
"""
import numpy as np
from typing import List, Optional, Tuple
from openbox import logger
from ConfigSpace import ConfigurationSpace
from openbox.utils.config_space.util import convert_configurations_to_array

from .base import BaseCompressor
from .utils import build_my_compressor


class DimensionCompressor(BaseCompressor):
    """Compressor that reduces the number of hyperparameters based on importance."""
    
    def __init__(self, config_space: ConfigurationSpace, 
                strategy: str = 'none', 
                topk: int = 35,
                expert_params: Optional[List[str]] = None,
                **kwargs):
        """
        Initialize dimension compressor.
        
        Args:
            config_space: Original configuration space
            strategy: Compression strategy ('none', 'ottertune', 'rover', etc.)
            topk: Number of top parameters to keep
            expert_params: List of expert-selected parameter names
        """
        super().__init__(config_space, **kwargs)
        self.strategy = strategy
        self.topk = topk
        self.expert_params = expert_params or []
        self.compressor = None
        self.selected_indices = None

    def compress(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Perform dimension compression.
        
        Args:
            space_history: Historical data for compression analysis
            
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        if self.strategy == 'none':
            return self._use_original_space("No dimension compression strategy specified")
        
        # don't set compression_info here, set it in the expert and algorithm compression methods
        # because it will be automatically set when using original space by _use_original_space method
        if self.strategy == 'expert':
            compressed_space, selected_indices = self._expert_compression()
        else:
            compressed_space, selected_indices = self._algorithm_compression(space_history)
        self.compressed_space = compressed_space
        self.selected_indices = selected_indices
        return self.compressed_space, self.selected_indices
    
    def _expert_compression(self) -> Tuple[ConfigurationSpace, List[int]]:
        """Perform expert-based compression using ExpertCompressor."""
        from .expert import ExpertCompressor
        expert_compressor = ExpertCompressor(self.origin_config_space, expert_params=self.expert_params)
        compressed_space, selected_indices = expert_compressor.compress()
        self.compression_info = expert_compressor.compression_info
        return compressed_space, selected_indices

    def _algorithm_compression(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, List[int]]:
        """Perform algorithm-based compression using historical data."""
        if space_history is None:
            return self._use_original_space("No space history provided")
            
        hist_x, hist_y = self._prepare_historical_data(space_history)
        if not hist_x or not hist_y:
            return self._use_original_space("Invalid space history data")
            
        self.compressor, algorithm_indices = build_my_compressor(
            hist_x, hist_y, self.topk, func_str=self.strategy
        )
        selected_indices = self._combine_with_expert_params(algorithm_indices)
        compressed_space = self._create_compressed_space(selected_indices)
        
        logger.info(f"Algorithm compression ({self.strategy}): selected {len(selected_indices)} parameters")
        self.compression_info = {
            'strategy': self.strategy,
            'algorithm_selected': len(algorithm_indices),
            'expert_selected': len(self.expert_params),
            'total_selected': len(selected_indices),
            'selected_params': self._get_hyperparameter_names(selected_indices)
        }
        return compressed_space, selected_indices
    
    def _prepare_historical_data(self, space_history: List) -> Tuple[List, List]:
        """Prepare historical data for compression analysis."""
        hist_x = []
        hist_y = []
        
        for idx, (X, y) in enumerate(space_history):
            if not idx:
                logger.info(f"Processing space_history[0] objectives: {np.array(y)}")
            hist_x.append(convert_configurations_to_array(X))
            hist_y.append(np.array(y))
            
        return hist_x, hist_y
    
    def _combine_with_expert_params(self, algorithm_indices: List[int]) -> List[int]:
        """Combine algorithm-selected indices with expert parameters."""
        if not self.expert_params:
            return algorithm_indices
        
        expert_indices = self._get_expert_indices_with_validation(self.expert_params)
        
        # Combine indices, prioritizing expert parameters
        combined_indices = expert_indices.copy()
        for idx in algorithm_indices:
            if idx not in combined_indices and len(combined_indices) < self.topk:
                combined_indices.append(idx)
                
        return sorted(combined_indices)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained compressor model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if self.compressor is None:
            raise ValueError("Compressor not trained. Call compress() first.")
            
        if isinstance(self.compressor, list):
            predictions = []
            for model in self.compressor:
                pred = model.predict(X)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
        else:
            return self.compressor.predict(X)
