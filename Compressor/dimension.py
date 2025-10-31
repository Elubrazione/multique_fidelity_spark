"""
Dimension-based compression for reducing the number of hyperparameters.
"""
from typing import List, Optional, Tuple
import numpy as np
from openbox import History, logger
from ConfigSpace import ConfigurationSpace, Configuration

from .base import BaseCompressor
from .utils import prepare_historical_data, load_expert_params


class DimensionCompressor(BaseCompressor):
    """Base Dimension-based compressor that reduces the number of hyperparameters based on importance."""
    
    def __init__(self, config_space: ConfigurationSpace, 
                strategy: str = 'dimension', 
                topk: int = 35,
                expert_params: Optional[List[str]] = None,
                expert_config_file: str = "config_space/expert_space.json",
                **kwargs):
        """
        Initialize dimension compressor.
        
        Args:
            config_space: Original configuration space
            strategy: Compression strategy (e.g. 'expert', 'none', 'dimension')
            topk: Number of top parameters to keep
            expert_params: List of expert-selected parameter names
            expert_config_file: Path to expert configuration file
        """
        super().__init__(config_space, **kwargs)
        self.strategy = strategy
        self.topk = topk
        self.expert_params = expert_params or []
        self.expert_config_file = expert_config_file
        self.selected_indices = None


    def compress_dimension(
        self,
        space_history: Optional[List[Tuple[List[Configuration], List[float]]]] = None,
    ) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Perform dimension compression.
        
        Args:
            space_history: Historical data for compression analysis
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        if self.strategy == 'none':
            return self._use_original_space("No dimension compression strategy specified")
        
        # Check cache first
        if (self.compressed_space is not None and
            self.selected_indices is not None and
            self.compression_info != {}):
            logger.info("Using existing compression results (no changes detected)")
            return self.compressed_space, self.selected_indices
        
        # Perform compression using strategy-specific implementation
        # don't set compression_info here, set it in the expert and algorithm compression methods
        # because it will be automatically set when using original space by _use_original_space method
        if self.strategy == 'expert':
            compressed_space, selected_indices = self._expert_compression()
        else:
            compressed_space, selected_indices = self._algorithm_compression(space_history)
        
        self._update_compression_state(compressed_space, selected_indices, space_history)
        return compressed_space, selected_indices


    def update_compression(self, new_strategy: Optional[str] = None,
                        new_topk: Optional[int] = None,
                        new_expert_params: Optional[List[str]] = None,
                        **kwargs) -> None:
        """
        Update compression parameters without re-running compression.
        
        Args:
            new_strategy: New compression strategy
            new_topk: New number of top parameters to keep
            new_expert_params: New expert-selected parameters
            **kwargs: Additional parameters
        """
        # Check if any parameters actually changed
        params_changed = False
        if new_strategy is not None and new_strategy != self.strategy:
            self.strategy = new_strategy
            params_changed = True
        if new_topk is not None and new_topk != self.topk:
            self.topk = new_topk
            params_changed = True
        if new_expert_params is not None and new_expert_params != self.expert_params:
            self.expert_params = new_expert_params
            params_changed = True
        
        # Clear cache if parameters changed
        if params_changed:
            self.compressed_space = None
            self.selected_indices = None
            self.compression_info = {}
            logger.info(f"Parameters changed, clearing cache. New strategy={self.strategy}, topk={self.topk}")
        else:
            logger.info("No parameter changes detected")


    def _expert_compression(self) -> Tuple[ConfigurationSpace, List[int]]:
        """Perform expert-based compression using expert parameters."""
        if not self.expert_params:
            self.expert_params = load_expert_params(self.expert_config_file)
            
        # Get indices for expert parameters with validation
        expert_indices, valid_params = self._get_expert_indices_with_validation(
            self.expert_params, return_valid_params=True
        )
        
        if not expert_indices:
            reason = "No expert parameters available" if not self.expert_params else "No valid expert parameters found"
            return self._use_original_space(reason)
            
        selected_indices = sorted(expert_indices)
        compressed_space = self._create_compressed_space(selected_indices)
        
        logger.info(f"Expert compression: selected {len(selected_indices)} parameters")
        
        # Set compression info with expert-specific details
        self._set_compression_info(
            compressed_space=compressed_space,
            selected_indices=selected_indices,
            space_history=None,
            strategy='expert',
            selected_params=valid_params,
            config_file=getattr(self, 'expert_config_file', None)
        )
        
        return compressed_space, selected_indices

    def _algorithm_compression(
        self, space_history: Optional[List[History]] = None
    ) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Perform algorithm-based compression using historical data.
        
        Args:
            space_history: Historical data for compression analysis
            
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        if not space_history:
            return self._use_original_space("No space history provided")
            
        hist_x, hist_y = prepare_historical_data(space_history)

        # 1. Get algorithm-specific parameter selection
        algorithm_indices = self._select_parameters(hist_x, hist_y)
        # 2. Combine with expert parameters if any
        selected_indices = self._combine_with_expert_params(algorithm_indices)
        # 3. Create compressed space
        compressed_space = self._create_compressed_space(selected_indices)
        
        logger.info(f"Algorithm compression ({self.strategy}): selected {len(selected_indices)} parameters")
        return compressed_space, selected_indices


    def _select_parameters(self, hist_x: List[np.ndarray], hist_y: List[np.ndarray]) -> List[int]:
        """
        Select parameters using algorithm-specific method.
        Subclasses must override this method to implement specific algorithms.
        
        Args:
            hist_x: Historical configuration data
            hist_y: Historical performance data
            
        Returns:
            List of selected parameter indices
        """
        if len(hist_x) == 0 or len(hist_y) == 0:
            logger.warning("No valid historical data for dimension compression")
            return list(range(len(self.origin_config_space.get_hyperparameters())))


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

    def _update_compression_state(self, compressed_space: ConfigurationSpace, 
                                selected_indices: List[int], 
                                space_history: Optional[List] = None):
        """
        Update internal compression state.
        
        Args:
            compressed_space: The compressed configuration space
            selected_indices: Indices of selected hyperparameters
            space_history: Historical data used for compression
        """
        self.compressed_space = compressed_space
        self.selected_indices = selected_indices

        self._set_compression_info(compressed_space, selected_indices, space_history)
        logger.info(f"Dimension compression completed: {len(selected_indices)} parameters selected")
