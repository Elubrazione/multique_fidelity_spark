"""
Dimension-based compression for reducing the number of hyperparameters.
"""
import numpy as np
from typing import List, Optional, Tuple
from openbox import logger
from ConfigSpace import ConfigurationSpace
from openbox.utils.config_space.util import convert_configurations_to_array

from .base import BaseCompressor


class DimensionCompressor(BaseCompressor):
    """Base Dimension-based compressor that reduces the number of hyperparameters based on importance."""
    
    def __init__(self, config_space: ConfigurationSpace, 
                strategy: str = 'none', 
                topk: int = 35,
                expert_params: Optional[List[str]] = None,
                expert_config_file: str = "configs/config_space/expert_space.json",
                **kwargs):
        """
        Initialize dimension compressor.
        
        Args:
            config_space: Original configuration space
            strategy: Compression strategy ('none', 'ottertune', 'rover', etc.)
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
        self.compressed_space = None
        self.compression_info = None


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
                        space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Update compression with new parameters and recompute.
        
        Args:
            new_strategy: New compression strategy
            new_topk: New number of top parameters to keep
            new_expert_params: New expert-selected parameters
            space_history: Historical data for compression analysis
            
        Returns:
            Tuple of (compressed_space, selected_indices)
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
            self.compression_info = None
            logger.info(f"Parameters changed, clearing cache. New strategy={self.strategy}, topk={self.topk}")
            return self.compress(space_history)
        else:
            logger.info("No parameter changes detected, using existing results")
            return self.compressed_space, self.selected_indices


    def _expert_compression(self) -> Tuple[ConfigurationSpace, List[int]]:
        """Perform expert-based compression using expert parameters."""
        if not self.expert_params:
            self.expert_params = self._load_expert_params()
            
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

    def _algorithm_compression(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Perform algorithm-based compression using historical data.
        
        Args:
            space_history: Historical data for compression analysis
            
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        if space_history is None:
            return self._use_original_space("No space history provided")
            
        hist_x, hist_y = self._prepare_historical_data(space_history)
        if not hist_x or not hist_y:
            return self._use_original_space("Invalid space history data")
        
        # 1. Get algorithm-specific parameter selection
        algorithm_indices = self._select_parameters(hist_x, hist_y)
        # 2. Combine with expert parameters if any
        selected_indices = self._combine_with_expert_params(algorithm_indices)
        # 3. Create compressed space
        compressed_space = self._create_compressed_space(selected_indices)
        
        logger.info(f"Algorithm compression ({self.strategy}): selected {len(selected_indices)} parameters")
        return compressed_space, selected_indices


    def _select_parameters(self, hist_x: List, hist_y: List) -> List[int]:
        """
        Select parameters using algorithm-specific method.
        Subclasses must override this method to implement specific algorithms.
        
        Args:
            hist_x: Historical configuration data
            hist_y: Historical performance data
            
        Returns:
            List of selected parameter indices
        """
        # Default implementation - subclasses should override
        logger.warning(f"Algorithm compression strategy '{self.strategy}' not implemented in base class")
        return list(range(min(self.topk, len(self.origin_config_space.get_hyperparameters()))))


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
    
    def _load_expert_params(self) -> List[str]:
        """Load expert parameters from configuration file."""
        import json
        expert_config_file = getattr(self, 'expert_config_file', "configs/config_space/expert_space.json")
        
        try:
            with open(expert_config_file, "r") as f:
                expert_params = json.load(f)                
            return expert_params
            
        except FileNotFoundError:
            logger.warning(f"Expert config file not found: {expert_config_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing expert config file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading expert parameters: {e}")
            return []
    
