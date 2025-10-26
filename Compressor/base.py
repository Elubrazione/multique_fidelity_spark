"""
Base compression interface and utilities.
"""

import copy
from typing import List, Optional, Tuple, Any
from openbox import logger
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import write
from .utils import filter_numeric_params


class BaseCompressor:
    """Base class for all compression strategies."""
    
    def __init__(self, config_space: ConfigurationSpace, **kwargs):
        """
        Initialize the compressor.
        
        Args:
            config_space: The original configuration space
            **kwargs: Additional compression-specific parameters
        """
        self.origin_config_space = config_space
        self.hyperparameter_names = [hp.name for hp in self.origin_config_space.get_hyperparameters()]
        self.numeric_hyperparameter_names = filter_numeric_params(self.origin_config_space)
        self.numeric_hyperparameter_indices = [self.hyperparameter_names.index(name) for name in self.numeric_hyperparameter_names]
    
        self.compressed_space = None
        self.compression_info = {}
        
    def get_compression_info(self) -> dict:
        """Get information about the compression process."""
        return self.compression_info.copy()
    
    def _create_compressed_space(self, indices: List[int]) -> ConfigurationSpace:
        """
        Create a compressed space based on selected indices.
        
        Args:
            indices: List of hyperparameter indices to keep
            
        Returns:
            Compressed configuration space
        """
        if not indices:
            logger.info("No compression applied, using original space")
            return copy.deepcopy(self.origin_config_space)
            
        compressed_space = ConfigurationSpace()
        for idx in indices:
            name = self.origin_config_space.get_hyperparameter_by_idx(idx)
            if name in self.origin_config_space:
                compressed_space.add_hyperparameter(self.origin_config_space[name])

        logger.info(f"Created compressed space with {len(compressed_space)} parameters")
        return compressed_space
    
    def _get_hyperparameter_names(self, indices: List[int]) -> List[str]:
        """Get hyperparameter names for given indices."""
        return [self.origin_config_space.get_hyperparameter_by_idx(idx) for idx in indices]
    
    def _get_hyperparameter_indices(self, names: List[str]) -> List[int]:
        """Get hyperparameter indices for given names."""
        return [self.origin_config_space.get_idx_by_hyperparameter_name(name) for name in names]
    
    def _get_expert_indices_with_validation(self, expert_params: List[str], return_valid_params: bool = False) -> tuple:
        """
        Get expert parameter indices with validation and error handling.
        
        This method provides a unified way to process expert parameters and get their indices,
        with proper error handling for missing parameters.
        
        Args:
            expert_params: List of expert parameter names
            return_valid_params: If True, also return the list of valid parameter names
            
        Returns:
            If return_valid_params is False: List[int] - valid expert parameter indices
            If return_valid_params is True: Tuple[List[int], List[str]] - (indices, valid_param_names)
        """
        expert_indices = []
        valid_params = []
        
        for param_name in expert_params:
            try:
                idx = self.origin_config_space.get_idx_by_hyperparameter_name(param_name)
                expert_indices.append(idx)
                valid_params.append(param_name)
            except KeyError:
                logger.warning(f"Expert parameter '{param_name}' not found in config space")
        
        if return_valid_params:
            return expert_indices, valid_params
        else:
            return expert_indices
    
    def _use_original_space(self, reason: str = "No compression applied") -> Tuple[ConfigurationSpace, List[int]]:
        """
        Fallback method to use the original configuration space without compression.
        
        This method is used when compression cannot be performed due to various reasons
        such as missing data, invalid parameters, or disabled compression strategy.
        
        Args:
            reason: Reason for using original space (for logging purposes)
            
        Returns:
            Tuple of (original_space_copy, all_indices)
        """
        logger.info(f"Using original configuration space: {reason}")
        self.selected_indices = list(range(len(self.origin_config_space)))
        self.compressed_space = copy.deepcopy(self.origin_config_space)
        
        self._set_compression_info(self.compressed_space, self.selected_indices, reason=reason)
        return self.compressed_space, self.selected_indices
    
    def _set_compression_info(self, compressed_space: ConfigurationSpace, 
                            selected_indices: Optional[List[int]] = None, 
                            space_history: Optional[List] = None,
                            **kwargs):
        """
        Set compression information with common fields.
        
        Args:
            compressed_space: The compressed configuration space
            selected_indices: Indices of selected hyperparameters (for dimension compression)
            space_history: Historical data used for compression
            **kwargs: Additional compression-specific information
        """
        # Common fields for all compressors
        common_info = {
            'strategy': getattr(self, 'strategy', 'none'),
            'original_params': len(self.origin_config_space.get_hyperparameters()),
            'compressed_params': len(compressed_space.get_hyperparameters()),
        }
        # Dimension compression specific fields
        if selected_indices is not None:
            common_info.update({
                'selected_indices': selected_indices,
                'selected_param_names': self._get_hyperparameter_names(selected_indices),
                'compression_ratio': len(selected_indices) / len(self.origin_config_space.get_hyperparameters()),
            })
        if space_history is not None:
            common_info['space_history_size'] = len(space_history)

        self.compression_info = {**common_info, **kwargs}
    
    @staticmethod
    def create_space_from_indices(source_space: ConfigurationSpace, indices: List[int]) -> ConfigurationSpace:
        """
        Create a new configuration space from selected indices of a source space.
        
        This method can be used both as a static method and as an instance method.
        
        Args:
            source_space: The source configuration space
            indices: List of hyperparameter indices to include
            
        Returns:
            New configuration space with only the selected parameters
        """
        new_space = ConfigurationSpace()
        for idx in indices:
            name = source_space.get_hyperparameter_by_idx(idx)
            if name in source_space:
                new_space.add_hyperparameter(source_space[name])
        return new_space
    
    def _create_space_from_indices(self, source_space: ConfigurationSpace, indices: List[int]) -> ConfigurationSpace:
        """
        Instance method wrapper for create_space_from_indices.
        """
        return self.create_space_from_indices(source_space, indices)
