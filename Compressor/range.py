"""
Range-based compression for reducing hyperparameter value ranges.
"""

import copy
import numpy as np
from typing import Optional, List, Tuple
from ConfigSpace import ConfigurationSpace, Configuration
from openbox import logger

from .base import BaseCompressor
from .utils import (
    update_hp_range,
    collect_compression_details,
    prepare_historical_data
)


class RangeCompressor(BaseCompressor):
    """Base compressor that reduces the range of hyperparameter values."""
    
    def __init__(self, config_space: ConfigurationSpace, 
                top_ratio: float = 0.8,
                sigma: float = 2.0,
                **kwargs):
        """
        Initialize range compressor.
        
        Args:
            config_space: Original configuration space
            top_ratio: Ratio of top-performing configurations to analyze
            sigma: Standard deviation multiplier for range filtering
        """
        super().__init__(config_space, **kwargs)
        
        # Set strategy for range compression
        self.strategy = 'range'
                
        # Store parameters for update_compression
        self.top_ratio = top_ratio
        self.sigma = sigma

        self.computed_space = None
        
    def compress_range(
        self, base_space: Optional[ConfigurationSpace] = None,
        space_history: Optional[List[Tuple[List[Configuration], List[float]]]] = None,
    ) -> ConfigurationSpace:
        """
        Perform range compression.
        
        Args:
            base_space: Base space to compress (if None, uses original space)
            space_history: Historical data for compression analysis
        Returns:
            Range-compressed configuration space
        """
        # Check cache first
        if (self.compressed_space is not None and 
            self.compression_info != {} and
            base_space is None and space_history is None):  # Only cache when range compressing original space
            logger.info("Using existing range compression results (no changes detected)")
            return self.compressed_space
        
        if base_space is None:  # if base_space is not provided, range compress original space
            base_space = copy.deepcopy(self.origin_config_space)

        if space_history is not None and len(space_history) > 0:
            logger.info("Computing range compression from new space history...")
            hist_x, hist_y = prepare_historical_data(space_history)
            self.computed_space = self._compute_range_compression(hist_x, hist_y)
            logger.info("Applying range compression to base space...")            
            self.compressed_space = self._apply_range_compression(base_space)
            range_compression_details = collect_compression_details(base_space, self.computed_space)
            computed_params_count = len(self.computed_space.get_hyperparameters())
        else:
            logger.info("No new space history provided to compute range compression, using base space as is")
            self.compressed_space = base_space
            range_compression_details = {}
            computed_params_count = 0

        self._set_compression_info(
            self.compressed_space,
            computed_params=computed_params_count,
            range_compression_details=range_compression_details
        )
        return self.compressed_space
    
    def update_compression(self, new_top_ratio: Optional[float] = None,
                        new_sigma: Optional[float] = None,
                        **kwargs) -> None:
        """
        Update range compression parameters without re-running compression.
        
        Args:
            new_top_ratio: New top ratio for analysis
            new_sigma: New sigma for range filtering
            **kwargs: Additional parameters
        """
        params_changed = False

        if new_top_ratio is not None and new_top_ratio != getattr(self, 'top_ratio', None):
            self.top_ratio = new_top_ratio
            params_changed = True
        if new_sigma is not None and new_sigma != getattr(self, 'sigma', None):
            self.sigma = new_sigma
            params_changed = True
        
        if params_changed:  # Clear cache if parameters changed
            self.compressed_space = None
            self.compression_info = {}
            self.computed_space = None
            logger.info(f"Range compression parameters changed, clearing cache")
        else:
            logger.info("No parameter changes detected")


    def _apply_range_compression(self, base_space: ConfigurationSpace) -> ConfigurationSpace:
        """Apply range compression to the base space."""
        compressed_space = copy.deepcopy(base_space)    # base_space is the original space or given space
        range_hp_names = [hp.name for hp in self.computed_space.get_hyperparameters()]
        
        hyperparameters_to_update = []
        for hp in list(compressed_space.get_hyperparameters()):
            name = hp.name
            if name in range_hp_names:
                new_hp = self.computed_space.get_hyperparameter(name)
                hyperparameters_to_update.append((name, new_hp))
        
        # Update all hyperparameters
        for name, new_hp in hyperparameters_to_update:
            update_hp_range(compressed_space, name, new_hp)
        return compressed_space
    
    
    def get_range_compression_details(self) -> dict:
        """
        Get detailed information about range compression.
        
        Returns:
            Dictionary containing compression details for each parameter
        """
        if hasattr(self, 'compression_info') and 'range_compression_details' in self.compression_info:
            return self.compression_info['range_compression_details']
        return {}


    def get_compressed_parameters(self) -> List[str]:
        """
        Get list of parameters that were range compressed.
        
        Returns:
            List of parameter names that were compressed
        """
        details = self.get_range_compression_details()
        return list(details.keys())
    
    
    def _compute_range_compression(
        self, hist_x: List[np.ndarray], hist_y: List[np.ndarray]
    ) -> ConfigurationSpace:
        """
        Abstract method to compute range compression space.
        Subclasses must implement this method to provide specific range compression algorithms.
        
        Args:
            hist_x: Historical configuration data
            hist_y: Historical performance data
        Returns:
            Range-compressed configuration space
        """
        if len(hist_x) == 0 or len(hist_y) == 0:
            logger.warning("No valid historical data for range compression")
            return self.origin_config_space