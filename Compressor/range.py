"""
Range-based compression for reducing hyperparameter value ranges.
"""

import copy
from typing import Optional, List
from ConfigSpace import ConfigurationSpace
from openbox import logger

from .base import BaseCompressor
from .utils import (
    load_performance_data, 
    update_hp_range,
    collect_compression_details,
    analyze_with_shap,
    filter_numeric_params
)


class RangeCompressor(BaseCompressor):
    """Compressor that reduces the range of hyperparameter values."""
    
    def __init__(self, config_space: ConfigurationSpace, 
                data_path: Optional[str] = None,
                target_column: str = 'spark_time',
                top_ratio: float = 0.8,
                sigma: float = 2.0,
                computed_space: Optional[ConfigurationSpace] = None,
                **kwargs):
        """
        Initialize range compressor.
        
        Args:
            config_space: Original configuration space
            data_path: Path to historical performance data (CSV file)
            target_column: Name of the target performance column
            top_ratio: Ratio of top-performing configurations to analyze
            sigma: Standard deviation multiplier for range filtering
            computed_space: Pre-computed range-compressed space (if provided, skips computation)
        """
        super().__init__(config_space, **kwargs)
        
        self.strategy = 'range'
        
        # Store parameters for update_compression
        self.data_path = data_path
        self.target_column = target_column
        self.top_ratio = top_ratio
        self.sigma = sigma
        
        if computed_space is not None:
            self.computed_space = computed_space
        elif data_path is not None:
            self.computed_space = self._compute_range_compression(data_path, target_column, top_ratio, sigma)
        else:
            self.computed_space = None
        
    def compress(self, base_space: Optional[ConfigurationSpace] = None) -> ConfigurationSpace:
        """
        Perform range compression.
        
        Args:
            base_space: Base space to compress (if None, uses original space)
            
        Returns:
            Range-compressed configuration space
        """
        # Check cache first
        if (self.compressed_space is not None and 
            self.compression_info != {} and
            base_space is None):  # Only cache when range compressing original space
            logger.info("Using existing range compression results (no changes detected)")
            return self.compressed_space
        
        if base_space is None:  # if base_space is not provided, range compress original space
            base_space = copy.deepcopy(self.origin_config_space)
            
        if self.computed_space is None:
            logger.info("No range compression data provided, using base space")
            self.compressed_space = base_space
        else:
            logger.info("Applying range compression to base space (original space or given space) using computed space...")
            self.compressed_space = self._apply_range_compression(base_space)

        # Collect detailed range compression information
        if self.computed_space is not None:
            range_compression_details = collect_compression_details(base_space, self.computed_space)
            computed_params_count = len(self.computed_space.get_hyperparameters())
        else:
            range_compression_details = {}
            computed_params_count = 0
        
        self._set_compression_info(
            self.compressed_space,
            computed_params=computed_params_count,
            range_compression_details=range_compression_details
        )
        return self.compressed_space
    
    def update_compression(self, new_data_path: Optional[str] = None,
                        new_target_column: Optional[str] = None,
                        new_top_ratio: Optional[float] = None,
                        new_sigma: Optional[float] = None,
                        base_space: Optional[ConfigurationSpace] = None) -> ConfigurationSpace:
        """
        Update range compression with new parameters and recompute.
        
        Args:
            new_data_path: New path to performance data
            new_target_column: New target column name
            new_top_ratio: New top ratio for analysis
            new_sigma: New sigma for range filtering
            base_space: Base space to compress
            
        Returns:
            Updated range-compressed configuration space
        """
        params_changed = False

        if new_data_path is not None and new_data_path != getattr(self, 'data_path', None):
            self.data_path = new_data_path
            params_changed = True
        if new_target_column is not None and new_target_column != getattr(self, 'target_column', None):
            self.target_column = new_target_column
            params_changed = True
        if new_top_ratio is not None and new_top_ratio != getattr(self, 'top_ratio', None):
            self.top_ratio = new_top_ratio
            params_changed = True
        if new_sigma is not None and new_sigma != getattr(self, 'sigma', None):
            self.sigma = new_sigma
            params_changed = True
        
        if params_changed:  # Clear cache if parameters changed
            self.compressed_space = None
            self.compression_info = {}
            
            if new_data_path is not None:   # Recompute computed_space if data_path changed
                self.computed_space = self._compute_range_compression(
                    new_data_path, 
                    new_target_column or self.target_column,
                    new_top_ratio or self.top_ratio,
                    new_sigma or self.sigma
                )
            logger.info(f"Range compression parameters changed, clearing cache")
        else:
            logger.info("No parameter changes detected, using existing results")

        return self.compress(base_space)


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
    
    
    def _compute_range_compression(self, data_path: str,
                                target_column: str = 'spark_time',
                                top_ratio: float = 0.8,
                                sigma: float = 2.0) -> ConfigurationSpace:
        """
        Compute range compression space from historical data using SHAP analysis.
        
        Args:
            data_path: Path to historical performance data (CSV file)
            target_column: Name of the target performance column
            top_ratio: Ratio of top-performing configurations to analyze
            sigma: Standard deviation multiplier for range filtering
            
        Returns:
            Range-compressed configuration space
        """
        try:
            logger.info("Computing range compression space from data...")
            data = load_performance_data(data_path)
            if data is None:
                logger.warning("Failed to load performance data")
                return self.origin_config_space    
            numeric_params = filter_numeric_params(self.origin_config_space)

            if not numeric_params:
                logger.warning("No numeric parameters found for range compression")
                return self.origin_config_space
                
            compressed_space = analyze_with_shap(data, numeric_params, target_column, top_ratio, sigma, self.origin_config_space)
            logger.info(f"Range compression completed: {len(numeric_params)} parameters analyzed")
            return compressed_space
            
        except Exception as e:
            logger.error(f"Error in range compression: {e}")
            return self.origin_config_space
