"""
Unified compressor that orchestrates different compression strategies.

This module provides a single entry point for all compression operations,
combining dimension compression, range compression, and expert compression
into a streamlined interface.
"""

import copy
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from openbox import logger
from openbox.utils.history import History, Observation
from ConfigSpace import ConfigurationSpace
# from ConfigSpace.read_and_write.json import write

from .dimension import DimensionCompressor
from .range import RangeCompressor
from .base import BaseCompressor


class Compressor:
    """
    Base compressor that coordinates different compression strategies.
    
    This class provides a single interface for:
    - Dimension compression (parameter selection)
    - Range compression (parameter range adjustment)
    - Expert compression (domain knowledge integration)
    - Data transformation and analysis
    
    Designed for inheritance pattern - subclasses should inherit from both
    Compressor and the specific compressor types (DimensionCompressor, RangeCompressor).
    """
    
    def __init__(self, config_space: ConfigurationSpace, **kwargs):
        """
        Initialize the base compressor.
        
        Args:
            config_space: Original configuration space
            **kwargs: Additional parameters for compressors
        """
        self.origin_config_space = config_space
        self.kwargs = kwargs
        
        # Compression results
        self.compression_results: Dict[str, Any] = {}
        self.compressed_space: Optional[ConfigurationSpace] = None
        self.surrogate_space: Optional[ConfigurationSpace] = None
        

    def compress_space(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, Dict[str, Any]]:
        """
        Perform complete space compression.
        
        This method orchestrates the compression process by delegating to the
        dimension_compressor and range_compressor implementations.
        
        Args:
            space_history: Historical data for compression
            
        Returns:
            Tuple of (compressed_space, compression_info)
        """
        logger.info("Starting space compression...")
        
        current_space = copy.deepcopy(self.origin_config_space)
        compression_info = {
            'original_params': len(self.origin_config_space.get_hyperparameters()),
            'steps': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Dimension compression
        logger.info("Performing dimension compression...")
        current_space, selected_indices = self.compress_dimension(space_history)
        self.surrogate_space = copy.deepcopy(current_space)
        
        step_info = {
            'type': 'dimension',
            'strategy': getattr(self, 'strategy', 'unknown'),
            'selected_indices': selected_indices,
            'num_params': len(current_space.get_hyperparameters()),
            'compression_ratio': len(selected_indices) / compression_info['original_params']
        }
        compression_info['steps'].append(step_info)
        compression_info['dimension_compression'] = getattr(self, 'compression_info', {})
            
        # Step 2: Range compression
        logger.info("Performing range compression...")
        current_space = self.compress_range(current_space, space_history)
        
        step_info = {
            'type': 'range',
            'num_params': len(current_space.get_hyperparameters()),
            'range_compression_details': getattr(self, 'compression_info', {}).get('range_compression_details', {})
        }
        compression_info['steps'].append(step_info)
        compression_info['range_compression'] = getattr(self, 'compression_info', {})

        # Final results
        compression_info['final_params'] = len(current_space.get_hyperparameters())
        compression_info['overall_compression_ratio'] = compression_info['final_params'] / compression_info['original_params']
        
        self.compression_results = compression_info
        self.compressed_space = current_space
        
        logger.info(f"Compression completed: {compression_info['original_params']} -> "
                    f"{compression_info['final_params']} parameters "
                    f"(ratio: {compression_info['overall_compression_ratio']:.3f})")
        
        return self.surrogate_space, current_space
    
    def get_expert_modified_space(self, expert_modified_space: Optional[ConfigurationSpace] = None) -> Optional[ConfigurationSpace]:
        """
        Get expert-modified space based on compression results.
        
        Args:
            expert_modified_space: Original expert-modified space
            
        Returns:
            Expert-modified space aligned with compression results
        """
        if not expert_modified_space:
            return expert_modified_space
            
        # Get selected indices from dimension compression if available
        selected_indices = None
        if self.dimension_compressor and hasattr(self.dimension_compressor, 'selected_indices'):
            selected_indices = self.dimension_compressor.selected_indices
            
        if selected_indices:
            # Use the static method to create space from indices
            return BaseCompressor.create_space_from_indices(expert_modified_space, selected_indices)
            
        return expert_modified_space
    
    def transform_source_data(self, source_hpo_data: Optional[List[History]]) -> Optional[List[History]]:
        """
        Transform source HPO data to match surrogate space.
        
        Args:
            source_hpo_data: List of source histories
            
        Returns:
            Transformed histories matching the surrogate space
        """
        if not source_hpo_data or not self.surrogate_space:
            return source_hpo_data
            
        logger.info(f"Transforming {len(source_hpo_data)} source histories to match surrogate space")
        
        target_param_names = [param.name for param in self.surrogate_space.get_hyperparameters()]
        new_histories = []
        
        for history in source_hpo_data:
            # Create new history with transformed observations
            data = {
                'task_id': history.task_id,
                'num_objectives': history.num_objectives,
                'num_constraints': history.num_constraints,
                'ref_point': history.ref_point,
                'meta_info': history.meta_info,
                'global_start_time': history.global_start_time.isoformat(),
                'observations': [obs.to_dict() for obs in history.observations]
            }
            
            # Transform observations to match target space
            for obs in data['observations']:
                new_conf = {}
                for name in target_param_names:
                    if name in obs['config']:
                        new_conf[name] = obs['config'][name]
                obs['config'] = new_conf
                
            # Reconstruct history
            global_start_time = data.pop('global_start_time')
            global_start_time = datetime.fromisoformat(global_start_time)
            observations = data.pop('observations')
            observations = [Observation.from_dict(obs, self.compressed_space) for obs in observations]
            
            new_history = History(**data)
            new_history.global_start_time = global_start_time
            new_history.update_observations(observations)
            
            new_histories.append(new_history)
            
        logger.info(f"Successfully transformed {len(new_histories)} histories")
        return new_histories
    
    def get_space_info(self, space: Optional[ConfigurationSpace] = None) -> Dict[str, Any]:
        """
        Get information about a configuration space.
        
        Args:
            space: Configuration space to analyze (if None, uses compressed space)
            
        Returns:
            Dictionary with space information
        """
        if space is None:
            space = self.compressed_space or self.origin_config_space
            
        info = {
            'num_hyperparameters': len(space.get_hyperparameters()),
            'hyperparameter_names': [hp.name for hp in space.get_hyperparameters()],
            'hyperparameter_types': {},
            'ranges': {}
        }
        
        for hp in space.get_hyperparameters():
            info['hyperparameter_types'][hp.name] = type(hp).__name__
            
            if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
                info['ranges'][hp.name] = (hp.lower, hp.upper)
            elif hasattr(hp, 'choices'):
                info['ranges'][hp.name] = list(hp.choices)
                
        return info
    
    def update_compression(self, **kwargs) -> None:
        """
        Update compression parameters without re-running compression.
        
        This method delegates to the dimension_compressor and range_compressor implementations
        to update their configurations. After calling this method, need to call compress_space()
        to re-run compression with the new parameters.
        
        Args:
            **kwargs: New compression parameters
        """
        logger.info("Updating compression parameters...")
        
        # Delegate update to dimension compressor implementation
        if hasattr(self, '_select_parameters'):
            DimensionCompressor.update_compression(self, **kwargs)
            
        # Delegate update to range compressor implementation
        if hasattr(self, '_compute_range_compression'):
            RangeCompressor.update_compression(self, **kwargs)
        
        # Clear internal state since parameters changed
        self.compression_results = {}
        self.compressed_space = None
        
        logger.info("Compression parameters updated. Call compress_space() to re-run compression.")
    
    def get_compression_info(self) -> Dict[str, Any]:
        """Get comprehensive compression information."""
        return self.compression_results.copy()
    
    def save_compression_info(self, filepath: str) -> None:
        """Save compression information to file."""
        with open(filepath, 'w') as f:
            json.dump(self.compression_results, f, indent=2, default=str)
        logger.info(f"Compression info saved to {filepath}")
    
    def load_compression_info(self, filepath: str) -> None:
        """Load compression information from file."""
        with open(filepath, 'r') as f:
            self.compression_results = json.load(f)
        logger.info(f"Compression info loaded from {filepath}")

    def get_selected_indices(self) -> Optional[List[int]]:
        """Get selected indices from dimension compression."""
        if hasattr(self, 'selected_indices'):
            return self.selected_indices
        return None
    
    def compress_dimension(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Dimension compression implementation.
        
        For classes that inherit from DimensionCompressor, this provides a unified interface.
        
        Args:
            space_history: Historical data for compression analysis
            
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        if hasattr(self, '_select_parameters'):
            return DimensionCompressor.compress_dimension(self, space_history)
        else:
            raise NotImplementedError("No dimension compression implementation available")
    
    def compress_range(self, base_space: Optional[ConfigurationSpace] = None,
                        space_history: Optional[List] = None) -> ConfigurationSpace:
        """
        Range compression implementation.
        
        For classes that inherit from RangeCompressor, this provides a unified interface.
        
        Args:
            base_space: Base space to compress
            space_history: Historical data for compression analysis
            
        Returns:
            Compressed configuration space
        """
        if hasattr(self, '_compute_range_compression'):
            return RangeCompressor.compress_range(self, base_space, space_history)
        else:
            raise NotImplementedError("No range compression implementation available")

    @property
    def config_space(self) -> ConfigurationSpace:
        """Get the current configuration space (compressed or original)."""
        return self.compressed_space or self.origin_config_space

