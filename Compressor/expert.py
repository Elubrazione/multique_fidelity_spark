"""
Expert-based compression using domain knowledge.
"""

from typing import List, Optional, Tuple
from openbox import logger
from ConfigSpace import ConfigurationSpace

from .dimension import DimensionCompressor


class ExpertCompressor(DimensionCompressor):
    """
    Expert compressor that inherits from DimensionCompressor.
    
    This is a convenience class that automatically sets the strategy to 'expert'
    and provides a simplified interface for expert-based compression.
    """
    
    def __init__(self, config_space: ConfigurationSpace, 
                expert_params: Optional[List[str]] = None,
                expert_config_file: str = "configs/config_space/expert_space.json",
                **kwargs):
        """
        Initialize expert compressor.
        
        Args:
            config_space: Original configuration space
            expert_params: List of expert parameter names
            expert_config_file: Path to expert configuration file
        """
        # Initialize with expert strategy
        super().__init__(
            config_space=config_space,
            strategy='expert',
            topk=len(expert_params) if expert_params else 0,  # Set topk based on expert params
            expert_params=expert_params,
            expert_config_file=expert_config_file,
            **kwargs
        )
        
    def compress(self, space_history: Optional[List] = None) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Perform expert-based compression.
        
        Args:
            space_history: Not used in expert compression, but kept for interface compatibility
            
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        # Expert compression doesn't use space_history, but we keep the parameter for compatibility
        return super().compress(space_history)
    
    def get_expert_space(self, expert_modified_space: Optional[ConfigurationSpace] = None) -> ConfigurationSpace:
        """
        Get expert-modified space based on selected indices from compression results.
        
        This method creates a new configuration space that combines:
        1. The parameter selection results from compression (which parameters to keep)
        2. The expert knowledge modifications (parameter ranges, constraints, etc.)
        
        The key difference from the `compress()` method:
        - `compress()`: Selects parameters from the ORIGINAL space based on expert knowledge
        - `get_expert_space()`: Selects parameters from an EXPERT-MODIFIED space based on compression results
        
        This is typically used for acquisition function optimizers that need both:
        - Reduced parameter dimensionality (from compression)
        - Expert-tuned parameter ranges and constraints
        
        Args:
            expert_modified_space: The original expert-modified configuration space that contains
                                expert knowledge about parameter ranges, constraints, and modifications.
                                If None, returns the compressed space from the compress() method.
        
        Returns:
            ConfigurationSpace: A new configuration space containing only the selected parameters
                            (from compression results) but with expert modifications applied.
                            This space is typically used for acquisition function optimization.
        """
        if expert_modified_space is None:
            return self.compressed_space
        
        if not self.selected_indices:
            return expert_modified_space
        
        return self._create_space_from_indices(expert_modified_space, self.selected_indices)
