"""
Expert-based compression using domain knowledge.
"""

import json
from typing import List, Optional, Tuple
from openbox import logger
from ConfigSpace import ConfigurationSpace

from .base import BaseCompressor


class ExpertCompressor(BaseCompressor):
    """Compressor that uses expert knowledge to select important parameters."""
    
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
        super().__init__(config_space, **kwargs)
        self.expert_params = expert_params or []
        self.expert_config_file = expert_config_file
        self.compressed_space = None
        self.selected_indices = None
        
    def compress(self) -> Tuple[ConfigurationSpace, List[int]]:
        """
        Perform expert-based compression.
        
        Returns:
            Tuple of (compressed_space, selected_indices)
        """
        if not self.expert_params:
            self.expert_params = self._load_expert_params()
            
        # Get indices for expert parameters with validation
        expert_indices, valid_params = self._get_expert_indices_with_validation(
            self.expert_params, return_valid_params=True
        )
        
        if not expert_indices:
            reason = "No expert parameters available" if not self.expert_params else "No valid expert parameters found"
            return self._use_original_space(reason)
            
        self.selected_indices = sorted(expert_indices)
        self.compressed_space = self._create_compressed_space(self.selected_indices)
        logger.info(f"Expert compression: selected {len(self.selected_indices)} parameters")

        self._set_compression_info(
            strategy='expert',
            selected_params=valid_params,
            config_file=self.expert_config_file
        )
        
        return self.compressed_space, self.selected_indices
    
    def _load_expert_params(self) -> List[str]:
        """Load expert parameters from configuration file."""
        try:
            with open(self.expert_config_file, "r") as f:
                expert_params = json.load(f)                
            return expert_params
            
        except FileNotFoundError:
            logger.warning(f"Expert config file not found: {self.expert_config_file}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing expert config file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading expert parameters: {e}")
            return []
    
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
