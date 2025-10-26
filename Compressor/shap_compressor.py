"""
SHAP-based compressor implementation.

This module provides a SHAP-based compressor that implements both dimension
and range compression using SHAP (SHapley Additive exPlanations) analysis.
"""
import shap
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from openbox import logger
from ConfigSpace import ConfigurationSpace
from xgboost import XGBRegressor

from .compressor import Compressor
from .dimension import DimensionCompressor
from .range import RangeCompressor
from .utils import (
    create_space_from_ranges,
    filter_numeric_params,
    compute_data_hash,
    compute_shap_based_ranges
)


class SHAPCompressor(Compressor, DimensionCompressor, RangeCompressor):
    """
    SHAP-based compressor that implements both dimension and range compression.
    
    This class inherits from Compressor and implements the DimensionCompressor
    and RangeCompressor interfaces, providing SHAP-based compression algorithms.
    """
    
    def __init__(self, config_space: ConfigurationSpace, **kwargs):
        """
        Initialize SHAP compressor.
        
        Args:
            config_space: Original configuration space
            **kwargs: Additional parameters
        """
        # Initialize base classes
        # Initialize BaseCompressor first to avoid attribute conflicts
        super(DimensionCompressor, self).__init__(config_space, **kwargs)
        super(RangeCompressor, self).__init__(config_space, **kwargs)
        
        # Set strategy and other attributes
        self.strategy = 'shap'
        self.topk = kwargs.get('topk', 5)
        self.expert_params = kwargs.get('expert_params', [])
        self.expert_config_file = kwargs.get('expert_config_file', None)
        self.top_ratio = kwargs.get('top_ratio', 0.8)
        self.sigma = kwargs.get('sigma', 2.0)
        self.computed_space = None
        
        # SHAP model cache
        self._shap_cache = {
            'models': None,
            'importances': None,
            'shap_values': None,
            'data_hash': None,
        }
        
        Compressor.__init__(
            self, 
            config_space=config_space,
            **kwargs
        )
    
    def _fetch_shap_cache(
        self, hist_x: List[np.ndarray], hist_y: List[np.ndarray]
    ) -> Tuple[List[Any], List[np.ndarray]]:
        """
        Get cached SHAP model or compute new one.
        
        Returns:
            Tuple of (models of different tasks, importances of different tasks, shap_values of different tasks)
        """
        # Compute data hash for caching
        data_hash = compute_data_hash(hist_x, hist_y)
        
        # Check if we can use cached results
        if (self._shap_cache['data_hash'] == data_hash and 
            self._shap_cache['models'] is not None and
            self._shap_cache['shap_values'] is not None and
            self._shap_cache['importances'] is not None):
            logger.info("Using cached SHAP model")
            return (self._shap_cache['models'],
                    self._shap_cache['importances'],
                    self._shap_cache['shap_values'])

        models = []
        importances = []
        shap_values = []
        param_names = [hp.name for hp in self.origin_config_space.get_hyperparameters()]

        for i in range(len(hist_x)):
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(hist_x[i], hist_y[i])
            explainer = shap.Explainer(model)
            shap_value = explainer(hist_x[i], check_additivity=False).values
            mean_shap = shap_value.mean(axis=0)
            df = pd.DataFrame({
                "feature": param_names,
                "importance": mean_shap,
                "shap_value": shap_value,
                # minimization problem
                "effect": np.where(mean_shap < 0, "increase_objective", "decrease_objective")
            }).sort_values("importance", ascending=True)
            logger.info(f"SHAP dimension compression feature importance: {df.to_string()}")

            models.append(model)
            importances.append(mean_shap)
            shap_values.append(shap_value)
        
        # all tasks' importances are averaged
        importances = np.mean(np.array(importances), axis=0)

        self._shap_cache.update({
            'models': models,
            'importances': importances,
            'shap_values': shap_values,
            'data_hash': data_hash,
        })
        return models, importances, shap_values


    def _clear_shap_cache(self):
        """Clear SHAP cache when parameters change."""
        for key in self._shap_cache.keys():
            self._shap_cache[key] = None
        logger.info(f"SHAP cache cleared")
    
    def _select_parameters(self, hist_x: List[np.ndarray], hist_y: List[np.ndarray]) -> List[int]:
        """
        Select parameters using SHAP analysis.
        
        Args:
            hist_x: Historical configuration data
            hist_y: Historical performance data
            
        Returns:
            List of selected parameter indices
        """
        super()._select_parameters(hist_x, hist_y)
        
        _, importances, _ = self._fetch_shap_cache(hist_x, hist_y)
        
        param_names = [hp.name for hp in self.origin_config_space.get_hyperparameters()]
        top_k = min(self.topk, len(param_names))
        selected_indices = np.argsort(importances)[: top_k]
        selected_param_names = [param_names[i] for i in selected_indices]
        logger.info(f"SHAP dimension compression selected {len(selected_param_names)} parameters: "
                    f"{selected_param_names}")
        return selected_indices

            
    def update_compression(self, **kwargs):
        """
        Update compression parameters and clear caches.
        
        Args:
            **kwargs: Parameters to update
        """
        self._clear_shap_cache()
        
        DimensionCompressor.update_compression(self, **kwargs)
        RangeCompressor.update_compression(self, **kwargs)