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
        # Check if we can use cached results
        if (self._shap_cache['models'] is not None and
            self._shap_cache['shap_values'] is not None and
            self._shap_cache['importances'] is not None):
            logger.info("Using cached SHAP model")
            return (self._shap_cache['models'],
                    self._shap_cache['importances'],
                    self._shap_cache['shap_values'])

        models = []
        importances = []
        shap_values = []

        for i in range(len(hist_x)):
            # Check if hist_x[i] is already numeric data or full data
            if hist_x[i].shape[1] == len(self.numeric_hyperparameter_names):
                hist_x_numeric = hist_x[i]  # Already numeric data, use directly
            else:   # Full data, extract numeric parameters
                hist_x_numeric = hist_x[i][:, self.numeric_hyperparameter_indices]
                # Force conversion to float to handle mixed-type arrays
                hist_x_numeric = hist_x_numeric.astype(float)
            from sklearn.ensemble import RandomForestRegressor
            # model = XGBRegressor(n_estimators=100, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(hist_x_numeric, hist_y[i])
            explainer = shap.Explainer(model)
            shap_value = -np.abs(explainer(hist_x_numeric, check_additivity=False).values)
            mean_shap = shap_value.mean(axis=0)
            df = pd.DataFrame({
                "feature": self.numeric_hyperparameter_names,
                "importance": mean_shap,
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
        
        top_k = min(self.topk, len(self.numeric_hyperparameter_names))
        selected_numeric_indices = np.argsort(importances)[: top_k].tolist()
        selected_param_names = [self.numeric_hyperparameter_names[i] for i in selected_numeric_indices]
        importances_selected = importances[selected_numeric_indices]

        # Convert numeric parameter indices to global parameter indices
        selected_indices = [self.numeric_hyperparameter_indices[i] for i in selected_numeric_indices]

        logger.info(f"SHAP dimension compression selected parameters: {selected_param_names}")
        logger.info(f"SHAP dimension compression importances: {importances_selected}")
        return selected_indices

    def _compute_range_compression(
        self, hist_x: List[np.ndarray], hist_y: List[np.ndarray]
    ) -> ConfigurationSpace:
        """
        Compute range compression using SHAP analysis.
        
        Args:
            hist_x: Historical configuration data
            hist_y: Historical performance data
            
        Returns:
            Compressed configuration space with adjusted ranges
        """
        super()._compute_range_compression(hist_x, hist_y)
        
        _, _, shap_values = self._fetch_shap_cache(hist_x, hist_y)

        all_x = []
        all_y = []
        all_shap_values = []
        
        for i in range(len(hist_x)):
            # if hist_x[i] is already numeric data, create DataFrame directly
            if hist_x[i].shape[1] == len(self.numeric_hyperparameter_names):
                df = pd.DataFrame(hist_x[i], columns=self.numeric_hyperparameter_names)
            else:
                # Extract numeric parameters and convert to float
                numeric_data = hist_x[i][:, self.numeric_hyperparameter_indices].astype(float)
                df = pd.DataFrame(numeric_data, columns=self.numeric_hyperparameter_names)

            df_with_target = df.copy()
            df_with_target['target'] = hist_y[i]
            sorted_data = df_with_target.sort_values('target')
            top_data = sorted_data.head(int(len(sorted_data) * self.top_ratio))

            top_indices = top_data.index.tolist()
            
            all_x.append(top_data[self.numeric_hyperparameter_names].values)
            all_y.append(top_data['target'].values)
            all_shap_values.append(shap_values[i][top_indices])
        
        X_combined = np.vstack(all_x)
        # y_combined = np.hstack(all_y)
        shap_values_combined = np.vstack(all_shap_values)
        
        compressed_ranges = compute_shap_based_ranges(
            X_combined, self.numeric_hyperparameter_names,
            shap_values_combined, self.sigma, self.origin_config_space
        )
        compressed_space = create_space_from_ranges(self.origin_config_space, compressed_ranges)
        
        logger.info(f"SHAP range compression completed: "
                    f"{len(self.origin_config_space.get_hyperparameters())} -> "
                    f"{len(compressed_space.get_hyperparameters())} parameters")
        return compressed_space
    
    def update_compression(self, **kwargs):
        """
        Update compression parameters and clear caches.
        
        Args:
            **kwargs: Parameters to update
        """
        self._clear_shap_cache()
        
        DimensionCompressor.update_compression(self, **kwargs)
        RangeCompressor.update_compression(self, **kwargs)