import numpy as np
import typing
from typing import Tuple
from openbox.utils.constants import VERY_SMALL_NUMBER
from Advisor.surrogate.base import BaseTLSurrogate
from LOCAT.dagp import DAGP


class DAGPWrapper(BaseTLSurrogate):
    def __init__(self, dagp: DAGP, config_space):
        super().__init__(
            config_space=config_space,
            source_hpo_data=None,
            seed=dagp.seed,
            surrogate_type='gp',
            norm_y=dagp.norm_y
        )
        
        self.dagp = dagp
        
        if dagp.types is not None and dagp.bounds is not None:
            self.types = dagp.types
            self.bounds = dagp.bounds
        else:
            from openbox.utils.util_funcs import get_types
            self.types, self.bounds = get_types(config_space)
            self.types = np.hstack((self.types, np.zeros(dagp.extra_dim, dtype=np.uint)))
            self.bounds = np.vstack((self.bounds, np.array([[0.0, 1.0]] * dagp.extra_dim)))
        
        self.y_normalize_mean = dagp.y_normalize_mean
        self.y_normalize_std = dagp.y_normalize_std
        self.var_threshold = VERY_SMALL_NUMBER
        self.hist_ws = []
    
    def train(self, X: np.ndarray, y: np.ndarray):
        self.dagp.train(X, y)
        self.y_normalize_mean = self.dagp.y_normalize_mean
        self.y_normalize_std = self.dagp.y_normalize_std
        
        if self.dagp.types is not None and self.dagp.bounds is not None:
            self.types = self.dagp.types
            self.bounds = self.dagp.bounds
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.dagp.predict(X)
    
    def predict_marginalized_over_instances(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
        
        mean, var = self.dagp.predict_marginalized_over_instances(X)
        var[var < self.var_threshold] = self.var_threshold
        var[np.isnan(var)] = self.var_threshold
        return mean, var
    
    def __getattr__(self, name):
        if hasattr(self.dagp, name):
            return getattr(self.dagp, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
