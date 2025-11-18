import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

from ..base import AcquisitionFunction
from .utils import convert_configurations_to_array


class AcquisitionOptimizer(ABC):
    def __init__(self, acquisition_function: AcquisitionFunction, config_space, rng=None):
        self.acq = acquisition_function
        self.config_space = config_space
        
        if rng is None:
            self.rng = np.random.RandomState(seed=42)
        else:
            self.rng = rng
        self.iter_id = 0
    
    @abstractmethod
    def _maximize(self, observations: List[Any], num_points: int, **kwargs) -> List[Tuple]:
        pass
    
    def maximize(self, observations: List[Any], num_points: int, **kwargs) -> List:
        results = self._maximize(observations, num_points, **kwargs)
        return [result[1] for result in results]
    
    def _sort_configs_by_acq_value(self, configs, **kwargs):
        acq_values = self._acquisition_function(configs, **kwargs).flatten()
        random_values = self.rng.rand(len(acq_values))
        # Sort by acquisition value (primary) and random tie-breaker (secondary)
        # Last column is primary sort key
        indices = np.lexsort((random_values.flatten(), acq_values.flatten()))
        # Return in descending order (highest acquisition first)
        return [(acq_values[ind], configs[ind]) for ind in indices[::-1]]
    
    def _acquisition_function(self, configs, **kwargs):
        X = convert_configurations_to_array(configs)
        return self.acq(X, **kwargs)