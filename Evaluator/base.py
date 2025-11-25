from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
import numpy as np
from ConfigSpace import Configuration

class Evaluator(ABC):
    """Evaluator interface
    
    The evaluator is responsible for executing evaluations at
    given configurations and resource levels, returning results.

    Methods:
        __call__: Evaluate configuration and return result dictionary
        build_result_dict: Helper method to build standard result dictionary
    """
    
    @abstractmethod
    def __call__(
        self, 
        config: Configuration, 
        resource_ratio: float,
        plan: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
    @staticmethod
    def build_result_dict(
        objective: float,
        start_time: float,
        extra_info: Optional[Dict[str, Any]] = None,
        traceback: Optional[str] = None,
    ) -> Dict[str, Any]:
        if extra_info is None:
            extra_info = {}
        
        return {
            'result': {'objective': objective},
            'timeout': not np.isfinite(objective),
            'traceback': traceback,
            'extra_info': extra_info,
            'elapsed_time': time.time() - start_time,
        }


class NoOpEvaluator(Evaluator):
    def __init__(self, default_objective: float = 1.0):
        self.default_objective = default_objective
    
    def __call__(
        self, 
        config: Configuration, 
        resource_ratio: float,
        plan: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import time
        start_time = time.time()
        return {
            'result': {'objective': self.default_objective},
            'timeout': False,
            'traceback': None,
            'elapsed_time': time.time() - start_time,
            'extra_info': {},
        }
