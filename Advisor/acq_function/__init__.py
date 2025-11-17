"""Acquisition Functions for Bayesian Optimization
"""

from typing import Dict, Type, List

from .base import (
    SurrogateModel,
    AcquisitionFunction,
    SingleObjectiveAcquisition,
    TransferLearningAcquisition,
)
from .ei import ExpectedImprovement
from .ucb import UpperConfidenceBound
from .weighted_rank import WeightedRank


_ACQ_REGISTRY: Dict[str, Type[AcquisitionFunction]] = {
    'ei': ExpectedImprovement,
    'ucb': UpperConfidenceBound,
}


def register(name: str, acq_class: Type[AcquisitionFunction]) -> None:
    _ACQ_REGISTRY[name.lower()] = acq_class


def get_acq(acq_type: str, model: SurrogateModel, **kwargs) -> AcquisitionFunction:
    acq_type = acq_type.lower()
    
    if acq_type.startswith('wrk'):
        parts = acq_type.split('_')
        inner_acq = parts[1] if len(parts) > 1 else 'ei'
        return WeightedRank(
            model=model,
            acq_func=inner_acq,
            temperature=kwargs.get('temperature', 0.1)
        )
    
    if acq_type in _ACQ_REGISTRY:
        acq_class = _ACQ_REGISTRY[acq_type]
        return acq_class(model=model, **kwargs)
    
    available = ', '.join(sorted(_ACQ_REGISTRY.keys()))
    raise ValueError(
        f"Unknown acquisition function: '{acq_type}'. "
        f"Available: {available}, or 'wrk_*' for transfer learning"
    )


def list_available() -> List[str]:
    standard = sorted(_ACQ_REGISTRY.keys())
    return standard + ['wrk_*  (transfer learning)']


__all__ = [
    'SurrogateModel',
    'AcquisitionFunction',
    'SingleObjectiveAcquisition',
    'TransferLearningAcquisition',
    
    'ExpectedImprovement',
    'UpperConfidenceBound',
    'WeightedRank',
    
    'get_acq',
    'register',
    'list_available',
]