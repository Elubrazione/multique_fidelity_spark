from typing import Dict, Type, List, Optional
import numpy as np
from ConfigSpace import ConfigurationSpace

from .utils import get_types

from .base import (
    Surrogate,
    SingleFidelitySurrogate,
    TransferLearningSurrogate,
    HistoryLike
)
from .prf import ProbabilisticRandomForest
from .gp import GaussianProcess
from .ensemble import MFGPE, RGPE, RGPE_avg

from .utils import (
    Normalizer,
    calculate_preserving_order_num
)

_SURROGATE_REGISTRY: Dict[str, Type[SingleFidelitySurrogate]] = {
    'prf': ProbabilisticRandomForest,
    'gp': GaussianProcess,
}


def register(name: str, surrogate_class: Type[SingleFidelitySurrogate]) -> None:
    _SURROGATE_REGISTRY[name.lower()] = surrogate_class


def build_surrogate(
    surrogate_type: str,
    config_space: ConfigurationSpace,
    rng: Optional[np.random.RandomState] = None,
    transfer_learning_history: Optional[List[HistoryLike]] = None,
    extra_dim: int = 0,
    norm_y: bool = True,
    **kwargs
) -> Surrogate:
    surrogate_type = surrogate_type.lower()
    
    # Alias mapping for backward compatibility
    # reacq_* -> rgpe_* (Ranking-weighted Ensemble with Acquisition)
    # re_* -> rgpe_* (Ranking-weighted Ensemble)
    # mceacq_* -> mfgpe_* (Multi-Fidelity Ensemble with Acquisition)
    # mce_* -> mfgpe_* (Multi-Fidelity Ensemble)
    # mfes_* -> mfgpe_* (Multi-Fidelity Ensemble, no transfer learning)
    if surrogate_type.startswith('mfes_'):
        # mfes_* means multi-fidelity without transfer learning
        # Map to mfgpe_* but set transfer_learning_history to None
        surrogate_type = surrogate_type.replace('mfes_', 'mfgpe_', 1)
        transfer_learning_history = None
    elif surrogate_type.startswith('reacq_'):
        surrogate_type = surrogate_type.replace('reacq_', 'rgpe_', 1)
    elif surrogate_type.startswith('re_'):
        surrogate_type = surrogate_type.replace('re_', 'rgpe_', 1)
    elif surrogate_type.startswith('mceacq_'):
        surrogate_type = surrogate_type.replace('mceacq_', 'mfgpe_', 1)
    elif surrogate_type.startswith('mce'):
        surrogate_type = surrogate_type.replace('mce', 'mfgpe', 1)
    
    if surrogate_type.startswith('rgpe'):
        parts = surrogate_type.split('_')
        inner_type = parts[1] if len(parts) > 1 else 'prf'
        return RGPE(
            config_space=config_space,
            source_hpo_data=transfer_learning_history,
            seed=rng.randint(2**31) if rng is not None else 42,
            surrogate_type=inner_type,
            num_src_hpo_trial=kwargs.get('num_src_hpo_trial', 50),
            only_source=kwargs.get('only_source', False),
            norm_y=norm_y
        )
    
    if surrogate_type.startswith('mfgpe'):
        parts = surrogate_type.split('_')
        inner_type = parts[1] if len(parts) > 1 else 'prf'
        return MFGPE(
            config_space=config_space,
            source_hpo_data=transfer_learning_history,
            seed=rng.randint(2**31) if rng is not None else 42,
            surrogate_type=inner_type,
            num_src_hpo_trial=kwargs.get('num_src_hpo_trial', 50),
            only_source=kwargs.get('only_source', False),
            norm_y=norm_y
        )
    
    # base surrogate
    types, bounds = get_types(config_space)
    if extra_dim > 0:
        types = np.hstack((types, np.zeros(extra_dim, dtype=np.uint)))
        bounds = np.vstack((bounds, np.array([[0, 1]] * extra_dim)))
    
    seed = kwargs.get('seed', rng.randint(2**31) if rng is not None else 42)
    
    # PRF
    if surrogate_type == 'prf':
        return ProbabilisticRandomForest(types=types, bounds=bounds, seed=seed)
    
    if surrogate_type.startswith('gp'):
        model_type = surrogate_type[:2] if len(surrogate_type) >= 2 else 'gp'
        return GaussianProcess(
            types=types,
            bounds=bounds,
            rng=rng,
            model_type=model_type
        )
    
    if surrogate_type in _SURROGATE_REGISTRY:
        surrogate_class = _SURROGATE_REGISTRY[surrogate_type]
        if surrogate_type == 'prf':
            return surrogate_class(types=types, bounds=bounds, seed=seed)
        elif surrogate_type.startswith('gp'):
            return surrogate_class(
                types=types,
                bounds=bounds,
                rng=rng,
                model_type=surrogate_type[:2]
            )
        else:
            return surrogate_class(**kwargs)
    
    available = ', '.join(sorted(_SURROGATE_REGISTRY.keys()))
    raise ValueError(
        f"Unknown surrogate type: '{surrogate_type}'. "
        f"Available: {available}, 'rgpe_*' (or 're_*', 'reacq_*'), "
        f"'mfgpe_*' (or 'mce_*', 'mceacq_*'), or 'gp*'"
    )


def list_available() -> List[str]:
    standard = sorted(_SURROGATE_REGISTRY.keys())
    return standard + [
        'rgpe_* (transfer learning, aliases: re_*, reacq_*)',
        'mfgpe_* (transfer learning, aliases: mce_*, mceacq_*)',
        'gp* (gaussian process variants)'
    ]


__all__ = [
    'Surrogate',
    'SingleFidelitySurrogate',
    'TransferLearningSurrogate',
    'HistoryLike',
    
    'ProbabilisticRandomForest',
    'GaussianProcess',
    
    'RGPE',
    'RGPE_avg',
    'MFGPE',
    
    'Normalizer',
    'calculate_preserving_order_num',
    
    'build_surrogate',
    'register',
    'list_available',
]
