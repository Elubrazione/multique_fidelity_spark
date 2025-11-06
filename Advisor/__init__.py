from typing import Dict, Optional
from .BO import BO
from .MFBO import MFBO

_ADVISOR_REGISTRY = {
    'bo': BO,
    'mfbo': MFBO,
}

class AdvisorConfig:
    def __init__(self, advisor_type: str, surrogate_type: str, acq_type: str):
        self.advisor_type = advisor_type
        self.surrogate_type = surrogate_type
        self.acq_type = acq_type
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'advisor_type': self.advisor_type,
            'surrogate_type': self.surrogate_type,
            'acq_type': self.acq_type
        }


def extract_base_surrogate(method_id: str) -> str:
    if method_id.endswith('_GP') or method_id == 'GP':
        return 'gp'
    elif method_id.endswith('_GPF') or method_id == 'GPF':
        return 'gpf'
    else:
        return 'prf'  # default: SMAC, MFES_SMAC, BOHB_SMAC


def get_surrogate_type(method_id: str, tl_strategy: str) -> str:
    base_type = extract_base_surrogate(method_id)
    if tl_strategy != 'none':
        surrogate_type = f'{tl_strategy}_{base_type}'
    else:
        surrogate_type = base_type
    if 'MFES' in method_id and tl_strategy == 'none':
        surrogate_type = f'mfes_{surrogate_type}'
    return surrogate_type


def get_acq_type(tl_strategy: str) -> str:
    if 'acq' in tl_strategy:
        return 'wrk_ei'
    else:
        return 'ei'


def get_advisor_config(
    method_id: str,
    tl_strategy: str = 'none'
) -> AdvisorConfig:
    if 'MFES' in method_id:
        advisor_type = 'mfbo'
    else:
        advisor_type = 'bo'
    
    surrogate_type = get_surrogate_type(method_id, tl_strategy)
    acq_type = get_acq_type(tl_strategy)
    return AdvisorConfig(advisor_type, surrogate_type, acq_type)


def get_advisor(advisor_type: str):
    if advisor_type not in _ADVISOR_REGISTRY:
        raise ValueError(f"Unknown advisor type: {advisor_type}. "
                        f"Available types: {list(_ADVISOR_REGISTRY.keys())}")
    
    return _ADVISOR_REGISTRY[advisor_type]


__all__ = [
    'BO',
    'MFBO',
    'AdvisorConfig',
    'get_advisor_config',
    'get_advisor',
    'get_surrogate_type',
    'get_acq_type',
    'extract_base_surrogate',
]