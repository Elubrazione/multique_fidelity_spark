from .base import BaseScheduler, FullFidelityScheduler
from .fidelity import FixedFidelityScheduler, BOHBFidelityScheduler, MFESFidelityScheduler

schedulers = {
    'fixed': FixedFidelityScheduler,
    'bohb': BOHBFidelityScheduler,
    'full': FullFidelityScheduler,
    'mfes': MFESFidelityScheduler
}

__all__ = [
    'BaseScheduler',
    'FullFidelityScheduler',
    'FixedFidelityScheduler',
    'BOHBFidelityScheduler',
    'MFESFidelityScheduler'
]