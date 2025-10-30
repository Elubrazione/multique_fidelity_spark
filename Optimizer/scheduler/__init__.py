from .base import BaseScheduler, FullFidelityScheduler
from .fidelity import FixedFidelityScheduler, BOHBFidelityScheduler

schedulers = {
    'fixed': FixedFidelityScheduler,
    'bohb': BOHBFidelityScheduler,
    'full': FullFidelityScheduler,
    'mfes': BOHBFidelityScheduler
}

__all__ = [
    'BaseScheduler',
    'FullFidelityScheduler',
    'FixedFidelityScheduler',
    'BOHBFidelityScheduler'
]