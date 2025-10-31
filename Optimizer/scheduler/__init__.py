from .base import BaseScheduler, FullFidelityScheduler
from .fidelity import FixedFidelityScheduler, BOHBFidelityScheduler, MFSEFidelityScheduler

schedulers = {
    'fixed': FixedFidelityScheduler,
    'bohb': BOHBFidelityScheduler,
    'full': FullFidelityScheduler,
    'mfes': MFSEFidelityScheduler
}

__all__ = [
    'BaseScheduler',
    'FullFidelityScheduler',
    'FixedFidelityScheduler',
    'BOHBFidelityScheduler',
    'MFSEFidelityScheduler'
]