from .base import BaseScheduler, FullFidelityScheduler

schedulers = {
    'full': FullFidelityScheduler,
}

__all__ = [
    'FullFidelityScheduler',
]