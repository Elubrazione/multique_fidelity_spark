from .BO import BO
from .MFBO import MFBO

advisors = {
    'bo': BO,
    'mfbo': MFBO,
}

__all__ = [
    'BO',
    'MFBO',
    'advisors'
]