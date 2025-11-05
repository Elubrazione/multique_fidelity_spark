from .BO import BO
from .MFBO import MFBO
from .LOCAT import LOCATAdvisor

advisors = {
    'bo': BO,
    'mfbo': MFBO,
    'locat': LOCATAdvisor,
}

__all__ = [
    'BO',
    'MFBO',
    'LOCATAdvisor',
    'advisors'
]