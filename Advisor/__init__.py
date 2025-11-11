from .Tuneful import Tuneful
from .Rover import Rover

advisors = {
    'tuneful': Tuneful,
    'rover': Rover,
}

__all__ = [
    'Tuneful',
    'Rover'
]