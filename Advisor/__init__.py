from .Rover import Rover
from .Tuneful import Tuneful
from .Toptune import Toptune

advisors = {
    'rover': Rover,
    'tuneful': Tuneful,
    'toptune': Toptune,
}

__all__ = [
    'Rover',
    'Tuneful',
    'Toptune',
]