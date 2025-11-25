import numpy as np
from typing import Tuple, Optional, Callable


VERY_SMALL_NUMBER = 1e-10

def zero_mean_unit_var_normalization(
    y: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """Zero Mean Unit Variance Normalization
    
    Parameters
    ----------
    y : np.ndarray
        Target values vector
    mean : float, optional
        If provided, use this mean; otherwise, calculate from data
    std : float, optional
        If provided, use this standard deviation; otherwise, calculate from data
        
    Returns
    -------
    y_normalized : np.ndarray
        Normalized target values
    mean : float
        Used mean
    std : float
        Used standard deviation
    """
    if mean is None:
        mean = np.mean(y)
    if std is None:
        std = np.std(y)
        if std < VERY_SMALL_NUMBER:
            std = 1.0
    
    y_normalized = (y - mean) / std
    return y_normalized, mean, std


def zero_mean_unit_var_unnormalization(
    y_normalized: np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """Unnormalization (from zero mean unit variance to original scale)
    
    Parameters
    ----------
    y_normalized : np.ndarray
        Normalized target values
    mean : float
        Original mean
    std : float
        Original standard deviation
        
    Returns
    -------
    y : np.ndarray
        Original scale target values
    """
    return y_normalized * std + mean


class Normalizer:    
    def __init__(self, norm_y: bool = True):
        self.norm_y = norm_y
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
    
    def fit(self, y: np.ndarray) -> None:
        if not self.norm_y:
            return
        
        if np.all(y == y[0]):
            y = y.copy()
            y[0] += 1e-4
        
        _, mean, std = zero_mean_unit_var_normalization(y)
        self.mean = mean
        self.std = std
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.norm_y:
            return y
        
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before transform")
        
        if np.all(y == y[0]):
            y = y.copy()
            y[0] += 1e-4
        
        y_normalized, _, _ = zero_mean_unit_var_normalization(y, self.mean, self.std)
        return y_normalized
    
    def inverse_transform(self, y_normalized: np.ndarray) -> np.ndarray:
        if not self.norm_y:
            return y_normalized
        
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        return zero_mean_unit_var_unnormalization(y_normalized, self.mean, self.std)
    
    def transform_variance(self, var_normalized: np.ndarray) -> np.ndarray:
        if not self.norm_y:
            return var_normalized
        
        if self.std is None:
            raise ValueError("Normalizer must be fitted before transform_variance")
        
        return var_normalized * (self.std ** 2)


def calculate_preserving_order_num(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[int, int]:
    array_size = len(y_pred)
    assert len(y_true) == array_size
    
    total_pair_num = 0
    order_preserving_num = 0
    
    for idx in range(array_size):
        for inner_idx in range(idx + 1, array_size):
            if not ((y_true[idx] > y_true[inner_idx]) ^ (y_pred[idx] > y_pred[inner_idx])):
                order_preserving_num += 1
            total_pair_num += 1
    return order_preserving_num, total_pair_num


def get_types(config_space) -> Tuple[np.ndarray, np.ndarray]:
    types = []
    bounds = []
    
    for hp in config_space.get_hyperparameters():
        if hasattr(hp, 'choices') and hp.choices is not None:
            types.append(len(hp.choices))
        else:
            types.append(0)
        
        if hasattr(hp, 'lower') and hasattr(hp, 'upper'):
            bounds.append([hp.lower, hp.upper])
        else:
            # Categorical parameter or parameter without bounds
            bounds.append([0.0, 1.0])
    return np.array(types, dtype=np.uint), np.array(bounds)
