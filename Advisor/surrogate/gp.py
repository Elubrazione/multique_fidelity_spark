import numpy as np
from typing import Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel

from .base import SingleFidelitySurrogate


class GaussianProcess(SingleFidelitySurrogate):
    """Gaussian Process surrogate model
    
    Based on sklearn's GaussianProcessRegressor.
    
    Parameters
    ----------
    types : np.ndarray
        Parameter type array
    bounds : np.ndarray
        Parameter boundary array
    rng : np.random.RandomState, optional
        Random number generator
    alpha : float, default=1e-10
        Value added to the diagonal of the kernel matrix during fitting
    kernel : str, default='matern'
        Kernel type ('matern' or 'rbf')
    """
    
    def __init__(
        self,
        types: np.ndarray,
        bounds: np.ndarray,
        rng: np.random.RandomState,
        alpha: float = 1e-10,
        kernel: str = 'matern',
        **kwargs
    ):
        self.types = types
        self.bounds = bounds
        self.rng = rng
        
        random_state = rng.randint(2**31)
        
        if kernel == 'matern':
            base_kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
        elif kernel == 'rbf':
            base_kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")
        
        self.gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel,
            alpha=alpha,
            normalize_y=False,  # normalize outside the model
            random_state=random_state,
            n_restarts_optimizer=10,
            **kwargs
        )
        
        self._is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the GP model
        
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix [n_samples, n_features]
        y : np.ndarray
            Target values [n_samples]
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.flatten()
        
        self.gp.fit(X, y)
        self._is_trained = True
    
    def predict(self, X: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance
        
        Parameters
        ----------
        X : np.ndarray
            Input feature matrix [n_samples, n_features]
            
        Returns
        -------
        mean : np.ndarray
            Predicted mean [n_samples, 1]
        var : np.ndarray
            Predicted variance [n_samples, 1]
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        mean, std = self.gp.predict(X, return_std=True)
        var = std ** 2
        var = np.maximum(var, 1e-10)
        return mean.reshape(-1, 1), var.reshape(-1, 1)
