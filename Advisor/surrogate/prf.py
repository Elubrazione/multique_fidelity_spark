import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
import threading
from joblib import Parallel, delayed

from .base import SingleFidelitySurrogate


class ProbabilisticRandomForest(SingleFidelitySurrogate):
    """Probabilistic Random Forest surrogate model
    
    Based on scikit-learn's RandomForestRegressor,
    providing mean and variance predictions by aggregating predictions from all trees.
    
    Parameters
    ----------
    types : np.ndarray
        Parameter type array (specifies categorical vs continuous)
    bounds : np.ndarray
        Parameter boundary array
    seed : int, default=42
        Random seed
    num_trees : int, default=10
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    min_samples_split : int, default=2
        Minimum samples required to split a node
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node
    max_features : str or int or float, default='sqrt'
        Number of features to consider for best split
    n_jobs : int, optional
        Number of parallel jobs
    """
    
    def __init__(
        self,
        types: np.ndarray,
        bounds: np.ndarray,
        seed: int = 42,
        num_trees: int = 10,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        n_jobs: Optional[int] = None,
        **kwargs
    ):
        self.types = types
        self.bounds = bounds
        self.seed = seed
        self.num_trees = num_trees
        self.n_jobs = n_jobs
        
        self.rf = RandomForestRegressor(
            n_estimators=num_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=seed,
            n_jobs=n_jobs,
            **kwargs
        )
        
        self._is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the random forest model
        
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
        
        self.rf.fit(X, y)
        self._is_trained = True
    
    def predict(self, X: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance
        
        Mean is the average prediction across all trees.
        Variance is the variance of predictions across all trees.
        
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
        
        check_is_fitted(self.rf)
        X = self.rf._validate_X_predict(X)
        
        all_predictions = []
        lock = threading.Lock()        
        Parallel(n_jobs=self.n_jobs, require="sharedmem")(
            delayed(self._collect_prediction)(tree.predict, X, all_predictions, lock)
            for tree in self.rf.estimators_
        )
        all_predictions = np.asarray(all_predictions, dtype=np.float64)        
        mean = np.mean(all_predictions, axis=0)
        var = np.var(all_predictions, axis=0)
        var = np.maximum(var, 1e-10)
        return mean.reshape(-1, 1), var.reshape(-1, 1)

    @staticmethod
    def _collect_prediction(predict, X, out, lock):
        prediction = predict(X, check_input=False)
        with lock:
            out.append(prediction)
