import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Protocol


class SurrogateModel(Protocol):
    """Protocol defining the interface for surrogate models
    
    This protocol allows any surrogate model implementation to be used
    with our acquisition functions, as long as it provides the predict method.
    """
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance for input points
        
        Parameters
        ----------
        X : np.ndarray of shape (N, D)
            Input points to predict
            
        Returns
        -------
        mean : np.ndarray of shape (N,) or (N, 1)
            Predicted mean values
        variance : np.ndarray of shape (N,) or (N, 1)
            Predicted variance values
        """
        ...


class AcquisitionFunction(ABC):
    def __init__(self, model: SurrogateModel, **kwargs):
        self.model = model
        self.long_name = self.__class__.__name__
    
    @abstractmethod
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    def __call__(self, X: np.ndarray, convert: bool = True, **kwargs) -> np.ndarray:
        return self._compute(X, **kwargs)
    
    def update(self, **kwargs) -> None:
        if 'model' in kwargs:
            self.model = kwargs['model']


class SingleObjectiveAcquisition(AcquisitionFunction):
    def __init__(self, model: SurrogateModel, **kwargs):
        super().__init__(model, **kwargs)
        self.eta = None
    
    def update(self, **kwargs) -> None:
        super().update(**kwargs)
        if 'eta' in kwargs:
            self.eta = kwargs['eta']


class TransferLearningAcquisition(AcquisitionFunction):    
    def __init__(self, model: SurrogateModel, **kwargs):
        super().__init__(model, **kwargs)
        self.source_acq_funcs = []
        self.target_acq_func = None
        self.weights = None
    
    @abstractmethod
    def _combine_acquisitions(self, source_values: np.ndarray, 
                            target_values: np.ndarray) -> np.ndarray:
        pass

