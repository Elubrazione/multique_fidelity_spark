import numpy as np

from .base import SingleObjectiveAcquisition, SurrogateModel


class UpperConfidenceBound(SingleObjectiveAcquisition):
    r"""Upper Confidence Bound (UCB) Acquisition Function
    
    UCB trades off exploitation (high mean prediction) with exploration 
    (high uncertainty) using a parameter :math:`\kappa`.
    
    For a point :math:`x`, UCB is defined as:
    
    .. math::
        \text{UCB}(x) = \mu(x) + \kappa \cdot \sigma(x)
    
    where:
    
    - :math:`\mu(x)` is the predicted mean
    - :math:`\sigma(x)` is the predicted standard deviation
    - :math:`\kappa` is the exploration parameter
    
    For minimization problems, we use:
    
    .. math::
        \text{UCB}(x) = -\mu(x) + \kappa \cdot \sigma(x)
    
    This gives higher acquisition where mean is low or uncertainty is high.
    
    Attributes
    ----------
    kappa : float
        Exploration-exploitation trade-off parameter
        Higher values encourage more exploration
    """
    
    def __init__(self, model: SurrogateModel, kappa: float = 2.0, **kwargs):
        super().__init__(model, **kwargs)
        self.long_name = 'Upper Confidence Bound'
        self.kappa = kappa
    
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        
        mean, var = self.model.predict(X)
        std = np.sqrt(var)
        
        # For minimization: we want to minimize, so we use -mean + kappa * std
        # This gives higher acquisition where mean is low or uncertainty is high
        ucb = -mean + self.kappa * std
        
        return ucb.reshape(-1, 1)
    
    def update(self, **kwargs) -> None:
        super().update(**kwargs)
        if 'kappa' in kwargs:
            self.kappa = kwargs['kappa']


class UCB(UpperConfidenceBound):
    pass

