import numpy as np
from scipy.stats import norm

from .base import SingleObjectiveAcquisition, SurrogateModel


class ExpectedImprovement(SingleObjectiveAcquisition):
    r"""Expected Improvement (EI) Acquisition Function
    
    EI quantifies the expected improvement over the current best observed value (incumbent).
    It balances exploration (high uncertainty) and exploitation (high predicted value).
    
    For a point :math:`x`, EI is defined as:
    
    .. math::
        \text{EI}(x) = \mathbb{E}[\max(f_{\min} - Y(x), 0)]
    
    where :math:`f_{\min}` is the incumbent and :math:`Y(x)` is the random variable 
    representing the objective at :math:`x` according to the surrogate model.
    
    For Gaussian processes, this has a closed-form solution:
    
    .. math::
        \text{EI}(x) = (f_{\min} - \mu(x)) \cdot \Phi(Z) + \sigma(x) \cdot \phi(Z)
    
    where:
    
    .. math::
        Z = \frac{f_{\min} - \mu(x)}{\sigma(x)}
    
    - :math:`\Phi` is the standard normal CDF
    - :math:`\phi` is the standard normal PDF
    - :math:`\mu(x)` and :math:`\sigma(x)` are the GP mean and std at :math:`x`
    
    Attributes
    ----------
    par : float
        Exploration-exploitation trade-off parameter (typically 0.0)
        Positive values encourage more exploration
    """
    
    def __init__(self, model: SurrogateModel, par: float = 0.0, **kwargs):
        super().__init__(model, **kwargs)
        self.long_name = 'Expected Improvement'
        self.par = par
    
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        
        mean, var = self.model.predict(X)
        std = np.sqrt(var)
        
        if self.eta is None:
            return np.zeros((X.shape[0], 1))
        
        z = (self.eta - mean - self.par) / (std + 1e-9)
        ei = (self.eta - mean - self.par) * norm.cdf(z) + std * norm.pdf(z)
        
        ei[std < 1e-9] = 0.0
        return ei.reshape(-1, 1)


class EI(ExpectedImprovement):
    pass

