import numpy as np
import pandas as pd
from .base import TransferLearningAcquisition, SurrogateModel
from ..surrogate.base import BaseTLSurrogate


class WeightedRank(TransferLearningAcquisition):
    """Weighted Rank Acquisition Function for Transfer Learning
    
    Combines acquisition function values from multiple source tasks and the target task
    using learned weights and ranking-based aggregation.
    
    This acquisition function:
    1. Computes acquisition values from K source tasks and 1 target task
    2. Converts values to rankings for each task
    3. Combines rankings using weighted sum with learned task weights
    
    Attributes
    ----------
    inner_acq_type : str
        Type of inner acquisition function to use (e.g., 'ei', 'ucb')
    temperature : float
        Temperature parameter for weight computation (currently unused)
    """
    
    def __init__(self, model: SurrogateModel, acq_func='ei', temperature=0.1):
        super(WeightedRank, self).__init__(model)
        self.long_name = 'Weighted Rank'

        self.eta = None
        self.temperature = temperature

        self.inner_acq_type = acq_func
        self.acq_funcs = None

    def update(self, model, eta, num_data):
        assert isinstance(model, BaseTLSurrogate)

        self.weights = np.array(model.get_weights())
        assert len(self.weights) == model.K + 1
        
        from . import get_acq
        self.source_acq_funcs = []
        for i in range(model.K):
            acq_func = get_acq(acq_type=self.inner_acq_type, model=model.source_surrogates[i])
            acq_func.update(
                eta=model.source_hpo_data[i].get_incumbent_value(),
                num_data=len(model.source_hpo_data[i])
            )
            self.source_acq_funcs.append(acq_func)

        self.target_acq_func = get_acq(acq_type=self.inner_acq_type, model=model.target_surrogate)
        self.target_acq_func.update(eta=eta, num_data=num_data)

        self.acq_funcs = self.source_acq_funcs + [self.target_acq_func]
        
        self.model = model
        self.eta = eta

    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        only_target = kwargs.get('only_target', True)
        if only_target:
            return self.acq_funcs[-1]._compute(X, **kwargs)

        all_scores = []
        for i in range(len(self.acq_funcs)):
            scores = self.acq_funcs[i]._compute(X, **kwargs).reshape(-1)
            all_scores.append(scores)
        all_rankings = np.array(calculate_ranking(all_scores))
        final_ranking = self._combine_acquisitions(all_rankings[: -1], all_rankings[-1])
        final_acq = np.max(final_ranking) - final_ranking
        return final_acq.reshape(-1, 1)
    
    def _combine_acquisitions(self, source_rankings: np.ndarray, 
                            target_ranking: np.ndarray) -> np.ndarray:
        all_rankings = np.vstack([source_rankings, target_ranking.reshape(1, -1)])
        final_ranking = np.sum(all_rankings * self.weights[:, np.newaxis], axis=0)
        return final_ranking

def calculate_ranking(score_list, ascending=False):
    rank_list = list()
    for i in range(len(score_list)):
        value_list = pd.Series(list(score_list[i]))
        rank_array = np.array(value_list.rank(ascending=ascending))
        rank_list.append(rank_array)
    return rank_list