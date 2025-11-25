import numpy as np
import pandas as pd
from typing import List
from .base import (
    TransferLearningAcquisition,
    AcquisitionFunction,
    SurrogateModel,
    AcquisitionContext
)


class WeightedRank(TransferLearningAcquisition):
    def __init__(self, model: SurrogateModel, acq_func='ei', temperature=0.1):
        super(WeightedRank, self).__init__(model)
        self.long_name = 'Weighted Rank'

        self.eta = None
        self.temperature = temperature

        self.inner_acq_type = acq_func
        self.acq_funcs: List[AcquisitionFunction] = None

    def update(self, context=None, **kwargs):
        if context is not None:
            self._update_from_context(context, **kwargs)
            return
        
    def _update_from_context(self, context: AcquisitionContext, **kwargs) -> None:
        if not context.is_multi_task():
            raise ValueError("WeightedRank requires multi-task context")
        
        self.weights = context.weights
        self.eta = context.get_target_task().get_incumbent_value()
        self.model = context.get_main_surrogate()
        
        from . import get_acq
        self.acq_funcs = []
        for task in context.tasks:
            acq_func = get_acq(
                acq_type=self.inner_acq_type,
                model=task.surrogate
            )
            acq_func.update(context=AcquisitionContext(tasks=[task], weights=None))
            self.acq_funcs.append(acq_func)
    
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