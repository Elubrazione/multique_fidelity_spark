import numpy as np

from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.surrogate.base.base_model import AbstractModel

from ..surrogate.base import BaseTLSurrogate
from ..utils import build_my_acq_func, calculate_ranking


class WeightedRank(AbstractAcquisitionFunction):
    def __init__(self, model: AbstractModel, acq_func='ei', temperature=0.1):
        super(WeightedRank, self).__init__(model)
        self.long_name = 'Weighted Rank'

        self.eta = None
        self.w = None

        self.inner_acq_type = acq_func
        self.acq_funcs = None

    def update(self, model, eta, num_data):

        assert isinstance(model, BaseTLSurrogate)

        self.w = np.array(model.get_weights())
        assert len(self.w) == model.K + 1
        self.acq_funcs = []
        for i in range(model.K):
            acq_func = build_my_acq_func(func_str=self.inner_acq_type, model=model.source_surrogates[i])
            acq_func.update(
                eta=model.source_hpo_data[i].get_incumbent_value(),
                num_data=len(model.source_hpo_data[i])
            )
            self.acq_funcs.append(acq_func)

        target_acq = build_my_acq_func(func_str=self.inner_acq_type, model=model.target_surrogate)
        target_acq.update(eta=eta, num_data=num_data)

        self.acq_funcs.append(target_acq)

        self.model = model

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        only_target = kwargs.get('only_target', True)
        if only_target:
            return self.acq_funcs[-1]._compute(X, **kwargs)

        all_scores = []
        for i in range(len(self.acq_funcs)):
            all_scores.append(self.acq_funcs[i]._compute(X, **kwargs).reshape(-1))

        all_rankings = np.array(calculate_ranking(all_scores))

        # TODO: Combine with weights
        final_ranking = np.sum(all_rankings * self.w[:, np.newaxis], axis=0)

        final_acq = np.max(final_ranking) - final_ranking

        return final_acq.reshape(-1, 1)
