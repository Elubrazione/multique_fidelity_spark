# License: MIT
import numpy as np
from openbox import logger

from .base import BaseTLSurrogate

_scale_method = 'scale'

class MFGPE(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed=0,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False, norm_y=True):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial,
                         norm_y=norm_y)
        self.method_id = 'mfgpe'
        self.only_source = only_source
        self.build_source_surrogates(normalize=_scale_method)

        self.scale = True
        self.hist_ws = list()
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array, **kwargs):
        self.target_surrogate = self.build_single_surrogate(X, y, normalize_y=self.norm_y)
        if self.source_hpo_data is None:
            return

        sample_num = y.shape[0]

        if self.source_hpo_data is None:
            raise ValueError('Source HPO data is None!')

        # Get the predictions of low-fidelity surrogates
        mu_list, var_list = list(), list()
        for id in range(self.K):
            mu, var = self.source_surrogates[id].predict(X)
            mu_list.append(mu.flatten())
            var_list.append(var.flatten())

        # Evaluate the generalization of the high-fidelity surrogate via CV
        if sample_num >= self.k_fold_num:
            _mu, _var = self.predict_target_surrogate_cv(X, y, k_fold_num=self.k_fold_num)
            mu_list.append(_mu)
            var_list.append(_var)
            self.w = self.get_w_ranking_pairs(mu_list, var_list, y)

        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        # logger.info('In iter-%d' % self.iteration_id)
        self.target_weight.append(w[-1])
        logger.info(f'weight: {weight_str}')
        self.iteration_id += 1

        w_str_list = []
        for i in range(self.K):
            w_str = "%s: sim%.4f" % (self.source_hpo_data[i].task_id, w[i])
            w_str_list.append(w_str)
        w_str_list.append("target: %.4f" % w[-1])
        self.hist_ws.append(w_str_list)

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        if self.source_hpo_data is None:
            return mu, var

        # Target surrogate predictions with weight.
        mu *= self.w[-1]
        var *= (self.w[-1] * self.w[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(self.K):
            mu_t, var_t = self.source_surrogates[i].predict(X)
            mu += self.w[i] * mu_t
            var += self.w[i] * self.w[i] * var_t
        return mu, var

    def get_w_ranking_pairs(self, mu_list, var_list, y_true):
        preserving_order_p, preserving_order_nums = list(), list()
        for i in range(self.K + 1):
            y_pred = mu_list[i]
            preorder_num, pair_num = self.calculate_preserving_order_num(y_pred, y_true)
            preserving_order_p.append(preorder_num / pair_num)
            preserving_order_nums.append(preorder_num)
        n_power = 3
        trans_order_weight = np.array(preserving_order_p)
        p_power = np.power(trans_order_weight, n_power)

        new_w = self.modify_w(p_power / np.sum(p_power))

        return new_w.tolist()

