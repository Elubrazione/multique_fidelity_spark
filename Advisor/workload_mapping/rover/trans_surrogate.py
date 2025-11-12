# License: MIT

import numpy as np
from openbox import logger
from openbox.surrogate.tlbo.base import BaseTLSurrogate

_scale_method = 'scale'

# 该迁移学习代理模型有固定的权重，即源任务平分权重，目标任务权重为0
class RGPE_avg(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'rgpe_avg'
        self.only_source = only_source
        self.build_source_surrogates(normalize=_scale_method)

        self.scale = True
        # self.num_sample = 100
        self.num_sample = 50

        if source_hpo_data is not None:
            # Weights for base surrogates and the target surrogate.
            self.w = [1. / self.K] * self.K + [0.]
            # Preventing weight dilution.
            self.ignored_flag = [False] * (self.K + 1)
        self.hist_ws = list()
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize_y=True)
        if self.source_hpo_data is None:
            return

        # Set weight dilution flag.
        # ranking_loss_caches = np.array(ranking_loss_caches)
        # threshold = sorted(ranking_loss_caches[:, -1])[int(self.num_sample * 0.95)]
        # for id in range(self.K):
        #     median = sorted(ranking_loss_caches[:, id])[int(self.num_sample * 0.5)]
        #     self.ignored_flag[id] = median > threshold
        # self.ignored_flag[-1] = self.only_source
        # if any(self.ignored_flag):
        #     logger.info(f'weight ignore flag: {self.ignored_flag}')

        w = self.w.copy()
        for id in range(self.K):
            if self.ignored_flag[id]:
                w[id] = 0.
        sum_w = np.sum(w)
        if sum_w == 0:
            w = [1. / self.K] * self.K + [0.] if self.only_source else [0.] * self.K + [1.]
        else:
            w = (np.array(w) / sum_w).tolist()
        weight_str = ','.join([('%.2f' % item) for item in w])
        # logger.info('In iter-%d' % self.iteration_id)
        self.target_weight.append(w[-1])
        logger.info(f'weight: {weight_str}')
        self.hist_ws.append(w)
        self.iteration_id += 1

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        if self.source_hpo_data is None:
            return mu, var

        # Target surrogate predictions with weight.
        mu *= self.w[-1]
        var *= (self.w[-1] * self.w[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(0, self.K):
            if not self.ignored_flag[i]:
                mu_t, var_t = self.source_surrogates[i].predict(X)
                mu += self.w[i] * mu_t
                var += self.w[i] * self.w[i] * var_t
        return mu, var

    def get_weights(self):
        return self.w
