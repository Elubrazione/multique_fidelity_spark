# License: MIT
import numpy as np
from openbox import logger

from .base import BaseTLSurrogate

_scale_method = 'scale'


# cal_w_strategy = mc(基于采样) / mean(只看均值)
class RGPE(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, seed=0,
                 surrogate_type='rf', num_src_hpo_trial=50, only_source=False, norm_y=True):
        super().__init__(config_space, source_hpo_data, seed,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial,
                         norm_y=norm_y)
        self.method_id = 'rgpe'
        self.cal_w_strategy = 'mc'
        self.only_source = only_source
        self.build_source_surrogates(normalize=_scale_method)

        self.scale = True
        # self.num_sample = 100
        self.num_sample = 50

        self.hist_ws = list()
        self.iteration_id = 0

    def train(self, X: np.ndarray, y: np.array):
        # Build the target surrogate.
        self.target_surrogate = self.build_single_surrogate(X, y, normalize_y=self.norm_y)
        if self.source_hpo_data is None:
            return

        # Train the target surrogate and update the weight w.
        mu_list, var_list = list(), list()
        for id in range(self.K):
            mu, var = self.source_surrogates[id].predict(X)
            mu_list.append(mu)
            var_list.append(var)

        # Pretrain the leave-one-out surrogates.
        instance_num = len(y)

        if instance_num >= self.k_fold_num:
            tar_mu, tar_var = self.predict_target_surrogate_cv(X, y)
            mu_list.append(tar_mu)
            var_list.append(tar_var)

            self.w = self.get_w_ranking_pairs(mu_list, var_list, y)

        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in self.w])
        # logger.info('In iter-%d' % self.iteration_id)
        self.target_weight.append(w[-1])
        logger.info(f'weight: {weight_str}')
        w_str_list = []
        for i in range(self.K):
            w_str = "%s: sim%.4f" % (self.source_hpo_data[i].task_id, w[i])
            w_str_list.append(w_str)
        w_str_list.append("target: %.4f" % w[-1])
        self.hist_ws.append(w_str_list)
        self.iteration_id += 1

    def predict(self, X: np.array):
        mu, var = self.target_surrogate.predict(X)
        if self.source_hpo_data is None:
            return mu, var

        # Target surrogate predictions with weight.
        mu *= self.w[-1]
        var *= (self.w[-1] * self.w[-1])

        # Base surrogate predictions with corresponding weights.
        for i in range(self.K):
            if not self.ignored_flag[i]:
                mu_t, var_t = self.source_surrogates[i].predict(X)
                mu += self.w[i] * mu_t
                var += self.w[i] * self.w[i] * var_t
        return mu, var

    def get_w_ranking_pairs(self, mu_list, var_list, y_true):
        instance_num = len(y_true)
        argmin_list = [0] * (self.K + 1)  # 记录每个源任务，相关性最高的次数
        ranking_loss_caches = list()  # num_sample, num_src_task+1 对每个源任务，预测当前所有观测的结果，采样num_sample次，每次计算的相关性
        for _ in range(self.num_sample):
            ranking_loss_list = list()
            for id in range(self.K):
                sampled_y = np.random.normal(mu_list[id], var_list[id])
                preorder_num, pair_num = self.calculate_preserving_order_num(sampled_y, y_true)
                rank_loss = pair_num - preorder_num
                ranking_loss_list.append(rank_loss)

            tar_mu, tar_var = mu_list[-1], var_list[-1]
            # Compute ranking loss for target surrogate.
            if instance_num >= self.k_fold_num:
                sampled_y = np.random.normal(tar_mu, tar_var)
                preorder_num, pair_num = self.calculate_preserving_order_num(sampled_y, y_true)
                rank_loss = pair_num - preorder_num
            else:
                rank_loss = instance_num * instance_num
            ranking_loss_list.append(rank_loss)
            ranking_loss_caches.append(ranking_loss_list)

            argmin_task = np.argmin(ranking_loss_list)
            argmin_list[argmin_task] += 1

        # Update the weights.
        w = np.array(argmin_list) / self.num_sample

        # Set weight dilution flag.
        ranking_loss_caches = np.array(ranking_loss_caches)
        threshold = sorted(ranking_loss_caches[:, -1])[int(self.num_sample * 0.95)]
        for id in range(self.K):
            median = sorted(ranking_loss_caches[:, id])[int(self.num_sample * 0.5)]
            self.ignored_flag[id] = median > threshold
        self.ignored_flag[-1] = self.only_source
        if any(self.ignored_flag):
            logger.info(f'weight ignore flag: {self.ignored_flag}')

        for id in range(self.K):
            if self.ignored_flag[id]:
                w[id] = 0.
        sum_w = np.sum(w)
        if sum_w == 0:
            w = [1. / self.K] * self.K + [0.] if self.only_source else [0.] * self.K + [1.]
        else:
            w = self.modify_w(np.array(w)/sum_w).tolist()

        return w

