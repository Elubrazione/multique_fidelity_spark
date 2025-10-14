# License: MIT

import abc
import time
import typing
import numpy as np
from typing import List
from sklearn.model_selection import KFold

from openbox import logger, History
from openbox.utils.util_funcs import get_types
from openbox.core.base import build_surrogate
from openbox.utils.constants import VERY_SMALL_NUMBER
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.transform import (
    zero_mean_unit_var_normalization, zero_mean_unit_var_unnormalization,
)

_scale_method = 'scale'

"""
norm_y 控制是否对 y 进行归一化
如果是在surrogate层面集成，那要归一化，在predict的时候会对预测结果重新放缩
如果实在acquisition function层面集成，那不需要归一化，因为acquisition function需要模型预测出原始的y值

"""
class BaseTLSurrogate(object):
    def __init__(self, config_space: ConfigurationSpace,
                 source_hpo_data: List,
                 seed: int = 0,
                 history_dataset_features: List = None,
                 num_src_hpo_trial: int = 50,
                 surrogate_type: str = 'rf',
                 k_fold_num: int = 5,
                 norm_y: bool = True):
        self.method_id = None
        self.config_space = config_space
        self.random_seed = seed
        self.num_src_hpo_trial = num_src_hpo_trial
        self.source_hpo_data = source_hpo_data
        self.source_surrogates = None
        self.target_surrogate = None
        self.history_dataset_features = history_dataset_features
        # The number of source problems.\
        self.K = 0
        if source_hpo_data is not None:
            self.K = len(source_hpo_data)
            if history_dataset_features is not None:
                assert len(history_dataset_features) == self.K
            # Preventing weight dilution.
            self.ignored_flag = [False] * (self.K + 1)
            
        if self.K == 0:
            self.w = [1.]
        else:
            self.w = [1. / self.K] * self.K + [0.]  # 高精度初始化权重为0

        self.surrogate_type = surrogate_type
        self.k_fold_num = k_fold_num

        self.types, self.bounds = get_types(config_space)

        self.var_threshold = VERY_SMALL_NUMBER

        self.eta_list = list()

        # meta features.
        self.meta_feature_scaler = None
        self.meta_feature_imputer = None

        self.y_normalize_mean = None
        self.y_normalize_std = None

        self.target_weight = list()
        self.norm_y = norm_y

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        pass

    def predict_marginalized_over_instances(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != len(self.bounds):
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (len(self.bounds), X.shape[1]))

        mean, var = self.predict(X)
        assert var is not None  # please mypy

        var[var < self.var_threshold] = self.var_threshold
        var[np.isnan(var)] = self.var_threshold

        if self.y_normalize_mean is not None and self.y_normalize_std is not None:
            mean = zero_mean_unit_var_unnormalization(mean, self.y_normalize_mean, self.y_normalize_std)
            var = var * self.y_normalize_std ** 2

        return mean, var

    def update_mf_trials(self, mf_hpo_data: List[History]):
        self.K = 0
        if mf_hpo_data is not None:
            self.K = len(mf_hpo_data)  # K is the number of low-fidelity groups
            # Preventing weight dilution.
            self.ignored_flag = [False] * (self.K + 1)
        if self.K == 0:
            self.w = [1.]
        else:
            self.w = [1. / self.K] * self.K + [0.]
        self.source_hpo_data = mf_hpo_data
        # Refit the base surrogates.
        self.build_source_surrogates(normalize=_scale_method)

    def build_source_surrogates(self, normalize='scale'):
        if self.source_hpo_data is None:
            logger.warning('No history BO data provided, resort to naive BO optimizer without TL.')
            return

        assert isinstance(self.source_hpo_data, list)

        logger.info('Start to train base surrogates.')
        start_time = time.time()
        self.source_surrogates = list()
        for task_history in self.source_hpo_data:
            assert isinstance(task_history, History)
            model = build_surrogate(self.surrogate_type, self.config_space,
                                    np.random.RandomState(self.random_seed))

            X = task_history.get_config_array(transform=normalize)[:self.num_src_hpo_trial]
            y = task_history.get_objectives(transform='infeasible')[:self.num_src_hpo_trial]
            y = y.reshape(-1)  # single objective

            if np.all(y == y[0]):
                y[0] += 1e-4
            if self.norm_y:
                y, _, _ = zero_mean_unit_var_normalization(y)

            self.eta_list.append(np.min(y))
            model.train(X, y)
            self.source_surrogates.append(model)
        logger.info('Building base surrogates took %.3fs.' % (time.time() - start_time))

    def build_single_surrogate(self, X: np.ndarray, y: np.array, normalize_y=False):
        model = build_surrogate(self.surrogate_type, self.config_space, np.random.RandomState(self.random_seed))

        if np.all(y == y[0]):
            y[0] += 1e-4
        if normalize_y:
            y, mean, std = zero_mean_unit_var_normalization(y)
            self.y_normalize_mean = mean
            self.y_normalize_std = std

        model.train(X, y)
        return model

    def predict_target_surrogate_cv(self, X, y, k_fold_num=5):
        _mu, _var = list(), list()

        # Conduct K-fold cross validation.
        kf = KFold(n_splits=k_fold_num)
        idxs = list()
        for train_idx, val_idx in kf.split(X):
            idxs.extend(list(val_idx))
            X_train, X_val, y_train, y_val = X[train_idx, :], X[val_idx, :], y[train_idx], y[val_idx]
            model = self.build_single_surrogate(X_train, y_train, normalize_y=False)
            mu, var = model.predict(X_val)
            mu, var = mu.flatten(), var.flatten()
            _mu.extend(list(mu))
            _var.extend(list(var))
        assert (np.array(idxs) == np.arange(X.shape[0])).all()
        return np.asarray(_mu), np.asarray(_var)

    def get_w_ranking_pairs(self, mu_list, var_list, y_true):

        raise NotImplementedError

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if not ((y_true[idx] > y_true[inner_idx]) ^ (y_pred[idx] > y_pred[inner_idx])):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def get_weights(self):
        return self.w

    def modify_w(self, new_w):
        if new_w[self.K] < self.w[self.K]:
            new_w[self.K] = self.w[self.K]
            new_w[:self.K] = new_w[:self.K] / np.sum(new_w[:self.K]) * (1 - new_w[self.K])

        return new_w