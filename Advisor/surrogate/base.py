from typing import Optional, List, Tuple
import numpy as np
import abc
from ConfigSpace import ConfigurationSpace
from sklearn.model_selection import KFold
from openbox import logger

from .weight import WeightCalculator, MFGPEWeightCalculator
from .utils import Normalizer
from ..acq_function import AcquisitionContext, TaskContext, HistoryLike

class Surrogate(abc.ABC):
    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        pass
    
    @abc.abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    def get_acquisition_context(self, history: HistoryLike) -> AcquisitionContext:
        return AcquisitionContext(
            tasks=[
                TaskContext(
                    surrogate=self,
                    history=history,
                    eta=history.get_incumbent_value(),
                    num_data=len(history)
                )
            ],
            weights=None
        )


class SingleFidelitySurrogate(Surrogate):
    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_type: str = 'prf',
        rng: np.random.RandomState = np.random.RandomState(42)
    ):
        self.config_space = config_space
        self.surrogate_type = surrogate_type
        self.rng = rng


class TransferLearningSurrogate(SingleFidelitySurrogate):
    def __init__(
        self,
        config_space: ConfigurationSpace,
        surrogate_type: str = 'prf',
        rng: np.random.RandomState = np.random.RandomState(42),
        num_src_trials: int = 50,
        weight_calculator: Optional[WeightCalculator] = None,
        source_data: Optional[List[HistoryLike]] = None,
        norm_y: bool = True,
        k_fold_num: int = 5,
        only_source: bool = False,
        **kwargs
    ):
        super().__init__(config_space, surrogate_type, rng)
        self.source_data = source_data or []
        self.num_src_trials = num_src_trials
        self.source_surrogates: List[Surrogate] = []
        self.target_surrogate: Optional[SingleFidelitySurrogate] = None

        self.weight_calculator = weight_calculator or MFGPEWeightCalculator()
        
        self.normalizer = Normalizer(norm_y=norm_y)

        self.k_fold_num = k_fold_num
        self.w: np.ndarray = np.array([1.0])
        self.current_target_weight = 0.0
        self.ignored_flags: List[bool] = []

        self.hist_ws = []
        self.target_weight = []
        self.iteration_id = 0
        self.only_source = only_source
        
        if self._get_num_tasks() > 0:
            self._build_source_surrogates()
    
    def update_mf_trials(self, history_list: List[HistoryLike]):
        self._clear_source_tasks()
        self._add_source_tasks(history_list)
        self._build_source_surrogates()

    def _build_source_surrogates(self):
        for task_history in self._get_all_tasks():
            X = task_history.get_config_array(transform='scale')[:self.num_src_trials]
            y = task_history.get_objectives(transform='infeasible')[:self.num_src_trials]
            y = y.reshape(-1)
            surrogate = self._build_single_surrogate(X, y)
            self.source_surrogates.append(surrogate)
    
    def _build_single_surrogate(self, X: np.ndarray, y: np.ndarray) -> Surrogate:
        from . import build_surrogate
        model = build_surrogate(
            surrogate_type=self.surrogate_type,
            config_space=self.config_space,
            rng=self.rng,
            transfer_learning_history=None, 
        )
        self.normalizer.fit(y)
        y = self.normalizer.transform(y)
        model.train(X, y)
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        # build target surrogate
        self.target_surrogate = self._build_single_surrogate(X, y)
        
        if self._get_num_tasks() == 0:
            return
        
        mu_list, var_list = [], []
        for surrogate in self.source_surrogates:
            mu, var = surrogate.predict(X)
            mu_list.append(mu.flatten())
            var_list.append(var.flatten())
        
        if len(y) >= self.k_fold_num:
            tar_mu, tar_var = self._predict_target_surrogate_cv(X, y)
            mu_list.append(tar_mu)
            var_list.append(tar_var)
            
            new_w = self.weight_calculator.calculate(
                mu_list, var_list, y, self._get_num_tasks() + 1,
                instance_num=len(y),
                k_fold_num=self.k_fold_num,
                only_source=self.only_source
            )
            self.ignored_flags = self.weight_calculator.get_ignored_flags()
            self.w, self.current_target_weight = self._modify_weights(
                new_w, 
                self.current_target_weight
            )
            self._record_weights()
        else:
            # If not enough data for CV, use default equal weights
            num_tasks = self._get_num_tasks() + 1
            self.w = np.ones(num_tasks) / num_tasks
            self.ignored_flags = [False] * num_tasks
    
    def predict(self, X: np.ndarray, **kwargs) -> tuple:
        mu, var = self.target_surrogate.predict(X)
        
        if self._get_num_tasks() == 0:
            return mu, var
        
        mu *= self.w[-1]
        var *= (self.w[-1] ** 2)
        for i, surrogate in enumerate(self.source_surrogates):
            if len(self.ignored_flags) > i and self.ignored_flags[i]:
                continue
            mu_t, var_t = surrogate.predict(X)
            mu += self.w[i] * mu_t
            var += self.w[i] * self.w[i] * var_t
        return mu, var
    
    def get_acquisition_context(self, history: HistoryLike) -> AcquisitionContext:
        tasks = []
        
        for i, task_history in enumerate(self._get_all_tasks()):
            tasks.append(
                TaskContext(
                    surrogate=self.source_surrogates[i],
                    history=task_history,
                    eta=task_history.get_incumbent_value(),
                    num_data=len(task_history)
                )
            )
        
        tasks.append(
            TaskContext(
                surrogate=self.target_surrogate,
                history=history,
                eta=history.get_incumbent_value(),
                num_data=len(history)
            )
        )
        
        context = AcquisitionContext(
            tasks=tasks,
            weights=self.w
        )
        context.set_main_surrogate(self)
        return context
    
    def get_weights(self) -> np.ndarray:
        return self.w.copy()
    
    def _predict_target_surrogate_cv(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """k-fold cross validation for surrogate model
        
        Parameters
        ----------
        X : np.ndarray
            feature matrix [n_samples, n_features]
        y : np.ndarray
            target values [n_samples]
            
        Returns
        -------
        mu : np.ndarray
            predicted mean values by cross validation [n_samples]
        var : np.ndarray
            predicted variance values by cross validation [n_samples]
        """
        if len(X) < self.k_fold_num:
            raise ValueError(f"Not enough samples ({len(X)}) for {self.k_fold_num}-fold CV")
        
        kf = KFold(n_splits=self.k_fold_num, shuffle=False)
        mu_list = []
        var_list = []
        indices = []    # indices of validation set
        
        for train_idx, val_idx in kf.split(X):
            indices.extend(list(val_idx))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, _ = y[train_idx], y[val_idx]
            
            model = self._build_single_surrogate(X_train, y_train)
            
            mu, var = model.predict(X_val)
            mu, var = mu.flatten(), var.flatten()
            
            mu_list.extend(list(mu))
            var_list.extend(list(var))
    
        assert (np.array(indices) == np.arange(X.shape[0])).all()
        return np.asarray(mu_list), np.asarray(var_list)


    def _add_source_tasks(self, history_list: List[HistoryLike]):
        self.source_data.extend(history_list)
    
    def _get_all_tasks(self) -> List[HistoryLike]:
        return self.source_data
    
    def _get_num_tasks(self) -> int:
        return len(self.source_data)
    
    def _clear_source_tasks(self):
        self.source_data = []
        self.source_surrogates = []

    def _modify_weights(
        self, 
        new_w: np.ndarray, 
        current_target_weight: float
    ) -> Tuple[np.ndarray, float]:
        if self._get_num_tasks() == 0:
            return new_w, new_w[0] if len(new_w) > 0 else 0.0
        
        # target task is the last task
        target_idx = self._get_num_tasks()
        if new_w[target_idx] < current_target_weight:
            # keep the target task weight non-decreasing
            new_w[target_idx] = current_target_weight
            if np.sum(new_w[: target_idx]) > 0:
                new_w[: target_idx] = (
                    new_w[: target_idx] / np.sum(new_w[: target_idx]) 
                    * (1 - new_w[target_idx])
                )
        new_target_weight = new_w[target_idx]
        return new_w, new_target_weight
    
    def _record_weights(self) -> None:
        if self._get_num_tasks() == 0:
            return
        
        w = self.w.copy()
        weight_str = ','.join([('%.2f' % item) for item in w])
        logger.info(f'weight: {weight_str}')
        
        ignored_flags = self.weight_calculator.get_ignored_flags()
        if ignored_flags and any(ignored_flags):
            logger.info(f'weight ignore flag: {ignored_flags}')
        
        w_str_list = []
        for i in range(self._get_num_tasks()):
            task = self._get_all_tasks()[i]
            task_id = task.task_id if hasattr(task, 'task_id') else f'task_{i}'
            w_str = "%s: sim%.4f" % (task_id, w[i])
            w_str_list.append(w_str)
        w_str_list.append("target: %.4f" % w[-1])
        
        self.hist_ws.append(w_str_list)
        self.target_weight.append(w[-1])
        self.iteration_id += 1