from typing import Optional, List, Tuple
import numpy as np
import abc
from ConfigSpace import ConfigurationSpace

from .weight_calculator import WeightCalculator, MFGPEWeightCalculator
from .weight_modifier import WeightModifier, NonDecreasingTargetWeightModifier
from .utils import cross_validate_surrogate, Normalizer
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
        weight_modifier: Optional[WeightModifier] = None,
        source_data: Optional[List[HistoryLike]] = None,
        norm_y: bool = True,
        k_fold_num: int = 5,
        **kwargs
    ):
        super().__init__(config_space, surrogate_type, rng)
        self.source_data = source_data or []
        self.num_src_trials = num_src_trials
        self.source_surrogates: List[Surrogate] = []
        self.target_surrogate: Optional[SingleFidelitySurrogate] = None

        self.weight_calculator = weight_calculator or MFGPEWeightCalculator()
        self.weight_modifier = weight_modifier or NonDecreasingTargetWeightModifier()
        
        self.normalizer = Normalizer(norm_y=norm_y)

        self.k_fold_num = k_fold_num
        self.w: np.ndarray = np.array([1.0])
        self.current_target_weight = 0.0
        self.ignored_flags: List[bool] = []
        
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
    
    def _build_single_surrogate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Surrogate:
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
            
            num_tasks = self._get_num_tasks() + 1
            new_w = self.weight_calculator.calculate(
                mu_list, var_list, y, num_tasks
            )
            
            if hasattr(self.weight_calculator, 'ignored_flags'):
                self.ignored_flags = self.weight_calculator.ignored_flags
            
            self.w, self.current_target_weight = self.weight_modifier.modify(
                new_w, 
                self._get_num_tasks(),
                self.current_target_weight
            )
    
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
    
    def _predict_target_surrogate_cv(self, X, y, k_fold_num=None):
        if k_fold_num is None:
            k_fold_num = self.k_fold_num
        
        def build_fn(X_train, y_train):
            return self.builder.build_single_surrogate(X_train, y_train, normalize_y=False)
        
        return cross_validate_surrogate(X, y, build_fn, k_fold=k_fold_num)

    def _add_source_tasks(self, history_list: List[HistoryLike]):
        self.source_data.extend(history_list)
    
    def _get_all_tasks(self) -> List[HistoryLike]:
        return self.source_data
    
    def _get_num_tasks(self) -> int:
        return len(self.source_data)
    
    def _clear_source_tasks(self):
        self.source_data = []
        self.source_surrogates = []