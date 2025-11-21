from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .utils import calculate_preserving_order_num


class WeightCalculator(ABC):
    def __init__(self):
        self.ignored_flags: List[bool] = []

    @abstractmethod
    def calculate(
        self,
        mu_list: List[np.ndarray],
        var_list: List[np.ndarray],
        y_true: np.ndarray,
        num_tasks: int
    ) -> np.ndarray:
        """Weight Calculator Interface
        
        Parameters
        ----------
        mu_list : List[np.ndarray]
            Predicted mean values for each task
        var_list : List[np.ndarray]
            Predicted variance values for each task
        y_true : np.ndarray
            True values
        num_tasks : int
            Number of tasks (including the target task)
            
        Returns
        -------
        weights : np.ndarray
            Weight vector [num_tasks]
        """
        pass

    def get_ignored_flags(self) -> List[bool]:
        return self.ignored_flags        


class MFGPEWeightCalculator(WeightCalculator):
    def __init__(self, n_power: int = 3):
        self.n_power = n_power
        super().__init__()

    def calculate(
        self,
        mu_list: List[np.ndarray],
        var_list: List[np.ndarray],
        y_true: np.ndarray,
        num_tasks: int
    ) -> np.ndarray:
        preserving_order_p = []
        for i in range(num_tasks):
            y_pred = mu_list[i]
            preorder_num, pair_num = calculate_preserving_order_num(y_pred, y_true)
            preserving_order_p.append(preorder_num / pair_num)
        
        trans_order_weight = np.array(preserving_order_p)
        p_power = np.power(trans_order_weight, self.n_power)
        return p_power / np.sum(p_power)


class RGPEWeightCalculator(WeightCalculator):
    def __init__(self, num_sample: int = 50, use_dilution: bool = False):
        self.num_sample = num_sample
        self.use_dilution = use_dilution
        super().__init__()
    
    def calculate(
        self,
        mu_list: List[np.ndarray],
        var_list: List[np.ndarray],
        y_true: np.ndarray,
        num_tasks: int
    ) -> np.ndarray:
        if self.use_dilution:
            return self.calculate_with_dilution(
                mu_list, var_list, y_true, num_tasks, len(y_true)
            )
        else:
            return self._calculate_basic(mu_list, var_list, y_true, num_tasks)
    
    def _calculate_basic(
        self,
        mu_list: List[np.ndarray],
        var_list: List[np.ndarray],
        y_true: np.ndarray,
        num_tasks: int
    ) -> np.ndarray:
        argmin_list = [0] * num_tasks
        ranking_loss_caches = []
        
        # Monte Carlo sampling
        for _ in range(self.num_sample):
            ranking_loss_list = []
            
            for i in range(num_tasks):
                sampled_y = np.random.normal(mu_list[i], var_list[i])
                preorder_num, pair_num = calculate_preserving_order_num(sampled_y, y_true)
                rank_loss = pair_num - preorder_num
                ranking_loss_list.append(rank_loss)
            
            ranking_loss_caches.append(ranking_loss_list)
            argmin_task = np.argmin(ranking_loss_list)
            argmin_list[argmin_task] += 1        
        w = np.array(argmin_list) / self.num_sample
        return w
    
    def calculate_with_dilution(
        self,
        mu_list: List[np.ndarray],
        var_list: List[np.ndarray],
        y_true: np.ndarray,
        num_tasks: int,
        instance_num: int,
        k_fold_num: int = 5,
        only_source: bool = False
    ) -> tuple[np.ndarray, List[bool]]:
        argmin_list = [0] * num_tasks
        ranking_loss_caches = []
        
        # Monte Carlo sampling
        for _ in range(self.num_sample):
            ranking_loss_list = []
            
            # Source tasks
            for i in range(num_tasks - 1):
                sampled_y = np.random.normal(mu_list[i], var_list[i])
                preorder_num, pair_num = calculate_preserving_order_num(sampled_y, y_true)
                rank_loss = pair_num - preorder_num
                ranking_loss_list.append(rank_loss)
            
            # Target task (if instance_num < k_fold_num, use the mean of the target task)
            tar_mu, tar_var = mu_list[-1], var_list[-1]
            if instance_num >= k_fold_num:
                sampled_y = np.random.normal(tar_mu, tar_var)
                preorder_num, pair_num = calculate_preserving_order_num(sampled_y, y_true)
                rank_loss = pair_num - preorder_num
            else:
                rank_loss = instance_num * instance_num
            ranking_loss_list.append(rank_loss)
            ranking_loss_caches.append(ranking_loss_list)
            
            # record the idx of the task with smallest ranking loss, count the frequency
            argmin_list[np.argmin(ranking_loss_list)] += 1
        
        # weights = frequency of task with smallest ranking loss
        # smaller ranking loss, larger weight
        w = np.array(argmin_list) / self.num_sample
        
        # Weight dilution logic (filter out the tasks with large ranking loss)
        ignored_flags = [False] * num_tasks
        ranking_loss_caches = np.array(ranking_loss_caches) # shape: (num_sample, num_tasks)
        # threshold is the 95% quantile of the ranking loss of the target task in each sample
        threshold = sorted(ranking_loss_caches[:, -1])[int(self.num_sample * 0.95)]
        for i in range(num_tasks - 1):
            # if the median of the ranking loss of the task is larger than the threshold, ignore the task
            median = sorted(ranking_loss_caches[:, i])[int(self.num_sample * 0.5)]
            ignored_flags[i] = median > threshold
        # if only_source is True, ignore the target task
        ignored_flags[-1] = only_source

        for i in range(num_tasks - 1):
            if ignored_flags[i]:
                w[i] = 0.0

        sum_w = np.sum(w)
        if sum_w == 0:
            w = [1.0 / (num_tasks - 1)] * (num_tasks - 1) + [0.0] \
                if only_source else [0.0] * (num_tasks - 1) + [1.0]
        else:
            w = w / sum_w
        
        self.ignored_flags = ignored_flags
        return w