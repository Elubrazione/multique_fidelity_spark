import os
from datetime import datetime
from openbox import logger

from .base import BaseOptimizer
from config import LIST_SPARK_NODES


class BOHB(BaseOptimizer):
    def __init__(self, config_space, eval_func, iter_num=200, per_run_time_limit=None,
                 method_id='smbo', task_id='test', target='redis',
                 cp_args={},
                 ws_strategy='none', ws_args={'init_num': 5}, 
                 tl_strategy='none', tl_args={'topk': 5},
                 backup_flag=False, save_dir='./results',
                 seed=42, rand_prob=0.15, rand_mode='ran',
                 task_manager=None,
                 scheduler_kwargs={},
                 _logger_kwargs=None):

        super().__init__(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                         per_run_time_limit=per_run_time_limit,
                         method_id=method_id, task_id=task_id, target=target,
                         ws_strategy=ws_strategy, ws_args=ws_args,
                         tl_strategy=tl_strategy, tl_args=tl_args,
                         cp_args=cp_args,
                         backup_flag=backup_flag, save_dir=save_dir,
                         seed=seed, rand_prob=rand_prob, rand_mode=rand_mode,
                         task_manager=task_manager,
                         _logger_kwargs=_logger_kwargs)

        self.inner_iter_id = 0
        from .scheduler.fidelity import FidelityScheduler
        self.scheduler = FidelityScheduler(**scheduler_kwargs)

    def _update_obs(self, config, results, resource_ratio):
        if 'BOHB' in self.method_id:
            if resource_ratio == 1:
                self.advisor.update(config=config, results=results)
        elif 'MFSE' in self.method_id:
            self.advisor.update(config=config, results=results, resource_ratio=resource_ratio)
        else:
            raise ValueError('Invalid method_id: %s!' % self.method_id)

    def _evaluate_configurations(self, candidates, resource_ratio, update=True):
        from concurrent.futures import ThreadPoolExecutor

        futures, performances = [], []
        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            for config in candidates:
                future = executor.submit(self.eval_func, config=config, resource_ratio=resource_ratio)
                futures.append((future, config))

            for future, config in futures:
                results = future.result()
                if update:
                    self._update_obs(config, results, resource_ratio=resource_ratio)
                performances.append(results['result']['objective'])
        return performances

    def _iterate(self, s):   
        iter_full_eval_configs, iter_full_eval_perfs = [], []
        candidates = []

        # Run each bracket.
        for i in range(s + 1):
            n_configs, n_resource = self.scheduler.get_stage_params(s, i, len(LIST_SPARK_NODES))
            logger.info(f"[BOHB] Stage {i}: n_configs={n_configs}, n_resource={n_resource}")
            
            if not i:
                candidates = list(set(self.advisor.samples(batch_size=n_configs * len(LIST_SPARK_NODES))))
                logger.info(f"[BOHB] Generated {len(candidates)} initial candidates")

            resource_ratio = self.scheduler.calculate_resource_ratio(n_resource)
            perfs = self._evaluate_configurations(candidates, resource_ratio)
            logger.info(f"[BOHB] Stage {i}: evaluated {len(perfs)} configs, resource_ratio={resource_ratio}")
            
            candidates, perfs = self.scheduler.eliminate_candidates(candidates, perfs, s, i)
            logger.info(f"[BOHB] Stage {i}: after elimination, {len(candidates)} candidates remain")
            
            if int(n_resource) == self.scheduler.R or i == s:
                iter_full_eval_configs.extend(candidates)
                iter_full_eval_perfs.extend(perfs)

        return iter_full_eval_configs, iter_full_eval_perfs


    def run_one_iter(self):
        self.iter_id += 1
        num_config_evaluated = len(self.advisor.history)
        if num_config_evaluated < self.advisor.init_num:
            remaining_configs = self.advisor.init_num - num_config_evaluated
            max_parallel = min(len(LIST_SPARK_NODES), remaining_configs)
            
            logger.info(f"[BOHB] Initialization phase: need to evaluate {remaining_configs} configs, using {max_parallel} parallel tasks")
            candidates = self.advisor.samples(batch_size=max_parallel)
            logger.info(f"[BOHB] Generated {len(candidates)} candidate configurations")
            perfs = self._evaluate_configurations(candidates, round(float(1.0), 5), update=True)
            
            iter_full_eval_configs = candidates
            iter_full_eval_perfs = perfs
        else:
            s = self.scheduler.s_values[self.inner_iter_id]
            iter_full_eval_configs, iter_full_eval_perfs = self._iterate(s)
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.scheduler.s_max + 1)

        self.log_iteration_results(iter_full_eval_configs, iter_full_eval_perfs)
        self.save_info(interval=1)