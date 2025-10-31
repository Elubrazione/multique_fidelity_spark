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
                 scheduler_type='bohb',
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
                         _logger_kwargs=_logger_kwargs)

        from .scheduler import schedulers
        self.scheduler = schedulers[scheduler_type](num_nodes=len(LIST_SPARK_NODES), **scheduler_kwargs)

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

    def _iterate(self):   
        iter_full_eval_configs, iter_full_eval_perfs = [], []
        candidates = []

        s = self.scheduler.get_bracket_index(self.iter_id - self.advisor.init_num)

        for i in range(s + 1):
            n_configs, n_resource = self.scheduler.get_stage_params(s=s, stage=i)
            logger.info(f"Stage {i}: n_configs={n_configs}, n_resource={n_resource}")
            
            if not i:
                candidates = list(set(self.advisor.samples(batch_size=n_configs)))
                logger.info(f"Generated {len(candidates)} initial candidates")

            resource_ratio = self.scheduler.calculate_resource_ratio(n_resource)
            perfs = self._evaluate_configurations(candidates, resource_ratio)
            logger.info(f"Stage {i}: evaluated {len(perfs)} configs, resource_ratio={resource_ratio}")
            
            candidates, perfs = self.scheduler.eliminate_candidates(candidates, perfs, s=s, stage=i)
            
            if i == s:
                iter_full_eval_configs.extend(candidates)
                iter_full_eval_perfs.extend(perfs)

        return iter_full_eval_configs, iter_full_eval_perfs


    def run_one_iter(self):
        self.iter_id += 1
        num_config_evaluated = len(self.advisor.history)
        if num_config_evaluated < self.advisor.init_num:
            candidates = self.advisor.samples(batch_size=self.scheduler.num_nodes)
            logger.info(f"Initialization phase: need to evaluate {self.scheduler.num_nodes} configs, generated {len(candidates)} initial candidates")
            perfs = self._evaluate_configurations(candidates, resource_ratio=round(float(1.0), 5), update=True)
        else:
            candidates, perfs = self._iterate()

        self.log_iteration_results(candidates, perfs)
        self.save_info(interval=1)