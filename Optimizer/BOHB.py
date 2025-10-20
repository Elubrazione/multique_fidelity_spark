import numpy as np
import os
from math import log, ceil
from datetime import datetime
from openbox import logger

from .base import BaseOptimizer
from config import LIST_SPARK_NODES


class BOHB(BaseOptimizer):
    def __init__(self, config_space, eval_func, iter_num=200, per_run_time_limit=None,
                 source_hpo_data=None, ep_args=None, ep_strategy='none',
                 method_id='smbo', task_id='test', target='redis', task_str='run',
                 space_history = None, cprs_strategy='none', cp_args=None,
                 initial_n_list=[16, 8, 2, 1], initial_r_list=[1, 2, 8, 64],
                #  initial_n_list=[2, 1, 1, 1], initial_r_list=[1, 1, 1, 1],
                 range_config_space=None,
                 ws_strategy='none', ws_args=None, tl_strategy='none', tl_args=None,
                 backup_flag=False, save_dir='./results', meta_feature=None, fixed_initial=True,
                 R=9, eta=3, seed=42, rand_prob=0.15, rand_mode='ran', initial_configs=64,
                 config_modifier=None, expert_modified_space=None, expert_params=[],
                 enable_range_compression=False, range_compress_data_path=None):

        super().__init__(config_space=config_space, eval_func=eval_func, iter_num=iter_num,
                         per_run_time_limit=per_run_time_limit, source_hpo_data=source_hpo_data,
                         method_id=method_id, task_id=task_id, target=target, task_str=task_str,
                         ws_strategy=ws_strategy, ws_args=ws_args, tl_strategy=tl_strategy, tl_args=tl_args,
                         ep_args=ep_args, ep_strategy=ep_strategy, range_config_space=range_config_space,
                         cprs_strategy=cprs_strategy, space_history=space_history, cp_args=cp_args,
                         backup_flag=backup_flag, save_dir=save_dir, meta_feature=meta_feature,
                         seed=seed, rand_prob=rand_prob, rand_mode=rand_mode,
                         expert_params=expert_params,
                         enable_range_compression=enable_range_compression, range_compress_data_path=range_compress_data_path)

        self.R = R
        self.eta = eta
        self.rand_prob = rand_prob
        self.initial_configs = initial_configs
        self.fixed_initial = fixed_initial
        self.initial_n_list = initial_n_list
        self.initial_r_list = initial_r_list

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))
        self.inner_iter_id = 0

        self.advisor.history.meta_info['R_eta'] = "%d_%d" % (self.R, self.eta)
        self._initialize_params()
        

    def _initialize_params(self):
        self.fidelity_levels = [int(x) for x in np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)]
        logger.info("Fidelity levels of %s: %s" % (self.method_id, self.fidelity_levels))

        self.flexible_n = list()
        self.flexible_r = list()
        for s in self.s_values:  # Same as bohb at the beginning
            if self.fixed_initial:
                n = self.initial_n_list[0]
                r = self.initial_r_list[0]
            else:
                n = int(ceil(self.B / self.R / (s + 1) * (self.eta ** s))) 
                r = int(self.R * self.eta ** (-s))
            self.flexible_n.append(n)
            self.flexible_r.append(r)        
        logger.info("Run %d brackets with number of configurations %s and resources %s. s_max = [%d]" 
                    % (len(self.flexible_n), self.flexible_n, self.flexible_r, self.s_max))


    def _gen_candidates(self, num_configs):
        if 'BOHB' in self.method_id:
            cans = self.advisor.sample(return_list=True)

            candidates = list()
            idx_acq = 0
            for _id in range(num_configs):
                if self.rng.random() < self.rand_prob or _id >= len(cans):
                    i = 0
                    config = self.advisor.sample_random_configs(
                        self.advisor.sample_space, 1, excluded_configs=self.advisor.history.configurations)[0]
                    while config in candidates:
                        config = self.advisor.sample_random_configs(
                            self.advisor.sample_space, 1, excluded_configs=self.advisor.history.configurations)[0]
                        i += 1
                        if i > 1000:
                            logger.warning('Cannot sample a new random configuration after 1000 iters.')
                            break
                    config.origin = 'BOHB Random Sample'
                else:
                    found = False
                    while idx_acq < len(cans):
                        config = cans[idx_acq]
                        idx_acq += 1
                        if config not in candidates:
                            config.origin = 'BOHB ' + config.origin
                            found = True
                            break
                    if not found:
                        # Fallback to random sample if all in cans are duplicates
                        config = self.advisor.sample_random_configs(
                            self.advisor.sample_space, 1,
                            excluded_configs=self.advisor.history.configurations + candidates
                        )[0]
                        config.origin = 'BOHB Random Fallback'
                candidates.append(config)
            return candidates
        elif 'MFSE' in self.method_id or 'FlexHB' in self.method_id:
            configs = self.advisor.samples(batch_size=num_configs)
            if self.config_modifier is not None:
                for config in configs:
                    config = self.config_modifier(config)
            return configs
        else:
            raise ValueError('Invalid method_id: %s!' % self.method_id)

    def _update_obs(self, config, results, resource_ratio, special_list=None):
        if 'BOHB' in self.method_id:
            if resource_ratio == 1:
                self.advisor.update(config=config, results=results)
        elif 'MFSE' in self.method_id or 'FlexHB' in self.method_id:
            if special_list is None:
                self.advisor.update(config=config, results=results, resource_ratio=resource_ratio)
            else:
                pass
        else:
            raise ValueError('Invalid method_id: %s!' % self.method_id)
    

    def _evaluate_configurations(self, candidates, resource_ratio, update=True):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        performances = []
        futures = []
        res_dirs = []

        with ThreadPoolExecutor(max_workers=len(candidates)) as executor:
            for config in candidates:
                res_dir = os.path.join(self.res_dir, datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
                res_dirs.append(res_dir)
                future = executor.submit(self.eval_func, config=config, resource_ratio=resource_ratio, res_dir=res_dir)
                futures.append((future, config, res_dir))

            for future, config, res_dir in futures:
                results = future.result()
                if update:
                    self._update_obs(config, results, resource_ratio=resource_ratio)
                performances.append(results['result']['objective'])

        return performances
                

    def _iterate(self, s, n, r, skip_last=0):        
        # Set initial number of configurations
        # n = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))  # (smax+1) / (s+1) * 3^s
        # initial number of iterations per config
        # r = int(self.R * self.eta ** (-s))                        # 81 * 3^(-s)

        # Choose a batch of configurations in different mechanisms.
        candidates = list(set(self._gen_candidates(n * len(LIST_SPARK_NODES))))
        logger.info("Get %d initial configurations" % (len(candidates)))

        iter_full_eval_configs, iter_full_eval_perfs = [], []
               
        # Run each bracket.
        for i in range((s + 1) - int(skip_last)):  # changed from s + 1
            # Run each of the n configs for <iterations>
            # and keep best (n_configs / eta) configurations
            n_configs = n * self.eta ** (-i)
            n_resource = r * self.eta ** i
            if len(self.initial_n_list) and len(self.initial_r_list):
                n_configs = self.initial_n_list[i] * len(LIST_SPARK_NODES)
                n_resource = self.initial_r_list[i]
            logger.info("Current number of configurations and resource: %d %d" % (n_configs, n_resource))

            resource_ratio = round(float(n_resource / self.R), 5)
            perfs = self._evaluate_configurations(candidates, resource_ratio)
            logger.info('iter: %d, resource: %d, val_loss: %s' % (i, n_resource, str(perfs)))
            
            reduced_num = int(n_configs / self.eta)
            if len(self.initial_n_list) and len(self.initial_r_list):
                reduced_num = self.initial_n_list[i + 1] if i < s else n_configs

            valid_count = np.sum([not np.isinf(p) for p in perfs])
            while i != s and valid_count < reduced_num:
                logger.warn("Valid performance count (%d) < expected (%d), resampling..." % (valid_count, reduced_num))

                additional_candidates = self._gen_candidates(reduced_num - valid_count)
                candidates.extend(additional_candidates)
                additional_perfs = self._evaluate_configurations(additional_candidates, resource_ratio)
                perfs.extend(additional_perfs)
                
                cur_valid_count = np.sum([not np.isinf(p) for p in perfs])
                logger.warning("Get %d valid configurations after resampling, still need %d ..." \
                    % (cur_valid_count - valid_count, reduced_num - cur_valid_count))
                valid_count = cur_valid_count
            
            logger.info("All performances of round-%d and iteration-%d: {%s}" % (self.iter_id, i + 1, str(perfs)))

            # Select a number of best configurations for the next loop.
            indices = np.argsort(perfs)
            candidates = [candidates[i] for i in indices]
            perfs = [perfs[i] for i in indices]
            logger.info("Sorted performances: %s" % (str(perfs)))
            candidates, perfs = candidates[: reduced_num], perfs[: reduced_num]
            
            if int(n_resource) == self.R or i == s - int(skip_last):
                iter_full_eval_configs.extend(candidates)
                iter_full_eval_perfs.extend(perfs)

        return iter_full_eval_configs, iter_full_eval_perfs


    def run_one_iter(self):
        self.iter_id += 1
        num_config_evaluated = len(self.advisor.history)
        if num_config_evaluated < self.advisor.init_num:
            remaining_configs = self.advisor.init_num - num_config_evaluated
            max_parallel = min(len(LIST_SPARK_NODES), remaining_configs)
            
            logger.info(f"[BOHB] 初始化阶段: 需要评估 {remaining_configs} 个配置，使用 {max_parallel} 个并行任务")
            candidates = self._gen_candidates(max_parallel)
            logger.info(f"[BOHB] 生成了 {len(candidates)} 个候选配置")
            perfs = self._evaluate_configurations(candidates, round(float(1.0), 5), update=True)
            
            iter_full_eval_configs = candidates
            iter_full_eval_perfs = perfs
        else:
            iter_full_eval_configs, iter_full_eval_perfs = self._iterate(
                self.s_max if self.fixed_initial else self.s_values[self.inner_iter_id],
                self.flexible_n[self.inner_iter_id],
                self.flexible_r[self.inner_iter_id]
            )
            self.inner_iter_id = (self.inner_iter_id + 1) % (self.s_max + 1)

        logger.info(
            "[{}] iter ------------------------------------------------{:5d}".format(self.task_str, self.iter_id))
        for idx, config in enumerate(iter_full_eval_configs):
            if config.origin:
                logger.warn("[{}] !!!!!!!!!! {} !!!!!!!!!!".format(self.task_str, config.origin))

            logger.info('[{}] Config: '.format(self.task_str) + str(config.get_dictionary()))
            logger.info('[{}] Obj: {}'.format(self.task_str, iter_full_eval_perfs[idx]))

        logger.info('[{}] best obj: {}'.format(self.task_str, self.advisor.history.get_incumbent_value()))

        logger.info(
            "[{}] ===================================================================================================================================================".format(
                self.task_str))

        self.save_info(interval=1)