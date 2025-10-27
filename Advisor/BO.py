import numpy as np
import copy
from openbox import logger
from ConfigSpace import Configuration, ConfigurationSpace

from .base import BaseAdvisor
from .utils import build_my_surrogate, build_my_acq_func
from .workload_mapping.rover.transfer import get_transfer_suggestion
from .acq_optimizer.local_random import InterleavedLocalAndRandomSearch
from Compressor import SHAPCompressor


class BO(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, source_hpo_data=None,
                surrogate_type='prf', acq_type='ei', task_id='test', meta_feature=None,
                ws_strategy='none', ws_args={'init_num': 5}, tl_args={'topk': 5},
                ep_args=None, ep_strategy='none', expert_params=[],
                cp_args=None, cprs_strategy='shap',
                safe_flag=False, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', 
                expert_modified_space=None, enable_range_compression=True,
                **kwargs):
        super().__init__(config_space, task_id=task_id, meta_feature=meta_feature,
                        ws_strategy=ws_strategy, ws_args=ws_args,
                        tl_args=tl_args, source_hpo_data=source_hpo_data,
                        ep_args=ep_args, ep_strategy=ep_strategy,
                        cprs_strategy=cprs_strategy, cp_args=cp_args,
                        seed=seed, rng=rng, rand_prob=rand_prob, rand_mode=rand_mode, **kwargs)

        self.safe_flag = safe_flag

        self.acq_type = acq_type
        self.surrogate_type = surrogate_type
        self.extra_dim = 0
        
        self.origin_expert_space = expert_modified_space
        self.expert_modified_space = copy.deepcopy(self.origin_expert_space)
        self.expert_params = expert_params

        self.norm_y = True
        if 'wrk' in acq_type:
            self.norm_y = False

        self.init_num = ws_args['init_num']
        
        self.compressor = SHAPCompressor(
            config_space=self.origin_config_space,
            expert_params=expert_params,
            **cp_args,
        )
        
        self.config_space, self.sample_space = self.compressor.compress_space(self.source_hpo_data)
        self._setup_optimizer()

    def warm_start(self):
        if self.ws_strategy == 'none':
            return

        sims = self.source_hpo_data_sims

        for i, sim in enumerate(sims):
            logger.info("The %d-th similar task(%s): %s" % (sim[0], self.source_hpo_data[i].task_id, sim[1]))

        warm_str_list = []
        for i in range(len(sims)):
            idx, sim = sims[i]
            task_str = self.source_hpo_data[idx].task_id
            warm_str = "%s: sim%.4f" % (task_str, sim)
            warm_str_list.append(warm_str)

        if 'warm_start' not in self.history.meta_info:
            self.history.meta_info['warm_start'] = [warm_str_list]
        else:
            self.history.meta_info['warm_start'].append(warm_str_list)

        num_evaluated = len(self.history)
        if self.ws_strategy.startswith('best'):
            for i, sim in enumerate(sims):
                sim_obs = copy.deepcopy(self.source_hpo_data[sim[0]].observations)
                sim_obs = sorted(sim_obs, key=lambda x: x.objectives[0])

                task_num = 3 if i == 0 else 1
                for j in range(task_num):
                    config_warm_old = sim_obs[j].config
                    # 在 sample_space 里创建新 config，并逐个拷贝参数
                    # 注意这里的搜索空间是 config_space 而不是 sample_space
                    config_warm = Configuration(self.config_space, values={
                        name: config_warm_old[name] for name in self.sample_space.get_hyperparameter_names()
                    })
                    config_warm.origin = self.ws_strategy + self.source_hpo_data[sim[0]].task_id
                    # 后加的更差，因为是从后往前取的，所以往前加
                    self.ini_configs = [config_warm] + self.ini_configs
                if len(self.ini_configs) + num_evaluated >= self.init_num:
                    break

            while len(self.ini_configs) + num_evaluated < self.init_num:
                config = self.sample_random_configs(self.sample_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
                self.ini_configs = [config] + self.ini_configs

            logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))

        elif self.ws_strategy.startswith('rgpe'):
            topk = self.ws_args.get('topk', 3)

            src_history = [self.source_hpo_data[sims[i][0]] for i in range(topk)]
            target_history = self.history

            while len(self.ini_configs) + num_evaluated < self.init_num:
                final_config = get_transfer_suggestion(src_history, target_history, _logger_kwargs=self._logger_kwargs)
                final_config.origin = self.ws_strategy
                if final_config not in self.history.configurations + self.ini_configs:
                    self.ini_configs.append(final_config)

            logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))

        else:

            raise ValueError('Invalid ws_strategy: %s' % self.ws_strategy)

    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """
    def sample(self, return_list=False):
        num_config_evaluated = len(self.history)
        if len(self.ini_configs) == 0 and (
            (self.init_num > 0 and num_config_evaluated < self.init_num)
            or (self.init_num == 0 and num_config_evaluated == 0)
        ):
            logger.info("Begin to warm start!")
            self.warm_start()

        logger.info("num_config_evaluated: [%d], init_num: [%d], init_configs: [%d]" % (num_config_evaluated, self.init_num, len(self.ini_configs)))
        if num_config_evaluated < self.init_num or (not self.init_num and not num_config_evaluated):
        # if num_config_evaluated <= self.init_num:
            if len(self.ini_configs) > 0:
                config = self.ini_configs[-1]
                self.ini_configs.pop()
            else:
                config = self.sample_random_configs(self.sample_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
            if return_list:
                return [config]
            else:
                return config
    
        X = self.history.get_config_array()
        Y = self.history.get_objectives()

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
            
        self.surrogate.train(X, Y)

        incumbent_value = np.sort(Y)[(num_config_evaluated - 1) // 5] \
            if self.ep_strategy == 'bo_pro' else self.history.get_incumbent_value()
        self.acq_func.update(model=self.surrogate, eta=incumbent_value, num_data=num_config_evaluated)

        observations = self.history.observations
        challengers = self.acq_optimizer.maximize(observations=observations, num_points=2000)
    
        if return_list:
            return challengers.challengers

        cur_config = challengers.challengers[0]
        recommend_flag = False
        for config in challengers.challengers:
            if config not in self.history.configurations:
                cur_config = config
                recommend_flag = True
                break
        if recommend_flag:
            logger.warn("Successfully recommend a configuration through Advisor!")
        else:
            logger.error("Failed to recommend am unique configuration ! Return a random config")
            cur_config = self.sample_random_configs(self.sample_space, 1, excluded_configs=self.history.configurations)

        logger.info("ret conf: %s" % (str(cur_config)))
        return cur_config

    def _setup_optimizer(self):
        """Setup surrogate model and acquisition optimizer after compression."""
        self.history.config_space = self.sample_space
        self.history.meta_info["compressor"] = self.compressor.compression_info

        self.ini_configs = list()
        self.sample_space.seed(self.seed)
        self.config_space.seed(self.seed)
        logger.info("ConfigSpace after whole compression (dimension + range): %s !!!" % (str(self.sample_space)))
        
        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space, rng=self.rng,
                                            transfer_learning_history=self.compressor.transform_source_data(self.source_hpo_data),
                                            extra_dim=self.extra_dim, norm_y=self.norm_y)
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate)
        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func,
                                                            rand_prob=self.rand_prob, rand_mode=self.rand_mode, rng=self.rng,
                                                            config_space=self.sample_space)