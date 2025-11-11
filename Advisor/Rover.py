import os
import time
import json
import numpy as np
from typing import Union
from copy import deepcopy
from ConfigSpace import ConfigurationSpace, Configuration
from sklearn.ensemble import RandomForestRegressor

from openbox import Advisor, Observation, History
from openbox.utils.config_space.util import convert_configurations_to_array

from openbox import logger
from .mtgp import MultiTaskGP
from .utils import build_observation

MAXINT = 2 ** 31 - 1

class Rover:
    default_openbox_kwargs = dict(
        surrogate_type='gp',
        acq_optimizer_type='local_random',
    )
    default_rfr_kwargs = dict(
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=0,
        n_jobs=1,
    )
    
    def __init__(self, config_space: ConfigurationSpace,
                 meta_feature=None, source_hpo_data=None,
                 surrogate_type='prf', acq_type='ei', task_id='test',
                 ws_strategy='none', ws_args={'init_num': 5}, tl_args={'topk': 5},
                 context_flag=False, safe_flag=False,
                 seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):
        
        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        self.config_space = config_space

        # 随机种子
        self.seed = seed
        if rng is None:
            rng = np.random.RandomState(self.seed)
        self.rng = rng
        config_space_seed = self.rng.randint(MAXINT)
        self.config_space.seed(config_space_seed)

        self.context_flag = context_flag

        self.last_context = meta_feature['ini_context']

        meta_feature['seed'] = seed
        meta_feature['rand_prob'] = rand_prob
        meta_feature['rand_mode'] = rand_mode
        # 历史数据
        self.history = History(task_id=task_id, config_space=config_space, meta_info=meta_feature)

        self.task_id = task_id
        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_args = tl_args

        self.source_hpo_data_sims = None
        self.source_hpo_data = self.filter_source_hpo_data(source_hpo_data)

        # 初始化
        self.ini_configs = list()
        default_config = self.config_space.get_default_configuration()
        default_config.origin = "default"
        self.ini_configs.append(default_config)

        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.safe_flag = safe_flag

        self.acq_type = acq_type
        self.surrogate_type = surrogate_type

        ini_context = meta_feature['ini_context']

        self.extra_dim = 0
        if self.context_flag:
            self.extra_dim = len(ini_context)

        self.norm_y = True
        if 'wrk' in acq_type:
            self.norm_y = False

        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space, rng=self.rng,
                                            transfer_learning_history=self.source_hpo_data,
                                            extra_dim=self.extra_dim, norm_y=self.norm_y)

        self.init_num = ws_args['init_num']

        self.acq_func = build_my_acq_func(func_str=acq_type, model=self.surrogate)

        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func, rand_prob=rand_prob, rand_mode=rand_mode,
                                                             config_space=self.config_space, rng=self.rng,
                                                             context=ini_context, context_flag=context_flag)
    

    def filter_source_hpo_data(self, source_hpo_data):

        if source_hpo_data is None or len(source_hpo_data) == 0:
            logger.info("No source hpo data provided.")
            return list()

        map_strategy = self.ws_strategy
        if map_strategy == 'none':
            map_strategy = 'rover'
        new_source_hpo_data = list()
        sims = map_source_hpo_data(map_strategy=map_strategy,
                                   target_his=self.history, source_hpo_data=source_hpo_data, **self.ws_args)
        tl_topk = len(source_hpo_data)
        if self.tl_args['topk'] > 0:
            tl_topk = self.tl_args['topk']

        self.source_hpo_data_sims = []  # 重新排序
        for i in range(tl_topk):
            new_source_hpo_data.append(source_hpo_data[sims[i][0]])
            self.source_hpo_data_sims.append([i, sims[i][1]])
            logger.info("The %d-th similar task(%s): %s" % (sims[i][0], source_hpo_data[sims[i][0]].task_id, sims[i][1]))

        logger.info("Successfully filter source hpo data to %d tasks." % len(new_source_hpo_data))

        return new_source_hpo_data

    @property
    def contexts(self):
        contexts = self.history.observations[0]['last_context']
        for idx in range(1, len(self.history)):
            contexts = np.vstack([contexts, self.history.observations[idx]['last_context']])

        return contexts

    def warm_start(self):

        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    @staticmethod
    def sample_random_configs(config_space, num_configs=1, excluded_configs=None):
        """
        Sample a batch of random configurations.

        Parameters
        ----------
        config_space: ConfigurationSpace
            Configuration space object.
        num_configs: int
            Number of configurations to sample.
        excluded_configs: optional, List[Configuration] or Set[Configuration]
            A list of excluded configurations.

        Returns
        -------
        configs: List[Configuration]
            A list of sampled configurations.
        """
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = config_space.sample_configuration()
            sample_cnt += 1
            if config not in configs and config not in excluded_configs:
                config.origin = "Random Sample!"
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                config.origin = "Random Sample(duplicate)!"
                configs.append(config)
                sample_cnt = 0
        return configs

    def update(self, config, results):

        obs = build_observation(config, results, last_context=self.last_context)

        self.history.update_observation(obs)

        self.last_context = results['result']['context']
        
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
                    config_warm = sim_obs[j].config
                    config_warm.origin = self.ws_strategy + self.source_hpo_data[sim[0]].task_id
                    # 后加的更差，因为是从后往前取的，所以往前加
                    self.ini_configs = [config_warm] + self.ini_configs
                if len(self.ini_configs) + num_evaluated >= self.init_num:
                    break

            while len(self.ini_configs) + num_evaluated < self.init_num:
                config = self.sample_random_configs(self.config_space, 1,
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
    evaluate后更新contextBO配置
    """

    def update(self, config, results):
        super(BO, self).update(config, results)
        if self.context_flag:
            self.acq_optimizer.update_contex(results['result']['context'])
            print("successfully update context!")

    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """

    def sample(self, return_list=False):
        num_config_evaluated = len(self.history)
        if len(self.ini_configs) == 0 and num_config_evaluated < self.init_num:
            self.warm_start()

        if num_config_evaluated < self.init_num:
            if len(self.ini_configs) > 0:
                config = self.ini_configs[-1]
                self.ini_configs.pop()
            else:
                config = self.sample_random_configs(self.config_space, 1,
                                                    excluded_configs=self.history.configurations)[0]
            if return_list:
                return [config]
            else:
                return config

        X = self.history.get_config_array()

        if self.context_flag:
            X = np.hstack((X, self.contexts))
        Y = self.history.get_objectives()

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.config_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
        self.surrogate.train(X, Y)

        self.acq_func.update(model=self.surrogate,
                             eta=self.history.get_incumbent_value(),
                             num_data=num_config_evaluated)

        challengers = self.acq_optimizer.maximize(observations=self.history.observations,
                                                  num_points=5000)

        if return_list:
            return challengers.challengers

        cur_config = challengers.challengers[0]

        # 对1个任务，在40轮之后引入安全约束（阈值85%)
        if self.safe_flag:
            recommend_flag = True
            if len(self.history) >= 10:
                recommend_flag = False
                for config in challengers.challengers:
                    X = convert_configurations_to_array([config])
                    if self.context_flag:
                        X = np.hstack((X, [self.last_context]))
                    pred_mean, pred_var = self.surrogate.predict(X)
                    if pred_mean[0] < 0.85 * self.history.get_incumbent_value():  # 满足约束 (perf是负数)
                        logger.warn(
                            '-----------The config_%d meet the security constraint-----------' % challengers.challengers.index(
                                config))
                        cur_config = config
                        recommend_flag = True
                        break

            if recommend_flag:
                logger.warn("Successfully recommend a configuration through Advisor!")
            else:
                logger.error(
                    "Failed to recommend a configuration that meets the security constraint! Return the incumbent_config")
                cur_config = self.history.get_incumbent_configs()[0]

        else:
            recommend_flag = False
            # 避免推荐已经评估过的配置
            for config in challengers:
                if config not in self.history.configurations:
                    cur_config = config
                    recommend_flag = True
                    break

            if recommend_flag:
                logger.warn("Successfully recommend a configuration through Advisor!")
            else:
                logger.error("Failed to recommend am unique configuration ! Return a random config")
                cur_config = self.sample_random_configs(self.config_space, 1, excluded_configs=self.history.configurations)

        return cur_config
