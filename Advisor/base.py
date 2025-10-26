import numpy as np
import json as js
from openbox import logger
from openbox.utils.history import History
from openbox.utils.constants import MAXINT
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from .utils import map_source_hpo_data, build_observation


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace,
                task_id='test',
                ep_args=None, ep_strategy='none',
                ws_strategy='none', ws_args=None, tl_args=None, source_hpo_data=None,
                cprs_strategy='none', cp_args=None,
                meta_feature=None, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):

        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        # 随机种子
        self.seed = seed
        self.rand_prob = rand_prob
        self.rand_mode = rand_mode
        if rng is None:
            rng = np.random.RandomState(self.seed)
        self.rng = rng
        
        self.origin_config_space = config_space
        self.config_space_seed = self.rng.randint(MAXINT)
        
        self.config_space = None

        # 历史数据
        meta_feature['seed'] = seed
        meta_feature['rand_prob'] = rand_prob
        meta_feature['rand_mode'] = rand_mode
        meta_feature['ori_space'] = js.loads(write(config_space))
        self.history = History(task_id=task_id, config_space=config_space, meta_info=meta_feature)

        self.task_id = task_id
        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_args = tl_args
        self.ep_args = ep_args
        self.ep_strategy = ep_strategy
        self.cprs_strategy = cprs_strategy
        self.cp_args = cp_args
        
        self.source_hpo_data_sims = None
        self.source_hpo_data = self.filter_source_hpo_data(source_hpo_data)


    def filter_source_hpo_data(self, source_hpo_data):

        if source_hpo_data is None or len(source_hpo_data) == 0:
            logger.info("No source hpo data provided.")
            return list()

        map_strategy = self.ws_strategy
        new_source_hpo_data = list()
        if map_strategy in ['none', 'best_all']:
            sims = [[i, 0.5] for i in range(len(source_hpo_data))]
        else:
            sims = map_source_hpo_data(map_strategy=map_strategy,
                                    target_his=self.history, source_hpo_data=source_hpo_data, **self.ws_args)
        tl_topk = len(source_hpo_data)
        if self.tl_args['topk'] > 0:
            tl_topk = min(self.tl_args['topk'], tl_topk)

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

        configs = set()

        while len(configs) < num_configs:
            sub_config = config_space.sample_configuration()

            if sub_config not in configs and sub_config not in excluded_configs:
                sub_config.origin = "Random Sample!"
                configs.add(sub_config)
            else:
                logger.warning("Duplicate configuration found or excluded, retrying.")

        sampled_configs = list(configs)
        return sampled_configs

    def update(self, config, results):
        obs = build_observation(config, results, last_context=self.last_context)
        self.history.update_observation(obs)

