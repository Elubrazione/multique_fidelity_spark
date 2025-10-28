import numpy as np
import json as js
from openbox import logger
from openbox.utils.history import History
from openbox.utils.constants import MAXINT
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from Compressor import SHAPCompressor
from .utils import build_observation


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace,
                task_id='test',
                ws_strategy='none', ws_args=None, tl_args=None, source_hpo_data=None,
                cprs_strategy='none', cp_args=None,
                meta_feature=None, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):

        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        self.seed = seed
        self.rand_prob = rand_prob
        self.rand_mode = rand_mode
        if rng is None:
            rng = np.random.RandomState(self.seed)
        self.rng = rng
        
        self.origin_config_space = config_space
        self.compressor = SHAPCompressor(config_space=self.origin_config_space, **cp_args)
        self.config_space, self.sample_space = self.compressor.compress_space(self.source_hpo_data)
        self.origin_config_space.seed(self.seed)
        self.sample_space.seed(self.seed)
        self.config_space.seed(self.seed)
        self.ini_configs = list()

        meta_feature['random'] = {'seed': seed, 'rand_prob': rand_prob, 'rand_mode': rand_mode}
        meta_feature['space'] = {'original': js.loads(write(config_space)),
                                'dimension': js.loads(write(self.config_space)),
                                'range': js.loads(write(self.sample_space))}
        meta_feature['compressor'] = self.compressor.compression_info
        self.history = History(task_id=task_id, config_space=self.sample_space, meta_info=meta_feature)

        self.task_id = task_id
        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_args = tl_args
        self.cprs_strategy = cprs_strategy
        self.cp_args = cp_args

        self.source_hpo_data_sims = None

    def filter_source_hpo_data(self, source_hpo_data):
        filtered, sims = self.task_manager.filter_source_hpo_data(self.history, source_hpo_data)
        return filtered

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