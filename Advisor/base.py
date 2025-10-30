import numpy as np
import json as js
from openbox import logger
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from Compressor import SHAPCompressor
from .utils import build_observation, is_valid_spark_config, sanitize_spark_config
from .task_manager import TaskManager


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace, task_manager: TaskManager,
                task_id='test',
                ws_strategy='none', ws_args=None, tl_args=None,
                cprs_strategy='none', cp_args=None,
                seed=42, rng=None, rand_prob=0.15, rand_mode='ran', 
                **kwargs):

        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        self.seed = seed
        self.rand_prob = rand_prob
        self.rand_mode = rand_mode
        self.rng = rng if rng is not None else np.random.RandomState(self.seed)
        
        self.task_manager = task_manager
        self.task_manager._update_similarity()
        self.compressor = SHAPCompressor(config_space=config_space, **cp_args)

        self.source_hpo_data, self.source_hpo_data_sims = self.task_manager.get_similar_tasks(topk=tl_args['topk'])

        # If compression strategy is 'none', skip both dimension and range compression
        if cprs_strategy == 'none':
            self.surrogate_space = config_space
            self.sample_space = config_space
            logger.info("Compression strategy is 'none'. Using original space for surrogate and sampling.")
        else:
            self.surrogate_space, self.sample_space = self.compressor.compress_space(self.source_hpo_data)
        
        self.sample_space.seed(self.seed)
        self.surrogate_space.seed(self.seed)
        self.ini_configs = list()

        meta_feature = {}
        meta_feature['random'] = {'seed': seed, 'rand_prob': rand_prob, 'rand_mode': rand_mode}
        meta_feature['space'] = {'original': js.loads(write(config_space)),
                                'dimension': js.loads(write(self.surrogate_space)),
                                'range': js.loads(write(self.sample_space))}
        meta_feature['compressor'] = self.compressor.compression_info
        self.task_manager.update_history_meta_info(meta_feature)

        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_args = tl_args
        self.cprs_strategy = cprs_strategy
        self.cp_args = cp_args
        self.history = self.task_manager.current_task_history


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

        _is_valid = is_valid_spark_config
        _sanitize = sanitize_spark_config

        trials = 0
        max_trials = max(100, num_configs * 20)
        while len(configs) < num_configs and trials < max_trials:
            trials += 1
            sub_config = config_space.sample_configuration()
            if not _is_valid(sub_config):
                sub_config = _sanitize(sub_config)
                if not _is_valid(sub_config):
                    continue

            if sub_config not in configs and sub_config not in excluded_configs:
                sub_config.origin = "Random Sample!"
                configs.add(sub_config)
            else:
                continue

        sampled_configs = list(configs)
        return sampled_configs

    def update(self, config, results):
        obs = build_observation(config, results)
        self.history.update_observation(obs)