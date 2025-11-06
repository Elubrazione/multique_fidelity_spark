import numpy as np
import json as js
from openbox import logger
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from Compressor import get_compressor
from .utils import build_observation, is_valid_spark_config, sanitize_spark_config


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace, method_id='unknown',
                task_id='test', ws_strategy='none', ws_args=None,
                tl_strategy='none', tl_args=None, cp_args=None,
                seed=42, rand_prob=0.15, rand_mode='ran', 
                **kwargs):
        # Delay import to avoid circular dependency
        from task_manager import TaskManager
        
        self.task_id = task_id
        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rand_prob = rand_prob
        self.rand_mode = rand_mode
        
        self.task_manager = TaskManager.instance()
        
        self.compressor = get_compressor(
            compressor_type=cp_args.get('strategy', 'none'),
            config_space=config_space,
            **(cp_args or {})
        )

        self.source_hpo_data, self.source_hpo_data_sims = self.task_manager.get_similar_tasks(topk=tl_args['topk']) if tl_strategy != 'none' else ([], [])
        if tl_strategy != 'none':
            # Compress space: pass source_hpo_data only if using transfer learning and compressor supports it
            self.surrogate_space, self.sample_space = self.compressor.compress_space(self.source_hpo_data)
        else:
            # For compressors that need compression even without transfer learning (e.g., LlamaTuneCompressor)
            self.surrogate_space, self.sample_space = self.compressor.compress_space()
        
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
        self.tl_strategy = tl_strategy
        self.tl_args = tl_args
        self.cp_args = cp_args
        
        self.method_id = method_id
        self.history = self.task_manager.current_task_history

        # init_num is equal to the number of topk similar tasks if use transfer learning,
        # otherwise it is the number of initial configurations for warm start
        self.init_num = ws_args['init_num'] if tl_strategy == 'none' else tl_args['topk']

        if tl_strategy != 'none':
            self._compress_history_observations()

    def get_num_evaluated_exclude_default(self):
        """
        Get the number of evaluated configurations excluding the default configuration.
        The default configuration is added in calculate_meta_feature and should not be counted.
        
        Returns
        -------
        int: Number of evaluated configurations excluding default config
        """
        if self.history is None or len(self.history) == 0:
            return 0
        self.has_default_config = any(
            hasattr(obs.config, 'origin') and obs.config.origin == 'Default Configuration'
            for obs in self.history.observations
        )
        num_evaluated = len(self.history)
        return max(0, num_evaluated - 1) if self.has_default_config else num_evaluated

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

    def update(self, config, results, **kwargs):
        if not kwargs.get('update', True):
            return
        obs = build_observation(config, results)
        self.history.update_observation(obs)

    def _compress_history_observations(self):
        if self.history is None or len(self.history) == 0:
            return

        surrogate_names = self.surrogate_space.get_hyperparameter_names()
        for obs in self.history.observations:
            config = obs.config
            if config.configuration_space == self.surrogate_space:
                continue

            config_dict = config.get_dictionary()
            filtered_values = {name: config_dict[name] for name in surrogate_names if name in config_dict}
            new_config = Configuration(self.surrogate_space, values=filtered_values)
            if hasattr(config, 'origin') and config.origin is not None:
                new_config.origin = config.origin
            obs.config = new_config

        self.history.config_space = self.surrogate_space