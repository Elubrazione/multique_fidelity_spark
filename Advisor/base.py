import numpy as np
import json as js
from openbox import logger
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from .utils import build_observation, is_valid_spark_config, sanitize_spark_config


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace, method_id='unknown',
                task_id='test', ws_strategy='none', ws_args=None,
                tl_strategy='none', tl_args=None,
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
        
        self.compressor = self.task_manager.get_compressor()
        if self.compressor is None:
            raise RuntimeError("Compressor must be initialized and registered to TaskManager before creating Advisor")

        self.source_hpo_data, self.source_hpo_data_sims = self.task_manager.get_similar_tasks(topk=tl_args['topk']) if tl_strategy != 'none' else ([], [])
        if tl_strategy != 'none':
            # Compress space: pass source_hpo_data only if using transfer learning and compressor supports it
            self.surrogate_space, self.sample_space = self.compressor.compress_space(self.source_hpo_data)
        else:
            self.surrogate_space, self.sample_space = self.compressor.compress_space()
        
        self.config_space = config_space
        self.config_space.seed(self.seed)
        self.sample_space.seed(self.seed)
        self.surrogate_space.seed(self.seed)
        self.ini_configs = list()

        meta_feature = {}
        meta_feature['random'] = {'seed': seed, 'rand_prob': rand_prob, 'rand_mode': rand_mode}
        meta_feature['space'] = {'original': js.loads(write(self.config_space)),
                                'dimension': js.loads(write(self.surrogate_space)),
                                'range': js.loads(write(self.sample_space))}
        meta_feature['compressor'] = self.compressor.compression_info
        self.task_manager.update_history_meta_info(meta_feature)

        self.ws_strategy = ws_strategy
        self.ws_args = ws_args
        self.tl_strategy = tl_strategy
        self.tl_args = tl_args        
        self.method_id = method_id
        self.history = self.task_manager.current_task_history
        
        # if self.history is not None:
        #     self.history.config_space = self.config_space

        # init_num is equal to the number of topk similar tasks if use transfer learning,
        # otherwise it is the number of initial configurations for warm start
        self.init_num = ws_args['init_num'] if tl_strategy == 'none' else tl_args['topk']

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

    def _cache_low_dim_config(self, config, obs):
        if not self.compressor.needs_unproject():
            return
        
        if hasattr(config, '_low_dim_config'):
            low_dim_dict = config._low_dim_config
            if obs.extra_info is None:
                obs.extra_info = {}
            obs.extra_info['low_dim_config'] = low_dim_dict
            logger.debug(f"Saved cached low_dim_config to observation (for record)")
        else:
            try:
                low_dim_dict = self.compressor.project_point(config)
                if low_dim_dict:
                    if obs.extra_info is None:
                        obs.extra_info = {}
                    obs.extra_info['low_dim_config'] = low_dim_dict
                    logger.debug(f"Computed and saved low_dim_config to observation (for record)")
            except Exception as e:
                logger.debug(f"Could not cache low-dim config for observation: {e}")
    
    def update(self, config, results, **kwargs):
        if not kwargs.get('update', True):
            return
        obs = build_observation(config, results)
        self._cache_low_dim_config(config, obs)
        self.history.update_observation(obs)