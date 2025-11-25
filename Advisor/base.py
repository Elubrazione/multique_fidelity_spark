import numpy as np
import json as js
from typing import Optional
from openbox import logger
from dimensio import Compressor
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write.json import write

from .utils import build_observation
from .validation import ValidationStrategy, SparkConfigValidation
from .warm_start import create_warm_starter


class BaseAdvisor:
    def __init__(self, config_space: ConfigurationSpace, method_id='unknown',
                task_id='test', ws_strategy='none',
                tl_strategy='none',
                seed=42, rand_prob=0.15, rand_mode='ran',
                validation_strategy: Optional[ValidationStrategy] = None,
                **kwargs):
        # Delay import to avoid circular dependency
        from manager import TaskManager
        
        self.task_id = task_id
        self._logger_kwargs = kwargs.get('_logger_kwargs', None)

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rand_prob = rand_prob
        self.rand_mode = rand_mode
        
        self.validation_strategy = validation_strategy \
                                    if validation_strategy is not None \
                                    else SparkConfigValidation()
        logger.info(f"Using validation strategy: {type(self.validation_strategy).__name__}")
        
        self.task_manager = TaskManager.instance()
        self.ws_args = self.task_manager.get_ws_args()
        self.tl_args = self.task_manager.get_tl_args()
        
        self.compressor: Compressor = self.task_manager.get_compressor()
        if self.compressor is None:
            raise RuntimeError("Compressor must be initialized and registered to TaskManager before creating Advisor")

        self.source_hpo_data, self.source_hpo_data_sims = self.task_manager. \
                                                        get_similar_tasks(topk=self.tl_args['topk']) \
                                                        if tl_strategy != 'none' else ([], [])
        if tl_strategy != 'none':
            # pass source_hpo_data as space_history and similarities
            self.surrogate_space, self.sample_space = self.compressor.compress_space(
                space_history=self.source_hpo_data,
                source_similarities={idx: sim for idx, sim in self.source_hpo_data_sims}
            )
        else:
            self.surrogate_space, self.sample_space = self.compressor.compress_space()
        
        self.config_space = config_space
        self.config_space.seed(self.seed)
        self.sample_space.seed(self.seed)
        self.surrogate_space.seed(self.seed)
        self.ini_configs = list()
        
        self.sampling_strategy = self.compressor.get_sampling_strategy()
        logger.info(f"Using sampling strategy: {type(self.sampling_strategy).__name__}")
        
        self.warm_starter = create_warm_starter(
            ws_strategy=ws_strategy,
            tl_strategy=tl_strategy,
            method_id=method_id,
            ws_args=self.ws_args
        )
        logger.info(f"Using warm starter: {type(self.warm_starter).__name__}")

        meta_feature = {}
        meta_feature['random'] = {'seed': seed, 'rand_prob': rand_prob, 'rand_mode': rand_mode}
        meta_feature['space'] = {'original': js.loads(write(self.config_space)),
                                'dimension': js.loads(write(self.surrogate_space)),
                                'range': js.loads(write(self.sample_space))}
        meta_feature['compressor'] = self.compressor.get_compression_summary()
        self.task_manager.update_history_meta_info(meta_feature)

        self.ws_strategy = ws_strategy
        self.tl_strategy = tl_strategy
        self.method_id = method_id
        self.history = self.task_manager.current_task_history

        # init_num is equal to the number of topk similar tasks if use transfer learning,
        # otherwise it is the number of initial configurations for warm start
        self.init_num = self.ws_args['init_num'] if tl_strategy == 'none' else self.tl_args['topk']

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

    def sample_random_configs(self, num_configs=1, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        configs = []

        trials = 0
        max_trials = max(100, num_configs * 20)
        while len(configs) < num_configs and trials < max_trials:
            trials += 1
            sampled = self.sampling_strategy.sample(1)[0]
            
            if not self.validation_strategy.is_valid(sampled):
                sampled = self.validation_strategy.sanitize(sampled)
                if not self.validation_strategy.is_valid(sampled):
                    continue

            if sampled not in configs and sampled not in excluded_configs:
                sampled.origin = "Random Sample!"
                configs.append(sampled)

        return configs

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