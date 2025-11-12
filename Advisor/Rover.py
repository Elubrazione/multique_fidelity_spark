import numpy as np
import copy
from openbox import logger
from ConfigSpace import Configuration, ConfigurationSpace

from .base import BaseAdvisor
from .utils import build_my_surrogate, build_my_acq_func, is_valid_spark_config, sanitize_spark_config
from .acq_optimizer.local_random import InterleavedLocalAndRandomSearch


class Rover(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, method_id='unknown',
                surrogate_type='prf', acq_type='ei', task_id='test',
                ws_strategy='none', ws_args={'init_num': 5},
                tl_strategy='none', tl_args={'topk': 5}, cp_args={},
                random_kwargs={}, **kwargs):
        super().__init__(config_space, task_id=task_id, method_id=method_id,
                        ws_strategy=ws_strategy, ws_args=ws_args,
                        tl_strategy=tl_strategy, tl_args=tl_args, cp_args=cp_args,
                        **random_kwargs, **kwargs)
        self.acq_type = acq_type
        # self.surrogate_type = surrogate_type
        self.surrogate_type = 'mce_gp'  # force to use RGPE
        self.norm_y = False if 'wrk' in self.acq_type else True
        
        # 这里需要保证build_my_surrogate建立的代理模型是rpge, 也就是需要func_str.startswith('mce')
        # 应该是'mce_gp'
        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.surrogate_space, rng=self.rng,
                                            transfer_learning_history=self.compressor.transform_source_data(self.source_hpo_data),
                                            extra_dim=0, norm_y=self.norm_y)
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate)
        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func,
                                                            rand_prob=self.rand_prob, rand_mode=self.rand_mode, rng=self.rng,
                                                            config_space=self.sample_space)

    def warm_start(self):
        # no warm start for rover situation
        return
       

    def sample(self, batch_size=1, prefix=''):
        # exclude default configuration from count
        num_evaluated_exclude_default = self.get_num_evaluated_exclude_default()

        if len(self.ini_configs) == 0 and num_evaluated_exclude_default < self.init_num:
            logger.info("Begin to warm start!")
            self.warm_start()

        logger.info("num_evaluated_exclude_default: [%d], init_num: [%d], init_configs: [%d]" \
                    % (num_evaluated_exclude_default, self.init_num, len(self.ini_configs)))

        # Check if called from MFBO (MFES uses MFBO, which handles initialization itself)
        # If prefix is 'MF', it means we're called from MFBO after initialization phase
        is_called_from_mfbo = prefix == 'MF'
        is_bohb = 'BOHB' in self.method_id
        
        # Initialization phase: only handle if not called from MFBO (BOHB uses BO directly)
        if num_evaluated_exclude_default < self.init_num and not is_called_from_mfbo:
            batch = []
            if is_bohb:
                # BOHB: full-fidelity warm start, take 1 config at a time for tl_args['topk'] rounds
                logger.info("BOHB: full-fidelity warm start, take 1 config at a time for tl_args['topk'] rounds")
                take_from_ws = min(1, batch_size, len(self.ini_configs))
                for _ in range(take_from_ws):
                    if len(self.ini_configs) > 0:
                        config = self.ini_configs[-1]
                        self.ini_configs.pop()
                        config.origin = prefix + 'BO Warm Start ' + str(config.origin)
                        logger.debug("BOHB: take config from warm start: %s" % config.origin)
                        batch.append(config)
                remaining = batch_size - len(batch)
                for _ in range(remaining):
                    config = self.sample_random_configs(self.sample_space, 1,
                                                        excluded_configs=self.history.configurations)[0]
                    config.origin = prefix + 'BO Warm Start Random Sample'
                    logger.debug("BOHB: take random config: %s" % config.origin)
                    batch.append(config)
            else:
                # Regular BO: take configs one by one during initialization
                logger.info("Regular BO: take configs one by one during initialization")
                for _ in range(batch_size):
                    if len(self.ini_configs) > 0:
                        config = self.ini_configs[-1]
                        self.ini_configs.pop()
                        config.origin = prefix + 'BO Warm Start ' + str(config.origin)
                        logger.debug("Regular BO: take config from warm start: %s" % config.origin)
                    else:
                        config = self.sample_random_configs(self.sample_space, 1,
                                                            excluded_configs=self.history.configurations)[0]
                        config.origin = prefix + 'BO Warm Start Random Sample'
                        logger.debug("Regular BO: take random config: %s" % config.origin)
                    batch.append(config)
            return batch
        
        # After initialization, use acquisition function for sampling
        X = self.history.get_config_array()
        Y = self.history.get_objectives()

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.surrogate_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
            
        self.surrogate.train(X, Y)

        incumbent_value = self.history.get_incumbent_value()
        self.acq_func.update(model=self.surrogate, eta=incumbent_value, num_data=len(self.history))

        observations = self.history.observations
        challengers = self.acq_optimizer.maximize(observations=observations, num_points=2000)
    
        _is_valid = is_valid_spark_config
        _sanitize = sanitize_spark_config

        batch = []
        # For BOHB/MFES in low-fidelity stage: take q configs from warm start, then fill rest with acquisition function
        # Note: MFES calls this through MFBO.sample() -> super().sample() after initialization, prefix='MF'
        if (is_bohb or is_called_from_mfbo) and len(self.ini_configs) > 0:
            # q: number of warm start configs to take in low-fidelity stage (default: 2, same as MFBO)
            logger.info("BOHB/MFES: take configs from warm start in low-fidelity stage")
            q = min(2, batch_size, len(self.ini_configs))
            for _ in range(q):
                config = self.ini_configs[-1]
                self.ini_configs.pop()
                config.origin = prefix + 'BO Warm Start ' + str(config.origin)
                logger.debug("BOHB/MFES: take config from warm start: %s" % config.origin)
                batch.append(config)
            logger.info(f"[BOHB/MFES] Take {q} configurations from warm start in low-fidelity stage, remaining: {len(self.ini_configs)}")
        
        # Fill remaining with acquisition function samples
        for config in challengers.challengers:
            if len(batch) >= batch_size:
                break
            if config in self.history.configurations:
                continue
            if not _is_valid(config):
                config = _sanitize(config)
            if _is_valid(config):
                config.origin = prefix + 'BO Acquisition'
                batch.append(config)
                logger.debug("BOHB/MFES: take config from acquisition function: %s" % config.origin)
        # Fill any remaining with random samples
        if len(batch) < batch_size:
            random_configs = self.sample_random_configs(
                self.sample_space, batch_size - len(batch),
                excluded_configs=self.history.configurations + batch
            )
            for config in random_configs:
                config.origin = prefix + 'BO Acquisition Random Sample'
                logger.debug("BOHB/MFES: take random config: %s" % config.origin)
                batch.append(config)
        return batch