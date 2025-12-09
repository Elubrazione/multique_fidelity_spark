import numpy as np
import copy
from openbox import logger
from openbox.utils.history import Observation
from ConfigSpace import ConfigurationSpace

from .base import BaseAdvisor
from .surrogate import build_surrogate
from .acq_function import get_acq
from .acq_function.optimizer import create_local_random_optimizer


class BO(BaseAdvisor):
    def __init__(self, config_space: ConfigurationSpace, method_id='unknown',
                surrogate_type='prf', acq_type='ei', task_id='test',
                ws_strategy='none', tl_strategy='none',
                random_kwargs={}, **kwargs):
        super().__init__(config_space, task_id=task_id, method_id=method_id,
                        ws_strategy=ws_strategy, tl_strategy=tl_strategy,
                        **random_kwargs, **kwargs)

        self.acq_type = acq_type
        self.surrogate_type = surrogate_type
        self.norm_y = False if 'wrk' in self.acq_type else True
        
        self.surrogate = build_surrogate(surrogate_type=self.surrogate_type, config_space=self.surrogate_space, rng=self.rng,
                                            transfer_learning_history=self.compressor.transform_source_data(self.source_hpo_data),
                                            extra_dim=0, norm_y=self.norm_y)
        self.acq_func = get_acq(acq_type=self.acq_type, model=self.surrogate)
        
        self.acq_optimizer = create_local_random_optimizer(
            acquisition_function=self.acq_func,
            config_space=self.sample_space,
            sampling_strategy=self.sampling_strategy,
            rand_prob=self.rand_prob,
            rng=self.rng,
            candidate_multiplier=3.0
        )

    def warm_start(self):
        if self.ws_strategy == 'none' or self.tl_strategy == 'none':
            return
        self._update_ws_info()
        num_evaluated = self.get_num_evaluated_exclude_default()
        logger.info("Begin using warm starter: %s" % type(self.warm_starter).__name__)
        
        ini_configs = self.warm_starter.get_initial_configs(
            source_hpo_data=self.source_hpo_data,
            source_hpo_data_sims=self.source_hpo_data_sims,
            init_num=self.init_num,
            compressor=self.compressor,
            num_evaluated=num_evaluated,
            sampling_func=lambda n: self.sample_random_configs(
                n, excluded_configs=self.history.configurations
            )
        )
        self.ini_configs = ini_configs + self.ini_configs
        
        logger.info("Successfully use warm starter %d configurations with %s!" 
                % (len(self.ini_configs), type(self.warm_starter).__name__))
    
    def _update_ws_info(self):
        warm_str_list = []
        for idx, sim in self.source_hpo_data_sims:
            task_str = self.source_hpo_data[idx].task_id
            warm_str = "%s: sim%.4f" % (task_str, sim)
            warm_str_list.append(warm_str)
        if 'warm_start' not in self.history.meta_info:
            self.history.meta_info['warm_start'] = [warm_str_list]
        else:
            self.history.meta_info['warm_start'].append(warm_str_list)
        logger.debug("Updated warm start meta info: %s" % warm_str_list)


    def sample(self, batch_size=1, prefix=''):
        # exclude default configuration from count
        num_evaluated_exclude_default = self.get_num_evaluated_exclude_default()

        if len(self.ini_configs) == 0 or num_evaluated_exclude_default < self.init_num:
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
                    config = self.sample_random_configs(1, excluded_configs=self.history.configurations)[0]
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
                        config = self.sample_random_configs(1, excluded_configs=self.history.configurations)[0]
                        config.origin = prefix + 'BO Warm Start Random Sample'
                        logger.debug("Regular BO: take random config: %s" % config.origin)
                    batch.append(config)
            
            self.compressor.unproject_points(batch)
            return batch
        
        X = self._get_surrogate_config_array()
        Y = self.history.get_objectives()
        self.surrogate.train(X, Y)
        self.acq_func.update(
            context=self.surrogate.get_acquisition_context(
                history=self.history
            )
        )
        challengers = self.acq_optimizer.maximize(
            observations=self._convert_observations_to_surrogate_space(
                self.history.observations
            ),
            num_points=2000
        )

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
        

        for config in challengers:
            if len(batch) >= batch_size:
                break
            if config in self.history.configurations:
                continue
            if not self.validation_strategy.is_valid(config):
                config = self.validation_strategy.sanitize(config)
            if self.validation_strategy.is_valid(config):
                config.origin = prefix + 'BO Acquisition ' + str(config.origin)
                batch.append(config)
                logger.debug("BOHB/MFES: take config from acquisition function: %s" % config.origin)
        # Fill any remaining with random samples
        if len(batch) < batch_size:
            random_configs = self.sample_random_configs(
                batch_size - len(batch),
                excluded_configs=self.history.configurations + batch
            )
            for config in random_configs:
                config.origin = prefix + 'BO Acquisition Random Sample'
                logger.debug("BOHB/MFES: take random config: %s" % config.origin)
                batch.append(config)
        
        self.compressor.unproject_points(batch)
        return batch
    
    
    def _get_surrogate_config_array(self):
        X_surrogate = []
        for obs in self.history.observations:
            surrogate_config = self.compressor.convert_config_to_surrogate_space(obs.config)
            X_surrogate.append(surrogate_config.get_array())
        return np.array(X_surrogate)
    
    def _convert_observations_to_surrogate_space(self, observations):
        converted_observations = []
        for obs in observations:
            surrogate_config = self.compressor.convert_config_to_surrogate_space(obs.config)
            converted_obs = Observation(
                config=surrogate_config,
                objectives=obs.objectives,
                constraints=obs.constraints,
                trial_state=obs.trial_state,
                elapsed_time=obs.elapsed_time,
                extra_info=obs.extra_info
            )
            converted_observations.append(converted_obs)
        return converted_observations
    
    def update_compression(self, history):
        updated = self.compressor.update_compression(history)
        if updated:
            logger.info("Compression updated, re-compressing space and retraining surrogate model")
            # compressor.update_compression already updated the spaces
            self.surrogate_space = self.compressor.surrogate_space
            self.sample_space = self.compressor.sample_space
            
            # Rebuild surrogate model with new space dimensions
            self.surrogate = build_surrogate(
                surrogate_type=self.surrogate_type,
                config_space=self.surrogate_space,
                rng=self.rng,
                transfer_learning_history=self.compressor.transform_source_data(self.source_hpo_data),
                extra_dim=0,
                norm_y=self.norm_y
            )
            logger.info(f"Successfully rebuilt the surrogate model ({self.surrogate_type}) with {len(self.surrogate_space.get_hyperparameters())} dimensions")
            
            self.sampling_strategy = self.compressor.get_sampling_strategy()
            
            self.acq_optimizer = create_local_random_optimizer(
                acquisition_function=self.acq_func,
                config_space=self.sample_space,
                sampling_strategy=self.sampling_strategy,
                rand_prob=self.rand_prob,
                rng=self.rng,
                candidate_multiplier=3.0
            )
            
            X_surrogate = self._get_surrogate_config_array()
            Y = self.history.get_objectives()
            self.surrogate.train(X_surrogate, Y)
            self.acq_func.update(
                context=self.surrogate.get_acquisition_context(
                    history=self.history
                )
            )
            
            logger.info("Surrogate model retrained after compression update")
            return True
        
        return False
    