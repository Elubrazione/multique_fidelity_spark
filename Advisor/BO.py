import numpy as np
import copy
from openbox import logger
from openbox.utils.history import Observation
from ConfigSpace import Configuration, ConfigurationSpace

from .base import BaseAdvisor
from .utils import build_my_surrogate, build_my_acq_func, \
    is_valid_spark_config, sanitize_spark_config
from .acq_optimizer.local_random import InterleavedLocalAndRandomSearch


class BO(BaseAdvisor):
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
        self.surrogate_type = surrogate_type
        self.norm_y = False if 'wrk' in self.acq_type else True
        
        self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.surrogate_space, rng=self.rng,
                                            transfer_learning_history=self.compressor.transform_source_data(self.source_hpo_data),
                                            extra_dim=0, norm_y=self.norm_y)
        self.acq_func = build_my_acq_func(func_str=self.acq_type, model=self.surrogate)
        self.acq_optimizer = InterleavedLocalAndRandomSearch(acquisition_function=self.acq_func,
                                                            rand_prob=self.rand_prob, rand_mode=self.rand_mode, rng=self.rng,
                                                            config_space=self.sample_space)

    def warm_start(self):
        # no warm start if ws_strategy or tl_strategy is none
        if self.ws_strategy == 'none' or self.tl_strategy == 'none':
            return
        
        sims = self.source_hpo_data_sims
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

        # warm_start strategy: select the best ws_args['topk'] configurations from each similar task
        # organize configurations by ranking, here K is the number of similar tasks (tl_args['topk'])
        #   task1_config1, task2_config1, task3_config1, ..., task_{K}_config1,
        #   task1_config2, task2_config2, task3_config2, ..., task_{K}_config2,
        #   ...
        #   task1_config{ws_topk}, task2_config{ws_topk}, task3_config{ws_topk}, ..., task_{K}_config{ws_topk},
        
        # For BOHB/MFES: ws_topk = ws_args['topk'], length of ini_configs = self.init_num * ws_topk
        # For others: ws_topk = 1, length of ini_configs = self.init_num
        ws_topk = int(self.ws_args['topk']) if 'BOHB' in self.method_id or 'MFES' in self.method_id else 1

        # prepare sorted configurations for each similar task
        source_observations = []
        for idx, sim in sims:
            sim_obs = copy.deepcopy(self.source_hpo_data[idx].observations)
            sim_obs = sorted(sim_obs, key=lambda x: x.objectives[0])
            # select the best ws_args['topk'] configurations
            top_obs = sim_obs[: min(ws_topk, len(sim_obs))]
            source_observations.append((idx, top_obs))
            logger.info("Source task %s: selected top %d configurations" \
                % (self.source_hpo_data[idx].task_id, len(top_obs)))

        ini_list = []
        target_length = self.init_num * ws_topk if ws_topk > 1 else self.init_num
        num_evaluated_exclude_default = self.get_num_evaluated_exclude_default()
        
        for rank in range(ws_topk):
            if len(ini_list) + num_evaluated_exclude_default >= target_length:
                break
            for idx, top_obs in source_observations:
                if len(ini_list) + num_evaluated_exclude_default >= target_length:
                    break
                if rank < len(top_obs):
                    config_warm_old = top_obs[rank].config
                    # Use compressor to convert config to surrogate space
                    # This handles both projection (if needed) and parameter filtering
                    config_warm = self.compressor.convert_config_to_surrogate_space(config_warm_old)
                    config_warm.origin = self.ws_strategy + "_" + self.source_hpo_data[idx].task_id + "_" + str(sims[idx][1]) + "_rank" + str(rank)
                    ini_list.append(config_warm)
                    logger.info("Warm start configuration from task %s, rank %d, objective: %s, %s" % 
                                (self.source_hpo_data[idx].task_id, rank, top_obs[rank].objectives[0], config_warm.origin))

        # the best configurations should be at the end of the list, so we need to reverse the list
        # the reversed order: task3_config2, ..., task3_config1, task2_config1, task1_config1
        # (the last one is the first one to be used)
        # the usage order: task1_config1, task2_config1, task3_config1, task1_config2, task2_config2
        self.ini_configs = ini_list[::-1] + self.ini_configs

        while len(self.ini_configs) + num_evaluated_exclude_default < target_length:
            config = self.sample_random_configs(self.sample_space, 1,
                                                excluded_configs=self.history.configurations)[0]
            config.origin = self.ws_strategy + " Warm Start Random Sample"
            logger.debug("Warm start configuration from random sample: %s" % config.origin)
            self.ini_configs = [config] + self.ini_configs

        logger.info("Successfully warm start %d configurations with %s!" \
                    % (len(self.ini_configs), self.ws_strategy))


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
            
            if self.compressor.needs_unproject():
                batch = self._unproject_batch(batch)
            
            return batch
        
        X = self._get_surrogate_config_array()
        Y = self.history.get_objectives()

        if self.surrogate_type == 'gpf':
            self.surrogate = build_my_surrogate(func_str=self.surrogate_type, config_space=self.surrogate_space,
                                                rng=self.rng,
                                                transfer_learning_history=self.source_hpo_data,
                                                extra_dim=self.extra_dim, norm_y=self.norm_y)
            logger.info("Successfully rebuild the surrogate model GP!")
            
        self.surrogate.train(X, Y)

        self.acq_func.update(
            model=self.surrogate,
            eta=self.history.get_incumbent_value(),
            num_data=len(self.history)
        )
        challengers = self.acq_optimizer.maximize(
            observations=self._convert_observations_to_surrogate_space(self.history.observations),
            num_points=2000
        )
    
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
        
        if self.compressor.needs_unproject():
            batch = self._unproject_batch(batch)

        return batch
    
    def _unproject_batch(self, batch):
        unprojected_batch = []
        for compressed_config in batch:
            compressed_dict = compressed_config.get_dictionary() if hasattr(compressed_config, 'get_dictionary') else dict(compressed_config)
            unprojected_dict = self.compressor.unproject_point(compressed_config)
            unprojected_config = Configuration(self.config_space, values=unprojected_dict)
            if hasattr(compressed_config, 'origin') and compressed_config.origin:
                unprojected_config.origin = compressed_config.origin
            unprojected_config._low_dim_config = compressed_dict
            unprojected_batch.append(unprojected_config)
        return unprojected_batch
    
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
            self.surrogate_space, self.sample_space = self.compressor.compress_space(history)
            self.acq_optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=self.acq_func,
                rand_prob=self.rand_prob,
                rand_mode=self.rand_mode,
                rng=self.rng,
                config_space=self.sample_space
            )
            if self.surrogate_type == 'gpf':
                self.surrogate = build_my_surrogate(
                    func_str=self.surrogate_type,
                    config_space=self.surrogate_space,
                    rng=self.rng,
                    transfer_learning_history=self.compressor.transform_source_data(self.source_hpo_data),
                    extra_dim=self.extra_dim,
                    norm_y=self.norm_y
                )
                logger.info("Successfully rebuilt the surrogate model GP!")
            
            X_surrogate = self._get_surrogate_config_array()
            Y = self.history.get_objectives()
            self.surrogate.train(X_surrogate, Y)
            
            incumbent_value = self.history.get_incumbent_value()
            self.acq_func.update(model=self.surrogate, eta=incumbent_value, num_data=len(self.history))
            
            logger.info("Surrogate model retrained after compression update")
            return True
        
        return False
    