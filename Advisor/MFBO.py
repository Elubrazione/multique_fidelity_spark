import copy
from typing import List
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import Configuration, ConfigurationSpace

from .BO import BO
from .utils import build_observation


class MFBO(BO):
    def __init__(self, config_space: ConfigurationSpace, method_id='unknown',
                surrogate_type='prf', acq_type='ei', task_id='test',
                ws_strategy='none', ws_args={'init_num': 5},
                tl_strategy='none', tl_args={'topk': 5},
                random_kwargs={}, **kwargs):
        super().__init__(config_space,
                        surrogate_type=surrogate_type, acq_type=acq_type,
                        task_id=task_id, method_id=method_id,
                        ws_strategy=ws_strategy, ws_args=ws_args,
                        tl_strategy=tl_strategy, tl_args=tl_args,
                        random_kwargs=random_kwargs,
                        **kwargs)

        self.history_list: List[History] = list()  # 低精度组的 history -> List[History]
        self.resource_identifiers = list()
        if self.source_hpo_data is not None and not self.surrogate_type.startswith('mfes'):
            self.history_list = self.compressor.transform_source_data(self.source_hpo_data)
            self.resource_identifiers = [-1] * len(self.source_hpo_data)  # 占位符


    def sample(self, batch_size):
        # exclude default configuration from count
        num_evaluated_exclude_default = self.get_num_evaluated_exclude_default()
        if len(self.ini_configs) == 0 and num_evaluated_exclude_default < self.init_num:
            logger.info("Begin to warm start!!!!")
            self.warm_start()

        logger.info("num_evaluated_exclude_default: [%d], init_num: [%d], init_configs: [%d]" \
                    % (num_evaluated_exclude_default, self.init_num, len(self.ini_configs)))

        # For MFES: full-fidelity initialization (tl_args['topk'] rounds), take 1 config at a time
        if num_evaluated_exclude_default < self.init_num:
            logger.info("MFBO: full-fidelity initialization, take 1 config at a time for tl_args['topk'] rounds")
            batch = []
            # MFES: full-fidelity warm start, take 1 config at a time for tl_args['topk'] rounds
            take_from_ws = min(1, batch_size, len(self.ini_configs))
            for _ in range(take_from_ws):
                if len(self.ini_configs) > 0:
                    # keep queue order: take from front (same as original MFBO)
                    config = self.ini_configs[-1]
                    self.ini_configs.pop()
                    config.origin = 'MFBO Warm Start ' + str(config.origin)
                    logger.debug("MFBO: take config from warm start: %s" % config.origin)
                    batch.append(config)
            # Fill remaining with random if needed
            remaining = batch_size - len(batch)
            for _ in range(remaining):
                config = self.sample_random_configs(self.sample_space, 1,
                                                    excluded_configs=self.history.configurations + batch)[0]
                config.origin = 'MFBO Warm Start Random Sample'
                logger.debug("MFBO: take random config: %s" % config.origin)
                batch.append(config)
            logger.info(f"[MFBO] Initialization: take {take_from_ws} from warm start, {remaining} random, remaining WS: {len(self.ini_configs)}")
            return batch

        # After initialization, update multi-fidelity trials and use parent BO's sampling logic
        self.surrogate.update_mf_trials(self.history_list)
        # self.surrogate.build_source_surrogates()

        # Already initialized after full-fidelity initialization, use parent BO's sampling logic
        return super().sample(batch_size=batch_size, prefix='MF')

    def update(self, config, results, resource_ratio=1, update=True):
        if not update:
            return
        
        obs = build_observation(config, results)
        self._cache_low_dim_config(config, obs)
        
        resource_ratio = round(resource_ratio, 5)
        if resource_ratio != 1:
            if resource_ratio not in self.resource_identifiers:
                self.resource_identifiers.append(resource_ratio)
                history = History(task_id="res%.5f_%s" % (resource_ratio, self.task_id),
                                  num_objectives=self.history.num_objectives,
                                  num_constraints=self.history.num_constraints,
                                  config_space=self.config_space)
                self.history_list.append(history)
            self.history_list[self.get_resource_index(resource_ratio)].update_observation(obs)
        else:
            self.history.update_observation(obs)


    def get_resource_index(self, resource_ratio):
        rounded_ratio = round(resource_ratio, 5)
        try:
            return self.resource_identifiers.index(rounded_ratio)
        except ValueError:
            return -1
