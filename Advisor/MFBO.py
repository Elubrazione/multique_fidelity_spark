from typing import List
from openbox import logger
from openbox.utils.history import History, Observation
from ConfigSpace import Configuration, ConfigurationSpace

from .BO import BO
from .utils import build_observation


class MFBO(BO):
    def __init__(self, config_space: ConfigurationSpace,
                 surrogate_type='prf', acq_type='ei', task_id='test', meta_feature=None,
                 ws_strategy='none', ws_args={'init_num': 5},
                 tl_args={'topk': 5}, source_hpo_data=None,
                 cprs_strategy='none', space_history = None, cp_topk=35, range_config_space=None,
                 ep_args=None, ep_strategy='none', expert_params=[],
                 enable_range_compression=False, range_compress_data_path=None,
                 safe_flag=False, seed=42, rng=None, rand_prob=0.15, rand_mode='ran', **kwargs):
        super().__init__(config_space, meta_feature=meta_feature,
                         surrogate_type=surrogate_type, acq_type=acq_type, task_id=task_id,
                         ws_strategy=ws_strategy, ws_args=ws_args,
                         tl_args=tl_args, source_hpo_data=source_hpo_data,
                         ep_args=ep_args, ep_strategy=ep_strategy, expert_params=expert_params,
                         cprs_strategy=cprs_strategy, space_history=space_history, cp_topk=cp_topk, range_config_space=range_config_space,
                         safe_flag=safe_flag, seed=seed, rng=rng, rand_prob=rand_prob, rand_mode=rand_mode,
                         enable_range_compression=enable_range_compression, range_compress_data_path=range_compress_data_path, **kwargs)

        self.history_list: List[History] = list()  # 低精度组的 history -> List[History]
        self.resource_identifiers = list()
        if self.source_hpo_data is not None and not surrogate_type.startswith('mfse'):
            self.history_list = self._source_hpo_data_in_new_space(self.source_hpo_data)
            self.resource_identifiers = [-1] * len(self.source_hpo_data)  # 占位符


    def warm_start(self):
        if self.ws_strategy == 'none':
            return
        ws_topk = int(self.ws_args.get('topk', max(1, self.init_num if hasattr(self, 'init_num') else 5)))
        if ws_topk <= 0:
            logger.error("ws_topk <= 0")
            return

        if not self.source_hpo_data or not self.source_hpo_data_sims:
            return

        num_source_tasks = len(self.source_hpo_data)
        if num_source_tasks == 1:
            best_src_idx = self.source_hpo_data_sims[0][0]
            src_history = self.source_hpo_data[best_src_idx]
            sim_obs = list(src_history.observations)
            if not sim_obs:
                return

            sim_obs.sort(key=lambda x: x.objectives[0])
            take = min(ws_topk, len(sim_obs))
            top_obs = sim_obs[:take]

            # generate center out index order
            def center_out_order(n: int):
                order = []
                m = n // 2
                indices = list(range(1, n + 1))
                def push(i):
                    if 1 <= i <= n and i not in order:
                        order.append(i)
                push(m)
                step = 1
                while len(order) < n:
                    push(m + step)  # m+1, m+2, ...
                    if len(order) >= n:
                        break
                    push(m - step)  # m-1, m-2, ...
                    step += 1
                for i in indices:
                    if i not in order:
                        order.append(i)
                return [i - 1 for i in order]

            reorder = center_out_order(take)

            ini_list = []
            for idx in reorder:
                old_conf = top_obs[idx].config
                logger.info("Warm start configuration, objective: %s" % (top_obs[idx].objectives[0]))
                new_conf = Configuration(self.config_space, values={
                    name: old_conf[name] for name in self.sample_space.get_hyperparameter_names()
                })
                new_conf.origin = f"mfbo_ws:{src_history.task_id}"
                ini_list.append(new_conf)

            self.ini_configs = ini_list
        else:
            ini_list = []
            
            source_observations = []
            for i, src_history in enumerate(self.source_hpo_data):
                sim_obs = list(src_history.observations)
                if sim_obs:
                    sim_obs.sort(key=lambda x: x.objectives[0])
                    source_observations.append(sim_obs)
                    logger.info("Source task %d (%s): %d observations" % (i, src_history.task_id, len(sim_obs)))
                else:
                    source_observations.append([])
                    logger.warning("Source task %d (%s): no observations" % (i, src_history.task_id))
            
            # 轮询选择：a.topk_1, b.topk_1, c.topk_1, a.topk_2, b.topk_2, c.topk_2, ...
            config_count = 0
            rank = 0
            while config_count < ws_topk:
                if all(rank >= len(obs_list) for obs_list in source_observations):
                    break
                
                for task_idx, obs_list in enumerate(source_observations):
                    if config_count >= ws_topk:
                        break
                    if rank < len(obs_list):
                        old_conf = obs_list[rank].config
                        logger.info("Warm start configuration from task %d, rank %d, objective: %s" % 
                                  (task_idx, rank, obs_list[rank].objectives[0]))
                        new_conf = Configuration(self.config_space, values={
                            name: old_conf[name] for name in self.sample_space.get_hyperparameter_names()
                        })
                        new_conf.origin = f"mfbo_ws:{self.source_hpo_data[task_idx].task_id}_rank{rank}"
                        ini_list.append(new_conf)
                        config_count += 1
                
                rank += 1

            self.ini_configs = ini_list
        logger.info("Successfully warm start %d configurations with %s!" % (len(self.ini_configs), self.ws_strategy))


    """
    采样(使用ini_configs进行热启动和普通采样)
    以及安全约束 (40轮后阈值为 0.85 * incumbent_value)
    """
    def samples(self, batch_size):
        num_config_evaluated = len(self.history)
        if len(self.ini_configs) == 0 and (
            (self.init_num > 0 and num_config_evaluated < self.init_num)
            or (self.init_num == 0 and num_config_evaluated == 0)
        ):
            logger.info("Begin to warm start!!!!")
            self.warm_start()

        # take two from warm start queue
        batch = []
        take_from_ws = min(2, batch_size, len(self.ini_configs))
        for _ in range(take_from_ws):
            # keep queue order: take from front
            batch.append(self.ini_configs.pop(0))
        logger.info("Take %d configurations from warm start, length of ini_configs after sampling: %d" % (take_from_ws, len(self.ini_configs)))

        remaining = batch_size - len(batch)
        if remaining == 0:
            return batch

        if num_config_evaluated == 0:
            batch.extend(
                self.sample_random_configs(
                    self.sample_space,
                    remaining,
                    excluded_configs=self.history.configurations + batch,
                )
            )
            logger.info("Random sample %d configurations" % (len(batch) - take_from_ws))
            return batch

        self.surrogate.update_mf_trials(self.history_list)
        self.surrogate.build_source_surrogates()
        candidates = super().sample(return_list=True)

        idx = 0
        while len(batch) < batch_size and idx < len(candidates):
            conf = candidates[idx]
            idx += 1
            if conf not in batch and conf not in self.history.configurations:
                batch.append(conf)

        # when candidates are not enough, random sample to fill the batch
        remaining = batch_size - len(batch)
        if remaining > 0:
            batch.extend(
                self.sample_random_configs(
                    self.sample_space, remaining, excluded_configs=self.history.configurations + batch
                )
            )
        logger.info("Random sample %d configurations when candidates are not enough" % (remaining))

        return batch

    def update(self, config, results, resource_ratio=1):
        obs = build_observation(config, results, last_context=self.last_context)
        resource_ratio = round(resource_ratio, 5)
        if resource_ratio != 1:
        # if resource_ratio ==  round(float(1/64), 5):
            if resource_ratio not in self.resource_identifiers:
                self.resource_identifiers.append(resource_ratio)
                history = History(task_id="res%.5f_%s" % (resource_ratio, self.task_id),
                                  num_objectives=self.history.num_objectives,
                                  num_constraints=self.history.num_constraints,
                                  config_space=self.sample_space)
                self.history_list.append(history)
                self.early_stop_histories.append([])
            self.history_list[self.get_resource_index(resource_ratio)].update_observation(obs)
            # self.history.update_observation(obs)
        else:
            self.history.update_observation(obs)


    def get_resource_index(self, resource_ratio):
        rounded_ratio = round(resource_ratio, 5)
        try:
            return self.resource_identifiers.index(rounded_ratio)
        except ValueError:
            return -1
