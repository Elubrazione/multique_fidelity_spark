import copy
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from openbox import logger
from ConfigSpace import Configuration
from dimensio import Compressor


class WarmStarter(ABC):    
    @abstractmethod
    def get_initial_configs(
        self,
        source_hpo_data: List,
        source_hpo_data_sims: List[Tuple[int, float]],
        init_num: int,
        compressor: Compressor,
        num_evaluated: int,
        sampling_func: Callable
    ) -> List[Configuration]:
        pass


class NoWarmStart(WarmStarter):    
    def get_initial_configs(
        self,
        source_hpo_data: List,
        source_hpo_data_sims: List[Tuple[int, float]],
        init_num: int,
        compressor: Compressor,
        num_evaluated: int,
        sampling_func: Callable
    ) -> List[Configuration]:
        return []


class BestConfigsWarmStart(WarmStarter):
    """Warm start strategy using best configurations from source tasks
    
    Selects the best k configurations from each similar task as initial configurations.
    """
    
    def __init__(self, ws_strategy: str, ws_topk: int = 1):
        self.ws_strategy = ws_strategy
        self.ws_topk = ws_topk
    
    def get_initial_configs(
        self,
        source_hpo_data: List,
        source_hpo_data_sims: List[Tuple[int, float]],
        init_num: int,
        compressor: Compressor,
        num_evaluated: int,
        sampling_func
    ) -> List[Configuration]:
        """Strategies:

        1. Selects the best ws_topk configurations from each similar task
        2. Organize by ranking: task1_config1, task2_config1, ..., task1_config2, task2_config2, ...
        3. If configurations are insufficient, supplement with random sampling
        """
        if not source_hpo_data or not source_hpo_data_sims:
            logger.info("No source data for warm start")
            return []
        
        warm_str_list = []
        for idx, sim in source_hpo_data_sims:
            task_str = source_hpo_data[idx].task_id
            warm_str = f"{task_str}: sim{sim:.4f}"
            warm_str_list.append(warm_str)
        
        logger.info(f"Warm start from similar tasks: {warm_str_list}")
        
        source_observations = []
        for idx, sim in source_hpo_data_sims:
            sim_obs = copy.deepcopy(source_hpo_data[idx].observations)
            sim_obs = sorted(sim_obs, key=lambda x: x.objectives[0])
            top_obs = sim_obs[: min(self.ws_topk, len(sim_obs))]
            source_observations.append((idx, top_obs))
            logger.info(f"Source task {source_hpo_data[idx].task_id}: selected top {len(top_obs)} configurations")
        
        ini_list = []
        target_length = init_num * self.ws_topk
        
        for rank in range(self.ws_topk):
            if len(ini_list) + num_evaluated >= target_length:
                break
            for idx, top_obs in source_observations:
                if len(ini_list) + num_evaluated >= target_length:
                    break
                if rank < len(top_obs):
                    config_warm_old = top_obs[rank].config
                    config_warm = compressor.conver_config_to_sample_space(config_warm_old)
                    config_warm.origin = (f"{self.ws_strategy}_{source_hpo_data[idx].task_id}_"
                                            f"{sim:.4f}_rank{rank}")
                    ini_list.append(config_warm)
                    logger.info(f"Warm start configuration from task {source_hpo_data[idx].task_id}, "
                            f"rank {rank}, objective: {top_obs[rank].objectives[0]}, {config_warm.origin}")
        
        ini_list = ini_list[::-1]        
        while len(ini_list) + num_evaluated < target_length:
            config = sampling_func(1)[0]
            config.origin = f"{self.ws_strategy} Warm Start Random Sample"
            logger.debug(f"Warm start configuration from random sample: {config.origin}")
            ini_list = [config] + ini_list
        return ini_list

def create_warm_starter(
    ws_strategy: str,
    tl_strategy: str,
    method_id: str,
    ws_args: dict
) -> WarmStarter:
    if ws_strategy == 'none' or tl_strategy == 'none':
        return NoWarmStart()
    ws_topk = int(ws_args.get('topk', 1)) if 'BOHB' in method_id or 'MFES' in method_id else 1

    if ws_strategy == 'best_all':
        return BestConfigsWarmStart(ws_strategy, ws_topk)
    else:
        return NoWarmStart()
