import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from Advisor.utils import map_source_hpo_data, build_observation
from config_manager import ConfigManager
from utils.spark import resolve_runtime_metrics


class TaskManager:
    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, 
                config_space: ConfigurationSpace,
                config_manager: ConfigManager,
                logger_kwargs,
                **kwargs):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        
        self._config_manager = config_manager
        self.history_dir = config_manager.history_dir
        self.spark_log_dir = config_manager.spark_log_dir
        self.similarity_threshold = config_manager.similarity_threshold

        method_args = config_manager.method_args
        self.ws_args = method_args.get('ws_args')
        self.tl_args = method_args.get('tl_args')
        self.scheduler_kwargs = method_args.get('scheduler_kwargs')
        self.logger_kwargs = logger_kwargs
        self.random_kwargs = method_args.get('random_kwargs')

        self.config_space = config_space
        
        self.historical_tasks: List[History] = []
        self.historical_meta_features: List[np.ndarray] = []
        self.current_task_history: Optional[History] = None
        self.current_meta_feature: Optional[np.ndarray] = None
        
        self.similar_tasks_cache: List[Tuple[int, float]] = []

        self._scheduler: Optional[object] = None
        self._sql_partitioner: Optional[object] = None
        self._planner: Optional[object] = None
        self._compressor: Optional[object] = None
        
        self._load_historical_tasks()


    def _load_historical_tasks(self):
        if not os.path.exists(self.history_dir):
            logger.warning(f"History directory {self.history_dir} does not exist.")
            return
        
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.history_dir, filename)
                history = History.load_json(filename=filepath, config_space=self.config_space)
                self.historical_tasks.append(history)
                meta_feature = history.meta_info.get('meta_feature')
                if meta_feature is not None:
                    logger.debug(f"Got meta_feature: {meta_feature} from {filename}")
                    self.historical_meta_features.append(np.array(meta_feature))
                else:
                    logger.warning(f"No meta_feature found in {filename}")     
        logger.info(f"Loaded {len(self.historical_tasks)} historical tasks from {self.history_dir}")


    def calculate_meta_feature(self, eval_func: Callable, task_id: str = "default", **kwargs):
        # skip meta_feature collecting and default config evaluation
        if kwargs.get('resume', None) is not None:
            self.current_task_history = History.load_json(
                                        filename=kwargs.get('resume'),
                                        config_space=self.config_space)
            self.current_meta_feature = np.array(self.current_task_history.meta_info.get('meta_feature'))
            logger.info(f"Current task meta feature: {self.current_meta_feature}")
            logger.info(f"Current task history: {self.current_task_history.objectives}")
            logger.info(f"Loaded current task history from {kwargs.get('resume')}")
            return

        # use default config writen in spark_default.conf
        default_config = self.config_space.get_default_configuration()
        default_config.origin = 'Default Configuration'
        result = eval_func(config=default_config, resource_ratio=1.0)
    
        if kwargs.get('test_mode', False):
            logger.info("Using test mode meta feature")
            self.current_meta_feature = np.random.rand(34)
            self.current_task_history = History(task_id=task_id, config_space=self.config_space,
                                                meta_info={'meta_feature': self.current_meta_feature.tolist()})
            self.current_task_history.update_observation(build_observation(default_config, result))
            self._update_similarity()
            return
        
        logger.info("Computing current task meta feature using default config...")

        metrics = resolve_runtime_metrics(spark_log_dir=self.spark_log_dir)

        self.current_meta_feature = metrics
        self.current_task_history = History(task_id=task_id, config_space=self.config_space,
                                            meta_info={'meta_feature': self.current_meta_feature.tolist()})
        self.current_task_history.update_observation(build_observation(default_config, result))
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")

        self._update_similarity()


    def update_history_meta_info(self, meta_info: dict):
        self.current_task_history.meta_info.update(meta_info)

    def register_scheduler(self, scheduler):
        # ensure scheduler is only registered once
        if self._scheduler is not None:
            logger.error("Scheduler already registered")
            return
        self._scheduler = scheduler
        logger.info(f"Registered scheduler: {self._scheduler}")
        self._mark_sql_plan_dirty()

    def get_scheduler(self) -> Optional[object]:
        return self._scheduler

    def register_sql_partitioner(self, partitioner) -> None:
        if self._sql_partitioner is not None and self._sql_partitioner is not partitioner:
            logger.warning("SQLPartitioner already registered; replacing with new instance")
        self._sql_partitioner = partitioner
        logger.info(f"Registered SQLPartitioner: {self._sql_partitioner}")
        self._mark_sql_plan_dirty()

    def get_sql_partitioner(self):
        return self._sql_partitioner

    def register_planner(self, planner) -> None:
        if self._planner is not None and self._planner is not planner:
            logger.warning("Planner already registered; replacing with new instance")
        self._planner = planner
        logger.info(f"Registered Planner: {self._planner}")

    def get_planner(self):
        return self._planner

    def register_compressor(self, compressor) -> None:
        if self._compressor is not None and self._compressor is not compressor:
            logger.warning("Compressor already registered; replacing with new instance")
        self._compressor = compressor
        logger.info(f"Registered Compressor: {self._compressor}")

    def get_compressor(self):
        return self._compressor

    def get_cp_string(self, config_space) -> str:
        return self._config_manager.get_cp_string(config_space)
    
    def generate_task_id(self, task_name: str, method_id: str, ws_strategy: str,
                        tl_strategy: str, scheduler_type: str, config_space,
                        rand_mode: str = 'ran', seed: int = 42) -> str:
        """Generate complete task ID string from ConfigManager"""
        return self._config_manager.generate_task_id(
            task_name, method_id, ws_strategy, tl_strategy, 
            scheduler_type, config_space, rand_mode, seed
        )

    def _mark_sql_plan_dirty(self) -> None:
        if self._sql_partitioner is not None:
            logger.warning("Marking SQL plan dirty")
            self._sql_partitioner.mark_plan_dirty()

    def get_ws_args(self) -> Dict[str, Any]:
        return dict(self.ws_args)

    def get_tl_args(self) -> Dict[str, Any]:
        return dict(self.tl_args)

    def get_cp_args(self, config_space=None) -> Dict[str, Any]:
        if config_space is None:
            config_space = self.config_space
        return self._config_manager.get_cp_args(config_space)

    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        return dict(self.scheduler_kwargs)

    def get_logger_kwargs(self) -> Dict[str, Any]:
        return dict(self.logger_kwargs)

    def get_random_kwargs(self) -> Dict[str, Any]:
        return dict(self.random_kwargs)

    def _update_similarity(self):
        self.similar_tasks_cache = map_source_hpo_data(
            target_his=self.current_task_history,
            source_hpo_data=self.historical_tasks,
            config_space=self.config_space,
            **self.ws_args,
        )
        
        filtered_sims = [(idx, sim) for idx, sim in self.similar_tasks_cache if sim >= self.similarity_threshold]
        self.similar_tasks_cache = filtered_sims
        
        logger.info(f"Updated similarity: {len(self.similar_tasks_cache)} tasks above threshold {self.similarity_threshold}")
        self._mark_sql_plan_dirty()


    def get_similar_tasks(self, topk: Optional[int] = None) -> Tuple[List[History], List[Tuple[int, float]]]:
        """
        Get filtered similar tasks.
        
        Args:
            topk: Top k similar tasks to return

        Returns:
            Tuple of (filtered_histories, similarity_scores)
        """
        if not self.similar_tasks_cache:
            return [], []
        if topk is None:
            topk = self.tl_args.get('topk') or len(self.similar_tasks_cache)
        topk = min(topk, len(self.similar_tasks_cache))
        filtered_histories = []
        filtered_sims = []
        for i in range(topk):
            idx, sim = self.similar_tasks_cache[i]
            filtered_histories.append(self.historical_tasks[idx])
            filtered_sims.append((i, sim))
            logger.info(f"Similar task {i}: {self.historical_tasks[idx].task_id} (similarity: {sim:.3f})")
        sims_sum = sum(sim for _, sim in filtered_sims)
        filtered_sims = [(idx, sim / sims_sum) for idx, sim in filtered_sims]
        logger.debug(f"Normalized similarities: {filtered_sims}")
        return filtered_histories, filtered_sims


    def update_current_task_history(self, config, results):
        if not self.current_task_history:
            logger.warning("Current task not initialized, cannot update history")
            return
            
        obs = build_observation(config, results)
        self.current_task_history.update_observation(obs)
        self._update_similarity()
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")


    def get_current_task_history(self) -> Optional[History]:
        return self.current_task_history