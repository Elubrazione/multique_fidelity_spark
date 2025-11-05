import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from Advisor.utils import build_observation
from config import ConfigManager
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
                cp_args,
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
        self.cp_args = cp_args
        self.scheduler_kwargs = method_args.get('scheduler_kwargs')
        self.logger_kwargs = logger_kwargs
        self.random_kwargs = method_args.get('random_kwargs')

        self.config_space = config_space
        
        self.historical_tasks: List[History] = []
        self.historical_meta_features: List[np.ndarray] = []
        self.current_task_history: Optional[History] = None
        
        self.similar_tasks_cache: List[Tuple[int, float]] = []

        self._scheduler: Optional[object] = None
        self._sql_partitioner: Optional[object] = None
        self._planner: Optional[object] = None
        

    def calculate_meta_feature(self, eval_func: Callable, task_id: str = "default", **kwargs):
        """
        Get runtime metric for current task by running default config and parsing latest Spark log.
        
        Args:
            eval_func: Evaluator function (ExecutorManager)
            task_id: Task ID
            kwargs: Additional keyword arguments
                - resume: Resume from a previous task
                - test_mode: Test mode
        """
        # skip meta_feature collecting and default config evaluation
        if kwargs.get('resume', None) is not None:
            self.current_task_history = History.load_json(
                                        filename=kwargs.get('resume'),
                                        config_space=self.config_space)
            logger.info(f"Current task history: {self.current_task_history.objectives}")
            logger.info(f"Loaded current task history from {kwargs.get('resume')}")
            return

        logger.info("Computing current task meta feature using default config...")
        self.current_task_history = History(task_id=task_id, config_space=self.config_space)
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")



    def update_history_meta_info(self, meta_info: dict):
        """
        Update meta information of historical tasks.
        
        Args:
            meta_info: Meta information of historical tasks
        """
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

    def _mark_sql_plan_dirty(self) -> None:
        if self._sql_partitioner is not None:
            logger.warning("Marking SQL plan dirty")
            self._sql_partitioner.mark_plan_dirty()

    def get_ws_args(self) -> Dict[str, Any]:
        return dict(self.ws_args)

    def get_tl_args(self) -> Dict[str, Any]:
        return dict(self.tl_args)

    def get_cp_args(self) -> Dict[str, Any]:
        return dict(self.cp_args)

    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        return dict(self.scheduler_kwargs)

    def get_logger_kwargs(self) -> Dict[str, Any]:
        return dict(self.logger_kwargs)

    def get_random_kwargs(self) -> Dict[str, Any]:
        return dict(self.random_kwargs)



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