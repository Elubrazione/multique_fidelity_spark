from typing import Any, Dict, Optional
import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox import logger
from core.interfaces import TargetSystem
from manager.config_manager import ConfigManager
from .evaluator import SparkEvaluatorManager
from .utils import resolve_runtime_metrics
from Optimizer.utils import load_space_from_json

class SparkTargetSystem(TargetSystem):
    def initialize(self, config_manager: ConfigManager, **kwargs):
        self.config_manager = config_manager
        self.system_config = config_manager.system_config
        
        # Fallback to old config structure if system_config is empty
        if not self.system_config:
             try:
                 self.system_config = config_manager.local_cluster
             except KeyError:
                 self.system_config = {}
    
    def get_evaluator_manager(self, config_space: ConfigurationSpace, **kwargs) -> Any:
        # Extract Spark-specific arguments
        evaluators = kwargs.pop('evaluators', None)
        
        return SparkEvaluatorManager(
            config_space=config_space,
            config_manager=self.config_manager,
            evaluators=evaluators,
            **kwargs
        )

    def get_default_config_space(self) -> ConfigurationSpace:
        return load_space_from_json(self.config_manager.config_space)

    def get_meta_feature(self, task_id: str, **kwargs) -> Any:
        if kwargs.get('test_mode', False):
            logger.info("Using test mode meta feature")
            return np.random.rand(34)

        spark_log_dir = self.system_config.get('spark_log_dir')
        
        if not spark_log_dir:
            logger.warning("Spark log dir not configured, cannot resolve runtime metrics.")
            raise ValueError("Spark log dir not configured")
            
        return resolve_runtime_metrics(spark_log_dir=spark_log_dir)

    def on_component_update(self, component_name: str, component: Any):
        if component_name == 'scheduler':
             # Avoid circular import at module level
             from manager.task_manager import TaskManager
             tm = TaskManager.instance()
             partitioner = tm.get_sql_partitioner()
             if partitioner is not None and hasattr(partitioner, 'mark_plan_dirty'):
                logger.warning("Marking SQL plan dirty due to component change")
                partitioner.mark_plan_dirty()

class SystemEntry(SparkTargetSystem):
    pass
