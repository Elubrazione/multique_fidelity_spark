from typing import Any, Dict, Optional, Tuple
import re
import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox import logger
from core.interfaces import TargetSystem
from manager.config_manager import ConfigManager
from .evaluator import SparkEvaluatorManager
from .utils import resolve_runtime_metrics
from Optimizer.utils import load_space_from_json


def parse_task_config(task_id: str, target: str) -> Tuple[int, int, int, int]:
    # Parse task_id: format is like '64u256n3'
    # Pattern: {cpu}u{memory}n{nodes}
    pattern = r'(\d+)u(\d+)n(\d+)'
    match = re.search(pattern, task_id)
    
    if match:
        cpu_cores = int(match.group(1))
        memory_gb = int(match.group(2))
        num_nodes = int(match.group(3))
    else:
        logger.warning(f"Cannot parse task_id '{task_id}', using default values")
        cpu_cores = 64
        memory_gb = 256
        num_nodes = 3
    
    db_pattern = r'(\d+)g'
    db_match = re.search(db_pattern, target, re.IGNORECASE)
    
    if db_match:
        database_size_gb = int(db_match.group(1))
    else:
        logger.warning(f"Cannot parse database size from target '{target}', using default value")
        database_size_gb = 100
    
    logger.info(f"Parsed task config: CPU={cpu_cores}, Memory={memory_gb}GB, "
                f"Nodes={num_nodes}, Database={database_size_gb}GB")
    return cpu_cores, memory_gb, num_nodes, database_size_gb


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
            base_feature = np.random.rand(34)
        else:
            spark_log_dir = self.system_config.get('spark_log_dir')
            
            if not spark_log_dir:
                logger.warning("Spark log dir not configured, cannot resolve runtime metrics.")
                raise ValueError("Spark log dir not configured")
                
            base_feature = resolve_runtime_metrics(spark_log_dir=spark_log_dir)
        
        db_name = self.config_manager.database
        cpu_cores, memory_gb, num_nodes, database_size_gb = parse_task_config(task_id, db_name)
        
        additional_features = np.array([cpu_cores, memory_gb, num_nodes, database_size_gb], dtype=float)
        meta_feature = np.concatenate([base_feature, additional_features])
        
        logger.info(f"Meta feature shape: {meta_feature.shape}, "
                   f"base_dim=34, additional=[CPU={cpu_cores}, Mem={memory_gb}, "
                   f"Nodes={num_nodes}, DB={database_size_gb}]")
        
        return meta_feature

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
