from abc import ABC, abstractmethod
from typing import List
from ConfigSpace import Configuration


class ValidationStrategy(ABC):
    
    @abstractmethod
    def is_valid(self, config: Configuration) -> bool:
        pass
    
    @abstractmethod
    def sanitize(self, config: Configuration) -> Configuration:
        pass


class NoOpValidation(ValidationStrategy):    
    def is_valid(self, config: Configuration) -> bool:
        return True
    
    def sanitize(self, config: Configuration) -> Configuration:
        return config


class SparkConfigValidation(ValidationStrategy):
    """Spark configuration validation strategy
    
    Validation constraints for Spark configurations:
    - spark.executor.cores >= spark.task.cpus
    - spark.executor.cores >= 1
    - spark.task.cpus >= 1
    """
    
    def _to_dict(self, config):
        try:
            if hasattr(config, 'get_dictionary'):
                return config.get_dictionary()
            return dict(config)
        except Exception:
            return {}
    
    def is_valid(self, config: Configuration) -> bool:
        d = self._to_dict(config)
        try:
            exec_cores = int(float(d.get('spark.executor.cores', 2)))
            task_cpus = int(float(d.get('spark.task.cpus', 1)))
            return exec_cores >= task_cpus and exec_cores >= 1 and task_cpus >= 1
        except Exception:
            return True
    
    def sanitize(self, config: Configuration) -> Configuration:
        try:
            d = self._to_dict(config)
            exec_cores = int(float(d.get('spark.executor.cores', 2)))
            task_cpus = int(float(d.get('spark.task.cpus', 1)))
            
            if exec_cores < 1:
                exec_cores = 1
            if task_cpus < 1:
                task_cpus = 1
            
            if exec_cores < task_cpus:
                config['spark.task.cpus'] = exec_cores
        except Exception:
            pass
        
        return config


class CompositeValidation(ValidationStrategy):
    """Composite validation strategy that combines multiple validation strategies
    
    Applies multiple validation strategies sequentially, and considers configuration valid only if all strategies pass.
    During sanitization, all strategies are applied sequentially.
    """
    
    def __init__(self, strategies: List[ValidationStrategy]):
        self.strategies = strategies
    
    def is_valid(self, config: Configuration) -> bool:
        return all(strategy.is_valid(config) for strategy in self.strategies)
    
    def sanitize(self, config: Configuration) -> Configuration:
        for strategy in self.strategies:
            config = strategy.sanitize(config)
        return config
