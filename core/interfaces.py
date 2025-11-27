from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from ConfigSpace import ConfigurationSpace
from manager.config_manager import ConfigManager

class TargetSystem(ABC):
    """
    Abstract base class for target systems (e.g., Spark, PyTorch, Database).
    This interface defines the contract that any extension must fulfill to be compatible
    with the tuning framework.
    """

    @abstractmethod
    def initialize(self, config_manager: ConfigManager, **kwargs):
        """
        Initialize the target system with configuration.
        
        Args:
            config_manager: The configuration manager instance.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def get_evaluator_manager(self, config_space: ConfigurationSpace, **kwargs) -> Any:
        """
        Return the configured EvaluatorManager for this system.
        
        Args:
            config_space: The configuration space to be used.
            **kwargs: Additional arguments needed for evaluator creation.
            
        Returns:
            An instance compatible with the framework's EvaluatorManager expectation.
        """
        pass

    @abstractmethod
    def get_default_config_space(self) -> ConfigurationSpace:
        """
        Return the default configuration space for this target system.
        
        Returns:
            ConfigurationSpace object.
        """
        pass

    @abstractmethod
    def get_meta_feature(self, task_id: str, **kwargs) -> Any:
        """
        Calculate or retrieve meta-features for the given task.
        
        Args:
            task_id: The identifier for the current task.
            **kwargs: Additional arguments (e.g., resume flag, test_mode).
            
        Returns:
            Meta-features (usually np.ndarray or list).
        """
        pass

    def on_component_update(self, component_name: str, component: Any):
        """
        Callback when a registered component is updated.
        
        Args:
            component_name: The name of the component (e.g., 'scheduler').
            component: The component instance.
        """
        pass
