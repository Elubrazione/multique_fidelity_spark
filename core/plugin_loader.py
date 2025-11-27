import importlib
import logging
from typing import Type
from .interfaces import TargetSystem

logger = logging.getLogger(__name__)

class PluginLoader:
    @staticmethod
    def load_target_system(system_name: str) -> TargetSystem:
        """
        Dynamically load a target system extension.
        
        Args:
            system_name: The name of the extension (e.g., 'spark').
            
        Returns:
            An instance of the TargetSystem implementation.
            
        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the module does not have a SystemEntry class.
            TypeError: If SystemEntry is not a subclass of TargetSystem.
        """
        module_path = f"extensions.{system_name}"
        try:
            logger.info(f"Attempting to load extension: {module_path}")
            module = importlib.import_module(module_path)
            
            if not hasattr(module, 'SystemEntry'):
                raise AttributeError(f"Module '{module_path}' does not have a 'SystemEntry' class.")
            
            system_class = getattr(module, 'SystemEntry')
            
            if not issubclass(system_class, TargetSystem):
                raise TypeError(f"'{system_name}.SystemEntry' must be a subclass of core.interfaces.TargetSystem")
            
            return system_class()
            
        except ImportError as e:
            logger.error(f"Failed to import extension '{system_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading extension '{system_name}': {e}")
            raise
