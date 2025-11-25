from typing import Optional, Dict, Any, Callable, List
from openbox import logger


class ComponentRegistry:
    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._listeners: Dict[str, List[Callable]] = {}
    
    def register(self, name: str, component: Any, replace: bool = False) -> None:
        if name in self._components and not replace:
            logger.error(f"Component '{name}' already registered. Use replace=True to override.")
            return
        
        if name in self._components:
            logger.warning(f"Replacing existing component '{name}'")
        
        self._components[name] = component
        logger.info(f"Registered component '{name}': {component}")
        
        self._notify_listeners(name, component)
    
    def get(self, name: str) -> Optional[Any]:
        return self._components.get(name)
    
    def has(self, name: str) -> bool:
        return name in self._components
    
    def unregister(self, name: str) -> bool:
        if name in self._components:
            del self._components[name]
            logger.info(f"Unregistered component '{name}'")
            return True
        return False
    
    def add_listener(self, component_name: str, callback: Callable[[Any], None]):
        if component_name not in self._listeners:
            self._listeners[component_name] = []
        self._listeners[component_name].append(callback)
    
    def _notify_listeners(self, component_name: str, component: Any):
        if component_name in self._listeners:
            for callback in self._listeners[component_name]:
                try:
                    callback(component)
                except Exception as e:
                    logger.error(f"Error calling listener for '{component_name}': {e}")
    
    def list_components(self) -> List[str]:
        return list(self._components.keys())

