from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Generic, TypeVar, Callable
from openbox import logger

PlanType = TypeVar('PlanType')


class Planner(ABC, Generic[PlanType]):
    """Planner interface for multi-fidelity planning
    
    The planner is responsible for generating evaluation plans based on resource ratios,
    deciding how to evaluate at given fidelity levels.
    
    Methods:
        plan: Generate evaluation plan based on resource ratio
        refresh_plan: Refresh the cached plan (optional, for planners with caching)
        _ensure_plan: Ensure plan is available (optional helper for caching)
    """
    
    def __init__(self):
        self._cached_plan: Optional[PlanType] = None
    
    @abstractmethod
    def plan(
        self, 
        resource_ratio: float, 
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Generate evaluation plan based on resource ratio
        
        Args:
            resource_ratio: Resource ratio (multi-fidelity level)
            **kwargs: Other planning parameters
        
        Returns:
            Evaluation plan dictionary, contains information needed for evaluation.
            The specific content is decided by the implementation.
            If cannot generate plan, return None.
        
        Example:
            >>> plan = planner.plan(resource_ratio=0.5)
            >>> if plan:
            ...     # Use plan for evaluation
            ...     pass
        """
        pass
    
    def refresh_plan(self, *, force: bool = False) -> Optional[PlanType]:
        return None
    
    def _ensure_plan(
        self, 
        *, 
        force_refresh: bool = False,
        check_dirty: Optional[Callable[[], bool]] = None,
    ) -> Optional[PlanType]:
        """Ensure plan is available (helper for caching logic)
        
        This helper method implements common caching logic:
        - If force_refresh is True, refresh the plan
        - If cached plan is None, refresh it
        - If check_dirty callback returns True, refresh it
        - Otherwise, return cached plan
        
        Args:
            force_refresh: Whether to force refresh
            check_dirty: Optional callback to check if plan is dirty
                Should return True if plan needs refresh
        
        Returns:
            Plan object, or None if refresh fails
        """
        if force_refresh:
            logger.debug("Force refreshing planner plan")
            return self.refresh_plan(force=True)
        
        if self._cached_plan is None:
            logger.debug("No cached plan, refreshing")
            return self.refresh_plan(force=True)
        
        if check_dirty is not None and check_dirty():
            logger.debug("Planner plan is dirty, refreshing")
            return self.refresh_plan(force=True)
        
        return self._cached_plan


class NoOpPlanner(Planner[None]):
    def plan(self, resource_ratio: float, **kwargs) -> Optional[Dict[str, Any]]:
        return None


__all__ = ['Planner', 'NoOpPlanner']