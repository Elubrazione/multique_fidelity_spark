from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, runtime_checkable, Generic, TypeVar, Optional

PlanType = TypeVar('PlanType')

class Partitioner(ABC, Generic[PlanType]):
    """Partitioner interface (multi-fidelity data partitioning)
    
    The partitioner is responsible for partitioning evaluation data/tasks into different fidelity levels.
    
    Methods:
        get_fidelity_subsets: Get data subsets at different fidelity levels
        refresh_plan: Refresh partition plan
        mark_plan_dirty: Mark plan as dirty (needs refresh)
        is_plan_dirty: Check if plan is dirty
    """
    
    def __init__(self):
        self._latest_plan: Optional[PlanType] = None
        self._plan_dirty: bool = True
    
    @abstractmethod
    def get_fidelity_subsets(self) -> Dict[float, Any]:
        """Get data subsets at different fidelity levels
        
        Returns:
            Dictionary, keys are fidelity levels (float), values are data subsets at that level.
            The specific type of data subsets is decided by the implementation.
        
        Example:
            >>> subsets = partitioner.get_fidelity_subsets()
            >>> low_fidelity_data = subsets[0.1]
            >>> high_fidelity_data = subsets[1.0]
        """
        pass
    
    @abstractmethod
    def refresh_plan(self, *, force: bool = False) -> PlanType:
        """Refresh partition plan
        
        Args:
            force: Whether to force refresh
        
        Returns:
            Partition plan object, the specific type is decided by the implementation
        """
        pass
    
    def mark_plan_dirty(self) -> None:
        self._plan_dirty = True
    
    def is_plan_dirty(self) -> bool:
        return self._plan_dirty
    
    @property
    def latest_plan(self) -> Optional[PlanType]:
        return self._latest_plan


@runtime_checkable
class PartitionPlanProtocol(Protocol):
    """Protocol for partition plan objects
    
    This protocol defines the minimum interface that partition plans should implement.
    Specific implementations (like Spark's PartitionPlan) can extend this with additional fields.
    
    Note: This is a Protocol, so any object with these attributes will satisfy the type checker.
    """
    fidelity_subsets: Dict[float, Any]
    metadata: Dict[str, Any]


class NoOpPartitioner(Partitioner[Dict[str, Any]]):
    def get_fidelity_subsets(self) -> Dict[float, Any]:
        return {}
    
    def refresh_plan(self, *, force: bool = False) -> Dict[str, Any]:
        return {}


__all__ = ['Partitioner', 'PartitionPlanProtocol', 'NoOpPartitioner']