import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any, Protocol, List
from dataclasses import dataclass, field

class SurrogateModel(Protocol):
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...


class AcquisitionFunction(ABC):
    def __init__(self, model: SurrogateModel, **kwargs):
        self.model = model
        self.long_name = self.__class__.__name__
    
    @abstractmethod
    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    def __call__(self, X: np.ndarray, convert: bool = True, **kwargs) -> np.ndarray:
        return self._compute(X, **kwargs)
    
    def update(self, context=None, **kwargs) -> None:
        if context is not None:
            self.model = context.get_main_surrogate()
            self._update_from_context(context, **kwargs)
            return
            
    def _update_from_context(self, context, **kwargs) -> None:
        pass


class SingleObjectiveAcquisition(AcquisitionFunction):
    def __init__(self, model: SurrogateModel, **kwargs):
        super().__init__(model, **kwargs)
        self.eta = None
    
    def update(self, context=None, **kwargs) -> None:
        super().update(context=context, **kwargs)
        
        if context is not None:
            target_task = context.get_target_task()
            self.eta = target_task.get_incumbent_value()
            return


class TransferLearningAcquisition(AcquisitionFunction):    
    def __init__(self, model: SurrogateModel, **kwargs):
        super().__init__(model, **kwargs)
        self.source_acq_funcs = []
        self.target_acq_func = None
        self.weights = None
    
    @abstractmethod
    def _combine_acquisitions(self, source_values: np.ndarray, 
                            target_values: np.ndarray) -> np.ndarray:
        pass


@dataclass
class TaskContext:
    surrogate: Any  # SurrogateModel
    history: Any  # HistoryLike
    eta: Any  # incumbent value
    num_data: int
    
    def get_incumbent_value(self):
        return self.eta


@dataclass
class AcquisitionContext:
    tasks: List[TaskContext]
    weights: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = np.array([1.0])
        else:
            self.weights = np.array(self.weights)
        
        if len(self.weights) != len(self.tasks):
            raise ValueError(
                f"Weights length ({len(self.weights)}) must match tasks length ({len(self.tasks)})"
            )
    
    def is_multi_task(self) -> bool:
        return len(self.tasks) > 1
    
    def get_target_task(self) -> TaskContext:
        return self.tasks[-1]
    
    def get_source_tasks(self) -> List[TaskContext]:
        return self.tasks[:-1] if self.is_multi_task() else []
    
    def get_main_surrogate(self):
        return self.get_target_task().surrogate