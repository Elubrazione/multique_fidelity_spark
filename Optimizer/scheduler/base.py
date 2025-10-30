import abc
import numpy as np
from typing import List, Tuple

class BaseScheduler(abc.ABC):
    def __init__(self, num_nodes: int = 1):
        self.num_nodes = num_nodes
        self.fidelity_levels = []

    @abc.abstractmethod
    def get_bracket_params(self) -> Tuple[int, int]:
        pass

    @abc.abstractmethod
    def get_elimination_count(self) -> int:
        pass

    def eliminate_candidates(
        self, candidates: List, perfs: List, **kwargs
    ) -> Tuple[List, List]:
        reduced_num = self.get_elimination_count(**kwargs) * self.num_nodes
        indices = np.argsort(perfs)
        sorted_candidates = [candidates[i] for i in indices]
        sorted_perfs = [perfs[i] for i in indices]
        return sorted_candidates[:reduced_num], sorted_perfs[:reduced_num]

    def get_fidelity_levels(self) -> List[float]:
        return self.fidelity_levels

    @abc.abstractmethod
    def calculate_resource_ratio(self) -> float:
        pass

class FullFidelityScheduler(BaseScheduler):
    def __init__(self, num_nodes: int = 1):
        super().__init__(num_nodes)

    def calculate_resource_ratio(self) -> float:
        return round(float(1.0), 5)

    def get_bracket_params(self) -> Tuple[int, int]:
        return self.num_nodes, 1

    def get_elimination_count(self) -> int:
        return self.num_nodes