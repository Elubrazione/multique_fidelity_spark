import numpy as np
from math import log, ceil
from typing import List, Tuple, Optional
from openbox import logger


class FidelityScheduler:
    """
    Multi-Fidelity Scheduler for BOHB optimization.
    
    This class provides the core scheduling logic:
    - Determines how many configurations to run (n)
    - Determines how much resource to allocate (r)
    - Manages bracket and stage structure
    - Calculates elimination counts
    """
    
    def __init__(self, R: int = 9, eta: int = 3, 
                 initial_n_list: Optional[List[int]] = None,
                 initial_r_list: Optional[List[int]] = None,
                 fixed_initial: bool = True):
        self.R = R
        self.eta = eta
        self.fixed_initial = fixed_initial
        self.initial_n_list = initial_n_list or []
        self.initial_r_list = initial_r_list or []
        
        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.R))
        self.B = (self.s_max + 1) * self.R
        self.s_values = list(reversed(range(self.s_max + 1)))

        self.fidelity_levels = [int(x) for x in np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)]
        
        logger.info("FidelityScheduler: run %d brackets with fidelity levels %s. s_max = [%d]. R = [%d], eta = [%d]" 
                    % (len(self.s_values), self.get_fidelity_levels(), self.s_max, self.R, self.eta))
    
    def get_bracket_params(self, s: int, num_nodes: int = 1) -> Tuple[int, int]:
        """
        Get bracket parameters for a given bracket index.
        
        Args:
            s: Bracket index (0 to s_max)
            num_nodes: Number of Spark nodes (for multi-node scaling)
            
        Returns:
            Tuple of (n_configs, n_resource)
        """
        if self.fixed_initial and len(self.initial_n_list) and len(self.initial_r_list):
            n_configs = self.initial_n_list[0] * num_nodes
            n_resource = self.initial_r_list[0]
        else:
            n_configs = int(ceil(self.B / self.R / (s + 1) * self.eta ** s))
            n_resource = int(self.R * self.eta ** (-s))
        
        return n_configs, n_resource
    
    def get_stage_params(self, s: int, stage: int, num_nodes: int = 1) -> Tuple[int, int]:
        """
        Get stage parameters within a bracket.
        
        Args:
            s: Bracket index
            stage: Stage index within bracket (0 to s)
            num_nodes: Number of Spark nodes (for multi-node scaling)
            
        Returns:
            Tuple of (n_configs, n_resource)
        """
        n_configs, base_resource = self.get_bracket_params(s, num_nodes)
        
        if self.fixed_initial and len(self.initial_n_list) and len(self.initial_r_list):
            n_configs_stage = self.initial_n_list[stage] * num_nodes
            n_resource_stage = self.initial_r_list[stage]
        else:
            n_configs_stage = int(n_configs * self.eta ** (-stage))
            n_resource_stage = int(base_resource * self.eta ** stage)
        
        return n_configs_stage, n_resource_stage
    
    def calculate_resource_ratio(self, n_resource: int) -> float:
        """
        Calculate resource ratio from resource allocation.
        
        Args:
            n_resource: Resource allocation
            
        Returns:
            Resource ratio (0.0 to 1.0)
        """
        return round(float(n_resource / self.R), 5)
    
    def get_elimination_count(self, s: int, stage: int) -> int:
        """
        Get number of configurations to eliminate after a stage.
        
        Args:
            s: Bracket index
            stage: Stage index
            
        Returns:
            Number of configurations to keep
        """
        n_configs, _ = self.get_stage_params(s, stage)
        
        if self.fixed_initial and len(self.initial_n_list):
            # Use fixed elimination
            if stage < s and stage + 1 < len(self.initial_n_list):
                return self.initial_n_list[stage + 1]
            else:
                return 0
        else:
            # Standard Successive Halving elimination
            return int(n_configs / self.eta)
    
    def get_fidelity_levels(self) -> List[float]:
        return self.fidelity_levels
    
    def eliminate_candidates(self, candidates: List, perfs: List, s: int, stage: int) -> Tuple[List, List]:
        """
        Eliminate candidates based on performance (Successive Halving logic).
        
        Args:
            candidates: List of candidates
            perfs: List of performances
            s: Bracket index
            stage: Current stage
            
        Returns:
            Tuple of (remaining_candidates, remaining_perfs)
        """
        reduced_num = self.get_elimination_count(s, stage)
        
        indices = np.argsort(perfs)
        sorted_candidates = [candidates[i] for i in indices]
        sorted_perfs = [perfs[i] for i in indices]

        return sorted_candidates[:reduced_num], sorted_perfs[:reduced_num]
