import numpy as np
from typing import List, Tuple, Any, Optional
from ConfigSpace import ConfigurationSpace

from .base import AcquisitionOptimizer
from .generator import SearchGenerator, LocalSearchGenerator, RandomSearchGenerator
from .selector import StrategySelector, FixedSelector, ProbabilisticSelector
from ..base import AcquisitionFunction


class CompositeOptimizer(AcquisitionOptimizer):
    """Composite Optimizer
    
    use the strategy pattern to combine multiple search strategies:
    1. use StrategySelector to select a strategy
    2. strategy generates candidate configurations
    3. batch evaluate all candidates' acquisition value
    4. select the best num_points configurations
    
    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        acquisition function
    config_space : ConfigurationSpace
        configuration space
    strategies : List[SearchGenerator]
        strategy list
    selector : StrategySelector
        strategy selector, if not provided, use FixedSelector(0)
    rng : np.random.RandomState
        random number generator, if not provided, use np.random.RandomState(42)
    candidate_multiplier : float, default=3.0
        candidate multiplier, generate num_points * candidate_multiplier candidates
        then select the best num_points configurations through acquisition function
        
    Examples
    --------
    >>> from .generator import LocalSearchGenerator, RandomSearchGenerator
    >>> from .selector import ProbabilisticSelector
    >>> 
    >>> # create strategy
    >>> local = LocalSearchGenerator(max_neighbors=50)
    >>> random = RandomSearchGenerator()
    >>> 
    >>> # create selector (85% local search, 15% random search)
    >>> selector = ProbabilisticSelector([0.85, 0.15])
    >>> 
    >>> # create composite optimizer
    >>> optimizer = CompositeOptimizer(
    ...     acquisition_function=acq_func,
    ...     config_space=config_space,
    ...     strategies=[local, random],
    ...     selector=selector
    ... )
    >>> 
    >>> # use
    >>> best_configs = optimizer.maximize(runhistory, num_points=10)
    """
    
    def __init__(self,
                 acquisition_function: AcquisitionFunction,
                 config_space: ConfigurationSpace,
                 strategies: List[SearchGenerator],
                 selector: StrategySelector = FixedSelector(0),
                 rng: np.random.RandomState = np.random.RandomState(42),
                 candidate_multiplier: float = 3.0):
        super().__init__(acquisition_function, config_space, rng)
        
        if not strategies:
            raise ValueError("At least one strategy is required")
        
        self.strategies = strategies
        self.selector = selector
        self.candidate_multiplier = candidate_multiplier
    
    def _maximize(self, observations: List[Any], num_points: int, **kwargs) -> List[Tuple]:
        """use strategy to generate candidates, then batch evaluate and select the best num_points configurations
        
        process:
        1. use selector to select a strategy
        2. strategy generates candidates (generate num_points * candidate_multiplier candidates)
        3. batch evaluate all candidates' acquisition value
        4. select the best num_points configurations
        
        Parameters
        ----------
        observations : List[Any]
            historical observations
        num_points : int
            number of configurations to return
        **kwargs
            additional arguments passed to acquisition function
            
        Returns
        -------
        List[Tuple[float, Configuration]]
            list of (acquisition_value, configuration) pairs
        """
        strategy = self.selector.select(self.strategies, self.iter_id)
        sorted_observations = self._prepare_observations_for_strategy(observations, strategy, **kwargs)
        n_candidates = int(num_points * self.candidate_multiplier)
        candidates = strategy.generate(
            observations=sorted_observations,
            num_points=n_candidates,
            rng=self.rng,
            **kwargs
        )
        
        if not candidates:
            raise RuntimeError(
                f"Strategy {type(strategy).__name__} generated no candidates. "
                "This should not happen if sampling_strategy is properly configured."
            )

        scores = self._evaluate_batch(candidates, **kwargs)
        sorted_indices = np.argsort(scores)[::-1][: num_points]
        results = [(scores[idx], candidates[idx]) for idx in sorted_indices]
        self.iter_id += 1
        
        return results
    
    def _prepare_observations_for_strategy(self, observations: List[Any], strategy, **kwargs) -> List[Any]:
        """Prepare observations for strategy by sorting by acquisition value
        
        This avoids duplicate evaluation: optimizer evaluates once, strategy uses sorted result.
        
        Parameters
        ----------
        observations : List[Observation]
            Historical observations
        strategy : SearchGenerator, LocalSearchGenerator, RandomSearchGenerator
            The strategy that will use these observations
        **kwargs
            Additional arguments for acquisition function
            
        Returns
        -------
        List[Observation]
            Observations sorted by acquisition value (descending)
        """
        # Only sort if strategy needs it (e.g., LocalSearchGenerator)
        # For RandomSearchGenerator, sorting is not needed
        if isinstance(strategy, LocalSearchGenerator) and observations:
            configs = [obs.config for obs in observations]
            sorted_configs_with_acq = self._sort_configs_by_acq_value(configs, **kwargs)
            config_to_obs = {obs.config: obs for obs in observations}
            sorted_observations = []
            for _, config in sorted_configs_with_acq:
                if config in config_to_obs:
                    sorted_observations.append(config_to_obs[config])
            return sorted_observations
        return observations
    
    def reset(self):
        self.iter_id = 0
        if hasattr(self.selector, 'reset'):
            self.selector.reset()


def create_local_random_optimizer(
    acquisition_function: AcquisitionFunction,
    config_space: ConfigurationSpace,
    sampling_strategy,
    rand_prob: float = 0.15,
    rng: np.random.RandomState = np.random.RandomState(42),
    candidate_multiplier: float = 3.0,
    local_max_neighbors: int = 50,
    local_n_start_points: int = 10
) -> CompositeOptimizer:
    """Create a CompositeOptimizer with LocalSearch and RandomSearch strategies
    
    This is a convenience function that creates a CompositeOptimizer configured
    with LocalSearchGenerator and RandomSearchGenerator, using ProbabilisticSelector
    to choose between them based on the given probability.
    
    Parameters
    ----------
    acquisition_function : AcquisitionFunction
        Acquisition function to maximize
    config_space : ConfigurationSpace
        Configuration space
    sampling_strategy
        Sampling strategy provided by compressor
    rand_prob : float, default=0.15
        Probability of using random search (1 - rand_prob for local search)
    rng : np.random.RandomState, optional
        Random number generator
    candidate_multiplier : float, default=3.0
        Multiplier for number of candidates to generate
    local_max_neighbors : int, default=50
        Maximum neighbors to explore in local search
    local_n_start_points : int, default=10
        Number of start points for local search
        
    Returns
    -------
    CompositeOptimizer
        Configured optimizer instance
        
    Examples
    --------
    >>> optimizer = create_local_random_optimizer(
    ...     acquisition_function=acq_func,
    ...     config_space=config_space,
    ...     sampling_strategy=sampling_strategy,
    ...     rand_prob=0.15
    ... )
    """    
    local_strategy = LocalSearchGenerator(
        max_neighbors=local_max_neighbors,
        n_start_points=local_n_start_points,
        sampling_strategy=sampling_strategy
    )
    random_strategy = RandomSearchGenerator(
        sampling_strategy=sampling_strategy
    )
    
    # create selector (1 - rand_prob for local, rand_prob for random)
    selector = ProbabilisticSelector(
        probabilities=[1 - rand_prob, rand_prob],
        rng=rng
    )
    
    optimizer = CompositeOptimizer(
        acquisition_function=acquisition_function,
        config_space=config_space,
        strategies=[local_strategy, random_strategy],
        selector=selector,
        rng=rng,
        candidate_multiplier=candidate_multiplier
    )
    
    return optimizer

