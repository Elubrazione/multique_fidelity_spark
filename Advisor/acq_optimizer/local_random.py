import numpy as np
from openbox.acq_optimizer.random_configuration_chooser import ChooserProb
from openbox.utils.config_space.util import convert_configurations_to_array

from .base import AcquisitionOptimizer
from .random import RandomSearch
from .local import LocalSearch
from ..acq_function.weighted_rank import WeightedRank


class InterleavedLocalAndRandomSearch(AcquisitionOptimizer):

    def __init__(self, acquisition_function, config_space, rng=None, max_steps=None,
                 n_steps_plateau_walk=10, n_sls_iterations=50, rand_prob=0.15, rand_mode='ran',
                 sampling_strategy=None):
        super().__init__(acquisition_function, config_space, rng)

        self.local_search = LocalSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.random_search = RandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )
        self.n_sls_iterations = n_sls_iterations
        self.random_chooser = ChooserProb(prob=rand_prob, rng=rng)
        self.rand_mode = rand_mode
        self.sampling_strategy = sampling_strategy

    def _maximize(self, observations, num_points: int, **kwargs):
        raise NotImplementedError

    def maximize(self, observations, num_points, random_configuration_chooser=None, **kwargs):
        need_acq = kwargs.get('need_acq', False)
        next_configs_by_local_search = self.local_search._maximize(
            observations, self.n_sls_iterations, **kwargs)

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            observations, num_points - len(next_configs_by_local_search),
            _sorted=True)

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of openbox. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
                next_configs_by_random_search_sorted
                + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        if not need_acq:
            next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        if isinstance(self.acq, WeightedRank):
            next_configs_by_weighted_rank_sorted = self._sort_configs_by_acq_value(next_configs_by_acq_value, only_target=False)
            next_configs_by_acq_value = next_configs_by_weighted_rank_sorted if need_acq else [_[1] for _ in next_configs_by_weighted_rank_sorted]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space,
                                     self.random_chooser, self.rand_mode,
                                     sampling_strategy=self.sampling_strategy)
        self.random_chooser.next_smbo_iteration()
        return challengers

class ChallengerList(object):
    def __init__(self, challengers, configuration_space, random_configuration_chooser, rand_mode='ran',
                 sampling_strategy=None):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self.random_configuration_chooser = random_configuration_chooser
        self.rand_mode = rand_mode
        self.sampling_strategy = sampling_strategy

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers):
            raise StopIteration
        else:
            if self.random_configuration_chooser.check(self._iteration):
                if self.rand_mode == 'ran':  # 纯随机
                    if self.sampling_strategy is not None:
                        config = self.sampling_strategy.sample(1)[0]
                    else:
                        config = self.configuration_space.sample_configuration()
                    config.origin = 'Random Search challenger!'
                elif self.rand_mode == 'rs':  # 用random search(sorted)
                    config = self.challengers[self._index]
                    if config.origin == 'Random Search (sorted)':
                        # 已经是了，就再降一级
                        if self.sampling_strategy is not None:
                            config = self.sampling_strategy.sample(1)[0]
                        else:
                            config = self.configuration_space.sample_configuration()
                        config.origin = 'Random Search challenger!'
                    else:
                        while config.origin != 'Random Search (sorted)':
                            self._index += 1
                            config = self.challengers[self._index]
                else:
                    raise ValueError('Invalid rand_mode: %s' % self.rand_mode)
            else:
                config = self.challengers[self._index]
                self._index += 1
            self._iteration += 1
            return config

    def get_challenger_array(self) -> np.ndarray:
        return convert_configurations_to_array(self.challengers)
