import numpy as np
from openbox.utils.config_space.util import convert_configurations_to_array


class AcquisitionOptimizer:
    def __init__(self, acquisition_function, config_space, rng):
        self.acq = acquisition_function
        self.config_space = config_space
        if rng is None:
            self.rng = np.random.RandomState(seed=42)
        else:
            self.rng = rng
        self.iter_id = 0


    def maximize(self, observations, num_points, **kwargs):
        return [t[1] for t in self._maximize(observations, num_points, **kwargs)]

    def _maximize(self, observations, num_points: int, **kwargs):
        raise NotImplementedError()

    def _sort_configs_by_acq_value(self, configs, **kwargs):
        acq_values = self.acquisition_function(configs, **kwargs)
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))
        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]

    def acquisition_function(self, configs, **kwargs):
        X = convert_configurations_to_array(configs)
        return self.acq(X, convert=False, **kwargs)