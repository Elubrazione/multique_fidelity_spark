import time
from openbox.utils.config_space import get_one_exchange_neighbourhood

from .base import AcquisitionOptimizer

class LocalSearch(AcquisitionOptimizer):
    def __init__(self, acquisition_function, config_space, rng=None, max_steps=None,
                 n_steps_plateau_walk=10):
        super().__init__(acquisition_function, config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk

    def _maximize(self, observations, num_points: int, **kwargs):
        init_points = self._get_initial_points(
            num_points, observations)

        acq_configs = []
        # Start N local search from different random start points
        for start_point in init_points:

            acq_val, configuration = self._one_iter(start_point, **kwargs)

            configuration.origin = "Local Search"
            acq_configs.append((acq_val, configuration))
            # acq_configs.append((self.acquisition_function([start_point])[0][0], start_point))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        return acq_configs[: num_points]

    def _get_initial_points(self, num_points, observations):

        if len(observations) == 0:
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = [observation.config for observation in observations]
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _one_iter(self, start_point, **kwargs):
        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent])[0][0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:
            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                print(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(incumbent, seed=self.rng.seed())

            for i, neighbor in enumerate(all_neighbors):
                if i > 50:
                    break

                s_time = time.time()
                acq_val = self.acquisition_function([neighbor])[0][0]
                neighbors_looked_at += 1
                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent:
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_steps is not None and local_search_steps == self.max_steps):
                break

        return acq_val_incumbent, incumbent
