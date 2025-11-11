import os
import time
import json
import numpy as np
from typing import Union
from copy import deepcopy
from ConfigSpace import ConfigurationSpace, Configuration
from sklearn.ensemble import RandomForestRegressor

from openbox import Advisor, Observation, History
from openbox.utils.config_space.util import convert_configurations_to_array

from openbox import logger
from .mtgp import MultiTaskGP
from .utils import build_observation


class Rover:
    default_openbox_kwargs = dict(
        surrogate_type='gp',
        acq_optimizer_type='local_random',
    )
    default_rfr_kwargs = dict(
        n_estimators=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=0,
        n_jobs=1,
    )

    def __init__(self, config_space: ConfigurationSpace,
                 sa_fraction=0.6,
                 sa_rounds=2,
                 sa_iterations=10,
                 num_objs=1,
                 num_constraints=0,
                 rfr_kwargs: dict = None,
                 openbox_kwargs: dict = None,
                 **kwargs):
                 
        self._logger_kwargs = kwargs.get('_logger_kwargs', None)
        
        from task_manager import TaskManager
        self.task_manager = TaskManager.instance()

        self.config_space = config_space
        self.sample_condition = None
        self.original_config_space = config_space
        self.source_hpo_data, _ = self.task_manager.get_similar_tasks(topk=1)
        self.source_hpo_data = self.source_hpo_data[0]

        self.rfr_kwargs = deepcopy(self.default_rfr_kwargs)
        if rfr_kwargs is not None:
            self.rfr_kwargs.update(rfr_kwargs)
        logger.info('rfr_kwargs: %s' % self.rfr_kwargs)

        self.openbox_kwargs = deepcopy(self.default_openbox_kwargs)
        if openbox_kwargs is not None:
            self.openbox_kwargs.update(openbox_kwargs)
        logger.info('openbox_kwargs: %s' % self.openbox_kwargs)

        self.sa_fraction = sa_fraction
        self.sa_rounds = sa_rounds
        self.sa_rounds_count = 0
        self.sa_iterations = sa_iterations
        logger.info('sa_fraction=%f, sa_rounds=%d, sa_iterations=%d' %
                    (self.sa_fraction, self.sa_rounds, self.sa_iterations))

        self.num_objs = num_objs
        self.num_constraints = num_constraints

        self.saved_configurations = []
        self.advisor = self.build_advisor(self.config_space)
        
        # task_manager里面的history有一个默认配置表现，初始化过来
        self.advisor.history = self.task_manager.current_task_history

    def build_config_space(self, parameter_names: list):
        cs = ConfigurationSpace()
        hps = [self.original_config_space.get_hyperparameter(name) for name in parameter_names]
        cs.add_hyperparameters(hps)
        logger.info('New config space built. size: %d. hps: %s.' % (len(hps), str(cs)))
        return cs

    def build_advisor(self, config_space, old_history_container=None):
        assert self.num_objs == 1
        advisor = Advisor(
            config_space=config_space,
            num_objectives=self.num_objs,
            num_constraints=self.num_constraints,
            initial_trials=2,
            logger_kwargs=self._logger_kwargs,
            **self.openbox_kwargs,
        )
        advisor.logger = logger

        new_history = History(task_id=f'history{id}', config_space=config_space)
        for obs in self.source_hpo_data.observations:
            conf = obs.config
            objs = obs.objectives
            new_config = self.build_config(conf, config_space)
            new_history.update_observation(Observation(config = new_config, objectives = objs))
        advisor.surrogate_model = MultiTaskGP(new_history)

        if old_history_container is not None:
            h = old_history_container
            assert len(h.configurations) == len(self.saved_configurations)
            for obs in h.observations:
                config = obs.config
                objs = obs.objectives
                new_config = self.build_config(config, config_space)
                obs = Observation(config = new_config, objectives = objs)
                advisor.update_observation(obs)
        logger.info('New advisor built. %d observations loaded.' % len(advisor.history.configurations))
        return advisor

    def build_config(self, config, config_space):
        default_config = config_space.get_default_configuration()
        config_dict = deepcopy(config.get_dictionary())
        new_config_dict = deepcopy(default_config.get_dictionary())

        # new_config_dict.update(config_dict)
        for k, v in config_dict.items():
            if k in new_config_dict:
                new_config_dict[k] = v

        new_config = Configuration(config_space, values=new_config_dict)
        return new_config

    def get_significant_parameters(self, top_k):
        h = self.advisor.history
        X = h.get_config_array(transform='scale')
        Y = h.get_objectives(transform='infeasible')
        model = RandomForestRegressor(**self.rfr_kwargs)
        model.fit(X, Y)
        importance = model.feature_importances_.tolist()
        parameter_names = self.config_space.get_hyperparameter_names()
        name_score = list(zip(parameter_names, importance))
        name_score.sort(key=lambda x: x[1], reverse=True)
        significant_parameters = [name for name, score in name_score[:top_k]]
        logger.info('Parameter importance:\n%s' % '\n'.join(map(str, name_score)))
        logger.info('Get %d significant parameters: %s' % (top_k, significant_parameters))
        return significant_parameters

    def decide_config_space(self):
        n = len(self.saved_configurations)
        if n == 0 or self.sa_rounds_count >= self.sa_rounds or n % self.sa_iterations != 0:
            return

        self.sa_rounds_count += 1
        n_parameters_old = len(self.config_space)
        n_parameters_new = int(np.ceil(n_parameters_old * self.sa_fraction))
        logger.info('SA round %d/%d. cs size: %d -> %d' %
                    (self.sa_rounds_count, self.sa_rounds, n_parameters_old, n_parameters_new))

        if n_parameters_new == n_parameters_old:
            logger.info('No need to change config space.')
            return

        significant_parameters = self.get_significant_parameters(top_k=n_parameters_new)
        self.config_space = self.build_config_space(significant_parameters)
        self.advisor = self.build_advisor(self.config_space, old_history_container=self.advisor.history)

    def sample(self, batch_size=1):
        self.decide_config_space()
        conf = self.advisor.get_suggestion()
        # Caution: advisor.get_suggestion() may still suggest config that doesn't meet condition
        return [self.build_config(conf, self.original_config_space)]

    def update(self, config, results, **kwargs):
        new_conf = self.build_config(config, self.config_space)
        new_obs = build_observation(new_conf, results)
        self.saved_configurations.append(new_obs.config)

        return self.advisor.update_observation(new_obs)


    @property
    def history(self):
        return self.advisor.history