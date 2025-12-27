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
from openbox.utils.space import Int, Float

from openbox import logger
from .mtgp import MultiTaskGP
from .utils import build_observation



# 将所有Project相关的功能写成一个小类，包括vector<->config的转换，类的init输入是config_space和d
class Projector:
    def __init__(self, config_space: ConfigurationSpace, d: int):
        self.config_space = config_space
        D = len(config_space)
        self.d = d
        self.D = D

        self.bounds = []
        self.para_types = []
        for name in self.config_space:
            para = self.config_space.get_hyperparameter(name)
            if isinstance(para, Int) :
                self.para_types.append('int')
                self.bounds.append((para.lower, para.upper))
            elif isinstance(para, Float):
                self.para_types.append('float')
                self.bounds.append((para.lower, para.upper))
            else:
                self.para_types.append('cat')
                self.bounds.append(para.choices)

            self.S = np.zeros((d, D), dtype=int)
            # 微微调整一下，保证每行都有一个非0元素
            for j in range(D):
                if j < d:
                    self.S[j, j] = np.random.choice([-1, 1])
                else:
                    i = np.random.randint(d)
                    self.S[i, j] = np.random.choice([-1, 1])

        self.pd_config_space = ConfigurationSpace()
        for i in range(d):
            self.pd_config_space.add_hyperparameter(Float('pd_dim_%d' % i, lower=-1.0, upper=1.0))

    def project_down(self, config): # 从config_space 到 pd_config_space

        theta = np.zeros(self.D)
        for i, name in enumerate(self.config_space):
            para = self.config_space.get_hyperparameter(name)
            if self.para_types[i] == 'int':
                val = config[name]
                low, high = self.bounds[i]
                theta[i] = 2.0 * (val - low) / (high - low) - 1.0
            elif self.para_types[i] == 'float':
                val = config[name]
                low, high = self.bounds[i]
                theta[i] = 2.0 * (val - low) / (high - low) - 1.0
            else:  # cat
                choices = self.bounds[i]
                val = config[name]
                idx = choices.index(val)
                theta[i] = 2.0 * idx / (len(choices) - 1) - 1.0

        theta_hat = self.S @ theta  # (d,D) · (D,) -> (d,)

        count = (self.S != 0).sum(axis=1)

        # 把0填充1
        count = np.where(count == 0, 1, count)
        theta_hat = theta_hat / count
        # clip到[-1, 1]
        theta_hat = np.clip(theta_hat, -1.0, 1.0)

        pd_config_dict = {}
        for i in range(self.d):
            pd_config_dict['pd_dim_%d' % i] = float(theta_hat[i])
        pd_config = Configuration(self.pd_config_space, values=pd_config_dict)
        return pd_config

    def project_up(self, vector): # 从pd_config_space 到 config_space
        theta_hat = np.array([vector['pd_dim_%d' % i] for i in range(self.d)])  # (d,)

        S_pinv = self.S.T  # (D,d)
        theta = S_pinv @ theta_hat  # (D,d) · (d,) -> (D,)

        config_dict = {}
        for i, name in enumerate(self.config_space):
            para = self.config_space.get_hyperparameter(name)
            if self.para_types[i] == 'int':
                low, high = self.bounds[i]
                val = int(np.round((theta[i] + 1.0) / 2.0 * (high - low) + low))
                val = min(max(val, low), high)
                config_dict[name] = val
            elif self.para_types[i] == 'float':
                low, high = self.bounds[i]
                val = (theta[i] + 1.0) / 2.0 * (high - low) + low
                val = min(max(val, low), high)
                config_dict[name] = val
            else:  # cat
                choices = self.bounds[i]
                idx = int(np.round((theta[i] + 1.0) / 2.0 * (len(choices) - 1)))
                idx = min(max(idx, 0), len(choices) - 1)
                config_dict[name] = choices[idx]

        config = Configuration(self.config_space, values=config_dict)

        return config



class Toptune:

    default_openbox_kwargs = dict(
        surrogate_type='gp',
        acq_optimizer_type='local_random',
    )
    def __init__(self, config_space: ConfigurationSpace,
                 d=25,
                 num_objs=1,
                 num_constraints=0,
                 rfr_kwargs: dict = None,
                 openbox_kwargs: dict = None,
                 **kwargs):
                 
        self._logger_kwargs = kwargs.get('_logger_kwargs', None)
        
        from task_manager import TaskManager
        self.task_manager = TaskManager.instance()

        self.config_space = config_space
        
        self.projector = Projector(config_space, d=25)

        self.num_objs = num_objs
        self.num_constraints = num_constraints
        
        self.openbox_kwargs = deepcopy(self.default_openbox_kwargs)
        if openbox_kwargs is not None:
            self.openbox_kwargs.update(openbox_kwargs)

        self.advisor = self.build_advisor()
        # task_manager里面的history有一个默认配置表现，初始化过来

        self.history = self.task_manager.current_task_history

        # 把已有的history里面的config转换成pd_config
        for obs in self.history.observations:
            config = obs.config
            pd_config = self.projector.project_down(config)
            new_obs = Observation(config=pd_config, objectives=[obs.objectives[0]], trial_state=obs.trial_state, elapsed_time=obs.elapsed_time,
                    extra_info=obs.extra_info)
            
            self.advisor.update_observation(new_obs)
            print("Update advisor with resume data", new_obs)

        self.last_pd_config = None



    def build_advisor(self):
        pd_config_space = self.projector.pd_config_space
        advisor = Advisor(
            config_space=pd_config_space,
            initial_trials=5,
            logger_kwargs=self._logger_kwargs,
            **self.openbox_kwargs
        )
        advisor.logger = logger
        return advisor


    def sample(self, batch_size=1):
        pd_config = self.advisor.get_suggestion()
        self.last_pd_config = pd_config
        config = self.projector.project_up(pd_config)
        return [config]

    def update(self, config, results, **kwargs):
        pd_config = self.last_pd_config
        obs = build_observation(pd_config, results, **kwargs)
        self.advisor.update_observation(obs)

        obs = build_observation(config, results, **kwargs)
        self.history.update_observation(obs)