import copy
import numpy as np
import pandas as pd
from openbox import logger
from openbox.utils.history import Observation
from openbox.utils.util_funcs import get_types
from openbox.utils.constants import SUCCESS, TIMEOUT, FAILED


def _to_dict(config):
    try:
        if hasattr(config, 'get_dictionary'):
            return config.get_dictionary()
        return dict(config)
    except Exception:
        return {}


def is_valid_spark_config(config) -> bool:
    d = _to_dict(config)
    try:
        exec_cores = int(float(d.get('spark.executor.cores', 2)))
        task_cpus = int(float(d.get('spark.task.cpus', 1)))
        return exec_cores >= task_cpus and exec_cores >= 1 and task_cpus >= 1
    except Exception:
        return True


def sanitize_spark_config(config):
    try:
        d = _to_dict(config)
        exec_cores = int(float(d.get('spark.executor.cores', 2)))
        task_cpus = int(float(d.get('spark.task.cpus', 1)))
        if exec_cores < 1:
            exec_cores = 1
        if task_cpus < 1:
            task_cpus = 1
        if exec_cores < task_cpus:
            config['spark.task.cpus'] = exec_cores
    except Exception:
        pass
    return config


def build_my_acq_func(func_str='ei', model=None, **kwargs):
    func_str = func_str.lower()
    if func_str.startswith('wrk'):
        inner_acq_func = func_str.split('_')[1]
        from .acq_function.weighted_rank import WeightedRank
        return WeightedRank(model=model, acq_func=inner_acq_func)
    elif func_str == 'ei':
        from openbox.acquisition_function import EI
        return EI(model=model)
    else:
        raise ValueError('Invalid string %s for acquisition function!' % func_str)

def build_my_surrogate(func_str='gp', config_space=None, rng=None, transfer_learning_history=None, **kwargs):
    extra_dim = kwargs.get('extra_dim', 0)
    seed = kwargs.get('seed', 42)
    norm_y = kwargs.get('norm_y', True)

    assert config_space is not None
    func_str = func_str.lower()
    types, bounds = get_types(config_space)
    if extra_dim > 0:
        types = np.hstack((types, np.zeros(extra_dim, dtype=np.uint)))
        bounds = np.vstack((bounds, np.array([[0, 1]] * extra_dim)))

    if func_str == 'prf':
        from openbox.surrogate.base.rf_with_instances_sklearn import skRandomForestWithInstances
        return skRandomForestWithInstances(types=types, bounds=bounds, seed=seed)
    elif func_str.startswith('gp'):
        from openbox.surrogate.base.build_gp import create_gp_model
        return create_gp_model(model_type=func_str[:2],
                               config_space=config_space,
                               types=types,
                               bounds=bounds,
                               rng=rng)
    elif func_str.startswith('mce'):
        from .surrogate.rgpe import RGPE
        inner_model = func_str.split('_')[1]
        return RGPE(config_space=config_space, source_hpo_data=transfer_learning_history, seed=seed,
                    surrogate_type=inner_model, norm_y=norm_y)
    elif func_str.startswith('re'):
        from .surrogate.mfgpe import MFGPE
        inner_model = func_str.split('_')[1]
        return MFGPE(config_space=config_space, source_hpo_data=transfer_learning_history, seed=seed,
                    surrogate_type=inner_model, norm_y=norm_y)
    elif func_str.startswith('mfes'):   # 没有迁移学习
        from .surrogate.mfgpe import MFGPE
        inner_model = func_str.split('_')[1]
        return MFGPE(config_space=config_space, source_hpo_data=None, seed=seed,
                    surrogate_type=inner_model, norm_y=norm_y)
    else:
        raise ValueError('Invalid string %s for surrogate!' % func_str)


def build_observation(config, results, **kwargs):
    ret, timeout_status, traceback_msg, elapsed_time, extra_info = (
        results['result'], results['timeout'], results['traceback'], results['elapsed_time'], results['extra_info'])
    perf = ret['objective']

    if timeout_status:
        trial_state = TIMEOUT
    elif traceback_msg is not None:
        trial_state = FAILED
        logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
    else:
        trial_state = SUCCESS

    extra_info_copy = copy.deepcopy(extra_info)
    obs = Observation(config=config, objectives=[perf], trial_state=trial_state, elapsed_time=elapsed_time,
                    extra_info={'origin': config.origin, **extra_info_copy})

    return obs
