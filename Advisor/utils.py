import copy
import numpy as np
import pandas as pd
from openbox import logger
from openbox.utils.history import Observation
from openbox.utils.util_funcs import get_types
from openbox.utils.constants import SUCCESS, TIMEOUT, FAILED

from .workload_mapping.rov import RoverMapper


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


def map_source_hpo_data(target_his, source_hpo_data, config_space, **kwargs):
    """
    计算目标任务与源任务之间的相似度
    
    Parameters:
    -----------
    target_his : History
        目标任务的历史观测数据
    source_hpo_data : List[History]
        源任务的历史观测数据列表
    config_space : ConfigurationSpace
        配置空间
    **kwargs : dict
        可选参数：
        - inner_surrogate_model : str, default='gp'
            内部代理模型类型
        - use_real : bool, default=False
            是否使用真实观测数据计算相似度（一致对占比）
            如果为True，使用map_with_observations方法（需要至少2个观测数据）
            如果为False，使用map_with_prediction方法（CatBoost预测，基于meta_feature）
    
    Returns:
    --------
    List[Tuple[int, float]]
        相似度列表，每个元素为(源任务索引, 相似度值)
    """
    inner_sm = kwargs.get('inner_surrogate_model', 'gp')
    use_real = kwargs.get('use_real', False)
    use_cached_model = kwargs.get('use_cached_model', False)
    
    rover = RoverMapper(surrogate_type=inner_sm)
    if not source_hpo_data:
        logger.warning('No source HPO data available. Returning empty similarity list.')
        return []
    rover.fit(source_hpo_data, config_space, use_cached_model=use_cached_model)
    
    # 调用map方法，它会根据use_real参数路由到相应的方法
    sims = rover.map(target_his, source_hpo_data, use_real=use_real)
    
    return sims


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
