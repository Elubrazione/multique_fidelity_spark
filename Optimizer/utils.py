import json
import time
import traceback
import numpy as np
from typing import Dict
from queue import Empty
from multiprocessing import Process, Queue
from openbox import logger, space as sp
from config import HUGE_SPACE_FILE

class AdvisorConfig:    
    def __init__(self, advisor_type: str, surrogate_type: str, acq_type: str):
        self.advisor_type = advisor_type
        self.surrogate_type = surrogate_type
        self.acq_type = acq_type
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'advisor_type': self.advisor_type,
            'surrogate_type': self.surrogate_type,
            'acq_type': self.acq_type
        }

def extract_base_surrogate(method_id: str) -> str:
    if method_id.endswith('_GP') or method_id == 'GP':
        return 'gp'
    elif method_id.endswith('_GPF') or method_id == 'GPF':
        return 'gpf'
    else:
        return 'prf'  # default: SMAC, MFES_SMAC, BOHB_SMAC

def get_surrogate_type(method_id: str, tl_strategy: str) -> str:
    base_type = extract_base_surrogate(method_id)
    if tl_strategy != 'none':
        surrogate_type = f'{tl_strategy}_{base_type}'
    else:
        surrogate_type = base_type
    if 'MFES' in method_id and tl_strategy == 'none':
        surrogate_type = f'mfes_{surrogate_type}'
    return surrogate_type

def get_acq_type(tl_strategy: str) -> str:
    if 'acq' in tl_strategy:
        return 'wrk_ei'
    else:
        return 'ei'

def get_advisor_config(
    method_id: str,
    tl_strategy: str = 'none'
) -> AdvisorConfig:
    if 'MFES' in method_id:
        advisor_type = 'mfbo'
    else:
        advisor_type = 'bo'
    
    surrogate_type = get_surrogate_type(method_id, tl_strategy)
    acq_type = get_acq_type(tl_strategy)
    return AdvisorConfig(advisor_type, surrogate_type, acq_type)

def build_optimizer(args, **kwargs):
    ws_args = kwargs.get('ws_args', None)
    tl_args = kwargs.get('tl_args', None)
    cp_args = kwargs.get('cp_args', None)
    random_kwargs = kwargs.get('random_kwargs', {})

    per_run_time_limit = kwargs.get('per_run_time_limit', None)

    from Optimizer.base import BaseOptimizer
    scheduler_kwargs = {
        'R': args.R,
        'eta': args.eta,
    }
    scheduler_type = 'mfes' if 'MFES' in args.opt else 'bohb' if 'BOHB' in args.opt else 'full'

    optimizer = BaseOptimizer(
        config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
        iter_num=args.iter_num, per_run_time_limit=per_run_time_limit,
        method_id=args.opt, task_id=kwargs['task'], target=kwargs['target'],
        cp_args=cp_args,
        ws_strategy=args.warm_start, ws_args=ws_args,
        tl_strategy=args.transfer, tl_args=tl_args,
        backup_flag=args.backup_flag,
        _logger_kwargs=kwargs.get('_logger_kwargs', None),
        random_kwargs=random_kwargs,
        scheduler_type=scheduler_type,
        scheduler_kwargs=scheduler_kwargs,
    )

    logger.info("[opt: {}] [warm_start_strategy: {}] [transfer_strategy: {}] [backup_flag: {}] [seed: {}] [rand_prob: {}]".format(
        args.opt, args.warm_start, args.transfer, args.backup_flag, args.seed, args.rand_prob)
    )
    logger.info("warm start args: %s: %s" % (args.warm_start, json.dumps(ws_args)))
    return optimizer


def wrapper_func(obj_func, queue, obj_args, obj_kwargs):
    try:
        ret = obj_func(*obj_args, **obj_kwargs)
    except Exception:
        result = {
            'result': {'objective': np.infty},
            'timeout': True,                 # 异常视为失败
            'traceback': traceback.format_exc()
        }
    else:
        obj = None
        try:
            if isinstance(ret, dict):
                obj = ret.get('objective', np.Inf)
            else:
                obj = np.Inf
        except Exception:
            obj = np.Inf
        timeout_flag = not np.isfinite(obj)
        result = {
            'result': ret,
            'timeout': timeout_flag,
            'traceback': None
        }
    queue.put(result)

def _check_result(result):
    if isinstance(result, dict) and set(result.keys()) == {'result', 'timeout', 'traceback'}:
        return result
    else:
        return {'result': {'objective': np.Inf}, 'timeout': True, 'traceback': None}


def run_without_time_limit(obj_func, obj_args, obj_kwargs):
    start_time = time.time()
    try:
        ret = obj_func(*obj_args, **obj_kwargs)
    except Exception:
        ret = {'result': {'objective': np.Inf}, 'timeout': False, 'traceback': traceback.format_exc()}
    return ret


def run_with_time_limit(obj_func, obj_args, obj_kwargs, timeout):
    start_time = time.time()
    queue = Queue()
    p = Process(target=wrapper_func, args=(obj_func, queue, obj_args, obj_kwargs))
    p.start()
    # wait until the process is finished or timeout is reached
    p.join(timeout=timeout)
    # terminate the process if it is still alive
    if p.is_alive():
        logger.info('Process timeout and is alive, terminate it')
        p.terminate()
        time.sleep(0.1)
        i = 0
        while p.is_alive():
            i += 1
            if i <= 10 or i % 100 == 0:
                logger.warning(f'Process is still alive, kill it ({i})')
            p.kill()
            time.sleep(0.1)
    # get the result
    try:
        result = queue.get(block=False)
    except Empty:
        result = None
    queue.close()
    result = _check_result(result)
    result['elapsed_time'] = time.time() - start_time
    return result


def run_obj_func(obj_func, obj_args, obj_kwargs, timeout=None):
    if timeout is None:
        result = run_without_time_limit(obj_func, obj_args, obj_kwargs)
    else:
        if timeout <= 0:
            timeout = None  # run by Process without timeout
        result = run_with_time_limit(obj_func, obj_args, obj_kwargs, timeout)
    return result


def load_space_from_json(json_file=None):
    
    if json_file is None:
        json_file = HUGE_SPACE_FILE
    
    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    space = sp.Space()
    
    for param_name, param_config in config.items():
        param_type = param_config["type"]
        default_value = param_config["default"]
        
        if param_type == "integer":
            space.add_variable(sp.Int(
                param_name,
                lower=param_config["min"],
                upper=param_config["max"],
                default_value=default_value
            ))
        elif param_type == "float":
            q = param_config.get("q", 0.05)
            space.add_variable(sp.Real(
                param_name,
                lower=param_config["min"],
                upper=param_config["max"],
                default_value=default_value,
                q=q
            ))
        elif param_type == "categorical":
            space.add_variable(sp.Categorical(
                param_name,
                choices=param_config["choice_values"],
                default_value=default_value
            ))
    
    return space