import json
import time
import traceback
import numpy as np
from queue import Empty
from multiprocessing import Process, Queue
from openbox import logger, space as sp
from config import HUGE_SPACE_FILE

def build_optimizer(args, **kwargs):
    ws_args = kwargs.get('ws_args', None)
    tl_args = kwargs.get('tl_args', None)
    cp_args = kwargs.get('cp_args', None)

    per_run_time_limit = kwargs.get('per_run_time_limit', None)

    optimizer = None
    if args.opt == 'RS':
        from Optimizer.SMBO import SMBO
        optimizer = SMBO(config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
                        iter_num=args.iter_num, per_run_time_limit=per_run_time_limit,
                        method_id='RS', task_id=kwargs['task'], target=kwargs['target'],
                        save_dir=kwargs['save_dir'],
                        task_manager=kwargs.get('task_manager'),
                        _logger_kwargs=kwargs.get('_logger_kwargs', None))
    if args.opt in ['GP', 'GPF', 'SMAC']:
        from Optimizer.SMBO import SMBO
        optimizer = SMBO(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
            iter_num=args.iter_num, per_run_time_limit=per_run_time_limit,
            method_id=args.opt, task_id=kwargs['task'], target=kwargs['target'],
            cprs_strategy=args.compress, cp_args=cp_args,
            ws_strategy=args.warm_start, ws_args=ws_args,
            tl_strategy=args.transfer, tl_args=tl_args,
            backup_flag=args.backup_flag, seed=args.seed,
            rand_prob=args.rand_prob, rand_mode=args.rand_mode,
            task_manager=kwargs.get('task_manager'),
            _logger_kwargs=kwargs.get('_logger_kwargs', None)
        )
    elif 'BOHB' in args.opt or 'MFSE' in args.opt:
        from Optimizer.BOHB import BOHB
        scheduler_kwargs = {
            'R': args.R,
            'eta': args.eta,
            'fixed_initial': False
        }
        optimizer = BOHB(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
            method_id=args.opt, task_id=args.task, target=kwargs['target'],
            iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, 
            cprs_strategy=args.compress, cp_args=cp_args,
            ws_strategy=args.warm_start, ws_args=ws_args,
            tl_strategy=args.transfer, tl_args=tl_args,
            backup_flag=args.backup_flag, seed=args.seed,
            rand_prob=args.rand_prob, rand_mode=args.rand_mode,
            save_dir=args.save_dir, 
            task_manager=kwargs.get('task_manager'),
            scheduler_kwargs=scheduler_kwargs,
            _logger_kwargs=kwargs.get('_logger_kwargs', None)
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


def process_src_data(his):
    default = his.observations[0].objectives[0]
    # 用默认配置的五倍弱来填inf
    fill_val = default * 5 if default > 0 else default / 5
    for obs in his.observations:
        if not np.isfinite(obs.objectives[0]):
            obs.objectives[0] = fill_val
            print("find inf objective in %s, fill %f" % (his.task_id, fill_val))

    return his

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