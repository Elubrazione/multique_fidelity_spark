import json
import time
import traceback
import numpy as np
from queue import Empty
from multiprocessing import Process, Queue
from openbox import logger

from Compressor.utils import load_expert_params


def build_optimizer(args, **kwargs):
    scene_str = kwargs.get('scene', 'none')
    assert scene_str in ["spark"]

    space_names = set(kwargs['config_space'].get_hyperparameter_names())
    expert_params = [p for p in load_expert_params(scene=scene_str) if p in space_names]

    task_str = kwargs.get('task_str', 'run')
    meta_feature = kwargs.get('meta_feature', None)
    source_hpo_data = kwargs.get('source_hpo_data', None)
    source_compress_data = kwargs.get('source_compress_data', None)
    range_config_space = kwargs.get('range_config_space', None)

    ws_args = kwargs.get('ws_args', None)
    tl_args = kwargs.get('tl_wargs', None)
    ep_args = kwargs.get('ep_args', None)
    cp_args = kwargs.get('cp_args', None)

    per_run_time_limit = kwargs.get('per_run_time_limit', None)

    optimizer = None
    if args.opt == 'RS':
        from Optimizer.SMBO import SMBO
        optimizer = SMBO(config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
                         iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, meta_feature=meta_feature,
                         method_id='RS', task_id=kwargs['task'], target=kwargs['target'], task_str=task_str, 
                         config_modifier=kwargs['config_modifier'], expert_modified_space=kwargs['expert_modified_space'],
                         save_dir=kwargs['save_dir'])
    if args.opt in ['GP', 'GPF', 'SMAC']:
        from Optimizer.SMBO import SMBO
        optimizer = SMBO(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'],
            iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, meta_feature=meta_feature,
            source_hpo_data=source_hpo_data, ep_args=ep_args, ep_strategy=args.expert,
            method_id=args.opt, task_id=kwargs['task'],target=kwargs['target'], task_str=task_str,
            cprs_strategy=args.compress, space_history=source_compress_data, cp_args=cp_args,
            ws_strategy=args.warm_start, ws_args=ws_args, tl_strategy=args.transfer, tl_args=tl_args,
            backup_flag=args.backup_flag, seed=args.seed, rand_prob=args.rand_prob, rand_mode=args.rand_mode, config_modifier=kwargs['config_modifier'],
            expert_params=expert_params
        )
    elif 'BOHB' in args.opt or 'MFSE' in args.opt or 'FlexHB' in args.opt:
        from Optimizer.BOHB import BOHB
        R = args.R
        eta = args.eta
        optimizer = BOHB(
            config_space=kwargs['config_space'], eval_func=kwargs['eval_func'], meta_feature=meta_feature,
            method_id=args.opt, task_id=args.task, target=kwargs['target'], task_str=task_str,
            iter_num=args.iter_num, per_run_time_limit=per_run_time_limit, 
            ep_args=ep_args, ep_strategy=args.expert, expert_params=expert_params, 
            cprs_strategy=args.compress, space_history=source_compress_data, cp_args=cp_args, range_config_space=range_config_space,
            ws_strategy=args.warm_start, ws_args=ws_args, tl_strategy=args.transfer, tl_args=tl_args, source_hpo_data=source_hpo_data,
            backup_flag=args.backup_flag, seed=args.seed, rand_prob=args.rand_prob, rand_mode=args.rand_mode,
            R=R, eta=eta, save_dir=args.save_dir, 
            config_modifier=kwargs['config_modifier'], expert_modified_space=kwargs['expert_modified_space'],
        )


    logger.info("[opt: {}] [warm_start_strategy: {}] [transfer_strategy: {}] [safe_flag: {}] [backup_flag: {}] [tasks: {}] [seed: {}] [rand_prob: {}]".format(
        args.opt, args.warm_start, args.transfer, args.safe_flag, args.backup_flag, task_str, args.seed, args.rand_prob)
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
        result = {'result': {'objective': np.Inf}, 'timeout': False, 'traceback': traceback.format_exc()}
    else:
        if np.isfinite(ret['objective']):
            result = {'result': ret, 'timeout': False, 'traceback': None}
        else:
            result = {'result': ret, 'timeout': True, 'traceback': None}
    result['elapsed_time'] = time.time() - start_time
    return result


def run_with_time_limit(obj_func, obj_args, obj_kwargs, timeout):
    start_time = time.time()
    queue = Queue()
    # Todo: 这里本来是多进程，但是多进程传入的参数要是可序列化的（SparkExecutor不满足），所以暂时改成多线程
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