import argparse
from openbox import logger

from executor import ExecutorManager, SparkSessionTPCDSExecutor
from Compressor.utils import load_expert_params
from Optimizer.utils import build_optimizer, load_space_from_json
from task_manager import TaskManager
from utils.spark import analyze_timeout_and_get_fidelity_details
from config import LOG_DIR, HUGE_SPACE_FILE, EXPERT_PARAMS_FILE, DATA_DIR


parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='MFES_SMAC',
                    choices=['BOHB_GP', 'BOHB_SMAC', 'MFES_GP', 'MFES_SMAC', 'SMAC', 'GP', 'BOHB_SMAC'])
parser.add_argument('--log_level', type=str, default='info', choices=['info', 'debug'])
parser.add_argument('--fidelity', type=float, default=1/9)
parser.add_argument('--iter_num', type=int, default=40)
parser.add_argument('--R', type=int, default=27)
parser.add_argument('--eta', type=int, default=3)
parser.add_argument('--timeout', type=int, default=100)

parser.add_argument('--save_dir', type=str, default=LOG_DIR)
parser.add_argument('--target', type=str, default='spark_hstest')

parser.add_argument('--compress', type=str, default='none', choices=['none', 'shap', 'expert'])
parser.add_argument('--cp_topk', type=int, default=40)
parser.add_argument('--warm_start', type=str, default='none', choices=['none', 'best_cos', 'best_euc', 'best_rover', 'best_all', 'rgpe_rover'])
parser.add_argument('--ws_init_num', type=int, default=4)
parser.add_argument('--ws_topk', type=int, default=4)
parser.add_argument('--ws_inner_surrogate_model', type=str, default='prf')

parser.add_argument('--transfer', type=str, default='none')
parser.add_argument('--tl_topk', type=int, default=3)

parser.add_argument('--src_data_path', type=str, default='')
parser.add_argument('--backup_flag', action='store_true', default=False)

parser.add_argument('--task', type=str, default='test_ws')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--rand_prob', type=float, default=0.15)
parser.add_argument('--rand_mode', type=str, default='ran', choices=['ran', 'rs'])

parser.add_argument('--test_mode', action='store_true', default=False)

args = parser.parse_args()

_logger_kwargs = {
    'name': "%s" % args.task,
    'logdir': f'{LOG_DIR}/{args.target}/{args.opt}',
    'level': args.log_level.upper()
}
logger.init(**_logger_kwargs)
_logger_kwargs['force_init'] = False

config_space = load_space_from_json(HUGE_SPACE_FILE)

fidelity_details, elapsed_timeout_dicts = analyze_timeout_and_get_fidelity_details(
    percentile=args.timeout, debug=False,
    ratio_list=[1, 1/8, 1/32], add_on_ratio=2.5
)
fidelity_details[round(float(1/64), 5)] = ['q48']
# fidelity_details[round(float(1.0), 5)] = ['q10', 'q12', 'q11']

executor = ExecutorManager(
    sqls=fidelity_details,
    timeout=elapsed_timeout_dicts,
    config_space=config_space,
    executor_cls=SparkSessionTPCDSExecutor,
    executor_kwargs={'sql_dir': DATA_DIR},
    test_mode=args.test_mode
)

ws_args = {
    'init_num': args.ws_init_num,
    'topk': args.ws_topk,
    'inner_surrogate_model': args.ws_inner_surrogate_model
}
tl_args = {
    'topk': args.tl_topk
}

task_manager = TaskManager.instance(
    history_dir=args.src_data_path,
    eval_func=executor,
    task_id=args.task,
    spark_log_dir="/root/codes/spark-log",
    ws_args=ws_args,
    similarity_threshold=0.5,
    config_space=config_space,
    test_mode=args.test_mode
)


cp_args = {
    'strategy': args.compress,
    'topk': args.cp_topk,
    'sigma': 2.0,
    'top_ratio': 0.8,
    'expert_params': [p for p in load_expert_params(EXPERT_PARAMS_FILE) \
                        if p in config_space.get_hyperparameter_names()],
}

random_kwargs = {
    'seed': args.seed,
    'rand_prob': args.rand_prob,
    'rand_mode': args.rand_mode,
}

opt_kwargs = {
    'config_space': config_space,
    'eval_func': executor,
    'target': args.target,
    'task': args.task,
    'ws_args': ws_args, 'tl_args': tl_args, 'cp_args': cp_args,
    'random_kwargs': random_kwargs,
    '__logger_kwargs': _logger_kwargs
}
optimizer = build_optimizer(args, **opt_kwargs)

if __name__ == '__main__':
    for i in range(optimizer.iter_num):
        optimizer.run_one_iter()