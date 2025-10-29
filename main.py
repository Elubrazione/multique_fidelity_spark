import argparse, os, json
import numpy as np
from openbox.utils.constants import MAXINT
from openbox.utils.history import History
from openbox import logger
from ConfigSpace import Configuration

from executor import ExecutorManager
from space_values import haisi_huge_spaces_from_json, load_space_from_json
from Compressor.utils import parse_combined_space, load_expert_params
from Optimizer.utils import build_optimizer
from Advisor.task_manager import TaskManager
from utils.spark import analyze_timeout_and_get_fidelity_details
from config import LOG_DIR, HISTORY_COMPRESS_DIR, \
    RANGE_COMPRESS_DATA, FILE_SQL_SEGMENTATION, HUGE_SPACE_FILE, OS_CONFIG_SPACE_FILE, EXPERT_PARAMS_FILE


parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='MFSE_SMAC',
                    choices=['BOHB_GP', 'BOHB_SMAC', 'MFSE_GP', 'MFSE_SMAC'])
parser.add_argument('--fidelity', type=float, default=1/9)
parser.add_argument('--iter_num', type=int, default=40)
parser.add_argument('--R', type=int, default=64)
parser.add_argument('--eta', type=int, default=4)
parser.add_argument('--timeout', type=int, default=100)

parser.add_argument('--save_dir', type=str, default=LOG_DIR)
parser.add_argument('--target', type=str, default='spark_hstest')

parser.add_argument('--expert', type=str, default='none', choices=['none', 'pibo', 'bo_pro', 'prior_band'])

parser.add_argument('--compress', type=str, default='none', choices=['none', 'shap'])
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

parser.add_argument('--test_mode', type=bool, default=False)

parser.add_argument('--disable_range_compress', type=bool, default=False,
                    help='禁用范围压缩功能，使用原始搜索空间')

parser.add_argument('--enable_os_tuning', type=bool, default=False)

args = parser.parse_args()

def modifier(config):
    if 'spark.io.compression.snappy.blockSize' in config:
        config['spark.io.compression.snappy.blockSize'] = max(64, config['spark.io.compression.snappy.blockSize'])
        logger.info("Debug: spark.io.compression.snappy.blockSize is modified to %d" % config['spark.io.compression.snappy.blockSize'])
    if 'spark.io.compression.lz4.blockSize' in config:
        config['spark.io.compression.lz4.blockSize'] = max(64, config['spark.io.compression.lz4.blockSize'])
        logger.info("Debug: spark.io.compression.lz4.blockSize is modified to %d" % config['spark.io.compression.lz4.blockSize'])
    return config


base_space = load_space_from_json(HUGE_SPACE_FILE)
old_space = base_space
logger.info(f"使用原始搜索空间：{len(base_space.get_hyperparameters())} 个参数")

fidelity_details, elapsed_timeout_dicts = analyze_timeout_and_get_fidelity_details(
    percentile=args.timeout, debug=False,
    ratio_list=[1, 1/8, 1/32], add_on_ratio=2.5
)
fidelity_details[round(float(1/64), 5)] = ['q48']
if args.test_mode:
    fidelity_details[round(float(1), 5)] = ['q48']


os_space = None
if args.enable_os_tuning:
    logger.info("OS参数调优功能已启用")
    os_space = load_space_from_json(OS_CONFIG_SPACE_FILE)
    logger.info(f"OS配置空间加载成功，包含 {len(os_space.get_hyperparameters())} 个参数")

    old_space = parse_combined_space(old_space, os_space)
    logger.info(f"OS参数拼接完成：原始空间 {len(old_space.get_hyperparameters())} 个参数")
else:
    logger.info("OS参数调优功能已禁用")

executor = ExecutorManager(
    sqls=fidelity_details, timeout=elapsed_timeout_dicts, config_space=old_space,
    enable_os_tuning=args.enable_os_tuning,
)

ws_args = {
    'init_num': args.ws_init_num,
    'topk': args.ws_topk,
    'inner_surrogate_model': args.ws_inner_surrogate_model
}
tl_args = {
    'topk': args.tl_topk
}

task_manager = TaskManager(
    history_dir=args.src_data_path,
    eval_func=executor,
    spark_log_dir="/root/codes/spark-log",
    ws_args=ws_args,
    similarity_threshold=0.5,
    config_space=old_space
)


cp_args = {
    'topk': args.cp_topk,
    'sigma': 2.0,
    'top_ratio': 0.8,
    'expert_params': [p for p in load_expert_params(EXPERT_PARAMS_FILE) if p in old_space.get_hyperparameter_names()],
}


config_space = old_space
logger.info(f"优化器配置：原始空间 {len(config_space.get_hyperparameters())} 个参数")

opt_kwargs = {
    'config_space': config_space,
    'eval_func': executor,
    'target': args.target,
    'task': args.task,
    'ws_args': ws_args, 'tl_args': tl_args, 'cp_args': cp_args,
    'task_manager': task_manager,
    'config_modifier': modifier,
    'expert_modified_space': None,
    'enable_range_compression': not args.disable_range_compress,
}
optimizer = build_optimizer(args, **opt_kwargs)

if __name__ == '__main__':
    if args.test_mode:
        optimizer.save_info()
        pass
    else:
        for i in range(optimizer.iter_num):
            optimizer.run_one_iter()