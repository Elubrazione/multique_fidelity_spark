"""
sample configs and evaluate performance script
validate random configs' performance under different data scales
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from openbox import logger

from executor import ExecutorManager
from space_values import load_space_from_json
from utils.spark import get_full_queries_tasks, analyze_sqls_timeout_from_csv
from config import HUGE_SPACE_FILE


def setup_openbox_logging(experiment_dir):
    logger_kwargs = {
        'name': 'data_volume_experiment',
        'logdir': experiment_dir
    }
    logger.init(**logger_kwargs)


def create_fidelity_database_mapping():
    return {
        0.03: "tpcds_30g",
        0.1: "tpcds_100g", 
        0.3: "tpcds_300g",
        0.6: "tpcds_600g",
        1.0: "tpcds_1000g"
    }


def create_fixed_sqls():
    return get_full_queries_tasks()


def filter_expert_config_space(original_config_space):
    import json
    from ConfigSpace import ConfigurationSpace

    expert_space_path = "/root/codes/multique_fidelity_spark/configs/config_space/expert_space.json"
    with open(expert_space_path, 'r') as f:
        expert_params = json.load(f)

    expert_param_names = set()
    expert_param_names.update(expert_params.get('spark', []))
    expert_param_names.update(expert_params.get('os', []))
    
    logger.info(f"expert params list: {len(expert_param_names)} params")
    logger.info(f"expert params: {sorted(expert_param_names)}")
    
    expert_config_space = ConfigurationSpace()
    
    original_hyperparams = original_config_space.get_hyperparameters()
    expert_count = 0
    
    for hyperparam in original_hyperparams:
        if hyperparam.name in expert_param_names:
            expert_config_space.add_hyperparameter(hyperparam)
            expert_count += 1
            logger.info(f"add expert param: {hyperparam.name}")
    
    logger.info(f"expert config space created, total {expert_count} params")
    return expert_config_space


def sample_configurations(config_space, num_samples=100, seed=42):
    np.random.seed(seed)
    configs = []
    
    for i in range(num_samples):
        config = config_space.sample_configuration()
        configs.append(config)
        logger.info(f"sample config {i+1}/{num_samples}: {config}")
    
    return configs


def evaluate_config_on_fidelity(executor, config, fidelity, config_idx, experiment_dir):
    try:
        logger.info(f"evaluate config {config_idx} on fidelity {fidelity}")
        
        config_dir = f"{experiment_dir}/config_{config_idx}"
        fidelity_dir = f"{config_dir}/{fidelity}"
        
        result = executor(config, fidelity, fidelity_dir)
        
        if result and 'result' in result:
            objective = result['result']['objective']
            elapsed_time = result.get('elapsed_time', 0)
            timeout_flag = result.get('timeout', False)
            
            return {
                'objective': objective,
                'elapsed_time': elapsed_time,
                'timeout': timeout_flag,
                'success': True
            }
        else:
            return {
                'objective': float('inf'),
                'elapsed_time': 0,
                'timeout': True,
                'success': False
            }
            
    except Exception as e:
        logger.error(f"评估配置时出错: {e}")
        return {
            'objective': float('inf'),
            'elapsed_time': 0,
            'timeout': True,
            'success': False,
            'error': str(e)
        }


def save_results(results, output_dir):    
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'config_validation_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"save results to: {csv_path}")
    
    json_path = os.path.join(output_dir, 'config_validation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"save results to: {json_path}")
    
    stats = {
        'total_configs': len(results),
        'fidelity_stats': {},
        'overall_stats': {
            'success_rate': df['success'].mean(),
            'avg_objective': df[df['success']]['objective'].mean(),
            'avg_elapsed_time': df['elapsed_time'].mean()
        }
    }
    
    for fidelity in df['fidelity'].unique():
        fidelity_data = df[df['fidelity'] == fidelity]
        stats['fidelity_stats'][str(fidelity)] = {
            'count': len(fidelity_data),
            'success_rate': fidelity_data['success'].mean(),
            'avg_objective': fidelity_data[fidelity_data['success']]['objective'].mean(),
            'avg_elapsed_time': fidelity_data['elapsed_time'].mean()
        }
    
    stats_path = os.path.join(output_dir, 'validation_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f": {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='sample configs and evaluate performance')
    parser.add_argument('--num_samples', type=int, default=100, help='sample configs number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--test_mode', action='store_true', help='test mode, only evaluate少量配置')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"/root/codes/multique_fidelity_spark/exps/data_volume_correlation/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    setup_openbox_logging(experiment_dir)
    
    logger.info("=" * 60)
    logger.info("sample configs and evaluate performance experiment starts")
    logger.info("=" * 60)
    logger.info(f"sample configs number: {args.num_samples}")
    logger.info(f"random seed: {args.seed}")
    logger.info(f"experiment directory: {experiment_dir}")
    
    if args.test_mode:
        args.num_samples = 1
        logger.info(f"test mode: only evaluate {args.num_samples} configs")
    
    fidelity_mapping = create_fidelity_database_mapping()
    fixed_sqls = create_fixed_sqls()
    
    logger.info(f"fidelity mapping: {fidelity_mapping}")
    logger.info(f"fixed sqls: {len(fixed_sqls)} queries")
    
    original_config_space = load_space_from_json(HUGE_SPACE_FILE)
    logger.info(f"original config space loaded, total {len(original_config_space.get_hyperparameters())} params")
    
    config_space = filter_expert_config_space(original_config_space)
    logger.info(f"expert config space created, total {len(config_space.get_hyperparameters())} params")
    
    logger.info("begin to sample configs...")
    configs = sample_configurations(config_space, args.num_samples, args.seed)
    logger.info(f"sample configs done, total {len(configs)} configs")
    
    fidelity_details = {}
    elapsed_timeout_dicts = analyze_sqls_timeout_from_csv(add_on_ratio=2.5)
    
    for fidelity in fidelity_mapping.keys():
        fidelity_details[fidelity] = fixed_sqls
    
    logger.info("create executor...")
    executor = ExecutorManager(
        sqls=fidelity_details,
        timeout=elapsed_timeout_dicts,
        config_space=config_space,
        enable_os_tuning=False,
        fidelity_database_mapping=fidelity_mapping,
        fixed_sqls=fixed_sqls
    )
    logger.info("executor created successfully")
    
    results = []
    total_evaluations = len(configs) * len(fidelity_mapping)
    current_evaluation = 0
    
    logger.info("begin to evaluate configs...")
    logger.info(f"total evaluations: {total_evaluations}")
    
    for config_idx, config in enumerate(configs):
        logger.info(f"evaluate config {config_idx + 1}/{len(configs)}")
        
        for fidelity_str in fidelity_mapping.keys():
            fidelity = round(float(fidelity_str), 5)
            current_evaluation += 1
            
            logger.info(f"   evaluation progress: {current_evaluation}/{total_evaluations} - Fidelity: {fidelity}")
            
            result = evaluate_config_on_fidelity(executor, config, fidelity, config_idx + 1, experiment_dir)
            
            result_record = {
                'config_id': config_idx,
                'fidelity': fidelity,
                'database': fidelity_mapping[fidelity_str],
                'config': config.get_dictionary() if hasattr(config, 'get_dictionary') else config,
                'objective': result['objective'],
                'elapsed_time': result['elapsed_time'],
                'timeout': result['timeout'],
                'success': result['success']
            }
            
            if 'error' in result:
                result_record['error'] = result['error']
            
            results.append(result_record)
            
            logger.info(f"result record: objective={result['objective']:.2f}, "
                       f"elapsed_time={result['elapsed_time']:.2f}s, "
                       f"success={result['success']}")
    
    logger.info("=" * 60)
    logger.info("all configs evaluated successfully")
    logger.info("=" * 60)
    
    logger.info("save evaluation results to experiment directory...")
    save_results(results, experiment_dir)
    
    df = pd.DataFrame(results)
    
    logger.info("=" * 60)
    for fidelity in sorted(df['fidelity'].unique()):
        fidelity_data = df[df['fidelity'] == fidelity]
        valid_fidelity_data = fidelity_data[fidelity_data['objective'] != float('inf')]
        
        success_rate = len(valid_fidelity_data) / len(fidelity_data)
        
        if len(valid_fidelity_data) > 0:
            avg_objective = valid_fidelity_data['objective'].mean()
            avg_obj_str = f"{avg_objective:.2f}"
            avg_elapsed_valid = valid_fidelity_data['elapsed_time'].mean()
            avg_elapsed_valid_str = f"{avg_elapsed_valid:.2f}"
        else:
            avg_obj_str = "N/A"
            avg_elapsed_valid_str = "N/A"
        
        logger.info(f"  Fidelity {fidelity}:")
        logger.info(f"    success rate: {success_rate:.2%} ({len(valid_fidelity_data)}/{len(fidelity_data)})")
        logger.info(f"    valid results: {len(valid_fidelity_data)} - avg objective: {avg_obj_str}, avg elapsed: {avg_elapsed_valid_str}s")
    
    logger.info("=" * 60)
    logger.info("sample configs and evaluate performance experiment done")
    logger.info(f"results saved to: {experiment_dir}")
    logger.info(f"logs saved to: {experiment_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
