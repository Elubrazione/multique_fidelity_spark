"""
Data volume correlation experiment using PySpark Session execution
Sample configs and evaluate performance under different data scales using single SparkSession per SQL file

Usage:
python data_volume_correlation_new.py --num_samples 10 --seed 42 --test_mode
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
from openbox import logger
from pyspark.sql import SparkSession

from Optimizer.utils import load_space_from_json
from utils.spark import (
    clear_cache_on_remote,
    custom_sort,
    format_spark_config_value,
    get_full_queries_tasks,
)
from config import HUGE_SPACE_FILE, SPARK_NODES, DATA_DIR


def extract_sql_performance_data(sql_results):
    performance_data = {}
    
    for sql_result in sql_results:
        sql_file = sql_result.get('sql_file', '')
        if not sql_file:
            continue
        
        if sql_result.get('status') == 'success':
            performance_data[f'et_{sql_file}'] = sql_result.get('total_elapsed_time', 0)
            
            queries = sql_result.get('queries', [])
            for query in queries:
                elapsed_time = query.get('elapsed_time', 0)
                performance_data[f'qt_{sql_file}'] = elapsed_time
        else:
            performance_data[f'et_{sql_file}'] = float('inf')
            performance_data[f'qt_{sql_file}'] = float('inf')
    
    return performance_data


def find_latest_experiment_dir(base_dir="./exps/data_volume_correlation"):
    if not os.path.exists(base_dir):
        return None
    
    experiment_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('20'):  # match the timestamp format
            all_inner_items = os.listdir(item_path)
            # filter experiments that contain .log files (current experiment runs)
            if any(inner_item.endswith('.log') for inner_item in all_inner_items):
                experiment_dirs.append(item_path)
    
    if not experiment_dirs:
        return None
    
    # sort by modification time, return the latest
    latest_dir = max(experiment_dirs, key=os.path.getmtime)
    logger.info(f"found latest experiment directory: {latest_dir}")
    return latest_dir


def add_error_sql_performance_data(result_data, sql_files):
    for sql_file in sql_files:
        result_data[f'et_{sql_file}'] = float('inf')
        result_data[f'qt_{sql_file}'] = float('inf')
    return result_data


def setup_openbox_logging(experiment_dir):
    logger_kwargs = {
        'name': 'data_volume_experiment',
        'logdir': experiment_dir
    }
    logger.init(**logger_kwargs)

def create_fidelity_database_mapping(test_mode=False):
    database_mapping = {
        0.03: "tpcds_30g",
        0.1: "tpcds_100g"
    }
    if not test_mode:
        database_mapping.update({
            0.3: "tpcds_300g",
            0.6: "tpcds_600g",
            1.0: "tpcds_1000g"
        })
    return database_mapping

def create_fixed_sqls(sql_dir, test_mode=False):
    queries = get_full_queries_tasks(sql_dir)
    if test_mode:
        return queries[: 2]
    return queries


def detect_completed_evaluations(experiment_dir, fidelity_mapping):
    """detect the completed configurations and fidelity combinations"""
    if not experiment_dir or not os.path.exists(experiment_dir):
        return set()
    
    completed = set()
    
    for item in os.listdir(experiment_dir):
        if item.startswith('config_'):
            config_path = os.path.join(experiment_dir, item)
            if not os.path.isdir(config_path):
                continue

            try:
                config_idx = int(item.split('_')[1])
            except (IndexError, ValueError):
                continue

            for fidelity_item in os.listdir(config_path):
                fidelity_path = os.path.join(config_path, fidelity_item)
                if not os.path.isdir(fidelity_path):
                    continue
                
                # check if there is a execution_results.json file
                result_file = os.path.join(fidelity_path, "execution_results.json")
                if os.path.exists(result_file):                      
                    fidelity = float(fidelity_item)
                    completed.add((config_idx, fidelity))
                    logger.info(f"detected completed evaluation: config_{config_idx}, fidelity_{fidelity}")
    
    logger.info(f"total completed evaluations: {len(completed)}")
    return completed

def rebuild_results_from_execution_files(experiment_dir, fidelity_mapping):
    """rebuild results list from existing execution_results.json files"""
    results = []
    
    if not experiment_dir or not os.path.exists(experiment_dir):
        return results
    
    # traverse config directories
    for item in os.listdir(experiment_dir):
        if item.startswith('config_'):
            config_path = os.path.join(experiment_dir, item)
            if not os.path.isdir(config_path):
                continue
            
            try:
                config_idx = int(item.split('_')[1])
            except (IndexError, ValueError):
                continue
            
            # traverse fidelity directories
            for fidelity_item in os.listdir(config_path):
                fidelity_path = os.path.join(config_path, fidelity_item)
                if not os.path.isdir(fidelity_path):
                    continue
                
                result_file = os.path.join(fidelity_path, "execution_results.json")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        # extract summary information
                        summary = result_data.get('summary', {})
                        fidelity = float(fidelity_item)
                        # try to get database from summary first, fallback to fidelity_mapping
                        database = summary.get('database', fidelity_mapping.get(str(fidelity), f"unknown_{fidelity_item}"))
                        
                        # calculate objective (total spark time)
                        total_spark_time = summary.get('total_spark_time', 0)
                        objective = total_spark_time if summary.get('successful_files', 0) > 0 else float('inf')
                        
                        # create result record
                        result_record = {
                            'config_id': config_idx - 1,  # convert to 0-based index
                            'fidelity': fidelity,
                            'database': database,
                            'config': summary.get('config', {}),  # extract config from summary
                            'objective': objective,
                            'elapsed_time': summary.get('overall_elapsed_time', 0),
                            'spark_time': total_spark_time,
                            'overhead': summary.get('overhead', 0),
                            'timeout': False,
                            'success': summary.get('successful_files', 0) == summary.get('total_files', 0),
                            'successful_files': summary.get('successful_files', 0),
                            'total_files': summary.get('total_files', 0)
                        }
                        
                        sql_results = result_data.get('results', [])
                        sql_performance = extract_sql_performance_data(sql_results)
                        result_record.update(sql_performance)
                        results.append(result_record)
                        logger.info(f"rebuilt result: config_{config_idx}, fidelity_{fidelity}, objective={objective:.2f}")
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"failed to rebuild result from {result_file}: {e}")
                        continue
    
    results.sort(key=lambda x: (x['config_id'], x['fidelity']))
    logger.info(f"results sorted by config_id and fidelity, total: {len(results)}")
    
    return results

def filter_expert_config_space(original_config_space):
    import json
    from ConfigSpace import ConfigurationSpace

    expert_space_path = "./configs/config_space/expert_space.json"
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
    config_space.seed(seed)
    configs = []
    
    i = 0
    while len(configs) < num_samples:        
        config = config_space.sample_configuration()
        if config not in configs:
            configs.append(config)
            logger.info(f"sample config {i+1}/{num_samples}: {config}")
            i += 1
        else:
            logger.info(f"duplicate config found, resampling...")
    return configs

def execute_sql_with_timing(spark, sql_content, sql_file, shuffle_seed=None):
    logger.info(f"execute sql file: {sql_file}")
    
    queries = [q.strip() for q in sql_content.split(';') if q.strip()]
    logger.info(f"  found {len(queries)} queries")
        
    query_times = []
    total_start_time = time.time()
    
    for i, query in enumerate(queries):
        if not query:
            continue 
        logger.info(f"  execute query {i+1}/{len(queries)}: {query[:50]}...")
        
        for node in SPARK_NODES:
            clear_cache_on_remote(node)

        query_start_time = time.time()
        try:
            result = spark.sql(query)
            logger.info(f"      select query...")
            collected_data = result.collect()
            logger.info(f"      query return {len(collected_data)} rows")
            
            if len(collected_data) > 0:
                logger.debug(f"      all collected data:")
                for j, row in enumerate(collected_data):
                    logger.debug(f"        row{j+1}: {row}")
            else:
                logger.debug(f"      results are empty")
            
            query_elapsed = time.time() - query_start_time
            query_times.append({
                "query_index": i,
                "query": sql_file[: -4] if sql_file.endswith('.sql') else sql_file + "_" + str(i),
                "elapsed_time": query_elapsed,
                "status": "success"
            })
            
            logger.info(f"      query {i+1} completed, time: {query_elapsed:.2f}s")
            spark.catalog.clearCache()
            
        except Exception as e:
            query_elapsed = time.time() - query_start_time
            query_times.append({
                "query_index": i,
                "query": sql_file[: -4] if sql_file.endswith('.sql') else sql_file + "_" + str(i),
                "elapsed_time": query_elapsed,
                "status": "error",
                "error": str(e)
            })
            
            logger.error(f"      query {i+1} failed: {str(e)}")
            break
    
    total_elapsed = time.time() - total_start_time
    
    return {
        "sql_file": sql_file,
        "total_elapsed_time": total_elapsed,
        "query_count": len(query_times),
        "queries": query_times,
        "status": "success" if all(q["status"] == "success" for q in query_times) else "error"
    }

def create_spark_session(config, app_name, database=None) -> SparkSession:
    spark_builder = SparkSession.builder.appName(app_name).enableHiveSupport()
    if hasattr(config, 'get_dictionary'):
        config_dict = config.get_dictionary()
    else:
        config_dict = config
    for key, value in config_dict.items():
        if key.startswith('spark.'):
            formatted_value = format_spark_config_value(key, value)
            spark_builder = spark_builder.config(key, formatted_value)
            logger.debug(f"set spark config: {key} = {formatted_value}")
    spark = spark_builder.getOrCreate()
    if database:
        spark.sql(f"USE {database}")
        logger.info(f"database set to: {database}")
    return spark

def evaluate_config_on_fidelity(config, fidelity, database, sql_files, 
                                config_idx, experiment_dir, sql_dir=DATA_DIR):
    try:
        logger.info(f"evaluate config {config_idx} on fidelity {fidelity} (database: {database})")
        
        config_dir = f"{experiment_dir}/config_{config_idx}"
        fidelity_dir = f"{config_dir}/{fidelity}"
        os.makedirs(fidelity_dir, exist_ok=True)
        
        spark = create_spark_session(config, app_name=f"DataVolumeCorrelation_{config_idx}_{fidelity}")
        logger.info("SparkSession created")
        logger.info(f"Master: {spark.conf.get('spark.master')}")
        logger.info(f"App ID: {spark.sparkContext.applicationId}")
        
        overall_start_time = time.time()
        results = []
        
        for sql_file in sql_files:
            sql_path = os.path.join(sql_dir, sql_file + '.sql')
            
            try:
                with open(sql_path, 'r') as f:
                    sql_content = f.read()
                
                new_spark = spark.newSession()
                new_spark.sql(f"USE {database}")
                
                result = execute_sql_with_timing(new_spark, sql_content, sql_file)
                results.append(result)
                logger.info(f"  {sql_file} completed, total time: {result['total_elapsed_time']:.2f}s")
                
            except Exception as e:
                logger.error(f"  error when processing {sql_file}: {str(e)}")
                results.append({
                    "sql_file": sql_file,
                    "total_elapsed_time": 0,
                    "query_count": 0,
                    "queries": [],
                    "status": "error",
                    "error": str(e)
                })
        
        overall_elapsed_time = time.time() - overall_start_time
        
        # Calculate total Spark execution time
        successful_results = [r for r in results if r["status"] == "success"]
        total_spark_time = 0
        for r in successful_results:
            for query in r.get('queries', []):
                if query.get('status') == 'success':
                    total_spark_time += query.get('elapsed_time', 0)
        
        overhead = overall_elapsed_time - total_spark_time
        
        # Calculate objective (total execution time)
        objective = total_spark_time if successful_results else float('inf')
        
        # Save detailed results
        detailed_results = {
            "summary": {
                "total_files": len(sql_files),
                "successful_files": len(successful_results),
                "failed_files": len(results) - len(successful_results),
                "overall_elapsed_time": overall_elapsed_time,
                "total_spark_time": total_spark_time,
                "overhead": overhead,
                "overhead_percentage": overhead/overall_elapsed_time*100 if overall_elapsed_time > 0 else 0,
                "config_idx": config_idx,
                "fidelity": fidelity,
                "database": database,
                "config": config.get_dictionary() if hasattr(config, 'get_dictionary') else config
            },
            "results": results
        }
        
        # Save results to file
        result_file = os.path.join(fidelity_dir, "execution_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        spark.stop()
        logger.info("SparkSession closed")
        
        result_data = {
            'objective': objective,
            'elapsed_time': overall_elapsed_time,
            'spark_time': total_spark_time,
            'overhead': overhead,
            'timeout': False,
            'success': len(successful_results) == len(sql_files),
            'successful_files': len(successful_results),
            'total_files': len(sql_files)
        }
        sql_performance = extract_sql_performance_data(results)
        result_data.update(sql_performance)
        
        return result_data
            
    except Exception as e:
        logger.error(f"error while evaluating config: {e}")
        spark.stop()
        error_result = {
            'objective': float('inf'),
            'elapsed_time': 0,
            'spark_time': 0,
            'overhead': 0,
            'timeout': True,
            'success': False,
            'error': str(e),
            'successful_files': 0,
            'total_files': len(sql_files)
        }
        add_error_sql_performance_data(error_result, sql_files)
        return error_result


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
            'avg_elapsed_time': df['elapsed_time'].mean(),
            'avg_spark_time': df[df['success']]['spark_time'].mean(),
            'avg_overhead': df['overhead'].mean()
        }
    }
    
    for fidelity in df['fidelity'].unique():
        fidelity_data = df[df['fidelity'] == fidelity]
        stats['fidelity_stats'][str(fidelity)] = {
            'count': len(fidelity_data),
            'success_rate': fidelity_data['success'].mean(),
            'avg_objective': fidelity_data[fidelity_data['success']]['objective'].mean(),
            'avg_elapsed_time': fidelity_data['elapsed_time'].mean(),
            'avg_spark_time': fidelity_data[fidelity_data['success']]['spark_time'].mean(),
            'avg_overhead': fidelity_data['overhead'].mean()
        }
    
    stats_path = os.path.join(output_dir, 'validation_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"save statistics to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description='Data volume correlation experiment using PySpark Session')
    parser.add_argument('--num_samples', type=int, default=100, help='sample configs number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--test_mode', action='store_true', default=False, help='test mode, only evaluate a few configs')
    parser.add_argument('--sql_dir', default=DATA_DIR, help='SQL file directory path')
    parser.add_argument('--resume', action='store_true', help='resume from latest experiment directory')
    parser.add_argument('--history', type=str, default=None, help='path to existing configs history file to load')
    parser.add_argument('--history_start_idx', type=int, default=1, help='starting index for history configs (default: 1)')
    args = parser.parse_args()
    
    if args.resume:
        experiment_dir = find_latest_experiment_dir()
        if experiment_dir:
            logger.info(f"resuming from existing experiment directory: {experiment_dir}")
        else:
            logger.info("no existing experiment directory found, creating new one")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_dir = f"./exps/data_volume_correlation/{timestamp}"
            os.makedirs(experiment_dir, exist_ok=True)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = f"./exps/data_volume_correlation/{timestamp}"
        os.makedirs(experiment_dir, exist_ok=True)
    
    setup_openbox_logging(experiment_dir)
    
    logger.info("=" * 60)
    logger.info("Data volume correlation experiment using PySpark Session starts")
    logger.info("=" * 60)
    logger.info(f"sample configs number: {args.num_samples}")
    logger.info(f"random seed: {args.seed}")
    logger.info(f"experiment directory: {experiment_dir}")
    logger.info(f"SQL directory: {args.sql_dir}")
    logger.info(f"resume mode: {args.resume}")
    logger.info(f"history file: {args.history}")
    
    if args.test_mode:
        args.num_samples = 2
        logger.info(f"test mode: only evaluate {args.num_samples} configs")
    
    fidelity_mapping = create_fidelity_database_mapping(args.test_mode)
    fixed_sqls = create_fixed_sqls(args.sql_dir, args.test_mode)
    
    logger.info(f"fidelity mapping: {fidelity_mapping}")
    logger.info(f"fixed sqls: {len(fixed_sqls)} queries")
    
    # detect the completed configurations and fidelity combinations
    completed_evaluations = detect_completed_evaluations(experiment_dir, fidelity_mapping)
    
    original_config_space = load_space_from_json(HUGE_SPACE_FILE)
    logger.info(f"original config space loaded, total {len(original_config_space.get_hyperparameters())} params")
    
    config_space = filter_expert_config_space(original_config_space)
    logger.info(f"expert config space created, total {len(config_space.get_hyperparameters())} params")
    
    logger.info("begin to sample configs...")
    if args.history:
        logger.info(f"load existing configs from history file: {args.history}")
        logger.info(f"history configs will start from index: {args.history_start_idx}")
        with open(args.history, 'r') as f:
            existing_configs_data = json.load(f)
        existing_configs = []
        for config_data in existing_configs_data:
            config = config_space.sample_configuration()
            for key, value in config_data.items():
                if key in config_space.get_hyperparameter_names():
                    config[key] = value
            existing_configs.append(config)
        configs = existing_configs[:]
        logger.info(f"loaded {len(configs)} configs from history")
    else:
        configs = sample_configurations(config_space, args.num_samples, args.seed)
        logger.info(f"sample configs done, total {len(configs)} configs")
    
    # load the existing results
    results = []
    existing_results_file = os.path.join(experiment_dir, 'config_validation_results.json')
    should_rebuild = False
    
    if os.path.exists(existing_results_file):
        try:
            with open(existing_results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"loaded {len(results)} existing results from previous run")
            
            # check if the number of completed executions is greater than current results
            completed_executions = len(completed_evaluations)
            logger.info(f"found {completed_executions} completed executions")
            
            if len(results) < completed_executions:
                logger.warning(f"existing results count ({len(results)}) is less than completed executions ({completed_executions}), will rebuild")
                should_rebuild = True
                results = []
            else:
                # also check CSV file for consistency
                csv_file = os.path.join(experiment_dir, 'config_validation_results.csv')
                if os.path.exists(csv_file):
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        csv_count = len(df)
                        if csv_count < completed_executions:
                            logger.warning(f"CSV file count ({csv_count}) is less than completed executions ({completed_executions}), will rebuild")
                            should_rebuild = True
                            results = []
                        else:
                            logger.info(f"both JSON ({len(results)}) and CSV ({csv_count}) results match completed executions ({completed_executions})")
                    except Exception as e:
                        logger.warning(f"failed to check CSV file: {e}, will rebuild")
                        should_rebuild = True
                        results = []
                else:
                    logger.warning("CSV file not found, will rebuild")
                    should_rebuild = True
                    results = []
                
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"failed to load existing results: {e}")
            should_rebuild = True
            results = []
    else:
        logger.info("no config_validation_results.json found")
        should_rebuild = True
    
    if should_rebuild:
        logger.info("rebuilding results from execution_results.json files")
        results = rebuild_results_from_execution_files(experiment_dir, fidelity_mapping)
        logger.info(f"rebuilt {len(results)} results from execution files")
        
        if results:
            save_results(results, experiment_dir)
            logger.info("saved rebuilt results to config_validation_results.json, csv and statistics")
    
    total_evaluations = len(configs) * len(fidelity_mapping)
    remaining_evaluations = total_evaluations - len(completed_evaluations)
    current_evaluation = len(completed_evaluations)
    
    logger.info("begin to evaluate configs...")
    logger.info(f"total evaluations: {total_evaluations}")
    logger.info(f"completed evaluations: {len(completed_evaluations)}")
    logger.info(f"remaining evaluations: {remaining_evaluations}")
    
    return 

    for config_idx, config in enumerate(configs):
        # Calculate actual config index based on history start index
        actual_config_idx = config_idx + args.history_start_idx - 1 if args.history else config_idx
        logger.info(f"evaluate config {actual_config_idx + 1} (config {config_idx + 1}/{len(configs)})")
        logger.info(f"  config: {config}")
        
        for fidelity_str in fidelity_mapping.keys():
            fidelity = float(fidelity_str)
            database = fidelity_mapping[fidelity_str]
            
            # check if the evaluation is completed
            if (actual_config_idx + 1, fidelity) in completed_evaluations:
                logger.info(f"  skip completed evaluation: config_{actual_config_idx + 1}, fidelity_{fidelity}")
                continue
            
            current_evaluation += 1
            
            logger.info(f"   evaluation progress: {current_evaluation}/{total_evaluations} - Fidelity: {fidelity}, Database: {database}")
            
            result = evaluate_config_on_fidelity(config, fidelity, database, fixed_sqls, 
                                                actual_config_idx + 1, experiment_dir, args.sql_dir)
            
            result_record = {
                'config_id': actual_config_idx,
                'fidelity': fidelity,
                'database': database,
                'config': config.get_dictionary() if hasattr(config, 'get_dictionary') else config,
            }
            
            for key, value in result.items():
                if key not in ['config_id', 'fidelity', 'database', 'config']:
                    result_record[key] = value
            
            results.append(result_record)
            
            # save the results to avoid losing data when interrupted
            save_results(results, experiment_dir)
            
            logger.info(f"result record: objective={result['objective']:.2f}, "
                       f"elapsed_time={result['elapsed_time']:.2f}s, "
                       f"spark_time={result['spark_time']:.2f}s, "
                       f"overhead={result['overhead']:.2f}s, "
                       f"success={result['success']}")
    
    logger.info("=" * 60)
    logger.info("all configs evaluated successfully")
    logger.info("=" * 60)
    
    logger.info("save evaluation results to experiment directory...")
    save_results(results, experiment_dir)
    
    df = pd.DataFrame(results)
    
    logger.info("=" * 60)
    logger.info("Final Results Summary:")
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
            avg_spark_time = valid_fidelity_data['spark_time'].mean()
            avg_spark_time_str = f"{avg_spark_time:.2f}"
            avg_overhead = valid_fidelity_data['overhead'].mean()
            avg_overhead_str = f"{avg_overhead:.2f}"
        else:
            avg_obj_str = "N/A"
            avg_elapsed_valid_str = "N/A"
            avg_spark_time_str = "N/A"
            avg_overhead_str = "N/A"
        
        logger.info(f"  Fidelity {fidelity} ({fidelity_mapping[fidelity]}):")
        logger.info(f"    success rate: {success_rate:.2%} ({len(valid_fidelity_data)}/{len(fidelity_data)})")
        logger.info(f"    valid results: {len(valid_fidelity_data)} - avg objective: {avg_obj_str}s")
        logger.info(f"    avg elapsed: {avg_elapsed_valid_str}s, avg spark time: {avg_spark_time_str}s, avg overhead: {avg_overhead_str}s")
    
    logger.info("=" * 60)
    logger.info("Data volume correlation experiment using PySpark Session completed")
    logger.info(f"results saved to: {experiment_dir}")
    logger.info(f"logs saved to: {experiment_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
