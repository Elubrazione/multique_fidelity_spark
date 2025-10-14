import re
import numpy as np
import pandas as pd
from openbox import logger
from config import FILE_TIMEOUT_CSV

def convert_to_spark_params(config: dict):
    memory_params = {
        'spark.executor.memory': 'g',
        'spark.driver.memory': 'g',
        'spark.executor.memoryOverhead': 'm',
        'spark.driver.maxResultSize': 'm',
        'spark.broadcast.blockSize': 'm',
        'spark.io.compression.lz4.blockSize': 'k',
        'spark.io.compression.snappy.blockSize': 'k',
        'spark.io.compression.zstd.bufferSize': 'k',
        'spark.shuffle.service.index.cache.size': 'm',
        
    }
    spark_params = []
    for k, v in config.items():
        if k in memory_params:
            spark_params.extend(["--conf", f"{k}={v}{memory_params[k]}"])
        elif k == 'spark.sql.autoBroadcastJoinThreshold':
            bytes_val = int(v) * 1024 ** 2
            spark_params.extend(["--conf", f"{k}={bytes_val}"])
        else:
            spark_params.extend(["--conf", f"{k}={v}"])
    return spark_params

def custom_sort(key):
    parts = re.findall(r'\d+|[a-zA-Z]+', key)
    sort_key = []
    for part in parts:
        if part.isdigit():
            digits = [int(d) for d in part] + [float('inf')]
            sort_key.extend(digits)
        else:
            sort_key.append(part)
    return tuple(sort_key)

def analyze_timeout_and_get_fidelity_details(file_path=FILE_TIMEOUT_CSV, percentile=100,
                                             ratio_list=[], round_num=5, debug=False, add_on_ratio=1.5):
    elapsed_timeout_dicts = analyze_sqls_timeout_from_csv(file_path=file_path, percentile=percentile)

    fidelity_details = {}
    excluded_sqls = set()
    for r in ratio_list:
        if int(r) == 1: continue
        selected_queries, total_time = off_line_greedy_selection(
            time_dicts=elapsed_timeout_dicts,
            ratio=r, excluded_sqls=excluded_sqls
        )
        if debug:
            print(total_time)
            for k, v in selected_queries.items():
                print(k, "  ", v)
        fidelity_details[round(r, round_num)] = sorted(list(selected_queries.keys()), key=lambda x: custom_sort(x))
        excluded_sqls.update(selected_queries.keys())
    fidelity_details[round(1, round_num)] = list(elapsed_timeout_dicts.keys())
    
    logger.info(fidelity_details)
    return fidelity_details, elapsed_timeout_dicts

def analyze_sqls_timeout_from_csv(file_path=FILE_TIMEOUT_CSV, percentile=33):
    df = pd.read_csv(file_path)
    df = df[df['status'] == 'complete']
    et_columns = [col for col in df.columns if col.startswith("et_q")]

    et_quantiles = {}
    for col in et_columns:
        values = df[col].dropna().values
        if len(values) > 0:
            clean_key = col.replace('et_', '', 1)
            et_quantiles[clean_key] = np.percentile(values, percentile)
    return et_quantiles

def off_line_greedy_selection(time_dicts: dict, ratio=1, excluded_sqls=None):
    sorted_queries = sorted(time_dicts.items(), key=lambda x: -x[1])
    time_target = sum(time_dicts.values()) * ratio
    
    selected_queries = {}
    total_time = 0
    for q, t in sorted_queries:
        if total_time + t <= time_target:
            if q not in excluded_sqls:
                selected_queries[q] = t
                total_time += t
        if total_time >= time_target * 0.98:
            break
    return selected_queries, total_time