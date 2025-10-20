import re
import numpy as np
import pandas as pd
from datetime import datetime
from openbox import logger
from config import FILE_TIMEOUT_CSV, DATABASE, DATA_DIR
import os, subprocess, paramiko

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
    elapsed_timeout_dicts = analyze_sqls_timeout_from_csv(file_path=file_path, percentile=percentile, add_on_ratio=add_on_ratio)

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

def analyze_sqls_timeout_from_csv(file_path=FILE_TIMEOUT_CSV, percentile=33, add_on_ratio=1.5):
    df = pd.read_csv(file_path)
    df = df[df['status'] == 'complete']
    et_columns = [col for col in df.columns if col.startswith("et_q")]

    et_quantiles = {}
    for col in et_columns:
        values = df[col].dropna().values
        if len(values) > 0:
            clean_key = col.replace('et_', '', 1)
            et_quantiles[clean_key] = np.percentile(values, percentile)
    et_quantiles = {k: v * add_on_ratio + v for k, v in et_quantiles.items()}
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

def run_spark(config, sql, result_dir):
    spark_cmd = [
        "spark-sql",
        "--master", "yarn",
        "--database", f"{DATABASE}",
        *convert_to_spark_params(config),
        "-f", f"{DATA_DIR}/{sql}.sql"
    ]

    log_file = f"{result_dir}/{sql}.log"
    try:
        with open(log_file, 'w') as f:
            subprocess.run(spark_cmd, check=True, stdout=f, stderr=f, text=True)
        return {"status": "success"}
    except subprocess.CalledProcessError as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def get_full_queries_tasks(query_dir=f"{DATA_DIR}/"):
    queries = os.listdir(query_dir)
    queries = sorted(
        [q[: -4] for q in queries if q.endswith('.sql')],
        key=lambda x: custom_sort(x)
    )
    return queries

def clear_cache_on_remote(server, username = "root", password = "root"):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, username=username, password=password)
        stdin, stdout, stderr = client.exec_command("echo 3 > /proc/sys/vm/drop_caches")
        error = stderr.read().decode()
        if error:
            print(f"[{server}] Error: {error}")
        else:
            print(f"[{server}] Cache cleared successfully.")

        stdin, stdout, stderr = client.exec_command("free -g")
        print(f"[{server}] Memory status:\n{stdout.read().decode()}")

        client.close()
    except Exception as e:
        print(f"[{server}] Error: {e}")

def parse_spark_log(log_dir, queries, suffix_type='log'):
    results = {}
    time_pattern = re.compile(r'Time taken: ([\d.]+) seconds')
    cost = 0
    for i in queries:
        results[i] = np.Inf
        if os.path.exists(os.path.join(log_dir, f"{i}.{suffix_type}")):
            with open(os.path.join(log_dir, f"{i}.{suffix_type}"), 'r') as f:
                for line in f:
                    time_match = time_pattern.search(line)
                    if time_match:
                        execution_time = float(time_match.group(1))
                        results[i] = execution_time
                        cost += execution_time

    sorted_results = dict(sorted(results.items(), key=lambda x: custom_sort(x[0])))
    return cost, sorted_results

def extract_performance_from_logs(log_dir, output_path):
    results = {}
    timestamps = []

    time_pattern = re.compile(r'Time taken: ([\d.]+) seconds')
    query_pattern = re.compile(r'(q\d+[a-z]?)')  # 匹配 q1, q1a, q1b 格式
    datetime_pattern = re.compile(r'(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})')

    for filename in os.listdir(log_dir):
        if filename.endswith('.log'):
            query_match = query_pattern.search(filename)
            if query_match:
                query_name = query_match.group(1)
                results[query_name] = np.Inf
                with open(os.path.join(log_dir, filename), 'r') as f:
                    for line in f:
                        time_match = time_pattern.search(line)
                        if time_match:
                            execution_time = float(time_match.group(1))
                            results[query_name] = execution_time

                        datetime_match = datetime_pattern.search(line)
                        if datetime_match:
                            timestamp = datetime.strptime(datetime_match.group(1), '%d/%m/%y %H:%M:%S')
                            timestamps.append(timestamp)

    with open(output_path, 'w') as out_file:
        if timestamps:
            earliest: datetime = min(timestamps)
            latest: datetime = max(timestamps)
            time_diff = latest - earliest
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)

            out_file.write(f"start: {earliest.strftime('%d/%m/%y %H:%M:%S')}\n")
            out_file.write(f"end: {latest.strftime('%d/%m/%y %H:%M:%S')}\n")
            out_file.write(f"cost: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n")

        sorted_results = dict(sorted(results.items(), key=lambda x: custom_sort(x[0])))
        cost = 0
        for _, (k, v) in enumerate(sorted_results.items()):
            cost += v
            out_file.write(f"{k}\t{v}\n")

        out_file.write(f'\ntotal: {cost}\n')

    print(f"Results saved to {output_path}")

    times = list(sorted_results.values())
    mean = np.mean(times)
    median = np.median(times)
    p90 = np.percentile(times, 90)
    p99 = np.percentile(times, 99)

    with open(output_path, 'a') as out_file:
        out_file.write("\n=== Statistics ===\n")
        out_file.write(f"Mean: {mean:.2f} s\n")
        out_file.write(f"Median: {median:.2f} s\n")
        out_file.write(f"P90: {p90:.2f} s\n")
        out_file.write(f"P99: {p99:.2f} s\n")


    outliers_p90 = {k: v for k, v in sorted_results.items() if v > p90}
    outliers_p99 = {k: v for k, v in sorted_results.items() if v > p99}

    with open(output_path, 'a') as out_file:
        out_file.write("\n=== Long Tail Queries (P90) ===\n")
        for k, v in outliers_p90.items():
            out_file.write(f"{k}\t{v}\n")

        out_file.write("\n=== Severe Long Tail Queries (P99) ===\n")
        for k, v in outliers_p99.items():
            out_file.write(f"{k}\t{v}\n")

if __name__ == '__main__':
    extract_performance_from_logs(log_dir=f"/root/codes/multique_fidelity_spark/exps/data_volume_correlation/20251019_163750/config_1/0.03", 
    output_path=f"/root/codes/multique_fidelity_spark/exps/data_volume_correlation/20251019_163750/config_1/performance_0.03.txt")