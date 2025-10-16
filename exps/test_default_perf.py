from random import shuffle
import csv
import time
import os
from datetime import datetime
import pandas as pd
from utils.spark import get_full_queries_tasks, custom_sort, \
    run_spark, parse_spark_log, clear_cache_on_remote
from config import SPARK_NODES


def execute_default_perf(queries, log_dir: str):
    servers = SPARK_NODES
    
    timestamp = log_dir.split('/')[-1]

    csv_path = f"{log_dir}/{timestamp}_default_perf.csv"

    default_config = {}

    with open(csv_path, 'w', newline='') as csvfile:
        all_queries = set()
        for query in queries:
            all_queries.add(query)
        all_queries = sorted(list(all_queries), key=custom_sort)
        
        fieldnames = ['queries', 'status', 'wall_time', 'spark_time', 'overhead'] + \
                    [f"qt_{q}" for q in all_queries] + [f"et_{q}" for q in all_queries]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for server in servers:
            clear_cache_on_remote(server)

        res_dir = f"{log_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(res_dir)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing default perf")

        etime_details = {}
        completed = True
        start_time = time.time()
        for query in queries:
            sql_start_time = time.time()
            test_result = run_spark(default_config, query, res_dir)
            if test_result['status'] != "success":
                completed = False
                print(test_result['error'])
            etime_details["et_" + query] = time.time() - sql_start_time
            if not completed:
                break
        elapsed = time.time() - start_time
        status = 'complete' if completed else test_result['status']
        spark_time, qtime_details = parse_spark_log(res_dir, queries)
        
        record = {
            'queries': ','.join(queries),
            'status': status,
            'wall_time': elapsed,
            'spark_time': spark_time,
            'overhead': elapsed - spark_time
        }
            
        for q in all_queries:
            record[f"qt_{q}"] = qtime_details.get(q, 0)
            record[f"et_{q}"] = etime_details.get(f"et_{q}", 0)
        
        writer.writerow(record)
        csvfile.flush()
        
        print(f"Default perf completed: status={status}, wall_time={elapsed:.2f}s, spark_time={spark_time:.2f}s\n")
    return csv_path

def main():
    queries = get_full_queries_tasks()
    # queries = sorted(['q10'], key=custom_sort)
    
    timestamp = "./exps/default_perf/test_" + datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(timestamp)
    
    csv_path = execute_default_perf(queries, timestamp)
    
    print(f"Test completed! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()

