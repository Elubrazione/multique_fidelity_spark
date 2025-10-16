from random import shuffle
import csv
import time
import os
from datetime import datetime
import pandas as pd
from utils.spark import get_full_queries_tasks, custom_sort, \
    run_spark, parse_spark_log, clear_cache_on_remote
from config import SPARK_NODES


def execute_shuffle_validation(queries_list, log_dir):
    """
    执行不同查询顺序的验证测试
    """
    servers = SPARK_NODES
    
    timestamp = log_dir.split('/')[-1]

    csv_path = f"{log_dir}/{timestamp}_shuffle.csv"

    default_config = {}

    with open(csv_path, 'w', newline='') as csvfile:
        all_queries = set()
        for query_seq in queries_list:
            all_queries.update(query_seq)
        all_queries = sorted(list(all_queries), key=custom_sort)
        
        fieldnames = ['sample_id', 'query_order', 'status', 'wall_time', 'spark_time', 'overhead'] + \
                    [f"qt_{q}" for q in all_queries] + [f"et_{q}" for q in all_queries]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for cnt, query_order in enumerate(queries_list):
            for server in servers:
                clear_cache_on_remote(server)
            
            res_dir = f"{log_dir}/order_{cnt}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(res_dir)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing query order {cnt+1}/{len(queries_list)}")
            print(f"Query order: {query_order}\n")
            
            etime_details = {}
            completed = True
            start_time = time.time()
            
            for sql in query_order:
                sql_start_time = time.time()
                test_result = run_spark(default_config, sql, res_dir)
                if test_result['status'] != "success":
                    completed = False
                    print(test_result['error'])

                etime_details["et_" + sql] = time.time() - sql_start_time
                if not completed:
                    break
            
            elapsed = time.time() - start_time
            status = 'complete' if completed else test_result['status']
            spark_time, qtime_details = parse_spark_log(res_dir, query_order)
            
            record = {
                'sample_id': cnt + 1,
                'query_order': ','.join(query_order),
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
            
            print(f"Order {cnt+1} completed: status={status}, wall_time={elapsed:.2f}s, spark_time={spark_time:.2f}s\n")
    return csv_path

def main():
    queries = get_full_queries_tasks()
    # queries = sorted(['q10'], key=custom_sort)

    print(f"Total queries: {len(queries)}")
    print(f"Queries: {queries}")
    
    shuffled_queries = [queries.copy()]
    
    for i in range(10):
        shuffled = queries.copy()
        shuffle(shuffled)
        shuffled_queries.append(shuffled)
    
    print(f"Generated {len(shuffled_queries)} different query orders")
    
    timestamp = "/root/codes/haisi_scripts/exps/shuffle/test_" + datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(timestamp)
    
    csv_path = execute_shuffle_validation(shuffled_queries, timestamp)
    
    print(f"Test completed! Results saved to: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print("\n=== Results Summary ===")
    print(f"Total orders tested: {len(df)}")
    print(f"Successful orders: {len(df[df['status'] == 'complete'])}")
    print(f"Average wall time: {df['wall_time'].mean():.2f}s")
    print(f"Average spark time: {df['spark_time'].mean():.2f}s")
    print(f"Average overhead: {df['overhead'].mean():.2f}s")
    
    print("\n=== Performance by Query Order ===")
    for idx, row in df.iterrows():
        print(f"Order {row['sample_id']}: {row['wall_time']:.2f}s (spark: {row['spark_time']:.2f}s, overhead: {row['overhead']:.2f}s)")

if __name__ == "__main__":
    main()

