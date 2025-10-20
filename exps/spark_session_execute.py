"""
SparkSession multiple SQL files execution test script

Use the same SparkSession to execute multiple SQL files, and record the detailed execution time statistics

Usage:
python spark_session_execute.py --sql-dir /root/codes/multique_fidelity_spark --database tpcds_30g --runs 5 --output-format both

Options:
  --sql-dir: SQL file directory path
  --database: target database name
  --limit: limit the number of SQL files to execute (for testing)
  --runs: number of runs; each run shuffles SQL execution order
  --shuffle-seed: random seed for shuffling (per run uses seed+run_index)
  --output-format: output results as json, csv, or both
"""

import os
import sys
import csv
import time
import json
import random
import argparse
from datetime import datetime
from openbox import logger
from pyspark.sql import SparkSession
from utils.spark import custom_sort, get_full_queries_tasks, clear_cache_on_remote
import builtins
from config import SPARK_NODES


def setup_openbox_logging(experiment_dir):
    logger_kwargs = {
        'name': 'spark_session_experiment',
        'logdir': experiment_dir
    }
    logger.init(**logger_kwargs)


def execute_sql_with_timing(spark, sql_content, sql_file, shuffle_queries=False, shuffle_seed=None):
    logger.info(f"execute sql file: {sql_file}")
    
    queries = [q.strip() for q in sql_content.split(';') if q.strip()]
    logger.info(f"  found {len(queries)} queries")
    
    # Shuffle queries if requested
    if shuffle_queries and len(queries) > 1:
        rng = random.Random(shuffle_seed)
        rng.shuffle(queries)
        logger.info(f"  shuffled {len(queries)} queries")
    
    query_times = []
    total_start_time = time.time()
    
    for i, query in enumerate(queries):
        if not query:
            continue
            
        logger.info(f"  execute query {i+1}/{len(queries)}: {query[:50]}...")
        
        query_start_time = time.time()
        try:
            result = spark.sql(query)
            logger.info(f"      selct query...")
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
                "query": query,
                "elapsed_time": query_elapsed,
                "status": "success"
            })
            
            logger.info(f"      query {i+1} completed, time: {query_elapsed:.2f}s")
            spark.catalog.clearCache()
            
        except Exception as e:
            query_elapsed = time.time() - query_start_time
            query_times.append({
                "query_index": i,
                "query": query,
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


def main():
    parser = argparse.ArgumentParser(description='SparkSession batch execute SQL files')
    parser.add_argument('--sql-dir', default='/home/hive-testbench-hdp3/spark-queries-tpcds/',
                       help='SQL file directory path')
    parser.add_argument('--database', default='tpcds_30g',
                       help='target database name')
    parser.add_argument('--limit', type=int, default=None,
                       help='limit the number of SQL files to execute (for testing)')
    parser.add_argument('--runs', type=int, default=1,
                       help='number of runs; each run shuffles SQL execution order')
    parser.add_argument('--shuffle-seed', type=int, default=42,
                       help='random seed for shuffling (per run uses seed+run_index)')
    parser.add_argument('--output-format', choices=['json', 'csv', 'both'], default='json',
                       help='output results as json, csv, or both')
    
    args = parser.parse_args()
    random.seed(args.shuffle_seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"/root/codes/multique_fidelity_spark/exps/spark_session_execute/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    setup_openbox_logging(experiment_dir)
    
    logger.info("=" * 60)
    logger.info("SparkSession batch execute SQL files experiment starts")
    logger.info("=" * 60)
    
    sql_dir = args.sql_dir
    
    if not os.path.exists(sql_dir):
        logger.error(f"SQL directory not found: {sql_dir}")
        sys.exit(1)
    
    spark = None
        
    
    try:
        if os.path.isfile(sql_dir) and sql_dir.lower().endswith('.sql'):
            base_dir = os.path.dirname(sql_dir)
            sql_files = [os.path.basename(sql_dir)]
            logger.info(f"single SQL file mode: {sql_dir}")
        else:
            base_dir = sql_dir
            if os.path.exists(base_dir):
                sql_files = [f + '.sql' for f in get_full_queries_tasks(base_dir)]
            else:
                logger.warning(f"SQL directory not found: {base_dir}, try to use default directory")
                sql_files = [f + '.sql' for f in get_full_queries_tasks()]
        
        if args.limit:
            sql_files = sql_files[:args.limit]
            logger.info(f"limit to execute {args.limit} SQL files")
        
        logger.info(f"found {len(sql_files)} SQL files")
        
        runs = builtins.max(1, int(args.runs))
        all_runs = []
        for run_index in range(runs):
            logger.info("=" * 60)
            logger.info(f"Run {run_index+1}/{runs}: shuffle SQL file execution order")
            logger.info("=" * 60)

            # Create a fresh SparkSession for each run
            spark = SparkSession.builder \
                .appName("MultiQueryExecution") \
                .enableHiveSupport() \
                .config("spark.default.parallelism", "876") \
                .config("spark.driver.cores", "10") \
                .config("spark.driver.maxResultSize", "2550m") \
                .config("spark.driver.memory", "77g") \
                .config("spark.executor.cores", "31") \
                .config("spark.executor.instances", "20") \
                .config("spark.executor.memory", "72g") \
                .config("spark.executor.memoryOverhead", "15241m") \
                .config("spark.locality.wait", "0") \
                .config("spark.network.timeout", "27004") \
                .config("spark.scheduler.minRegisteredResourcesRatio", "1.0") \
                .config("spark.sql.autoBroadcastJoinThreshold", "267386880") \
                .config("spark.sql.broadcastTimeout", "3244") \
                .config("spark.sql.shuffle.partitions", "546") \
                .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "85") \
                .config("spark.task.cpus", "2") \
                .getOrCreate()

            for node in SPARK_NODES:
                clear_cache_on_remote(node)

            # Only shuffle if more than 1 run, and not on the last run
            if runs > 1 and run_index < runs - 1:
                rng = random.Random(args.shuffle_seed + run_index if args.shuffle_seed is not None else None)
                sql_files_order = list(sql_files)
                rng.shuffle(sql_files_order)
                logger.info(f"  shuffled execution order for run {run_index+1}")
            else:
                sql_files_order = list(sql_files)
                if runs > 1:
                    logger.info(f"  using original order for final run {run_index+1}")
                else:
                    logger.info(f"  using original order (single run)")

            overall_start_time = time.time()
            results = []

            for sql_file in sql_files_order:
                sql_path = os.path.join(base_dir, sql_file)

                try:
                    with open(sql_path, 'r') as f:
                        sql_content = f.read()

                    # Create a new session for each SQL file execution
                    new_spark = spark.newSession()
                    new_spark.sql(f"USE {args.database}")
                    logger.info(f"database set to: {args.database}")
                    
                    # Determine if we should shuffle queries within the file
                    should_shuffle_queries = runs > 1 and run_index < runs - 1
                    query_shuffle_seed = args.shuffle_seed + run_index if args.shuffle_seed is not None else None
                    
                    result = execute_sql_with_timing(new_spark, sql_content, sql_file, 
                                                   shuffle_queries=should_shuffle_queries, 
                                                   shuffle_seed=query_shuffle_seed)
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

            logger.info("calculate execution time statistics...")
            successful_results = [r for r in results if r["status"] == "success"]
            total_spark_time = 0
            for r in successful_results:
                total_spark_time += r['total_elapsed_time']
            logger.info(f"Spark execution time total: {total_spark_time:.2f}s")

            overhead = overall_elapsed_time - total_spark_time

            successful_files = [r for r in results if r["status"] == "success"]
            failed_files = [r for r in results if r["status"] != "success"]

            logger.info("=" * 60)
            logger.info("execution result statistics")
            logger.info("=" * 60)
            logger.info(f"total SQL files: {len(sql_files_order)}")
            logger.info(f"successful execution: {len(successful_files)}")
            logger.info(f"execution failed: {len(failed_files)}")
            logger.info(f"overall execution time: {overall_elapsed_time:.2f}s")
            logger.info(f"Spark execution time total: {total_spark_time:.2f}s")
            logger.info(f"Overhead: {overhead:.2f}s ({overhead/overall_elapsed_time*100:.1f}%)")

            if failed_files:
                logger.info("failed SQL files:")
                for failed in failed_files:
                    logger.info(f"  - {failed['sql_file']}: {failed.get('error', 'Unknown error')}")

            detailed_results = {
                "summary": {
                    "total_files": len(sql_files_order),
                    "successful_files": len(successful_files),
                    "failed_files": len(failed_files),
                    "overall_elapsed_time": overall_elapsed_time,
                    "total_spark_time": total_spark_time,
                    "overhead": overhead,
                    "overhead_percentage": overhead/overall_elapsed_time*100 if overall_elapsed_time > 0 else 0,
                    "run_index": run_index + 1,
                    "runs": runs
                },
                "results": results,
                "execution_order": sql_files_order
            }
            all_runs.append(detailed_results)

            logger.info("execution time of each SQL file:")
            sorted_successful_files = sorted(successful_files, key=lambda x: custom_sort(x['sql_file']))
            for result in sorted_successful_files:
                logger.info(f"  {result['sql_file']}: {result['total_elapsed_time']:.2f}s")

            try:
                spark.stop()
                logger.info("SparkSession closed")
            finally:
                spark = None

        # write combined outputs once after all runs
        combined_base = f"spark_session_results_{timestamp}"
        if args.output_format in ("json", "both"):
            combined_json_path = os.path.join(experiment_dir, f"{combined_base}.json")
            combined_payload = {
                "meta": {
                    "runs": runs,
                    "database": args.database,
                    "sql_dir": sql_dir,
                    "shuffle_seed": args.shuffle_seed,
                    "generated_at": timestamp
                },
                "run_results": all_runs
            }
            with open(combined_json_path, 'w', encoding='utf-8') as f:
                json.dump(combined_payload, f, indent=2, ensure_ascii=False)
            logger.info(f"combined JSON results saved to: {combined_json_path}")

        if args.output_format in ("csv", "both"):
            combined_csv_path = os.path.join(experiment_dir, f"{combined_base}.csv")
            with open(combined_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["run_index", "order_index", "sql_file", "status", "total_elapsed_time", "query_count"]) 
                for run in all_runs:
                    run_idx = run["summary"]["run_index"]
                    order = run["execution_order"]
                    results_map = {r["sql_file"]: r for r in run["results"]}
                    for idx, sql_file in enumerate(order, start=1):
                        rec = results_map.get(sql_file)
                        if rec is None:
                            writer.writerow([run_idx, idx, sql_file, "missing", "", ""]) 
                        else:
                            writer.writerow([
                                run_idx,
                                idx,
                                sql_file,
                                rec.get("status", ""),
                                f"{rec.get('total_elapsed_time', 0):.6f}",
                                rec.get("query_count", 0)
                            ])
            logger.info(f"combined CSV summary saved to: {combined_csv_path}")

        logger.info("=" * 60)
        logger.info("SparkSession batch execute SQL files experiment completed")
        logger.info(f"results saved to: {experiment_dir}")
        logger.info("=" * 60)
        
    finally:
        if spark is not None:
            spark.stop()
            logger.info("SparkSession closed")


if __name__ == "__main__":
    main()
