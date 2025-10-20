"""
SparkSession批量执行SQL文件测试脚本
使用同一个SparkSession执行多个SQL文件，记录详细的执行时间统计
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from openbox import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from utils.spark import custom_sort, get_full_queries_tasks


def setup_openbox_logging(experiment_dir):
    logger_kwargs = {
        'name': 'spark_session_experiment',
        'logdir': experiment_dir
    }
    logger.init(**logger_kwargs)


def execute_sql_with_timing(spark, sql_content, sql_file):
    logger.info(f"execute sql file: {sql_file}")
    
    queries = [q.strip() for q in sql_content.split(';') if q.strip()]
    logger.info(f"  found {len(queries)} queries")
    
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
                logger.info(f"      all collected data:")
                for i, row in enumerate(collected_data):
                    logger.info(f"        row{i+1}: {row}")
            else:
                logger.info(f"      results are empty")
            
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
    parser.add_argument('--output-dir', default='.',
                       help='result file output directory')
    
    args = parser.parse_args()
    
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
        
    
    try:
        spark.sql(f"USE {args.database}")
        logger.info(f"database set to: {args.database}")
        
        if os.path.exists(sql_dir):
            sql_files = [f + '.sql' for f in get_full_queries_tasks(sql_dir)]
        else:
            logger.warning(f"SQL directory not found: {sql_dir}, try to use default directory")
            sql_files = [f + '.sql' for f in get_full_queries_tasks()]
        
        if args.limit:
            sql_files = sql_files[:args.limit]
            logger.info(f"limit to execute {args.limit} SQL files")
        
        logger.info(f"found {len(sql_files)} SQL files")
        
        overall_start_time = time.time()
        
        results = []
        
        for sql_file in sql_files:
            sql_path = os.path.join(sql_dir, sql_file)
            
            try:
                with open(sql_path, 'r') as f:
                    sql_content = f.read()
                
                result = execute_sql_with_timing(spark, sql_content, sql_file)
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
        
        # 计算overhead
        overhead = overall_elapsed_time - total_spark_time
        
        # 统计信息
        successful_files = [r for r in results if r["status"] == "success"]
        failed_files = [r for r in results if r["status"] != "success"]
        
        logger.info("=" * 60)
        logger.info("execution result statistics")
        logger.info("=" * 60)
        logger.info(f"total SQL files: {len(sql_files)}")
        logger.info(f"successful execution: {len(successful_files)}")
        logger.info(f"execution failed: {len(failed_files)}")
        logger.info(f"overall execution time: {overall_elapsed_time:.2f}s")
        logger.info(f"Spark execution time total: {total_spark_time:.2f}s")
        logger.info(f"Overhead: {overhead:.2f}s ({overhead/overall_elapsed_time*100:.1f}%)")
        
        if failed_files:
            logger.info("failed SQL files:")
            for failed in failed_files:
                logger.info(f"  - {failed['sql_file']}: {failed.get('error', 'Unknown error')}")
        
        result_file = os.path.join(experiment_dir, f"spark_session_results_{timestamp}.json")
        
        detailed_results = {
            "summary": {
                "total_files": len(sql_files),
                "successful_files": len(successful_files),
                "failed_files": len(failed_files),
                "overall_elapsed_time": overall_elapsed_time,
                "total_spark_time": total_spark_time,
                "overhead": overhead,
                "overhead_percentage": overhead/overall_elapsed_time*100 if overall_elapsed_time > 0 else 0
            },
            "results": results
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"detailed results saved to: {result_file}")
        
        logger.info("execution time of each SQL file:")
        sorted_successful_files = sorted(successful_files, key=lambda x: custom_sort(x['sql_file']))
        for result in sorted_successful_files:
            logger.info(f"  {result['sql_file']}: {result['total_elapsed_time']:.2f}s")
        
        logger.info("=" * 60)
        logger.info("SparkSession batch execute SQL files experiment completed")
        logger.info(f"results saved to: {experiment_dir}")
        logger.info("=" * 60)
        
    finally:
        spark.stop()
        logger.info("SparkSession closed")


if __name__ == "__main__":
    main()
