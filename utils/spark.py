from typing import Any, Dict, Iterable, Optional, Tuple
import os
import re
import subprocess
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from openbox import logger
from pyspark import SparkContext
from pyspark.sql import SparkSession
from config import ConfigManager
import paramiko


SPARK_SIZE_SUFFIXES: Dict[str, str] = {
    'spark.executor.memory': 'g',
    'spark.driver.memory': 'g',
    'spark.executor.memoryOverhead': 'm',
    'spark.driver.memoryOverhead': 'm',
    'spark.driver.maxResultSize': 'm',
    'spark.broadcast.blockSize': 'm',
    'spark.io.compression.snappy.blockSize': 'k',
    'spark.shuffle.service.index.cache.size': 'm',
    'spark.sql.autoBroadcastJoinThreshold': 'm',
    'spark.memory.offHeap.size': 'g',
    'spark.storage.memoryMapThreshold': 'g',
    'spark.kryoserializer.buffer.max': 'm',
    'spark.shuffle.file.buffer': 'k',
    'spark.shuffle.unsafe.file.output.buffer': 'k',
}


def format_spark_config_value(key: str, value: Any) -> str:
    suffix = SPARK_SIZE_SUFFIXES.get(key)
    value_str = str(value)
    if not suffix:
        return value_str

    if value_str.lower().endswith(suffix.lower()):
        return value_str
    return f"{value_str}{suffix}"


def is_spark_context_valid(spark) -> bool:
    try:
        sc = getattr(spark, "sparkContext", None)
        if sc is None or getattr(sc, "_jsc", None) is None:
            return False
        _ = sc.version
        return True
    except Exception:
        return False


def stop_active_spark_context(sleep_seconds: float = 3.0) -> bool:
    try:
        sc = SparkContext._active_spark_context
    except Exception:
        logger.debug("[SparkSession] Unable to access active SparkContext reference")
        return False

    if sc is None:
        return False

    logger.warning("[SparkSession] Found active SparkContext, stopping it")
    try:
        sc.stop()
        logger.info("[SparkSession] SparkContext stopped")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return True
    except Exception as exc:
        logger.debug(f"[SparkSession] Could not stop SparkContext: {type(exc).__name__}: {str(exc)}")
        return False


def stop_active_spark_session() -> None:
    try:
        active_session = SparkSession.getActiveSession()
    except AttributeError:
        logger.warning("[SparkSession] getActiveSession() not available, skipping active session check")
        stop_active_spark_context()
        return
    except Exception as exc:
        logger.warning(f"[SparkSession] Could not check for active session: {type(exc).__name__}: {str(exc)}")
        stop_active_spark_context()
        return

    if active_session is None:
        stop_active_spark_context()
        return

    logger.warning("[SparkSession] Found active SparkSession, stopping it before creating new one")
    try:
        active_session.stop()
        logger.info("[SparkSession] SparkSession stopped")
    except Exception as exc:
        logger.warning(f"[SparkSession] Failed to stop existing session: {type(exc).__name__}: {str(exc)}")
    finally:
        stop_active_spark_context(sleep_seconds=1.0)


def use_database(spark: SparkSession, database: Optional[str]) -> bool:
    if not database:
        return True

    db_name = str(database).strip()
    if not db_name:
        logger.warning("[SparkSession] Empty database name, skipping USE.")
        return False

    logger.info(f"[SparkSession] Attempting to set database: '{db_name}'")
    try:
        spark.sql(f"USE `{db_name}`")
        logger.info(f"[SparkSession] Database set to: {db_name}")
        return True
    except Exception as exc:
        error_msg = str(exc).lower()
        error_type = type(exc).__name__

        if 'does not exist' in error_msg or ('database' in error_msg and 'not found' in error_msg):
            logger.warning(f"[SparkSession] Database '{db_name}' does not exist. Skipping USE.")
            return False

        if 'hivesessionstatebuilder' in error_msg or 'illegalargumentexception' in error_msg:
            logger.error(f"[SparkSession] Database operation failed with session state error: {error_type}: {str(exc)}")
            raise RuntimeError(f"SparkSession state error during database operation: {str(exc)}") from exc

        logger.warning(f"[SparkSession] Failed to set database to '{db_name}': {error_type}: {str(exc)}")
        return False


def create_spark_session(
    config_dict: Dict[str, Any],
    app_name: str,
    database: Optional[str] = None,
    max_retries: int = 2,
) -> SparkSession:
    stop_active_spark_session()

    def _build_spark_builder() -> SparkSession.Builder:
        builder = SparkSession.builder.appName(app_name).enableHiveSupport()
        for key, value in config_dict.items():
            if str(key).startswith('spark.'):
                builder = builder.config(key, format_spark_config_value(key, value))
        return builder

    spark_builder = _build_spark_builder()

    last_exception: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"[SparkSession] Retry attempt {attempt + 1}: ensuring clean state")
                stop_active_spark_session()
                time.sleep(3)
                spark_builder = _build_spark_builder()

            spark = spark_builder.getOrCreate()

            try:
                db_set = use_database(spark, database)
                if not db_set and database is not None:
                    logger.warning(
                        "[SparkSession] Database setting failed but continuing with session. "
                        "This may cause query failures if database is required."
                    )
            except RuntimeError:
                logger.error("[SparkSession] Database operation revealed session state problem, will retry")
                spark.stop()
                raise

            return spark
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                logger.warning(
                    f"[SparkSession] Attempt {attempt + 1} failed, trying to clean up and retry: "
                    f"{type(exc).__name__}: {str(exc)}"
                )
                stop_active_spark_session()
                time.sleep(3)
            else:
                logger.error(
                    f"[SparkSession] Failed to create Spark session with app_name={app_name} "
                    f"after {max_retries} attempts: {type(exc).__name__}: {str(exc)}"
                )
                raise

    if last_exception:
        raise last_exception

    raise RuntimeError("Failed to create Spark session for unknown reasons")


def clear_cluster_cache(spark_nodes: Iterable[str], username: str, password: str) -> None:
    if not spark_nodes:
        return

    for node in spark_nodes:
        try:
            clear_cache_on_remote(node, username=username, password=password)
        except Exception as exc:
            logger.error(f"[SparkSession] Failed to clear cache on {node}: {exc}")


def execute_sql_with_timing(spark, sql_content: str, sql_file: str, *, check_context=is_spark_context_valid):
    queries = [q.strip() for q in sql_content.split(';') if q.strip()]

    total_start_time = time.time()
    per_qt_time = 0.0
    status = 'success'

    for idx, query in enumerate(queries):
        if not query:
            continue
        logger.debug(f"  execute query {idx + 1}/{len(queries)}: {query[:50]}...")

        if not check_context(spark):
            logger.error(f"     {sql_file} query {idx + 1} failed: SparkContext was shut down")
            status = 'error'
            per_qt_time = float('inf')
            raise RuntimeError("SparkContext was shut down")

        query_start_time = time.time()
        try:
            result = spark.sql(query)
            collected = result.collect()
            logger.debug(f"     {sql_file} query {idx + 1} returned {len(collected)} rows")
            per_qt_time += (time.time() - query_start_time)
            logger.info(f"     {sql_file} query {idx + 1} completed")
        except Exception as exc:
            _ = time.time() - query_start_time
            status = 'error'
            py_err = type(exc).__name__
            jvm_info = ""
            try:
                java_exc = getattr(exc, 'java_exception', None)
                if java_exc is not None:
                    jvm_err = java_exc.getClass().getName()
                    try:
                        jvm_msg = str(java_exc.getMessage()) or ""
                        jvm_info = f", jvm={jvm_err}, msg={jvm_msg[:150]}"
                    except Exception:
                        jvm_info = f", jvm={jvm_err}"
            except Exception:
                pass
            logger.error(f"     {sql_file} query {idx + 1} failed (py_err={py_err}{jvm_info})")

            error_msg = str(exc).lower()
            if 'sparkcontext' in error_msg and ('shut down' in error_msg or 'cancelled' in error_msg):
                logger.warning(f"     SparkContext was shut down, will attempt to recreate SparkSession")
                raise RuntimeError("SparkContext was shut down")

            break

    total_elapsed = time.time() - total_start_time
    return {
        'sql_file': sql_file,
        'per_et_time': total_elapsed,
        'per_qt_time': per_qt_time if status == 'success' else float('inf'),
        'status': status
    }


def convert_to_spark_params(config: dict):
    spark_params = []
    for k, v in config.items():
        formatted_value = format_spark_config_value(k, v)
        spark_params.extend(["--conf", f"{k}={formatted_value}"])
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

def run_spark(config, sql, result_dir, database, sql_dir):
    spark_cmd = [
        "spark-sql",
        "--master", "yarn",
        "--database", f"{database}",
        *convert_to_spark_params(config),
        "-f", f"{sql_dir}/{sql}.sql"
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

def get_full_queries_tasks(query_dir):
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
            logger.error(f"[{server}] Error: {error}")
        else:
            logger.info(f"[{server}] Cache cleared successfully.")

        stdin, stdout, stderr = client.exec_command("free -g")
        logger.info(f"[{server}] Memory status:\n{stdout.read().decode()}")

        client.close()
    except Exception as e:
        logger.error(f"[{server}] Error: {e}")


def decode_results_spark(results: str) -> np.ndarray:
    """
    Decode Spark application logs to extract runtime metrics.
    
    Args:
        results: JSON content from Spark log file
        
    Returns:
        Tuple of (run_time, metrics_array)
    """
    result_list = results.split('\n')
    logs = []
    for line in result_list:
        if line.strip() == '':
            continue
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f'Skipping invalid JSON line: {line[:100]}... (error: {e})')
            continue

    start_time, end_time = None, None
    task_metrics = dict()

    cnt = 0
    for event in logs:
        if event['Event'] == "SparkListenerApplicationStart":
            start_time = event['Timestamp']
        elif event['Event'] == "SparkListenerApplicationEnd":
            end_time = event['Timestamp']
        elif event['Event'] == "SparkListenerTaskEnd":
            # Some tasks (e.g., resubmitted tasks) may not have Task Metrics
            if 'Task Metrics' not in event:
                logger.debug(f"Skipping TaskEnd event without Task Metrics (Task End Reason: {event.get('Task End Reason', 'N/A')})")
                continue
            cnt += 1
            metrics_dict = event['Task Metrics']
            for key, value in metrics_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, dict):
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                final_key = "%s_%s_%s" % (key, sub_key, sub_sub_key)
                                task_metrics[final_key] = task_metrics.get(final_key, 0) + sub_sub_value
                        else:
                            final_key = "%s_%s" % (key, sub_key)
                            task_metrics[final_key] = task_metrics.get(final_key, 0) + sub_value
                elif isinstance(value, list):
                    continue
                else:
                    task_metrics[key] = task_metrics.get(key, 0) + value

    if start_time is None or end_time is None:
        logger.warning('Cannot find start or end time in log')
    else:
        run_time = (end_time - start_time) / 1000
        logger.info(f"Application run time: {run_time:.2f} seconds")

    if cnt == 0:
        logger.warning('No TaskEnd events found in log, using fallback metrics')
        raise ValueError('No TaskEnd events found')

    keys = list(task_metrics.keys())
    keys.sort()
    for k, v in task_metrics.items():
        logger.debug(f"{k}: {v / cnt}")
    metrics = np.array([task_metrics[key] / cnt for key in keys])
    logger.info(f"Metrics array shape: {metrics.shape}")

    return metrics

def get_latest_application_id(spark_log_dir: str = "/root/codes/spark-log") -> Optional[str]:
    """
    Get the latest application_id from spark-log directory by finding the newest zstd file.
    
    Returns:
        Application ID extracted from the newest zstd filename, or None if not found
    """
    if not os.path.exists(spark_log_dir):
        logger.warning(f"Spark log directory {spark_log_dir} does not exist.")
        return None
        
    zstd_files = []
    for filename in os.listdir(spark_log_dir):
        if filename.endswith('.zstd'):
            filepath = os.path.join(spark_log_dir, filename)
            mtime = os.path.getmtime(filepath)
            zstd_files.append((filename, mtime))
    
    if not zstd_files:
        logger.warning(f"No zstd files found in {spark_log_dir}")
        return None
        
    zstd_files.sort(key=lambda x: x[1], reverse=True)
    latest_filename = zstd_files[0][0]

    if latest_filename.startswith('application_') and latest_filename.endswith('.zstd'):
        application_id = latest_filename[:-5]
        logger.info(f"Found latest application_id: {application_id}")
        return application_id
    else:
        logger.warning(f"Unexpected filename format: {latest_filename}")
        return None

def resolve_runtime_metrics(
    spark_log_dir: str = "/root/codes/spark-log",
) -> np.ndarray:
    application_id = get_latest_application_id(spark_log_dir)
    if not application_id:
        logger.warning("No application_id found, using fallback metrics")
        raise ValueError("No application_id found")

    zstd_file = os.path.join(spark_log_dir, f"{application_id}.zstd")
    if not os.path.exists(zstd_file):
        logger.warning(f"Zstd file not found: {zstd_file}, using fallback metrics")
        raise ValueError(f"Zstd file not found: {zstd_file}")
    logger.info(f"Found zstd file: {zstd_file}")

    json_file = os.path.join(spark_log_dir, "app.json")
    if os.path.exists(json_file):
        os.remove(json_file)
        logger.info(f"Removed existing json file: {json_file}")
    logger.info(f"Decoding zstd file: {zstd_file} to json file: {json_file}")
    try:
        subprocess.run(['zstd', '-d', zstd_file, '-o', json_file], check=True)
    except Exception as e:
        logger.error(f"Failed to decode zstd file: {e}, using fallback metrics")
        raise ValueError(f"Failed to decode zstd file: {e}")

    json_content = ""
    with open(json_file, 'r') as f:
        json_content = f.read()
    logger.info(f"Read json file: {json_file}")
    metrics = decode_results_spark(json_content)
    os.remove(zstd_file)
    os.remove(json_file)
    logger.info(f"Initialized current task default with meta feature shape: {metrics.shape}")
    return metrics