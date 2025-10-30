
from typing import Dict, Any, List
import re, os, time, csv, json
from queue import Queue
import threading
import numpy as np
from datetime import datetime
from openbox import logger
from ConfigSpace import Configuration
from pyspark.sql import SparkSession

from utils.spark import convert_to_spark_params, custom_sort, clear_cache_on_remote
from config import ENV_SPARK_SQL_PATH, DATABASE, DATA_DIR, RESULT_DIR, \
    LIST_SPARK_NODES, LIST_SPARK_SERVER, LIST_SPARK_USERNAME, LIST_SPARK_PASSWORD, \
    SPARK_NODES, SPARK_SERVER_NODE, SPARK_USERNAME, SPARK_PASSWORD


class ExecutorManager:
    def __init__(self, sqls: dict, timeout: dict, config_space,
                 spark_sql=ENV_SPARK_SQL_PATH, spark_nodes=LIST_SPARK_NODES,
                 servers=LIST_SPARK_SERVER, usernames=LIST_SPARK_USERNAME, passwords=LIST_SPARK_PASSWORD,
                 fidelity_database_mapping=None, fixed_sqls=None,
                 executor_cls=None, executor_kwargs=None):
        self.sqls = sqls
        self.timeout = timeout
        self.config_space = config_space
        self.child_num = len(spark_nodes)
        self.executor_queue = Queue()
        self.fidelity_database_mapping = fidelity_database_mapping or {}
        self.fixed_sqls = fixed_sqls
        self.executor_cls = executor_cls or SparkSessionTPCDSExecutor
        self.executor_kwargs = executor_kwargs or {}
        
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
        self.csv_path = os.path.join(RESULT_DIR, datetime.now().strftime('%Y%m%d-%H%M%S') + ".csv")
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.all_queries = sorted(sqls[round(float(1), 5)], key=lambda x: custom_sort(x))
        self.header = ['resource', 'query_time', 'elapsed_time', 'overhead'] + \
                      [f'qt_{q}' for q in self.all_queries] + \
                      [f'et_{q}' for q in self.all_queries] + \
                      [f'{k.name}_param' for k in config_space.get_hyperparameters()]

        if os.stat(self.csv_path).st_size == 0:
            self.csv_writer.writerow(self.header)

        self.write_queue = Queue()
        self.write_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.write_thread.start()
        self.closed = False

        self._init_child_executor(spark_sql, spark_nodes, servers, usernames, passwords)
        
        for item in sqls[round(float(1), 5)]:
            logger.info("[timeout]: %s, %f" % (item, timeout.get(item, np.inf)))

    def _init_child_executor(self, spark_sql, spark_nodes, servers, usernames, passwords):
        self.executors = []
        for idx in range(self.child_num):
            self.executor_queue.put(idx)
            executor_kwargs = {
                'sqls': self.sqls,
                'timeout': self.timeout,
                'write_queue': self.write_queue,
                'all_queries': self.all_queries,
                'spark_sql': spark_sql,
                'spark_nodes': spark_nodes[idx],
                'server': servers[idx],
                'username': usernames[idx],
                'password': passwords[idx],
                'fidelity_database_mapping': self.fidelity_database_mapping,
                'fixed_sqls': self.fixed_sqls
            }
            executor_kwargs.update(self.executor_kwargs)
            self.executors.append(
                self.executor_cls(**executor_kwargs)
            )
            
    def _writer_loop(self):
        while True:
            row_dict = self.write_queue.get()
            if row_dict is None:
                break
            row = [row_dict.get(h, 0) for h in self.header]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            self.write_queue.task_done()

    def __call__(self, config, resource_ratio):
        idx = self.executor_queue.get()  # 阻塞直到有空闲 executor
        logger.info(f"Got free executor: {idx}")

        result_queue = Queue()

        def run():
            start_time = time.time()
            try:
                result = self.executors[idx](config, resource_ratio)
            except Exception:
                result = {
                    'result': {'objective': float('inf')},
                    'timeout': True,
                    'traceback': None,
                    'elapsed_time': time.time() - start_time,
                }
                logger.error(f"[Executor {idx}] Execution raised exception, continue with INF objective.")
            finally:
                result_queue.put(result)
                self.executor_queue.put(idx)  # 标记为“空闲”
                logger.info(f"[Executor {idx}] Marked as free again.")

        thread = threading.Thread(target=run)
        thread.start()

        result = result_queue.get()  # 等待结果
        thread.join()
        return result
    
    def __del__(self):
        self.close()

    def close(self):
        if self.closed:
            return
        try:
            self.write_queue.put(None)
            self.write_thread.join()
        finally:
            try:
                self.csv_file.close()
            finally:
                self.closed = True
                logger.info("ExecutorManager closed and CSV file saved.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



class SparkSessionTPCDSExecutor:
    def __init__(self, sqls: dict, timeout: dict, write_queue: Queue, all_queries=None,
                 spark_sql=ENV_SPARK_SQL_PATH, spark_nodes=SPARK_NODES,
                 server=SPARK_SERVER_NODE, username=SPARK_USERNAME, password=SPARK_PASSWORD,
                 config_space=None, fidelity_database_mapping=None, fixed_sqls=None,
                 sql_dir=DATA_DIR, app_name_prefix="SparkSessionTPCDSExecutor"):
        self.sqls = sqls
        self.timeout = timeout
        self.write_queue = write_queue
        self.all_queries = all_queries or []
        self.spark_nodes = spark_nodes or []
        if isinstance(self.spark_nodes, str):
            self.spark_nodes = [self.spark_nodes]
        self.server = server
        self.username = username
        self.password = password
        self.sql_dir = sql_dir
        self.app_name_prefix = app_name_prefix

        self.fidelity_database_mapping = fidelity_database_mapping or {}
        if fixed_sqls is not None:
            self.fixed_sqls = fixed_sqls
        else:
            try:
                self.fixed_sqls = self.get_sqls_by_fidelity(resource=1.0)
            except Exception:
                # fallback to the largest fidelity available
                max_fidelity = max(self.sqls.keys()) if self.sqls else 1.0
                self.fixed_sqls = self.sqls.get(max_fidelity, [])

    def __call__(self, config, resource_ratio):
        return self.run_spark_session_job(config, resource_ratio)

    def _normalize_fidelity_key(self, fidelity):
        if fidelity in self.sqls:
            return fidelity
        rounded = round(float(fidelity), 5)
        if rounded in self.sqls:
            return rounded
        for key in self.sqls.keys():
            try:
                if abs(float(key) - float(fidelity)) < 1e-8:
                    return key
            except (TypeError, ValueError):
                continue
        raise KeyError(f"Fidelity {fidelity} not found in SQL definitions")

    def _resolve_database(self, fidelity):
        if not self.fidelity_database_mapping:
            return DATABASE

        candidates = [
            fidelity,
            round(float(fidelity), 5),
            str(fidelity),
            str(round(float(fidelity), 5))
        ]
        for key in candidates:
            if key in self.fidelity_database_mapping:
                return self.fidelity_database_mapping[key]
        for mapping_key in self.fidelity_database_mapping.keys():
            try:
                if abs(float(mapping_key) - float(fidelity)) < 1e-8:
                    return self.fidelity_database_mapping[mapping_key]
            except (TypeError, ValueError):
                continue
        return DATABASE

    def get_sqls_by_fidelity(self, resource, use_delta=False):
        if self.fidelity_database_mapping:
            try:
                fidelity_key = self._normalize_fidelity_key(resource)
            except KeyError:
                fidelity_key = resource
            logger.info(f"[SparkSession] Using fixed SQL set for fidelity {resource}: {len(self.fixed_sqls)} queries")
            return self.fixed_sqls

        fidelity_key = self._normalize_fidelity_key(resource)
        original_sqls = list(self.sqls[fidelity_key])
        if use_delta:
            for k, v in self.sqls.items():
                if k == fidelity_key:
                    break
                original_sqls = [i for i in original_sqls if i not in v]
        return original_sqls

    def get_database_by_fidelity(self, fidelity):
        database = self._resolve_database(fidelity)
        logger.info(f"[SparkSession] Using database for fidelity {fidelity}: {database}")
        return database

    def _config_to_dict(self, config):
        if hasattr(config, 'get_dictionary'):
            return config.get_dictionary()
        return dict(config)

    def _format_config_value(self, key, value):
        memory_params = {
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
        suffix = memory_params.get(key, '')
        return f"{value}{suffix}" if suffix and not str(value).endswith(suffix) else str(value)

    def create_spark_session(self, config_dict, app_name, database=None):
        spark_builder = SparkSession.builder.appName(app_name).enableHiveSupport()
        for key, value in config_dict.items():
            if str(key).startswith('spark.'):
                spark_builder = spark_builder.config(key, self._format_config_value(key, value))
        spark = spark_builder.getOrCreate()
        if database is not None:
            db_name = str(database).strip()
            if db_name:
                logger.info(f"[SparkSession] Attempting to set database: '{db_name}'")
                try:
                    count = spark.sql(f"SHOW DATABASES LIKE '{db_name}'").count()
                    logger.info(f"[SparkSession] Database existence check for '{db_name}': {count} match(es)")
                    if count == 0:
                        logger.warning(f"[SparkSession] Database '{db_name}' does not exist. Skipping USE.")
                    else:
                        spark.sql(f"USE `{db_name}`")
                        logger.info(f"[SparkSession] Database set to: {db_name}")
                except Exception:
                    logger.warning(f"[SparkSession] Failed to set database to '{db_name}'. Continuing without changing database.")
            else:
                logger.warning("[SparkSession] Empty database name resolved; skipping USE.")
        return spark

    def _clear_cluster_cache(self):
        if not self.spark_nodes:
            return
        for node in self.spark_nodes:
            try:
                clear_cache_on_remote(node, username=self.username, password=self.password)
            except Exception as exc:
                logger.error(f"[SparkSession] Failed to clear cache on {node}: {exc}")

    def execute_sql_with_timing(self, spark, sql_content, sql_file):
        logger.info(f"[SparkSession] Execute SQL file: {sql_file}")
        queries = [q.strip() for q in sql_content.split(';') if q.strip()]

        total_start_time = time.time()
        per_sql_time = 0.0
        status = 'success'

        for idx, query in enumerate(queries):
            if not query:
                continue
            logger.debug(f"  execute query {idx + 1}/{len(queries)}: {query[:50]}...")
            query_start_time = time.time()
            try:
                result = spark.sql(query)
                collected = result.collect()
                logger.debug(f"      query returned {len(collected)} rows")
                per_sql_time += (time.time() - query_start_time)
                logger.info(f"      query {idx + 1} completed")
            except Exception as exc:
                _ = time.time() - query_start_time
                status = 'error'
                py_err = type(exc).__name__
                jvm_err = None
                try:
                    java_exc = getattr(exc, 'java_exception', None)
                    if java_exc is not None:
                        jvm_err = java_exc.getClass().getName()
                except Exception:
                    jvm_err = None
                if jvm_err:
                    logger.error(f"      query {idx + 1} failed (py_err={py_err}, jvm_err={jvm_err})")
                else:
                    logger.error(f"      query {idx + 1} failed (py_err={py_err})")
                break

        total_elapsed = time.time() - total_start_time
        return {
            'sql_file': sql_file,
            'total_elapsed_time': total_elapsed,
            'per_sql_time': per_sql_time,
            'status': status
        }

    def run_spark_session_job(self, config, resource):
        start_time = time.time()
        queries = self.get_sqls_by_fidelity(resource=resource)

        database = self.get_database_by_fidelity(resource)
        config_dict = self._config_to_dict(config)

        logger.info(f"[SparkSession] Evaluating fidelity {resource} on database {database}")
        logger.debug(f"[SparkSession] Configuration: {config_dict}")

        spark = None
        results = []
        elapsed_time_map = {sql: np.inf for sql in queries}
        qtime_details = {sql: np.inf for sql in queries}
        total_status = True

        app_name = f"{self.app_name_prefix}_{resource}"
        try:
            spark = self.create_spark_session(config_dict, app_name=app_name, database=database)
        except Exception:
            logger.error("[SparkSession] Failed to create Spark session, skip remaining queries.")
            return self.build_ret_dict(float('inf'), start_time)

        total_spark_time = 0.0
        for sql in queries:
            sql_path = os.path.join(self.sql_dir, f"{sql}.sql")
            if not os.path.exists(sql_path):
                logger.error(f"[SparkSession] SQL file not found: {sql_path}")
                total_status = False
                break

            with open(sql_path, 'r', encoding='utf-8') as f:
                    sql_content = f.read()

            result = self.execute_sql_with_timing(spark, sql_content, sql)
            results.append(result)

            elapsed_time_map[sql] = result['total_elapsed_time']
            per_sql_time = result.get('per_sql_time', 0.0)
            qtime_details[sql] = per_sql_time if result['status'] == 'success' else float('inf')

            if result['status'] == 'success':
                total_spark_time += per_sql_time

            if result['status'] != 'success':
                total_status = False
                break

        spark.stop()

        elapsed_total_time = sum(v for v in elapsed_time_map.values() if np.isfinite(v))
        if not total_status:
            total_spark_time = float('inf')

        row_dict = {
            'resource': resource,
            'query_time': total_spark_time if np.isfinite(total_spark_time) else float('inf'),
            'elapsed_time': elapsed_total_time if np.isfinite(elapsed_total_time) else float('inf'),
            'overhead': (elapsed_total_time - total_spark_time)
            if np.isfinite(total_spark_time) and np.isfinite(elapsed_total_time) else float('inf')
        }

        for sql in self.all_queries:
            row_dict[f'qt_{sql}'] = qtime_details.get(sql, 0)
            row_dict[f'et_{sql}'] = elapsed_time_map.get(sql, 0)

        for key, value in config_dict.items():
            row_dict[f'{key}_param'] = value

        if self.write_queue is not None:
            self.write_queue.put(row_dict)

        objective = total_spark_time if np.isfinite(total_spark_time) else float('inf')
        return self.build_ret_dict(objective, start_time)

    @staticmethod
    def build_ret_dict(perf, start_time):
        result = {
            'result': {'objective': perf},
            'timeout': not np.isfinite(perf),
            'traceback': None
        }
        result['elapsed_time'] = time.time() - start_time
        return result
