
import os, time
from queue import Queue
import threading
import numpy as np
from openbox import logger
from ConfigSpace import Configuration
from pyspark.sql import SparkSession

from utils.spark import clear_cache_on_remote
from config import ENV_SPARK_SQL_PATH, DATABASE, DATA_DIR, \
    LIST_SPARK_NODES, LIST_SPARK_SERVER, LIST_SPARK_USERNAME, LIST_SPARK_PASSWORD, \
    SPARK_NODES, SPARK_SERVER_NODE, SPARK_USERNAME, SPARK_PASSWORD


class ExecutorManager:
    def __init__(self, sqls: dict, timeout: dict, config_space,
                 spark_sql=ENV_SPARK_SQL_PATH, spark_nodes=LIST_SPARK_NODES,
                 servers=LIST_SPARK_SERVER, usernames=LIST_SPARK_USERNAME, passwords=LIST_SPARK_PASSWORD,
                 fidelity_database_mapping=None, fixed_sqls=None,
                 executor_cls=None, executor_kwargs=None,
                 **kwargs):
        self.sqls = sqls
        self.timeout = timeout
        self.config_space = config_space
        self.child_num = len(spark_nodes)
        self.executor_queue = Queue()
        self.fidelity_database_mapping = fidelity_database_mapping or {}
        self.fixed_sqls = fixed_sqls
        self.executor_cls = executor_cls or SparkSessionTPCDSExecutor
        self.executor_kwargs = executor_kwargs or {}
        self._init_child_executor(spark_sql, spark_nodes, servers, usernames, passwords, **kwargs)
        

    def _init_child_executor(self, spark_sql, spark_nodes, servers, usernames, passwords, **kwargs):
        self.executors = []
        for idx in range(self.child_num):
            self.executor_queue.put(idx)
            if kwargs.get('test_mode', True):
                self.executors.append(TestExecutor())
                continue
            executor_kwargs = {
                'sqls': self.sqls,
                'timeout': self.timeout,
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
            
    def __call__(self, config, resource_ratio):
        idx = self.executor_queue.get()  # 阻塞直到有空闲 executor
        logger.debug(f"Got free executor: {idx}")

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
                logger.debug(f"[Executor {idx}] Marked as free again.")

        thread = threading.Thread(target=run)
        thread.start()

        result = result_queue.get()  # 等待结果
        thread.join()
        return result


class SparkSessionTPCDSExecutor:
    def __init__(self, sqls: dict, timeout: dict,
                 spark_sql=ENV_SPARK_SQL_PATH, spark_nodes=SPARK_NODES,
                 server=SPARK_SERVER_NODE, username=SPARK_USERNAME, password=SPARK_PASSWORD,
                 config_space=None, fidelity_database_mapping=None, fixed_sqls=None,
                 sql_dir=DATA_DIR, app_name_prefix="SparkSessionTPCDSExecutor"):
        self.sqls = sqls
        self.timeout = timeout
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
            self.fixed_sqls = self.get_sqls_by_fidelity(resource=1.0)

    def __call__(self, config, resource_ratio):
        return self.run_spark_session_job(config, resource_ratio)

    def _normalize_fidelity_key(self, fidelity):
        if fidelity in self.sqls:
            return fidelity
        rounded = round(float(fidelity), 5)
        if rounded in self.sqls:
            return rounded

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

    def get_sqls_by_fidelity(self, resource):
        if self.fidelity_database_mapping:
            fidelity_key = self._normalize_fidelity_key(resource)
            logger.info(f"[SparkSession] Using fixed SQL set for fidelity {resource}: {len(self.fixed_sqls)} queries")
            return self.fixed_sqls
        fidelity_key = self._normalize_fidelity_key(resource)
        return list(self.sqls[fidelity_key])

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
    
    def use_database(self, spark, database):
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

    def create_spark_session(self, config_dict, app_name, database=None):
        spark_builder = SparkSession.builder.appName(app_name).enableHiveSupport()
        for key, value in config_dict.items():
            if str(key).startswith('spark.'):
                spark_builder = spark_builder.config(key, self._format_config_value(key, value))
        spark = spark_builder.getOrCreate()
        self.use_database(spark, database)
        return spark

    def _clear_cluster_cache(self):
        if not self.spark_nodes:
            return
        for node in self.spark_nodes:
            try:
                clear_cache_on_remote(node, username=self.username, password=self.password)
            except Exception as exc:
                logger.error(f"[SparkSession] Failed to clear cache on {node}: {exc}")

    def _check_spark_context_valid(self, spark):
        try:
            sc = spark.sparkContext
            if sc is None or sc._jsc is None:
                return False
            _ = sc.version
            return True
        except Exception:
            return False

    def execute_sql_with_timing(self, spark, sql_content, sql_file):
        queries = [q.strip() for q in sql_content.split(';') if q.strip()]

        total_start_time = time.time()
        per_qt_time = 0.0
        status = 'success'

        for idx, query in enumerate(queries):
            if not query:
                continue
            logger.debug(f"  execute query {idx + 1}/{len(queries)}: {query[:50]}...")
            
            if not self._check_spark_context_valid(spark):
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

    def run_spark_session_job(self, config, resource):
        """
        Workflow:
        - if query executed successfully: continue to next query
        - if query executed failed (not SparkContext problem): set status to False and break the loop
        - if SparkContext is closed and can be retried:
            - stop old session
            - try to create new session
                - if successful: retry the query
                - if failed: set status to False and break the loop
        - if SparkContext is closed and cannot be retried: set status to False and break the loop
        - if other exception: set status to False and break the loop
        """
        start_time = time.time()
        queries = self.get_sqls_by_fidelity(resource=resource)

        database = self.get_database_by_fidelity(resource)
        config_dict = self._config_to_dict(config)
        logger.info(f"[SparkSession] Evaluating fidelity {resource} on database {database}")
        logger.debug(f"[SparkSession] Configuration: {config_dict}")

        spark = None
        total_status = True
        app_name = f"{self.app_name_prefix}_{resource}"
        try:
            spark = self.create_spark_session(config_dict, app_name=app_name, database=database)
        except Exception:
            logger.error("[SparkSession] Failed to create Spark session, skip remaining queries.")
            return self.build_ret_dict(float('inf'), start_time)

        total_spark_time = 0.0
        extra_info = {'qt_time': {}, 'et_time': {}}
        max_retry_attempts = 1
        
        for sql in queries:
            sql_path = os.path.join(self.sql_dir, f"{sql}.sql")
            if not os.path.exists(sql_path):
                logger.error(f"[SparkSession] SQL file not found: {sql_path}")
                total_status = False
                break

            with open(sql_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()

            retry_count = 0
            query_executed = False
            result = None
            
            while retry_count <= max_retry_attempts and not query_executed:
                try:
                    result = self.execute_sql_with_timing(spark, sql_content, sql)
                    per_qt_time = result.get('per_qt_time', float('inf'))
                    per_et_time = result.get('per_et_time', float('inf'))

                    if result['status'] == 'success':
                        total_spark_time += per_qt_time
                        extra_info['qt_time'][sql] = per_qt_time
                        extra_info['et_time'][sql] = per_et_time
                        query_executed = True
                    else:
                        total_status = False
                        extra_info['qt_time'][sql] = float('inf')
                        extra_info['et_time'][sql] = float('inf')
                        query_executed = True
                        
                except RuntimeError as e:
                    if "SparkContext was shut down" in str(e) and retry_count < max_retry_attempts:
                        logger.warning(f"[SparkSession] SparkContext was shut down during {sql}, attempting to recreate (retry {retry_count + 1}/{max_retry_attempts})")

                        self.stop_spark_session(spark)
                        try:
                            spark = self.create_spark_session(config_dict, app_name=app_name, database=database)
                            logger.info(f"[SparkSession] Successfully recreated SparkSession, retrying {sql}")
                            retry_count += 1
                        except Exception as recreate_exc:
                            logger.error(f"[SparkSession] Failed to recreate SparkSession: {recreate_exc}")
                            total_status = False
                            extra_info['qt_time'][sql] = float('inf')
                            extra_info['et_time'][sql] = float('inf')
                            query_executed = True
                    else:
                        logger.error(f"[SparkSession] Failed to execute {sql}: {e}")
                        total_status = False
                        extra_info['qt_time'][sql] = float('inf')
                        extra_info['et_time'][sql] = float('inf')
                        query_executed = True
                        
                except Exception as unexpected_exc:
                    logger.error(f"[SparkSession] Unexpected error during {sql}: {unexpected_exc}")
                    total_status = False
                    extra_info['qt_time'][sql] = float('inf')
                    extra_info['et_time'][sql] = float('inf')
                    query_executed = True
            
            if not query_executed or (result is not None and result['status'] != 'success'):
                break

        self.stop_spark_session(spark)

        if not total_status:
            total_spark_time = float('inf')
        objective = total_spark_time if np.isfinite(total_spark_time) else float('inf')
        return self.build_ret_dict(objective, start_time, extra_info)

    def stop_spark_session(self, spark):
        if spark is not None:
            try:
                spark.stop()
                logger.info("SparkSession closed")
            except Exception:
                pass

    @staticmethod
    def build_ret_dict(perf, start_time, extra_info={'qt_time': {}, 'et_time': {}}):
        result = {
            'result': {'objective': perf},
            'timeout': not np.isfinite(perf),
            'traceback': None,
            'extra_info': extra_info,
            'elapsed_time': time.time() - start_time
        }
        return result

class TestExecutor:
    def __init__(self):
        pass

    def __call__(self, config, resource_ratio):
        extra_info = {'qt_time': {}, 'et_time': {}}
        sql_list = ['q10', 'q11', 'q12', 'q13', 'q14a', 'q14b', 'q15', 'q16', 'q17', 'q18', 'q19', 'q1', 'q20', 'q21', 'q22', 'q23a', 'q23b', 'q24a', 'q24b', 'q25', 'q26', 'q27', 'q28', 'q29', 'q2', 'q30', 'q31', 'q32', 'q33', 'q34', 'q35', 'q36', 'q37', 'q38', 'q39a', 'q39b', 'q3', 'q40', 'q41', 'q42', 'q43', 'q44', 'q45', 'q46', 'q47', 'q48', 'q49', 'q4', 'q50', 'q51', 'q52', 'q53', 'q54', 'q55', 'q56', 'q57', 'q58', 'q59', 'q5', 'q60', 'q61', 'q62', 'q63', 'q64', 'q65', 'q66', 'q67', 'q68', 'q69', 'q6', 'q70', 'q71', 'q72', 'q73', 'q74', 'q75', 'q76', 'q77', 'q78', 'q79', 'q7', 'q80', 'q81', 'q82', 'q83', 'q84', 'q85', 'q86', 'q87', 'q88', 'q89', 'q8', 'q90', 'q91', 'q92', 'q93', 'q94', 'q95', 'q96', 'q97', 'q98', 'q99', 'q9']
        for it in sql_list:
            extra_info['qt_time'][it] = np.random.rand()
            extra_info['et_time'][it] = np.random.rand()
        return {
            'result': {'objective': sum(extra_info['qt_time'].values())},
            'timeout': not np.isfinite(np.random.rand()),
            'traceback': None,
            'extra_info': extra_info,
            'elapsed_time': np.random.rand()
        }