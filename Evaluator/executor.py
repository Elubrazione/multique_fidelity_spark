
import copy
import os
import time
import traceback
from queue import Queue
import threading
from typing import Optional
import numpy as np
from openbox import logger
from ConfigSpace import Configuration

from task_manager import TaskManager
from .planner import SparkSQLPlanner
from .partitioner import SQLPartitioner
from .mock_executor import MockExecutor
from .utils import config_to_dict
from utils.spark import create_spark_session, execute_sql_with_timing, stop_active_spark_session
from config import ConfigManager


_DEFAULT_EXTRA_INFO = {'qt_time': {}, 'et_time': {}}


def _create_default_result(start_time):
    return {
        'result': {'objective': float('inf')},
        'timeout': True,
        'traceback': None,
        'elapsed_time': time.time() - start_time,
        'extra_info': copy.deepcopy(_DEFAULT_EXTRA_INFO),
    }


class ExecutorManager:
    def __init__(self,
                config_space,
                config_manager: ConfigManager,
                **kwargs):
        self.config_manager = config_manager
        
        self.config_space = config_space
        self.child_num = len(config_manager.multi_nodes)
        self.executor_queue = Queue()
        self.sql_dir = config_manager.data_dir
        self._planner = None
        self._partitioner = None
        self._task_manager = None
        self._init_child_executor(**kwargs)
        

    def _init_child_executor(self, **kwargs):
        self.executors = []
        for idx in range(self.child_num):
            self.executor_queue.put(idx)
            if kwargs.get('test_mode', True):
                base_seed = kwargs.get('seed')
                if base_seed is None:
                    base_seed = time.time_ns()
                else:
                    base_seed = int(base_seed) + time.time_ns()

                self.executors.append(MockExecutor(seed=base_seed + idx))
                continue
            executor_kwargs = {
                'nodes': self.config_manager.multi_nodes[idx],
                'server': self.config_manager.multi_servers[idx],
                'username': self.config_manager.multi_usernames[idx],
                'password': self.config_manager.multi_passwords[idx],
                'database': self.config_manager.database,
                'sql_dir': self.sql_dir,
                'debug': kwargs.get('debug', False),
            }
            self.executors.append(SparkSessionTPCDSExecutor(**executor_kwargs))
            
    def __call__(self, config, resource_ratio):
        idx = self.executor_queue.get()  # 阻塞直到有空闲 executor
        logger.debug(f"Got free executor: {idx}")

        planner = None
        plan = None
        try:
            planner = self._ensure_planner()
            if planner is not None:
                plan = planner.plan(resource_ratio, force_refresh=False, allow_fallback=True)
        except Exception as exc:
            logger.error(f"[ExecutorManager] Failed to obtain workload plan for resource {resource_ratio}: {exc}")
            logger.debug(traceback.format_exc())

        if plan is None:
            fallback_sqls = []
            partitioner = self._ensure_partitioner()
            if partitioner is not None:
                fallback_sqls = partitioner.get_all_sqls()
            logger.info(
                "[ExecutorManager] Using fallback plan for resource %.5f (sqls=%d)",
                resource_ratio, len(fallback_sqls)
            )
            plan = {
                "sqls": fallback_sqls,
                "timeout": {},
                "selected_fidelity": 1.0,
                "plan_source": "executor-fallback",
            }
    
        result_queue = Queue()

        def run():
            start_time = time.time()
            result = None
            try:
                result = self.executors[idx](config, resource_ratio, plan)
            except Exception as e:
                result = _create_default_result(start_time)
                logger.error(f"[Executor {idx}] Execution raised exception, continue with INF objective. Exception: {type(e).__name__}: {str(e)}")
            finally:
                if result is not None:
                    result_queue.put(result)
                else:
                    result = _create_default_result(start_time)
                    result_queue.put(result)
                    logger.error(f"[Executor {idx}] Result was None, using default INF result.")
                self.executor_queue.put(idx)  # 标记为"空闲"
                logger.debug(f"[Executor {idx}] Marked as free again.")

        thread = threading.Thread(target=run)
        thread.start()

        result = result_queue.get()  # 等待结果
        thread.join()
        return result

    def _get_task_manager(self) -> Optional[TaskManager]:
        if self._task_manager is not None:
            return self._task_manager
        task_mgr = getattr(TaskManager, "_instance", None)
        if task_mgr is not None and getattr(task_mgr, "_initialized", False):
            self._task_manager = task_mgr
            return self._task_manager
        return None

    def _ensure_partitioner(self) -> Optional[SQLPartitioner]:
        """Initialise and cache the SQLPartitioner.

        When TaskManager is not ready yet (bootstrap phase), return `None`
        and executor will rely on the static SQL list; once TaskManager has
        finished initialisation we register the real partitioner exactly once.
        """
        if self._partitioner is not None:
            return self._partitioner
        task_manager = self._get_task_manager()
        if task_manager is None:
            return None
        partitioner = task_manager.get_sql_partitioner()
        if partitioner is None:
            # When calculating meta features, the task_manager is ready,
            # but the sql_partitioner is not ready yet.
            # So we create a new partitioner and register it with the task_manager.
            logger.info("No partitioner found, creating a new one")
            partitioner = SQLPartitioner(sql_dir=self.sql_dir)
            task_manager.register_sql_partitioner(partitioner)
        self._partitioner = partitioner
        return partitioner
    
    def _ensure_planner(self) -> Optional[SparkSQLPlanner]:
        if self._planner is not None:
            return self._planner
        task_manager = self._get_task_manager()
        if task_manager is None:
            return None # No probability to happen
        partitioner = self._ensure_partitioner()
        if partitioner is None:
            return None # No probability to happen

        fallback = {1.0: partitioner.get_all_sqls()}
        logger.info("Fallback to full SQL list when first called to calculate meta features")

        planner = task_manager.get_planner()
        if planner is None:
            logger.info("No planner found, creating a new one")
            planner = SparkSQLPlanner(partitioner, fallback_sqls=fallback)
            task_manager.register_planner(planner)
        if getattr(planner, "latest_plan", None) is None:
            planner.refresh_plan(force=True)
            logger.info("Planner refreshed because there is no cached plan")
        self._planner = planner
        logger.info(
            "[ExecutorManager] Planner ready (resource levels=%s)",
            list(planner._cached_plan.fidelity_subsets.keys()) \
                if getattr(planner, "_cached_plan", None) else "<none>"
        )
        return planner


class SparkSessionTPCDSExecutor:
    def __init__(self,
                nodes, server, username, password,
                database, sql_dir, app_name_prefix="SessionExecutor",
                **kwargs):

        self.nodes = nodes
        self.server = server
        self.username = username
        self.password = password
        self.database = database
        self.sql_dir = sql_dir
        self.app_name_prefix = app_name_prefix

        self.debug = kwargs.get('debug', False)
        if self.debug:
            logger.info(f"SessionExecutor initialized in debug mode")

    def __call__(self, config, resource_ratio, plan=None):
        return self.run_spark_session_job(config, resource_ratio, plan)


    def run_spark_session_job(self, config, resource, plan=None):
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
        plan_sqls = plan.get('sqls') if isinstance(plan, dict) else None
        if not plan_sqls:
            plan_sqls = []
        timeout_overrides = plan.get('timeout', {}) if isinstance(plan, dict) else {}

        # only execute 2 sqls when debug mode is enabled
        if self.debug:
            plan_sqls = copy.deepcopy(plan_sqls)[: 2]
            timeout_overrides = copy.deepcopy(timeout_overrides)[: 2]

        config_dict = config_to_dict(config)
        logger.info(f"[SparkSession] Evaluating fidelity {resource} on database {self.database}")
        logger.debug(f"[SparkSession] Configuration: {config_dict}")

        spark = None
        total_status = True
        app_name = f"{self.app_name_prefix}_{resource}"
        try:
            spark = create_spark_session(config_dict, app_name=app_name, database=self.database)
        except Exception as e:
            logger.error(f"[SparkSession] Failed to create Spark session, skip remaining queries. Error: {type(e).__name__}: {str(e)}")
            logger.error(f"[SparkSession] Traceback: {traceback.format_exc()}")
            return self.build_ret_dict(float('inf'), start_time)

        total_spark_time = 0.0
        extra_info = copy.deepcopy(_DEFAULT_EXTRA_INFO)
        extra_info['plan_sqls'] = list(plan_sqls)
        extra_info['plan_timeout'] = timeout_overrides
        max_retry_attempts = 1
        
        for sql in plan_sqls:
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
                    result = execute_sql_with_timing(spark, sql_content, sql)
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
                        stop_active_spark_session()
                        try:
                            spark = create_spark_session(config_dict, app_name=app_name, database=self.database)
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

        # Stop SparkSession and ensure proper cleanup
        try:
            stop_active_spark_session()
        except Exception as cleanup_exc:
            logger.warning(f"[SparkSession] Exception during session cleanup: {cleanup_exc}")

        if not total_status:
            total_spark_time = float('inf')
        objective = total_spark_time if np.isfinite(total_spark_time) else float('inf')
        return self.build_ret_dict(objective, start_time, extra_info)

    @staticmethod
    def build_ret_dict(perf, start_time, extra_info=None):
        if extra_info is None:
            extra_info = copy.deepcopy(_DEFAULT_EXTRA_INFO)
        result = {
            'result': {'objective': perf},
            'timeout': not np.isfinite(perf),
            'traceback': None,
            'extra_info': extra_info,
            'elapsed_time': time.time() - start_time
        }
        return result
