
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

from .utils import config_to_dict
from utils.spark import create_spark_session, execute_sql_with_timing, stop_active_spark_session


_DEFAULT_EXTRA_INFO = {'qt_time': {}, 'et_time': {}}

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

        stop_active_spark_session()

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
