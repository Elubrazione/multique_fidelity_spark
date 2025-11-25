import copy
import imp
import os
import time
import traceback
from typing import Optional, Dict, Any, List, Iterable
import numpy as np
from openbox import logger
from ConfigSpace import Configuration
from Evaluator import (
    Evaluator,
    EvaluatorManager,
    NoOpPlanner,
    Planner,
    Partitioner,
)
from .planner import SparkSQLPlanner
from .partitioner import SQLPartitioner
from .utils import (
    create_spark_session,
    execute_sql_with_timing,
    stop_active_spark_session,
    config_to_dict,
)

_DEFAULT_EXTRA_INFO = {'qt_time': {}, 'et_time': {}}


class SparkSessionEvaluator(Evaluator):
    def __init__(self,
                database, sql_dir, app_name_prefix="SparkSessionEvaluator",
                **kwargs):
        self.database = database
        self.sql_dir = sql_dir
        self.app_name_prefix = app_name_prefix

        self.debug = kwargs.get('debug', False)
        if self.debug:
            logger.info(f"SessionExecutor initialized in debug mode, only execute 2 sqls when debug mode is enabled")

    def __call__(self, config: Configuration, resource_ratio: float, plan: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return self.run_spark_session_job(config, resource_ratio, plan)

    def run_spark_session_job(self, config: Configuration, resource: float, plan: Optional[Dict[str, Any]] = None):
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
            return self.build_ret_dict(float('inf'), start_time, traceback=traceback.format_exc())

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

    def build_ret_dict(self, perf, start_time, extra_info=None):
        if extra_info is None:
            extra_info = copy.deepcopy(_DEFAULT_EXTRA_INFO)
        return self.build_result_dict(perf, start_time, extra_info=extra_info)


class SparkEvaluatorManager(EvaluatorManager):
    def __init__(
        self,
        config_space,
        config_manager,
        *,
        evaluators: Optional[List] = None,
        planner: Optional[Planner] = None,
        partitioner: Optional[Partitioner] = None,
        sql_dir: Optional[str] = None,
        num_evaluator: Optional[int] = None,
        planner_timeout: Optional[Dict[str, float]] = None,
        fallback_sqls: Optional[Dict[float, Iterable[str]]] = None,
        evaluator_kwargs: Optional[Dict] = None,
        partitioner_kwargs: Optional[Dict] = None,
        planner_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        sql_dir = sql_dir or getattr(config_manager, "data_dir", None)
        if sql_dir is None:
            raise ValueError("SparkEvaluatorManager requires sql_dir (config_manager.data_dir).")

        partitioner_kwargs = partitioner_kwargs or {}
        planner_kwargs = planner_kwargs or {}
        evaluator_kwargs = evaluator_kwargs or {}

        if partitioner is None:
            partitioner = SQLPartitioner(sql_dir=sql_dir, **partitioner_kwargs)

        if planner is None:
            if fallback_sqls is None and partitioner is not None:
                fallback_sqls = {1.0: partitioner.get_all_sqls()}
                logger.info("Fallback to full SQL list when first called to calculate meta features")
            planner = SparkSQLPlanner(
                partitioner,
                timeout=planner_timeout,
                fallback_sqls=fallback_sqls,
                **planner_kwargs,
            )

        if evaluators is None:
            evaluators = [self._create_spark_evaluator(config_manager, sql_dir, evaluator_kwargs)]

        super().__init__(
            config_space=config_space,
            evaluators=evaluators,
            planner=planner,
            partitioner=partitioner,
            config_manager=config_manager,
            num_evaluator=num_evaluator or len(evaluators),
            **kwargs,
        )
        self.sql_dir = sql_dir

    def _create_spark_evaluator(self, config_manager, sql_dir: str, extra_kwargs: Dict):
        try:
            executor_kwargs = {
                "database": config_manager.database,
                "sql_dir": sql_dir
            }
        except (AttributeError, IndexError) as exc:
            raise ValueError("ConfigManager missing Spark connection information.") from exc
        executor_kwargs.update(extra_kwargs)
        return SparkSessionEvaluator(**executor_kwargs)

    def _build_fallback_plan(self, resource_ratio: float):
        if self.partitioner is None:
            return None
        try:
            subsets = self.partitioner.get_fidelity_subsets()
        except Exception as exc:
            logger.warning(f"[SparkEvaluatorManager] Failed to get fallback subsets: {exc}")
            return None

        fallback_sqls = subsets.get(resource_ratio) or subsets.get(1.0, [])
        if not fallback_sqls:
            return None

        logger.info(
            "[SparkEvaluatorManager] Using fallback plan for resource %.5f (items=%d)",
            resource_ratio,
            len(fallback_sqls),
        )
        timeout = {}
        if isinstance(self.planner, SparkSQLPlanner):
            timeout = {sql: self.planner.timeout.get(sql) for sql in fallback_sqls if sql in self.planner.timeout}

        return {
            "sqls": fallback_sqls,
            "timeout": timeout,
            "selected_fidelity": float(resource_ratio),
            "plan_source": "executor-fallback",
        }

    def attach_task_manager(self, task_manager=None):
        task_manager = task_manager or self._get_task_manager()
        if task_manager is None:
            return
        if self.partitioner is not None:
            if hasattr(self.partitioner, '_task_manager'):
                self.partitioner._task_manager = task_manager
            task_manager.register_sql_partitioner(self.partitioner)
        if self.planner is not None and not isinstance(self.planner, NoOpPlanner):
            task_manager.register_planner(self.planner)
            # Refresh planner plan if no cached plan exists (following old version pattern)
            if isinstance(self.planner, SparkSQLPlanner):
                if getattr(self.planner, "_cached_plan", None) is None:
                    logger.info("Planner refreshed because there is no cached plan")
                    self.planner.refresh_plan(force=True)
                    logger.info(
                        "[SparkEvaluatorManager] Planner ready (resource levels=%s)",
                        list(self.planner._cached_plan.fidelity_subsets.keys())
                        if getattr(self.planner, "_cached_plan", None) else "<none>"
                    )

    def _get_task_manager(self):
        try:
            from manager import TaskManager

            task_mgr = getattr(TaskManager, "_instance", None)
            if task_mgr is not None and getattr(task_mgr, "_initialized", False):
                return task_mgr
        except Exception:
            pass
        return None

