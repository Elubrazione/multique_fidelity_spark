import os
import json
import subprocess
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from Advisor.utils import map_source_hpo_data, build_observation

META_FEATURE = np.random.rand(34)

class TaskManager:
    """
    Manages historical tasks and current task similarity computation.
    Handles:
    - Loading historical tasks from directory
    - Computing current task runtime metrics
    - Dynamic similarity updates
    - Providing filtered similar tasks to optimizer/compressor
    - Updating current task history from evaluator results

    Args:
        - history_dir: Directory containing historical tasks
        - spark_log_dir: Directory containing Spark log files
        - similarity_threshold: Similarity threshold for filtering similar tasks
        - ws_args: Warm start arguments
        - config_space: Original configuration space without compression
    """

    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, 
                 history_dir: str,
                 spark_log_dir: str = "/root/codes/spark-log",
                 similarity_threshold: float = 0.5,
                 config_space: Optional[ConfigurationSpace] = None,
                 **kwargs):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.history_dir = history_dir
        self.spark_log_dir = spark_log_dir

        self.ws_args = dict(kwargs.get('ws_args') or {})
        self.tl_args = dict(kwargs.get('tl_args') or {})
        self.cp_args = dict(kwargs.get('cp_args') or {})
        self.scheduler_kwargs = dict(kwargs.get('scheduler_kwargs') or {})
        self.logger_kwargs = dict(kwargs.get('logger_kwargs') or {})
        self.random_kwargs = dict(kwargs.get('random_kwargs') or {})

        self.similarity_threshold = similarity_threshold
        self.config_space = config_space
        
        self.historical_tasks: List[History] = []
        self.historical_meta_features: List[np.ndarray] = []
        
        self.current_task_history: Optional[History] = None
        self.current_meta_feature: Optional[np.ndarray] = None
        
        self.similar_tasks_cache: List[Tuple[int, float]] = []

        self._scheduler: Optional[object] = None
        self._sql_partitioner: Optional[object] = kwargs.get('sql_partitioner')
        self._planner: Optional[object] = kwargs.get('planner')
        self.tl_topk: Optional[int] = self.tl_args.get('topk') if isinstance(self.tl_args, dict) else None
        
        self._load_historical_tasks()


    def _load_historical_tasks(self):
        if not os.path.exists(self.history_dir):
            logger.warning(f"History directory {self.history_dir} does not exist.")
            return
        
        # 默认History里面observations的配置是全空间的配置而非压缩过的
        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.history_dir, filename)
                history = History.load_json(filename=filepath, config_space=self.config_space)
                self.historical_tasks.append(history)
                meta_feature = history.meta_info.get('meta_feature')
                if meta_feature is not None:
                    logger.debug(f"Got meta_feature: {meta_feature} from {filename}")
                    self.historical_meta_features.append(np.array(meta_feature))
                else:
                    logger.warning(f"No meta_feature found in {filename}")     
        logger.info(f"Loaded {len(self.historical_tasks)} historical tasks from {self.history_dir}")


    def decode_results_spark(self, results: str) -> Tuple[float, np.ndarray]:
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

    def get_latest_application_id(self) -> Optional[str]:
        """
        Get the latest application_id from spark-log directory by finding the newest zstd file.
        
        Returns:
            Application ID extracted from the newest zstd filename, or None if not found
        """
        if not os.path.exists(self.spark_log_dir):
            logger.warning(f"Spark log directory {self.spark_log_dir} does not exist.")
            return None
            
        zstd_files = []
        for filename in os.listdir(self.spark_log_dir):
            if filename.endswith('.zstd'):
                filepath = os.path.join(self.spark_log_dir, filename)
                mtime = os.path.getmtime(filepath)
                zstd_files.append((filename, mtime))
        
        if not zstd_files:
            logger.warning(f"No zstd files found in {self.spark_log_dir}")
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


    def calculate_meta_feature(self, eval_func: Callable, task_id: str = "default", **kwargs):
        """
        Get runtime metric for current task by running default config and parsing latest Spark log.
        
        Args:
            eval_func: Evaluator function (ExecutorManager)
            task_id: Task ID
            kwargs: Additional keyword arguments
                - resume: Resume from a previous task
                - test_mode: Test mode
        """
        # skip meta_feature collecting and default config evaluation
        if kwargs.get('resume', None) is not None:
            self.current_task_history = History.load_json(
                                        filename=kwargs.get('resume'),
                                        config_space=self.config_space)
            self.current_meta_feature = np.array(self.current_task_history.meta_info.get('meta_feature'))
            logger.info(f"Current task meta feature: {self.current_meta_feature}")
            logger.info(f"Current task history: {self.current_task_history.objectives}")
            logger.info(f"Loaded current task history from {kwargs.get('resume')}")
            return

        # use default config writen in spark_default.conf
        default_config = self.config_space.get_default_configuration()
        default_config.origin = 'Default Configuration'
        result = eval_func(config=default_config, resource_ratio=1.0)
    
        if kwargs.get('test_mode', False):
            logger.info("Using test mode meta feature")
            self.current_meta_feature = META_FEATURE
            self.current_task_history = History(task_id=task_id, config_space=self.config_space,
                                                meta_info={'meta_feature': self.current_meta_feature.tolist()})
            self.current_task_history.update_observation(build_observation(default_config, result))
            self._update_similarity()
            return
        
        logger.info("Computing current task meta feature using default config...")

        application_id = self.get_latest_application_id()
        if not application_id:
            logger.warning("No application_id found, using fallback metrics")
            raise ValueError("No application_id found")

        zstd_file = os.path.join(self.spark_log_dir, f"{application_id}.zstd")
        if not os.path.exists(zstd_file):
            logger.warning(f"Zstd file not found: {zstd_file}, using fallback metrics")
            raise ValueError(f"Zstd file not found: {zstd_file}")
        logger.info(f"Found zstd file: {zstd_file}")

        json_file = os.path.join(self.spark_log_dir, "app.json")
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
        try:
            metrics = self.decode_results_spark(json_content)
        except Exception as e:
            logger.error(f"Failed to decode Spark log: {e}")
            logger.warning("Using fallback meta feature")
            metrics = META_FEATURE
        finally:
            try:
                os.remove(zstd_file)
            except Exception:
                pass
            try:
                os.remove(json_file)
            except Exception:
                pass

        logger.info(f"Initialized current task default with meta feature shape: {metrics.shape}")

        self.current_meta_feature = metrics
        self.current_task_history = History(task_id=task_id, config_space=self.config_space,
                                            meta_info={'meta_feature': self.current_meta_feature.tolist()})
        self.current_task_history.update_observation(build_observation(default_config, result))
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")

        self._update_similarity()


    def update_history_meta_info(self, meta_info: dict):
        """
        Update meta information of historical tasks.
        
        Args:
            meta_info: Meta information of historical tasks
        """
        self.current_task_history.meta_info.update(meta_info)

    def register_scheduler(self, scheduler):
        # ensure scheduler is only registered once
        if self._scheduler is not None:
            logger.error("Scheduler already registered")
            return
        self._scheduler = scheduler
        logger.info(f"Registered scheduler: {self._scheduler}")

    def get_scheduler(self) -> Optional[object]:
        return self._scheduler

    def register_sql_partitioner(self, partitioner) -> None:
        if self._sql_partitioner is not None and self._sql_partitioner is not partitioner:
            logger.warning("SQLPartitioner already registered; replacing with new instance")
        self._sql_partitioner = partitioner
        logger.info(f"Registered SQLPartitioner: {self._sql_partitioner}")

    def get_sql_partitioner(self):
        return self._sql_partitioner

    def register_planner(self, planner) -> None:
        if self._planner is not None and self._planner is not planner:
            logger.warning("Planner already registered; replacing with new instance")
        self._planner = planner
        logger.info(f"Registered Planner: {self._planner}")

    def get_planner(self):
        return self._planner

    def get_ws_args(self) -> Dict[str, Any]:
        return dict(self.ws_args)

    def get_tl_args(self) -> Dict[str, Any]:
        return dict(self.tl_args)

    def get_cp_args(self) -> Dict[str, Any]:
        return dict(self.cp_args)

    def get_scheduler_kwargs(self) -> Dict[str, Any]:
        return dict(self.scheduler_kwargs)

    def get_logger_kwargs(self) -> Dict[str, Any]:
        return dict(self.logger_kwargs)

    def get_random_kwargs(self) -> Dict[str, Any]:
        return dict(self.random_kwargs)

    def _update_similarity(self):
        self.similar_tasks_cache = map_source_hpo_data(
            target_his=self.current_task_history,
            source_hpo_data=self.historical_tasks,
            config_space=self.config_space,
            **self.ws_args,
        )
        
        filtered_sims = [(idx, sim) for idx, sim in self.similar_tasks_cache if sim >= self.similarity_threshold]
        self.similar_tasks_cache = filtered_sims
        
        logger.info(f"Updated similarity: {len(self.similar_tasks_cache)} tasks above threshold {self.similarity_threshold}")


    def get_similar_tasks(self, topk: Optional[int] = None) -> Tuple[List[History], List[Tuple[int, float]]]:
        """
        Get filtered similar tasks.
        
        Args:
            topk: Top k similar tasks to return

        Returns:
            Tuple of (filtered_histories, similarity_scores)
        """
        if not self.similar_tasks_cache:
            return [], []
        if topk is None:
            topk = self.tl_topk or len(self.similar_tasks_cache)
        topk = min(topk, len(self.similar_tasks_cache))
        filtered_histories = []
        filtered_sims = []
        for i in range(topk):
            idx, sim = self.similar_tasks_cache[i]
            filtered_histories.append(self.historical_tasks[idx])
            filtered_sims.append((i, sim))
            logger.info(f"Similar task {i}: {self.historical_tasks[idx].task_id} (similarity: {sim:.3f})")
        return filtered_histories, filtered_sims


    def update_current_task_history(self, config, results):
        """
        Update current task history with new evaluation result.
        This is called by optimizer after evaluator execution.
        
        Args:
            config: Configuration that was evaluated
            results: Evaluation results from executor
        """
        if not self.current_task_history:
            logger.warning("Current task not initialized, cannot update history")
            return
            
        from .utils import build_observation
        obs = build_observation(config, results)
        self.current_task_history.update_observation(obs)
        self._update_similarity()
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")


    def get_current_task_history(self) -> Optional[History]:
        return self.current_task_history