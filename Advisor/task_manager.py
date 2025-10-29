import os
import json
import subprocess
import tempfile
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace
from .utils import map_source_hpo_data


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

    def __init__(self, 
                 history_dir: str,
                 eval_func: Callable,
                 spark_log_dir: str = "/root/codes/spark-log",
                 similarity_threshold: float = 0.5,
                 ws_args: Optional[dict] = None, 
                 config_space: Optional[ConfigurationSpace] = None):
        self.history_dir = history_dir
        self.spark_log_dir = spark_log_dir
        self.ws_args = ws_args or {}
        self.similarity_threshold = similarity_threshold
        self.config_space = config_space
        
        self.historical_tasks: List[History] = []
        self.historical_meta_features: List[np.ndarray] = []
        
        self.current_task_history: Optional[History] = None # initialized in initialize_current_task, used in _update_similarity
        self.current_meta_feature: Optional[np.ndarray] = None # initialized in calculate_meta_feature, used in initialize_current_task
        
        # For multi-fidelity optimization (MFBO)
        self.multi_fidelity_history_list: List[History] = []
        self.resource_identifiers: List[int] = []
        
        self.similar_tasks_cache: List[Tuple[int, float]] = []
        
        self._load_historical_tasks()
        self.calculate_meta_feature(eval_func)


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
            if line == '':
                continue
            try:
                logs.append(json.loads(line))
            except:
                raise ValueError('Cannot decode json data: %s' % line)

        start_time, end_time = None, None
        task_metrics = dict()

        cnt = 0
        for event in logs:
            if event['Event'] == "SparkListenerApplicationStart":
                start_time = event['Timestamp']
            elif event['Event'] == "SparkListenerApplicationEnd":
                end_time = event['Timestamp']
            elif event['Event'] == "SparkListenerTaskEnd":
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

        run_time = (end_time - start_time) / 1000

        keys = list(task_metrics.keys())
        keys.sort()
        for k, v in task_metrics.items():
            logger.debug(f"{k}: {v / cnt}")
        metrics = np.array([task_metrics[key] / cnt for key in keys])

        if start_time is None or end_time is None:
            raise ValueError('Cannot find start or end time in log')

        return run_time, metrics

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


    def calculate_meta_feature(self, eval_func: Callable):
        """
        Get runtime metric for current task by running default config and parsing latest Spark log.
        
        Args:
            eval_func: Evaluator function (ExecutorManager)

        """
        logger.info("Computing current task meta feature using default config...")

        # use default config writen in spark_default.conf
        default_config = {}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            _ = eval_func(config=default_config, resource_ratio=1.0, res_dir=tmp_dir)

            application_id = self.get_latest_application_id()
            if not application_id:
                logger.warning("No application_id found, using fallback metrics")
                raise ValueError("No application_id found")
            zstd_file = os.path.join(self.spark_log_dir, f"{application_id}.zstd")
            if not os.path.exists(zstd_file):
                logger.warning(f"Zstd file not found: {zstd_file}")
                raise ValueError(f"Zstd file not found: {zstd_file}")
            logger.info(f"Found zstd file: {zstd_file}")
            json_file = os.path.join(tmp_dir, "app.json")
            subprocess.run(['zstd', '-d', zstd_file, '-o', json_file], check=True)
            with open(json_file, 'r') as f:
                json_content = f.read()
            run_time, metrics = self.decode_results_spark(json_content)
            
            logger.info(f"Application run time: {run_time:.2f} seconds")
            logger.info(f"Metrics array shape: {metrics.shape}")
            self.current_meta_feature = metrics

    def update_history_meta_info(self, meta_info: dict):
        """
        Update meta information of historical tasks.
        
        Args:
            meta_info: Meta information of historical tasks
        """
        self.current_task_history.meta_info.update(meta_info)


    def initialize_current_task(self, task_id: str, meta_info: dict = None):
        """
        Initialize current task, called in Optimizer.base.BaseOptimizer.__init__

        Args:
            task_id: Current task identifier, used for history task_id
            eval_func: Evaluator function
            meta_info: Meta information of current task
        """
        meta_info = meta_info or {}
        logger.info(f"Current meta feature: {self.current_meta_feature}")
        meta_info['meta_feature'] = self.current_meta_feature.tolist()
        self.current_task_history = History(task_id=task_id, config_space=self.config_space, meta_info=meta_info)
        logger.info(f"Initialized current task {task_id} with meta feature shape: {self.current_meta_feature.shape}")
        self._update_similarity() # update similarity between current task and historical tasks


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


    def get_similar_tasks(self, topk: int = 3) -> Tuple[List[History], List[Tuple[int, float]]]:
        """
        Get filtered similar tasks.
        
        Args:
            topk: Top k similar tasks to return

        Returns:
            Tuple of (filtered_histories, similarity_scores)
        """
        if not self.similar_tasks_cache:
            return [], []
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