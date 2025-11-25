import os
import numpy as np
from typing import List, Tuple, Optional
from openbox import logger
from openbox.utils.history import History
from ConfigSpace import ConfigurationSpace


class HistoryManager:
    def __init__(self, 
                 config_space: ConfigurationSpace,
                 history_dir: str,
                 similarity_threshold: float = 0.0):
        self.config_space = config_space
        self.history_dir = history_dir
        self.similarity_threshold = similarity_threshold
        
        self.historical_tasks: List[History] = []
        self.historical_meta_features: List[np.ndarray] = []
        
        self.current_task_history: Optional[History] = None
        self.current_meta_feature: Optional[np.ndarray] = None
        
        self.similar_tasks_cache: List[Tuple[int, float]] = []
        
        self._load_historical_tasks()
    
    def _load_historical_tasks(self):
        if not os.path.exists(self.history_dir):
            logger.warning(f"History directory {self.history_dir} does not exist.")
            return
        
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
    
    def initialize_current_task(self, task_id: str, meta_feature: np.ndarray = None):
        meta_info = {}
        if meta_feature is not None:
            self.current_meta_feature = meta_feature
            meta_info['meta_feature'] = meta_feature.tolist()
        
        self.current_task_history = History(
            task_id=task_id,
            config_space=self.config_space,
            meta_info=meta_info
        )
        logger.info(f"Initialized current task history: {task_id}")
    
    def resume_current_task(self, history_file: str):
        self.current_task_history = History.load_json(
            filename=history_file,
            config_space=self.config_space
        )
        meta_feature = self.current_task_history.meta_info.get('meta_feature')
        if meta_feature is not None:
            self.current_meta_feature = np.array(meta_feature)
        logger.info(f"Resumed current task from {history_file}")
        logger.info(f"Current task history: {self.current_task_history.objectives}")
    
    def update_current_history(self, observation):    
        if self.current_task_history is None:
            logger.warning("Current task not initialized, cannot update history")
            return
        self.current_task_history.update_observation(observation)
        logger.info(f"Updated current task history, total observations: {len(self.current_task_history)}")
    
    def update_history_meta_info(self, meta_info: dict):
        if self.current_task_history is None:
            logger.warning("Current task not initialized, cannot update meta info")
            return
        self.current_task_history.meta_info.update(meta_info)
    
    def compute_similarity(self, similarity_func, **kwargs):
        if self.current_task_history is None:
            logger.warning("Current task not initialized, cannot compute similarity")
            return
        if not self.historical_tasks:
            logger.info("No historical tasks available for similarity computation")
            return
        self.similar_tasks_cache = similarity_func(
            target_his=self.current_task_history,
            source_hpo_data=self.historical_tasks,
            config_space=self.config_space,
            **kwargs
        )
        filtered_sims = [
            (idx, sim) for idx, sim in self.similar_tasks_cache 
            if sim >= self.similarity_threshold
        ]
        self.similar_tasks_cache = filtered_sims
        logger.info(f"Updated similarity: {len(self.similar_tasks_cache)} tasks above threshold {self.similarity_threshold}")
    
    def get_similar_tasks(self, topk: Optional[int] = None) -> Tuple[List[History], List[Tuple[int, float]]]:
        if not self.similar_tasks_cache:
            return [], []
        
        if topk is None:
            topk = len(self.similar_tasks_cache)
        topk = min(topk, len(self.similar_tasks_cache))
        
        filtered_histories = []
        filtered_sims = []
        
        for i in range(topk):
            idx, sim = self.similar_tasks_cache[i]
            filtered_histories.append(self.historical_tasks[idx])
            filtered_sims.append((i, sim))
            logger.info(f"Similar task {i}: {self.historical_tasks[idx].task_id} (similarity: {sim:.3f})")
        
        # Normalize similarities
        sims_sum = sum(sim for _, sim in filtered_sims)
        if sims_sum > 0:
            filtered_sims = [(idx, sim / sims_sum) for idx, sim in filtered_sims]
        logger.info(f"Normalized similarities: {filtered_sims}")
        return filtered_histories, filtered_sims
    
    def get_current_history(self) -> Optional[History]:
        return self.current_task_history
    
    def get_current_meta_feature(self) -> Optional[np.ndarray]:
        return self.current_meta_feature