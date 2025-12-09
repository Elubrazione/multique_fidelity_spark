import os
import numpy as np
from typing import List, Tuple
from openbox.utils.config_space.util import convert_configurations_to_array
from ConfigSpace import ConfigurationSpace
from openbox import logger, History
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

from .base import *
from .rover.train_model import calculate_similarity, train_model, calculate_relative
from .rover.transfer import get_transfer_tasks


class RoverMapper(BaseMapper):
    def __init__(self, surrogate_type: str, seed: int = 0):
        super().__init__()

        self.method_id = 'rover'
        self.surrogate_type = surrogate_type
        self.surrogate_models = list()
        self.ts_meta_features = None
        self.seed = seed

        self.scaler = MinMaxScaler()

        self.model = None
        self.already_fit = False

    @staticmethod
    def history_to_list(history: History) -> List[List]:
        his = []
        for j in range(len(history)):
            obs = history.observations[j]
            config, perf = obs.config, obs.objectives[0]
            
            if perf is None or (isinstance(perf, float) and (np.isnan(perf) or np.isinf(perf))):
                continue
            
            config = convert_configurations_to_array([config])[0]
            his.append([config, perf])
        return his

    @staticmethod
    def get_src_history(
        source_hpo_data: List[History]
    ) -> Tuple[np.ndarray, List[List[int]]]:
        # meta_feature是一个n * l的numpy array,
        # 每行表示一个任务的meta feature, l是meta feature维数
        ts_meta_features = []

        # history是一个list, 里面的每个元素his表示一个任务的历史观测数据
        # 每个his也是一个list, 其中每个元素是形如 [conf, perf] 的list, 代表一轮的历史观测数据
        # 其中conf是一个numpy array, 代表该轮的配置, perf是一个float值, 代表该轮的观测结果
        ts_his = []
        
        skipped_count = 0
        for history in source_hpo_data:
            his_list = RoverMapper.history_to_list(history)
            
            if len(his_list) < 3:
                logger.warning(f"Skipping history task {history.task_id}: only {len(his_list)} valid observations after filtering (need at least 3)")
                skipped_count += 1
                continue
            
            meta_feature = history.meta_info['meta_feature']
            ts_meta_features.append(meta_feature)
            ts_his.append(his_list)

        if skipped_count > 0:
            logger.info(f"Filtered out {skipped_count} history tasks with insufficient valid observations, kept {len(ts_his)} valid tasks")

        ts_meta_features = np.array(ts_meta_features).copy()
        ts_meta_features[np.isnan(ts_meta_features)] = 0

        return ts_meta_features, ts_his

    def fit(
        self,
        source_hpo_data: List[History],
        config_space: ConfigurationSpace,
        use_cached_model: bool = False
    ):
        if self.already_fit:
            logger.warning('RoverMapper has already been fitted!')
            return

        ts_meta_features, ts_his = self.get_src_history(source_hpo_data)

        logger.info(f"Loaded {len(source_hpo_data)} source tasks")
        logger.info(f"Meta features shape: {ts_meta_features.shape}")
        if ts_meta_features.size > 0:
            logger.info(f"Meta features stats: min={ts_meta_features.min():.4f}, max={ts_meta_features.max():.4f}, std={ts_meta_features.std():.4f}")
        else:
            logger.warning("No meta features available from source tasks. Skipping stats and model training.")
            self.ts_meta_features = ts_meta_features
            self.model = None
            self.already_fit = True
            return

        if len(source_hpo_data) < 2:
            logger.warning(f"Only {len(source_hpo_data)} historical task(s) available. RoverMapper requires at least 2 tasks for similarity calculation.")
            self.ts_meta_features = ts_meta_features  # 不进行缩放
            self.model = None
            self.already_fit = True
            return

        self.scaler.fit(ts_meta_features)
        self.ts_meta_features = self.scaler.transform(ts_meta_features)
        logger.info(f"Scaled meta features stats: min={self.ts_meta_features.min():.4f}, max={self.ts_meta_features.max():.4f}, std={self.ts_meta_features.std():.4f}")

        if use_cached_model:
            model_path = 'catboost_info/model.cbm'
            if os.path.exists(model_path):
                logger.info(f"--use_cached_model flag is set. Loading existing model from {model_path}...")
                self.model = CatBoostRegressor()
                self.model.load_model(model_path)
                logger.info(f"Successfully loaded cached model from {model_path}")
                self.already_fit = True
                return
            else:
                logger.info(f"--use_cached_model flag is set but model not found at {model_path}. Training new model...")
        
        sim = calculate_similarity(ts_his, config_space)
        logger.info(f"Similarity matrix shape: {sim.shape}")
        logger.info(f"Similarity matrix stats: min={sim.min():.4f}, max={sim.max():.4f}, std={sim.std():.4f}")

        self.model = train_model(self.ts_meta_features, sim)
        self.already_fit = True

    def map(
        self,
        target_history: History,
        source_hpo_data: List[History],
        use_real: bool = False,
    ) -> List[Tuple[int, float]]:
        if use_real:
            logger.info("Using real observations to calculate similarity (consistency ratio)")
            return self.map_with_observations(target_history, source_hpo_data)
        else:
            logger.info("Using CatBoost prediction to calculate similarity (based on meta_feature)")
            return self.map_with_prediction(target_history, source_hpo_data)
    
    def map_with_prediction(self, target_history: History, source_hpo_data: List[History]) -> List[Tuple[int, float]]:
        target_meta_feature = np.array(target_history.meta_info.get('meta_feature', []))

        if len(target_meta_feature) == 0 or target_meta_feature.size == 0:
            logger.warning("Target meta_feature is empty, returning all tasks with similarity 1.0")
            return [(i, 1.0) for i in range(len(source_hpo_data))]
        
        if self.model is None:
            logger.warning("No trained model available (single task case), returning all tasks with similarity 1.0")
            return [(i, 1.0) for i in range(len(source_hpo_data))]
        
        if len(target_meta_feature) != self.ts_meta_features.shape[1]:
            logger.warning(f"Target meta_feature dimension ({len(target_meta_feature)}) doesn't match trained features ({self.ts_meta_features.shape[1]}), returning all tasks with similarity 1.0")
            return [(i, 1.0) for i in range(len(source_hpo_data))]
        
        target_meta_feature = self.scaler.transform([target_meta_feature])[0]
        target_meta_feature[np.isnan(target_meta_feature)] = 0

        idxes, sims = get_transfer_tasks(self.ts_meta_features, target_meta_feature, num=len(self.ts_meta_features), theta=-float('inf'))

        return list(zip(idxes, sims))
    
    def map_with_observations(
        self,
        target_history: History,
        source_hpo_data: List[History]
    ) -> List[Tuple[int, float]]:
        """
        使用真实观测数据计算目标任务与源任务之间的相似度（一致对占比）
        
        该方法使用真实观测值计算一致对占比，而不是使用CatBoost预测值。
        适用于任务开始后有足够观测数据时。
        
        计算方式：
        1. 对目标任务和源任务，分别用真实观测数据训练GP模型
        2. 在相同的25个随机测试配置上预测性能
        3. 计算预测序列之间的一致对占比作为相似度
        """
        target_his = self.history_to_list(target_history)
        _, source_his_list = self.get_src_history(source_hpo_data)
        
        # 构建包含目标任务和所有源任务的历史列表，用于calculate_similarity
        # calculate_similarity需要所有任务的历史数据来计算相似度矩阵
        all_history = [target_his] + source_his_list
        
        # 使用calculate_similarity计算相似度矩阵
        # 返回的sim矩阵中，sim[0, i+1]表示目标任务与第i个源任务的相似度
        sim_matrix = calculate_similarity(all_history, target_history.config_space)
        
        sims = []
        for idx in range(len(source_hpo_data)):
            # sim_matrix[0, idx+1] 是目标任务与第idx个源任务的相似度
            sim = sim_matrix[0, idx + 1]
            sims.append((idx, sim))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Calculated similarity with real observations: {[(idx, f'{sim:.4f}') for idx, sim in sims[:min(5, len(sims))]]}")
        return sims

