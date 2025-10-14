import numpy as np
from typing import List, Tuple
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox import logger, History
from sklearn.preprocessing import MinMaxScaler

from .base import *
from .rover.train_model import calculate_similarity, train_model
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
    def get_src_history(source_hpo_data: List[History]) -> Tuple[np.ndarray, List[List[int]]]:
        # meta_feature是一个n * l的numpy array,
        # 每行表示一个任务的meta feature, l是meta feature维数
        ts_meta_features = []

        # history是一个list, 里面的每个元素his表示一个任务的历史观测数据
        # 每个his也是一个list, 其中每个元素是形如 [conf, perf] 的list, 代表一轮的历史观测数据
        # 其中conf是一个numpy array, 代表该轮的配置, perf是一个float值, 代表该轮的观测结果
        ts_his = []

        for history in source_hpo_data:
            meta_feature = history.meta_info['meta_feature']
            ts_meta_features.append(meta_feature)

            ts_his.append([])
            for j in range(len(history)):
                obs = history.observations[j]
                config, perf = obs.config, obs.objectives[0]
                config = convert_configurations_to_array([config])[0]

                ts_his[-1].append([config, perf])

        ts_meta_features = np.array(ts_meta_features).copy()
        ts_meta_features[np.isnan(ts_meta_features)] = 0

        return ts_meta_features, ts_his

    def fit(self, source_hpo_data: List[History]):
        if self.already_fit:
            logger.warning('RoverMapper has already been fitted!')
            return

        config_space = source_hpo_data[0].config_space

        ts_meta_features, ts_his = self.get_src_history(source_hpo_data)

        self.scaler.fit(ts_meta_features)
        self.ts_meta_features = self.scaler.transform(ts_meta_features)

        # 计算相似度矩阵
        sim = calculate_similarity(ts_his, config_space)
        # print(sim)

        # 训练模型并保存
        self.model = train_model(self.ts_meta_features, sim)

        self.already_fit = True

    def map(self, target_history: History, source_hpo_data: List[History]) -> List[Tuple[int, float]]:
        # 获取当前任务的context
        target_meta_feature = target_history.meta_info['meta_feature']
        target_meta_feature = self.scaler.transform([target_meta_feature])[0]
        target_meta_feature[np.isnan(target_meta_feature)] = 0

        idxes, sims = get_transfer_tasks(self.ts_meta_features, target_meta_feature, num=len(self.ts_meta_features), theta=-float('inf'))

        return list(zip(idxes, sims))

