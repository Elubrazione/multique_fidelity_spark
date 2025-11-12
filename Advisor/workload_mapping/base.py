from typing import Tuple, List
from openbox.utils.history import History


class BaseMapper:
    def __init__(self):
        self.map_strategy = "base"

    def fit(self, source_hpo_data: List[History]):

        raise NotImplementedError

    # 返回相似性降序排列编号，已经对应的相似度
    def map(self, target_history: History, source_hpo_data: List[History]) -> List[Tuple[int, float]]:

        raise NotImplementedError
