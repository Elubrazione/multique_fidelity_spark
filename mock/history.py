# 一个单例模式情况下的历史数据, 负责初始化, 维护, 更新和查看
# 本质是为了替代MySQL而使用pandas进行数据处理
import pandas as pd
from config.knobs_list import KNOB_DETAILS
from util import clear_scale_dict, gen_task_embedding

class GlobalData:
    _instance = None
    # 改成dataframe之后, 没必要像原来那么麻烦维护三个表, task_history, task_best_config, task_embeddings
    # 只需要维护一个history_data就行了
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.history_data = None
        return cls._instance
    
    def init_history_data(self, file_path):
        """初始化历史数据"""
        try:
            self.history_data = pd.read_csv(file_path)
            return True
        except Exception as e:
            self.history_data = pd.DataFrame()
            return False
    
    def get_history_data(self)->pd.DataFrame:
        """获取历史数据"""
        return self.history_data
    
    def update_history_data(self, new_data):
        """更新历史数据"""
        if self.history_data is None:
            self.history_data = pd.DataFrame(new_data)
        else:
            new_data_df = pd.DataFrame(new_data)
            self.history_data = pd.concat([self.history_data, new_data_df], ignore_index=True)

# 创建全局单例实例
global_data = GlobalData()

# 提供便捷函数
def fake_init_tuning_data(file_path):
    return global_data.init_history_data(file_path)

def get_history_data()->pd.DataFrame:
    return global_data.get_history_data()

def fake_update_data(app_id, task_id, config, duration, logger):
    new_data = {"app_id": app_id, "task_id": task_id, "duration": duration, "status": 1}
    new_data.update(clear_scale_dict(config))
    embedding_vector = gen_task_embedding(task_id)
    new_data.update(embedding_vector)
    global_data.update_history_data([new_data])

def fake_get_task_embedding(task_id):
    embeddings = global_data.get_history_data()
    task_embedding = embeddings[embeddings['task_id'] == task_id]
    if not task_embedding.empty:
        embedding_columns = [col for col in embeddings.columns if col not in ['app_id', 'task_id', 'duration', 'status'] + list(KNOB_DETAILS.keys())]
        return task_embedding[embedding_columns].values.tolist()[0]
    else:
        return None

def fake_get_task_best_config(task_id):
    history_data = global_data.get_history_data()
    task_data = history_data[history_data['task_id'] == task_id]
    if not task_data.empty:
        best_row = task_data.loc[task_data['duration'].idxmin()]
        best_config = {knob: best_row[knob] for knob in KNOB_DETAILS.keys()}
        return best_config
    else:
        return None