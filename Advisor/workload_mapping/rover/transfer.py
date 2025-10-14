import numpy as np
import time
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox import History, Advisor
from catboost import CatBoostRegressor
from typing import List

from .trans_surrogate import RGPE_avg


def get_transfer_tasks(src_meta_feature, target_meta_feature, num = 5, theta = 0.7, model=None):
    """
    该函数用于筛选与目标任务最相近的num个相似度大于阈值theta的任务, 并返回一个它们在src_meta_feature中的下标构成的列表
    如果没有相似度大于阈值的任务, 则会返回一个空列表

    src_meta_feature是一个n * l的numpy array, 每行表示一个任务的meta feature, l是meta feature维数
    target_meta_feature是一个长为l的一维numpy array, 表示目标任务的meta feature
    """
    n_src = src_meta_feature.shape[0]

    target_X = []
    for i in range(n_src):
        target_X.append(np.concatenate([src_meta_feature[i], target_meta_feature]))
    target_X = np.array(target_X)

    surrogate_cat = model
    if model is None:
        surrogate_cat = CatBoostRegressor()
        surrogate_cat.load_model('model.cbm') # 加载之前训练的预测模型

    pred_Y = surrogate_cat.predict(target_X)
    
    idx = np.argsort(-pred_Y)
    res = []
    res_pred_Y = []
    j = 0
    while j < num and j < n_src:
        if j > 0 and pred_Y[idx[j]] < theta:
            break
        res.append(idx[j])
        res_pred_Y.append(pred_Y[idx[j]])
        j += 1

    return res, res_pred_Y


def get_transfer_suggestion(src_historys:List[History], target_history:History, **kwargs):
    """
    该函数用于根据筛选出的源任务的历史观察和目标任务的历史观察, 给出迁移学习模型的推荐配置

    src_history是源任务的历史观察结果构成的列表, target_history是目标任务的历史观察结果

    每个任务的观察结果his格式如下;
        每个his是一个list, 其中每个元素是形如[conf, perf]的list, 代表一轮的历史观测数据
        其中conf是一个numpy array, 代表该轮的配置, perf是一个float值, 代表该轮的观测结果
        返回值是一个numpy array, 代表迁移学习模型推荐的配置
    """
    config_space = target_history.config_space

    # 将src_history转换为openbox支持的格式
    new_his = []
    for i, his in enumerate(src_historys):
        tmp = History(task_id=f'history{i}', config_space=config_space)
        for obs in his.observations:
            tmp.update_observation(obs)
        new_his.append(tmp)

    # 创建一个openbox的advisor, 替换代理模型, 更新观察历史并给出推荐配置
    _logger_kwargs = kwargs.get('_logger_kwargs', None)
    advisor = Advisor(config_space, num_objectives=1, num_constraints=0, initial_trials=1,
                      surrogate_type='gp', acq_type='ei', rand_prob=0, logger_kwargs=_logger_kwargs)  # acq_optimizer_type='random_scipy'
    surrogate = RGPE_avg(config_space, new_his, int(time.time()), surrogate_type='gp', num_src_hpo_trial=-1)
    advisor.surrogate_model = surrogate

    for obs in target_history.observations:
        advisor.update_observation(obs)

    raw_configs = advisor.get_suggestion(return_list=True)
    configs = convert_configurations_to_array(raw_configs)
    evaluated_configs = target_history.get_config_array()

    for i, config in enumerate(configs):
        flag = True
        for e_conf in evaluated_configs:
            if np.sum(np.abs(config-e_conf) <= 1e-3):
                flag = False
                break
        if flag:
            return raw_configs[i]

    return raw_configs[-1]