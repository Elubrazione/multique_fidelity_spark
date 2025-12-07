import pandas as pd
import numpy as np
import json
import random

from config.knobs_list import KNOBS
from mock.history import get_history_data
from config.encoder_config import sql_embedding_dim

def fake_recommend_config(data:dict, logger):
    history_data = get_history_data()
    similar_task_id, similarity = find_most_similar_by_dot_product(data, history_data, sql_embedding_dim, logger)

    if similar_task_id is None:
        logger.info("未找到相似任务")
        return None
    
    logger.info(f"新任务 = {data['task_id']}, 相似任务 = {similar_task_id}, 相似度 = {similarity}")
    
    # 从history_data中查找相似任务的配置
    config = find_config_by_task_id(similar_task_id, history_data, logger)
    if config is None:
        logger.info(f"未找到相似任务 {similar_task_id} 的配置")
        return None
    
    # 仅保留配置相关的信息
    config = {key: value for key, value in config.items() if key in KNOBS}  
    
    return config, similar_task_id


def find_most_similar_by_dot_product(data, history_data, embedding_dim, logger):
    """
    使用点积计算当前任务与历史任务的相似度
    """
    try:
        # 提取当前任务的嵌入向量
        current_embedding = np.array([data[f'task_embedding_{i}'] for i in range(embedding_dim)])
        
        # 计算与每个历史任务的点积相似度
        similarities = []
        for _, row in history_data.iterrows():
            # 提取历史任务的嵌入向量
            history_embedding = np.array([row[f'task_embedding_{i}'] for i in range(embedding_dim)])
            # 计算点积相似度
            similarity = np.dot(current_embedding, history_embedding)
            similarities.append((row['task_id'], similarity))
        
        # 按相似度降序排序，取最相似的任务
        if similarities:
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_task_id, max_similarity = similarities[0]
            return similar_task_id, max_similarity
        else:
            return None, 0.0
            
    except Exception as e:
        logger.error(f"计算相似度时出错: {e}")
        return None, 0.0


def find_config_by_task_id(task_id, history_data, logger)-> dict:
    """
    根据task_id在history_data中查找对应的配置
    """
    try:
        # 查找匹配的任务配置
        matched_rows = history_data[history_data['task_id'] == task_id]
        
        if len(matched_rows) == 0:
            logger.warning(f"未找到任务 {task_id} 的配置")
            return None
        
        # 取history_data中duration字段最小的配置
        config_row = matched_rows.loc[matched_rows['duration'].idxmin()]

        # 转换为字典
        config_dict = config_row.to_dict()
        
        return config_dict
        
    except Exception as e:
        logger.error(f"查找配置时出错: {e}")
        return None