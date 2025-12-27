# 绘制tpch-600g的性能曲线

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys
import time
import random
import math
import re



# 机器越来越慢，写一个系数，随着时间增加，系数越来越小，时间为0时候系数为1，时间为96时候系数为0.9
def get_coefficient(time):
    # 写成分段，0-24小时，系数为1，24-48小时，系数为0.9，48-72小时，系数为0.8，72-96小时，系数为0.7
    # time是一个numpy
    time = np.array(time)
    coefficient = np.ones_like(time)
    # coefficient[time <= 96] = 0.7
    # coefficient[time <= 72] = 0.8
    # coefficient[time <= 48] = 0.9
    # coefficient[time <= 24] = 1

    return coefficient


METHODS = [
    "rover",
    "tuneful",
    # "loftune",
    # "toptune",
]

target = 'tpch_600g_crossbench'

DATA_DIR = f"./{target}"
SAVE_DIR = f"./images/{target}"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

plt.figure(figsize=(10, 6))
for method in METHODS:
    data_path = os.path.join(DATA_DIR, method)
    # 找到其中的.json文件
    json_files = sorted([f for f in os.listdir(data_path) if f.endswith('.json')])
    if len(json_files) == 0:
        raise ValueError(f"No json files found in {data_path}")
    json_file = os.path.join(data_path, json_files[-1])
    
    data = json.load(open(json_file))

    # 读出所有observations中的objectives 和 elapsed_time，横坐标是时间，纵坐标是objectives画图
    objectives = [20000]
    elapsed_time = [0]
    for observation in data["observations"]:
        objectives.append(observation["objectives"][0])
        elapsed_time.append(observation["elapsed_time"])

    # objectives 取收敛曲线 
    ori_objectives = np.array(objectives)
    # AttributeError: module 'numpy' has no attribute 'cummin'
    # 时间累加
    elapsed_time = np.cumsum(elapsed_time)
    # 转化为h
    hours = elapsed_time / 3600

    # 计算系数
    coefficient = get_coefficient(hours)
    ori_objectives *= coefficient
    objectives = np.minimum.accumulate(ori_objectives)

    # 把点也画出来
    plt.plot(hours, objectives, label=method)
    plt.scatter(hours, ori_objectives, marker='o', label=method)

plt.legend()
plt.xlabel("Time")
plt.ylabel("Objectives")
plt.title(f"{method} Performance")
# x每4个小时一个刻度，标到48小时，包括0和48
plt.xticks(np.arange(0, 98, 4))
plt.xlim(0, 96)
# 设置y最大为 6000
plt.ylim(0, 10000)
plt.savefig(os.path.join(SAVE_DIR, f"{target}.png"))
plt.show()