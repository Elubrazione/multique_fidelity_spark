import numpy as np
import time
import ConfigSpace as CS
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.surrogate.base.build_gp import create_gp_model
from openbox.utils.util_funcs import get_types
from catboost import CatBoostRegressor

# 该函数用于从任务的meta_features生成pairwise的训练数据
# 其中A和B分别是nA * l和nB * l的numpy array, 每行表示一个任务的meta feature，l是meta feature维数
# S是一个nA * nB的numpy array，表示A和B中的任务两两之间的相似度
# 返回一个Tuple(X, Y)，X是将任务的meta feature进行pairwise的组合后得到的训练输入，shape为(nA * nB, l)
# Y是每条训练输入对应的输出，即任务之间的相似度


def generate_pairwise_data(A, B, S):
    X = []
    Y = []
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            X.append(np.concatenate([A[i], B[j]]))
            Y.append(S[i, j])
    return np.array(X), np.array(Y)

# 该函数用于计算A，B两个序列中排序关系正确的偏序对比例


def calculate_relative(A, B):
    P = 0
    l = len(A)
    for i in range(l):
        for j in range(i):
            if (A[i] <= A[j] and B[i] <= B[j]) or (A[i] > A[j] and B[i] > B[j]):
                P += 1
    return 2 * P / (l * (l - 1))


def calculate_similarity(history, config_space):
    """
    该函数提供一个计算任务间相似度的示例，主要展示相似度的计算方式

    输入参数history的格式如下:
    history是一个list, 里面的每个元素his表示一个任务的历史观测数据
    每个his也是一个list, 其中每个元素是形如 [conf, perf] 的list, 代表一轮的历史观测数据
    其中conf是一个numpy array, 代表该轮的配置, perf是一个float值, 代表该轮的观测结果

    相似度的计算方式如下：
    在搜索空间中随机采样若干个配置, 对每个任务用已观测的历史数据训练一个surrogate, 并用该surrogate对随机配置进行预测, 得到一个结果序列
    对每个任务对，计算它们的结果序列之间，预测的排序关系正确的偏序对比例，作为它们的相似度。
    """
    types, bounds = get_types(config_space)
    seed = int(time.time())
    rng = np.random.RandomState(seed)

    # 采样25个随机样本用于预测计算相似度
    testX = []
    for i in range(25):
        testX.append(convert_configurations_to_array(
                [config_space.sample_configuration()])[0]
            )
    testX = np.array(testX)

    predYs = []
    for his in history:
        # 这里的50是设置用于计算相似度的历史轮数上限
        l = min(50, len(his))
        trainX = np.array([his[i][0] for i in range(l)])
        trainY = np.array([his[i][1] for i in range(l)])

        # 用历史数据训练GP，并对testX中的样本进行预测
        surrogate_gp = create_gp_model(
            model_type='gp', config_space=config_space, types=types, bounds=bounds, rng=rng)
        surrogate_gp.train(trainX, trainY)

        predY, _ = surrogate_gp.predict(testX)
        predYs.append(predY.reshape(-1).tolist())

    # 利用预测结果计算两两之间排序正确的偏序对比例作为相似度
    n = len(history)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            sim[i][j] = calculate_relative(predYs[i], predYs[j])

    return sim



def train_model(src_meta_feature, sim):
    """
    该函数根据输入的任务meta_feature和相似度ground_truth信息训练预测相似度的CatBoost模型

    src_meta_feature是一个n * l的numpy array, 
        每行表示一个任务的meta feature, l是meta feature维数
    sim是一个n * n的numpy array, 
        表示训练使用的任务两两间相似度的ground truth信息

    注意src_meta_feature和sim中的任务顺序需要对齐
    """
    train_X, train_Y = generate_pairwise_data(
        src_meta_feature, src_meta_feature, sim)
    surrogate_cat = CatBoostRegressor()
    surrogate_cat.fit(train_X, train_Y, silent=True)

    surrogate_cat.save_model('model.cbm')  # 保存训练得到的模型

    return surrogate_cat
