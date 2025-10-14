import pandas as pd
import shap
import copy
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from openbox import space as sp, logger
from config import HUGE_SPACE_FILE, RANGE_COMPRESS_DATA, ROOT_DIR


SPECIAL_MIN = {
    "spark.io.compression.lz4.blockSize": 64,
    "spark.io.compression.snappy.blockSize": 64,
}

def construct_huge_spaces(extra_int_params=None, extra_cat_params=None,
                          extra_float_params=None, space=None):
    if space is None:
        space = sp.Space()
    for name, lower, upper, default in (extra_int_params or []):
        space.add_variable(sp.Int(name, lower=int(lower), upper=int(upper), default_value=default))
    for name, choices, default in (extra_cat_params or []):
        space.add_variable(sp.Categorical(name, choices, default_value=default))
    for name, lower, upper, default in (extra_float_params or []):
        space.add_variable(sp.Real(name, lower=float(lower), upper=float(upper), default_value=float(default)))
    return space


def get_haisi_spark_space(fixed_in=False, custom_ranges=None,
                          extra_int_params=None, extra_cat_params=None, extra_float_params=None):
    space = sp.Space()
    def add_real_param(name, lower, upper, default, type="int"):
        if custom_ranges and name in custom_ranges:
            lower, upper = custom_ranges[name]

        default = max(int(lower), min(int(default), int(upper)))
        if type == "int":
            space.add_variable(sp.Int(name, lower=int(lower), upper=int(upper), default_value=default))
        elif type == "float":
            space.add_variable(sp.Real(name, lower=float(lower), upper=float(upper), default_value=float(default)))

    base_int_params = [
        ("spark.task.cpus", 1, 8, 1),
        ("spark.locality.wait", 0, 3, 0),
        ("spark.executor.memory", 1, 180, 52),
        ("spark.executor.cores", 1, 32, 16),
        ("spark.executor.instances", 1, 24, 12),
        ("spark.executor.memoryOverhead", 384, 20480, 1024),
        ("spark.driver.cores", 1, 16, 1),
        ("spark.driver.memory", 10, 120, 20),
        ("spark.default.parallelism", 100, 1200, 600),
        ("spark.sql.shuffle.partitions", 100, 1200, 600),
        ("spark.sql.autoBroadcastJoinThreshold", 10, 1000, 100),
        ("spark.network.timeout", 120, 30000, 600),
        ("spark.sql.broadcastTimeout", 300, 30000, 600),
        ("spark.sql.sources.parallelPartitionDiscovery.parallelism", 10, 200, 60),
    ]

    for param in base_int_params + (extra_int_params or []):
        add_real_param(*param, type="int")

    if fixed_in:
        base_cat_params = [
            ("spark.serializer", ["org.apache.spark.serializer.KryoSerializer"], "org.apache.spark.serializer.KryoSerializer"),
            ("spark.executor.extraJavaOptions", ["-XX:+UseG1GC"], "-XX:+UseG1GC"),
            ("spark.master", ['yarn'], 'yarn'),
            ("spark.sql.orc.impl", ['native'], 'native'),
            ("spark.sql.adaptive.enabled", ['true'], 'true'),
            ("spark.nodemanager.numas", ['2'], '2'),
        ]
    return space

def modify_spaces_range(space: sp.Space, new_ranges: dict):
    for name, (low, high) in new_ranges.items():
        if name in space:
            hp = space.get_hyperparameter(name)
            if isinstance(hp, sp.Real):
                hp.lower, hp.upper = float(low), float(high)
                if not (hp.lower <= hp.default_value <= hp.upper):
                    hp.default_value = (hp.lower + hp.upper) / 2
            elif isinstance(hp, sp.Int):
                hp.lower, hp.upper = int(low), int(high)
                if not (hp.lower <= hp.default_value <= hp.upper):
                    hp.default_value = (hp.lower + hp.upper) // 2
            else:
                print(f"参数 {name} 不是数值型 (type={type(hp)}), 跳过")
    return space

def haisi_huge_spaces(space, old_data_path: str, new_data_path: str):
    # space = get_haisi_spark_space()
    # space = construct_huge_spaces(extra_int_params, extra_cat_params, extra_float_params, get_haisi_spark_space())

    output_dir = f"{ROOT_DIR}/shap_plots"
    param_names = [hp.name for hp in space.get_hyperparameters()]
    
    old_top, new_top = load_and_prepare_data(old_data_path, new_data_path, param_names, ratio=0.8)
    new_space = analyze_space(old_top, new_top, param_names, output_dir, space=space)
    # new_space = construct_huge_spaces(extra_int_params, extra_cat_params, extra_float_params, new_space)
    logger.info(space)
    logger.info(new_space)

    return new_space

def get_range_by_std_filter(data, sigma=2):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    filtered = data[(data >= mean - sigma * std) & (data <= mean + sigma * std)]
    if len(filtered) > 0:
        return np.min(filtered), np.max(filtered)
    else:
        return np.min(data), np.max(data)

def clean_spark_columns(df):
    import re
    rename_map = {}
    for col in df.columns:
        if col.startswith("spark."):
            new_col = re.sub(r'_\w+$', '', col)
            rename_map[col] = new_col
    return df.rename(columns=rename_map)

def load_from_history_json(json_path, feature_cols):
    """
    从 History JSON 文件中加载数据并转换为 DataFrame 格式
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    observations = data.get('observations', [])
    if not observations:
        logger.warning(f"JSON 文件中没有找到 observations 数据: {json_path}")
        return pd.DataFrame()
    
    configs = []
    objectives = []
    
    for obs in observations:
        config = obs.get('config', {})
        objective = obs.get('objectives', [None])[0] if obs.get('objectives') else None
        
        if objective is not None and np.isfinite(objective):
            configs.append(config)
            objectives.append(objective)
    
    if not configs:
        logger.warning(f"JSON 文件中没有找到有效的配置数据: {json_path}")
        return pd.DataFrame()
    
    # 创建 DataFrame
    df = pd.DataFrame(configs)
    
    # 添加性能列
    if objectives:
        df['query_time'] = objectives
        df['spark_time'] = objectives  # 为了兼容性，也添加 spark_time 列
    
    for col in feature_cols:
        if col not in df.columns:
            # 尝试从配置空间获取默认值
            logger.warning(f"特征列 {col} 在 JSON 数据中不存在，将用 NaN 填充")
            df[col] = np.nan
    
    return df

def load_from_multiple_json_files(directory_path, feature_cols, recursive=True):
    """
    从目录中加载多个 JSON 文件并合并数据
    
    Args:
        directory_path: JSON 文件所在目录
        feature_cols: 特征列名列表
        recursive: 是否递归搜索子目录
    
    Returns:
        合并后的 DataFrame
    """
    import glob
    
    # 构建搜索模式
    if recursive:
        pattern = os.path.join(directory_path, "**", "*.json")
    else:
        pattern = os.path.join(directory_path, "*.json")
    
    # 查找所有 JSON 文件
    json_files = glob.glob(pattern, recursive=recursive)
    
    if not json_files:
        logger.warning(f"在目录 {directory_path} 中没有找到 JSON 文件")
        return pd.DataFrame()
    
    logger.info(f"找到 {len(json_files)} 个 JSON 文件")
    
    # 加载并合并所有 JSON 文件的数据
    all_dataframes = []
    successful_files = 0
    
    for json_file in json_files:
        try:
            df = load_from_history_json(json_file, feature_cols)
            if not df.empty:
                # 添加来源文件信息
                df['source_file'] = os.path.basename(json_file)
                all_dataframes.append(df)
                successful_files += 1
                logger.info(f"成功加载文件: {os.path.basename(json_file)} ({len(df)} 条记录)")
            else:
                logger.warning(f"文件 {os.path.basename(json_file)} 没有有效数据")
        except Exception as e:
            logger.error(f"加载文件 {json_file} 时出错: {e}")
            continue
    
    if not all_dataframes:
        logger.warning("没有成功加载任何 JSON 文件")
        return pd.DataFrame()
    
    # 合并所有 DataFrame
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    logger.info(f"成功合并 {successful_files} 个文件，总共 {len(merged_df)} 条记录")
    return merged_df

def load_and_prepare_data(old_path, new_path, feature_cols, ratio=0.3):
    # 检查 old_path 是否为目录
    if os.path.isdir(old_path):
        old_df = load_from_multiple_json_files(old_path, feature_cols)
    elif old_path.endswith('.json'):
        old_df = load_from_history_json(old_path, feature_cols)
    else:
        old_df = clean_spark_columns(pd.read_csv(old_path))
    
    # 检查 new_path 是否为目录
    if os.path.isdir(new_path):
        new_df = load_from_multiple_json_files(new_path, feature_cols)
    elif new_path.endswith('.json'):
        new_df = load_from_history_json(new_path, feature_cols)
    else:
        new_df = clean_spark_columns(pd.read_csv(new_path))
    
    target_query_key_old = 'query_time'
    target_query_key_new = 'query_time'
    if 'status' in old_df.columns:
        old_df = old_df[old_df['status'] == 'complete']
        target_query_key_old = 'spark_time'
    if 'status' in new_df.columns:
        new_df = new_df[new_df['status'] == 'complete']
        target_query_key_new = 'spark_time'

    old_top = old_df.nsmallest(int(len(old_df) * ratio), target_query_key_old)
    new_top = new_df.nsmallest(int(len(new_df) * ratio), target_query_key_new)
    return old_top, new_top

def train_shap_model(df, target_col, feature_cols):
    if target_col not in df.columns:
        if 'query_time' in df.columns:
            target_col = 'query_time'
        else:
            raise ValueError("Invalid target_col: %s" % (target_col))

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.info(f"非数值列将被丢弃: {list(non_numeric_cols)}")
        df = df.drop(columns=non_numeric_cols)
    
    # 只使用实际存在的特征列，排除完全为 NaN 的列
    available_feature_cols = []
    for col in df.columns:
        if col.startswith("spark.") and not df[col].isnull().all():
            available_feature_cols.append(col)
    
    logger.info(f"可用的特征列数量: {len(available_feature_cols)}")
    logger.info(f"原始特征列数量: {len([col for col in df.columns if col.startswith('spark.')])}")
    
    if len(available_feature_cols) == 0:
        raise ValueError("没有可用的特征列进行训练")
    
    X = df[available_feature_cols]
    y = df[target_col]
    
    # 处理 NaN 值：删除包含 NaN 的行
    nan_mask = X.isnull().any(axis=1)
    if nan_mask.any():
        logger.info(f"发现 {nan_mask.sum()} 行包含 NaN 值，将被删除")
        X = X.dropna()
        y = y[~nan_mask]
        logger.info(f"删除 NaN 后剩余 {len(X)} 行数据")
    
    if len(X) == 0:
        raise ValueError("删除 NaN 后没有剩余数据")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return model, explainer, shap_values, X, available_feature_cols

def compute_shap_stats(shap_vals_array, feature_cols):
    important_stats = []
    for i, param in enumerate(feature_cols):
        shap_col = shap_vals_array[:, i]
        neg_ratio = np.mean(shap_col < 0)       # SHAP 为负的比例
        mean_shap = np.mean(shap_col)           # 平均 SHAP 值
        min_shap = np.min(shap_col)             # 最负值
        important_stats.append((param, neg_ratio, mean_shap, min_shap))
    important_stats.sort(key=lambda x: x[2])
    return important_stats

def print_shap_summary(important_stats, print_nums=15):
    logger.info(f"{'参数':<60} {'SHAP为负占比':<15} {'平均SHAP值':<15} {'最小SHAP值':<15}")
    logger.info("=" * 130)
    for param, neg_ratio, mean_val, min_val in important_stats[: print_nums]:
        logger.info(f"{param:<60} {neg_ratio:<15.4f} {mean_val:<15.4f} {min_val:<15.4f}")

def get_shap_based_ranges(shap_values, X, feature_cols, shap_vals_array):
    shap_based_ranges = {}
    for i, param in enumerate(feature_cols):
        values = X[param].values
        shap_effect = shap_vals_array[:, i]
        beneficial_values = values[shap_effect < 0]
        if len(beneficial_values) > 0:
            min_val, max_val = get_range_by_std_filter(beneficial_values)
        else:
            min_val, max_val = get_range_by_std_filter(values)

        if param in SPECIAL_MIN:
            min_val = max(min_val, SPECIAL_MIN[param])
            if max_val < min_val:
                max_val = min_val + 5
        shap_based_ranges[param] = (min_val, max_val)
    return shap_based_ranges

def print_combined_space_table_numeric(space, shap_based_ranges, new_data_ranges):
    current_ranges = {}
    for var in space.get_hyperparameters():
        if hasattr(var, 'lower') and hasattr(var, 'upper'):
            current_ranges[var.name] = (var.lower, var.upper)
    def overlap_ratio(range1, range2):
        l1, r1 = range1
        l2, r2 = range2
        inter_left = max(l1, l2)
        inter_right = min(r1, r2)
        if inter_right < inter_left:
            return 0.0
        intersection = inter_right - inter_left
        union = max(r1, r2) - min(l1, l2)
        if union == 0:
            return 1.0
        return round(intersection / union, 4)

    logger.info(f"{'参数':<60} {'SHAP区间':<20} {'新数据区间':<20} {'重合度':<10} {'原搜索空间区间':<20}")
    logger.info("=" * 130)

    all_params = set(shap_based_ranges) | set(new_data_ranges) | set(current_ranges)

    rows = []
    for param in all_params:
        if param not in current_ranges:
            continue
        r1 = shap_based_ranges.get(param, ("N/A", "N/A"))
        r2 = new_data_ranges.get(param, ("N/A", "N/A"))
        if r1 == ("N/A", "N/A") or r2 == ("N/A", "N/A"):
            ratio = "N/A"
        else:
            ratio = overlap_ratio(r1, r2)
        curr_range = current_ranges[param]

        rows.append((param, str(r1), str(r2), ratio, str(curr_range)))

    rows = sorted(rows, key=lambda x: x[3] if isinstance(x[3], float) else -1, reverse=True)

    for param, r1_str, r2_str, ratio, curr_str in rows:
        logger.info(f"{param:<60} {r1_str:<20} {r2_str:<20} {str(ratio):<10} {curr_str:<20}")

def plot_shap_summary(shap_values, X, output_dir):
    short_name_map = {col: col[6: ] for col in X.columns}
    X_short = X.rename(columns=short_name_map)
    shap_values.feature_names = list(X_short.columns)
    plt.figure(figsize=(20, 3))
    shap.summary_plot(shap_values, X_short, show=False)
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()

def plot_individual_shap(shap_values, X, feature_cols, output_dir):
    print("shap_values.shape:", shap_values)
    print("len(feature_cols):", len(feature_cols))
    for i, param in enumerate(feature_cols):
        # print(i, param, shap_values[:, i])
        fig, ax = plt.subplots()
        shap.plots.scatter(
            shap_values[:, i],
            color=X[param].values,
            ax=ax,
            show=False
        )
        ax.set_title(f"SHAP Value Distribution: {param}")
        fig.savefig(os.path.join(output_dir, f"shap_{param.replace('.', '_')}.png"), bbox_inches='tight')
        plt.close(fig)

def analyze_space(old_top, new_top, feature_cols, output_dir, space=None, return_space=True, use_old=True):
    model, explainer, shap_values, X, feature_cols = train_shap_model(old_top, 'spark_time', feature_cols)
    shap_vals_array = shap_values.values
    logger.info("数值型参数个数: %d, 参数类型: %s" % (len(feature_cols), str(feature_cols)))

    important_stats = compute_shap_stats(shap_vals_array, feature_cols)
    logger.info("\n对提升 TPS 最关键的参数(SHAP 值越负越好)")
    print_shap_summary(important_stats)

    shap_based_ranges = get_shap_based_ranges(shap_values, X, feature_cols, shap_vals_array)

    if use_old:
        X_new = old_top[feature_cols]
    else:
        # todo: modify feature_col since columns maybe different in new_top
        X_new = new_top[feature_cols]
    shap_values_new = explainer(X_new)
    shap_vals_new_array = shap_values_new.values
    new_data_ranges = get_shap_based_ranges(shap_values_new, X_new, feature_cols, shap_vals_new_array)

    if space is None:
        logger.info("No space provided, using default expert space!")
        space = get_haisi_spark_space(fixed_in=False)
    else:
        space = copy.deepcopy(space)
    print_combined_space_table_numeric(space, shap_based_ranges, new_data_ranges)

    logger.info("\n根据 [%s] 表现好的结果构建建议搜索空间" % ("两节点" if use_old else "三节点"))
    logger.info(f"{'参数':<60} {'原搜索空间':<25} {'建议新空间':<25}")
    logger.info("=" * 110)

    suggested_ranges = {}
    for param, (min_val, max_val) in new_data_ranges.items():
        orig_range = space.get_hyperparameter(param)
        orig_range_str = f"({orig_range.lower}, {orig_range.upper})"
        try:
            min_int = int(round(float(min_val)))
            max_int = int(round(float(max_val)))
        except Exception:
            logger.info(f"{param:<60} {orig_range_str:<25} 跳过(建议区间无效: 非法数字)")
            continue
        if min_int > max_int:
            logger.info(f"{param:<60} {orig_range_str:<25} 跳过(建议区间无效: min > max)")
            continue
        if min_int == max_int:
            logger.info(f"{param:<60} {orig_range_str:<25} 跳过(建议区间无效: min == max)")
            continue
        suggested_ranges[param] = (min_val, max_val)
        logger.info(f"{param:<60} {orig_range_str:<25} ({min_val:.2f}, {max_val:.2f})")

    new_space = modify_spaces_range(space=space, new_ranges=suggested_ranges)

    # plot_shap_summary(shap_values, X, output_dir)
    # plot_individual_shap(shap_values, X, feature_cols, output_dir)
    
    if return_space:
        return new_space

    return model, explainer, shap_based_ranges, new_data_ranges, suggested_ranges, new_space


def load_space_from_json(json_file=None):
    
    if json_file is None:
        json_file = HUGE_SPACE_FILE
    
    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    space = sp.Space()
    
    for param_name, param_config in config.items():
        param_type = param_config["type"]
        default_value = param_config["default"]
        
        if param_type == "integer":
            space.add_variable(sp.Int(
                param_name,
                lower=param_config["min"],
                upper=param_config["max"],
                default_value=default_value
            ))
        elif param_type == "float":
            q = param_config.get("q", 0.05)
            space.add_variable(sp.Real(
                param_name,
                lower=param_config["min"],
                upper=param_config["max"],
                default_value=default_value,
                q=q
            ))
        elif param_type == "categorical":
            space.add_variable(sp.Categorical(
                param_name,
                choices=param_config["choice_values"],
                default_value=default_value
            ))
    
    return space


def haisi_huge_spaces_from_json(old_data_path=None, new_data_path=None, json_file=None):    
    if old_data_path is None:
        old_data_path = RANGE_COMPRESS_DATA
    
    old_space = load_space_from_json(json_file)
    new_space = haisi_huge_spaces(old_space, old_data_path=old_data_path, new_data_path=new_data_path)

    return old_space, new_space


if __name__ == "__main__":
    pass