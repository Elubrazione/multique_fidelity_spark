import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import logging

def setup_basic_logger():
    """基础日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # logging.StreamHandler(sys.stdout), # 输出到控制台
            logging.FileHandler('app.log') # 输出到文件
        ]
    )

    return logging.getLogger(__name__)

logger = setup_basic_logger()

def normalize_performance(performance_list):
    """
    对性能数据进行min-max归一化
    """
    performance_array = np.array(performance_list)
    min_val = np.min(performance_array)
    max_val = np.max(performance_array)
    
    # 防止除以0
    if max_val == min_val:
        return np.ones_like(performance_array) * 0.5
    
    normalized_data = (performance_array - min_val) / (max_val - min_val)
    return normalized_data

def plot_performance_scatter(performance_list, title="Performance Distribution"):
    """
    绘制性能数据在[0,1]线段上的散点图
    """
    valid_performance = [p for p in performance_list if not np.isinf(p)]
    inf_count = len(performance_list) - len(valid_performance)
    ratio = len(valid_performance) / len(performance_list)
    logger.info(f"Original data size: {len(performance_list)}")
    logger.info(f"inf data size: {inf_count}")
    logger.info(f"valid ratio: {ratio:.2%}")

    # 如果没有inf数据，提示用户
    if inf_count == 0:
        logger.warning("There is no inf data, two ways would be the same")
    
    # 将inf替换为极大值
    max_finite = np.max([p for p in valid_performance])
    performance_list = [p if not np.isinf(p) else max_finite for p in performance_list]

    method_data = {
        "Valid Data": valid_performance,
        "Transfered Data": performance_list
    }
    
    # 计算实际的分位数位置
    quantiles = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 添加分段线 - 使用实际的分位数位置
    colors = ['red', 'orange', 'green', 'blue', 'yellow', 'purple']
    quantile_lables = ['20%', '40%', '60%', '70%', '80%', '90%']

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    results = {}

    for idx, (method_name, data) in enumerate(method_data.items()):
        ax = axes[idx]

        # 归一化
        normalized_data = normalize_performance(data)

        quantile_values = np.percentile(normalized_data, [q * 100 for q in quantiles])
    
        # 绘制[0,1]线段
        ax.plot([0, 1], [0, 0], 'k-', linewidth=3, alpha=0.7)
    
        # 绘制散点（没有纵坐标偏移）
        scatter = ax.scatter(normalized_data, np.zeros_like(normalized_data), 
                            alpha=0.7, s=50, c=normalized_data, 
                            cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
    
    
    
        for q_value, q_label, color in zip(quantile_values, quantile_lables, colors):
            ax.axvline(x=q_value, color=color, linestyle='--', alpha=0.8, linewidth=2)
            
            # 在分段线上方添加标注
            ax.text(q_value, 0.1, f'{q_label}\n({q_value:.3f})', 
                    fontsize=10, color=color, ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.15, 0.15)
        ax.set_xlabel('Normalized Performance Score', fontsize=12)
        ax.set_title(f"{method_name}", fontsize=13, fontweight='bold')
    
        # 隐藏y轴
        ax.set_yticks([])
        ax.set_ylabel('')
    
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.15)
        cbar.set_label('Normalized Performance (0=min, 1=max)', fontsize=10)
    
        # 添加详细的统计信息
        original_stats = f'Original Statistics:\n'
        original_stats += f'N={len(data)}\n'
        original_stats += f'Min={np.min(data):.3f}\n'
        original_stats += f'Max={np.max(data):.3f}\n'
    
        # 在左侧添加原始数据统计
        ax.text(0.85, 0.95, original_stats, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightblue', alpha=0.8))
    
        # 添加分位数信息
        quantile_info = 'Quantile Positions:\n'
        for q_label, q_value in zip(quantile_lables, quantile_values):
            quantile_info += f'{q_label}: {q_value:.3f}\n'
    
        ax.text(0.85, 0.05, quantile_info, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='wheat', alpha=0.8))
        
        # 存储结果
        results[method_name] = {
            'data': data,
            'normalized_data': normalized_data,
            'quantiles': dict(zip(quantile_lables, quantile_values)),
            'stats': {
                'n': len(data),
                'min': np.min(data),
                'max': np.max(data)
            }
        }
    
    # 调整布局
    plt.tight_layout()

    return fig, axes, results

# 示例使用
if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.info("Usage: python draw_ratio.py <json_file_path> <save_fig_path>")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    save_fig_path = sys.argv[2]

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"success to load file: {json_file_path}")
    except Exception as e:
        logger.error(f"fail to load file: {json_file_path}")
        logger.error(str(e))
        sys.exit(1)

    observations = data['observations']
    performance_data = [observation['objectives'][0] for observation in observations]
    
    # 绘制图形
    fig, ax, results = plot_performance_scatter(
        performance_data, 
        title="Performance Distribution with Actual Quantile Markers"
    )

    fig.savefig(save_fig_path)

    plt.close()