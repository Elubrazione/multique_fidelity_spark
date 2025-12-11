import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

class PerformancePlotter:
    """性能数据绘图器"""
    
    def __init__(self, figsize=(12, 8), dpi=100):
        """
        初始化绘图器
        
        Args:
            figsize: 图形大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        self._setup_style()
        
    def _setup_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8-darkgrid')
        matplotlib.rcParams['font.size'] = 12
        matplotlib.rcParams['axes.titlesize'] = 16
        matplotlib.rcParams['axes.labelsize'] = 14
        matplotlib.rcParams['legend.fontsize'] = 12
        
    def load_performance_data(self, json_path: str) -> List[float]:
        """
        从JSON文件加载性能数据
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            性能数据列表
        """
        data = None
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if (data != None):
            raw_values = [observation['objectives'][0] for observation in data['observations']]
            processed_values = []
            for val in raw_values:
                if isinstance(val, str) and val.lower() in ['inf', '+inf', 'infinity']:
                    processed_values.append(float('inf'))
                elif isinstance(val, str) and val.lower() in ['-inf', '-infinity']:
                    processed_values.append(float('-inf'))
                elif isinstance(val, str) and val.lower() == 'nan':
                    processed_values.append(float('nan'))
                else:
                    # 尝试转换为float
                    try:
                        processed_values.append(float(val))
                    except (ValueError, TypeError):
                        # 如果转换失败，当作nan处理
                        processed_values.append(float('nan'))            
            return processed_values
        else:
            raise ValueError(f"fail to load data from {json_path}")
            
    def compute_cumulative_min_with_inf_skip(self, 
                                           data: List[Union[float, int]]) -> List[float]:
        """
        计算当前轮最小值，遇到inf时跳过
        
        例如: [inf, 3, inf, 5, 2] -> [?, 3, 3, 3, 2]
        
        Args:
            data: 原始数据列表，可能包含inf
            
        Returns:
            当前轮最小值列表
        """
        if not data:
            return []
        
        result = []
        current_min = None
        data = np.array(data)
        
        for i, value in enumerate(data):
            # 检查是否为有效数值
            is_valid = (isinstance(value, (int, float)) and 
                       not np.isinf(value) and 
                       not np.isnan(value))
            
            if is_valid:
                # 如果是第一个有效值
                if current_min is None:
                    current_min = value
                else:
                    current_min = min(current_min, value)
                result.append(current_min)
            else:
                # 如果是无效值（inf/nan），使用前一个最小值
                if current_min is not None:
                    result.append(current_min)
                else:
                    # 如果还没有有效的最小值，用None占位
                    result.append(None)
        
        return result
    
    def plot_cumulative_min_tasks(self, 
                                 tasks: List[str], 
                                 json_files: List[str], 
                                 colors: Optional[List[str]] = None,
                                 title: str = "Performance over iterations",
                                 xlabel: str = "Iterations",
                                 ylabel: str = "Best Performance (s)",
                                 save_path: Optional[str] = None,
                                 show_grid: bool = True,
                                 line_width: float = 2.5,
                                 marker_size: int = 8,
                                 alpha: float = 0.8,
                                 show_invalid_points: bool = True,
                                 invalid_points_marker: str = 'x',
                                 invalid_points_color: str = 'red'):
        """
        绘制多个任务的当前轮最小值
        
        Args:
            tasks: 任务名称列表
            json_files: JSON文件路径列表
            colors: 颜色列表
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            fill_under: 是否填充曲线下方
            show_original_points: 是否显示原始数据点
            original_points_alpha: 原始数据点透明度
            show_invalid_points: 是否标记无效点(inf/nan)
            invalid_points_marker: 无效点标记样式
            invalid_points_color: 无效点颜色
        """
        if len(tasks) != len(json_files):
            raise ValueError("任务数量和JSON文件数量必须相同")
            
        if colors is None:
            colors = ['blue', 'red', 'green', 'orange', 'purple', 
                     'brown', 'pink', 'gray', 'olive', 'cyan']
            colors = colors[:len(tasks)]
        elif len(colors) != len(tasks):
            raise ValueError("颜色数量必须和任务数量相同")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        all_cumulative_data = []  # 存储所有累积最小值数据
        all_original_data = []    # 存储所有原始数据
        
        # 处理并绘制每个任务
        for i, (task, json_file, color) in enumerate(zip(tasks, json_files, colors)):
            try:
                # 加载原始数据
                original_data = self.load_performance_data(json_file)
                # print(original_data)
                all_original_data.append(original_data)
                
                # 统计无效值
                data_array = np.array(original_data, dtype=np.float64)
                inf_count = np.sum(np.isinf(data_array))
                
                print(f"\n任务: {task}")
                print(f"   原始数据点: {len(data_array)} 个")
                print(f"   inf值数量: {inf_count}")
                
                # 计算当前轮最小值
                cumulative_min = self.compute_cumulative_min_with_inf_skip(original_data)
                all_cumulative_data.append(cumulative_min)
                
                # 提取有效点
                valid_indices = []
                valid_cumulative_values = []
                # print(len(cumulative_min))
                
                for idx, val in enumerate(cumulative_min):
                    if val is not None:
                        valid_indices.append(idx + 1)
                        valid_cumulative_values.append(val)
                
                if not valid_cumulative_values:
                    print(f"   ✗ 无有效累积最小值数据")
                    continue
                
                # 计算统计信息
                valid_array = np.array(valid_cumulative_values)
                print(f"   有效累积最小值: {len(valid_cumulative_values)} 个")
                print(f"   最终最小值: {valid_cumulative_values[-1]:.4f}")
                print(f"   累积最小值范围: [{np.min(valid_array):.4f}, {np.max(valid_array):.4f}]")
                
                # 准备x轴数据
                x = list(range(1, len(cumulative_min) + 1))
                
                # 绘制累积最小值曲线
                line_label = f"{task} (best perf)"
                line, = ax.plot(x, cumulative_min, 
                              color=color, 
                              linewidth=line_width,
                              marker='o',
                              markersize=marker_size,
                              alpha=alpha,
                              label=line_label,
                              zorder=3)
                
                
                
                # 标记无效点
                if show_invalid_points:
                    invalid_indices = []
                    
                    for idx, val in enumerate(original_data):
                        if isinstance(val, (int, float)):
                            if np.isinf(val) or np.isnan(val):
                                invalid_indices.append(idx + 1)
                    
                    if invalid_indices:
                        # 在无效点位置绘制标记
                        invalid_x = invalid_indices
                        invalid_y = []
                        
                        for idx in invalid_indices:
                            idx = idx - 1
                            # 找到无效点对应的累积最小值
                            if idx < len(cumulative_min) and cumulative_min[idx] is not None:
                                invalid_y.append(cumulative_min[idx])
                            else:
                                # 如果没有累积最小值，尝试找到最近的有效值
                                for j in range(idx-1, -1, -1):
                                    if j < len(cumulative_min) and cumulative_min[j] is not None:
                                        invalid_y.append(cumulative_min[j])
                                        break
                                else:
                                    invalid_y.append(0)
                        
                        # print(invalid_x)
                        # print(invalid_y)
                        
                        ax.scatter(invalid_x, invalid_y,
                                 color=invalid_points_color,
                                 marker=invalid_points_marker,
                                 s=marker_size*20,
                                 zorder=4,
                                 linewidths=1,
                                 label=f"(inf/nan)" if i == 0 else "")
                        
                        # 在无效点位置添加垂直虚线
                        for idx in invalid_indices:
                            if idx < len(cumulative_min) and cumulative_min[idx] is not None:
                                ax.axvline(x=idx, color='gray', 
                                         linestyle=':', 
                                         alpha=0.5, 
                                         linewidth=1, 
                                         zorder=1)
                
            except Exception as e:
                print(f"✗ 处理任务 '{task}' 时出错: {str(e)}")
                all_cumulative_data.append([])
                all_original_data.append([])
        
        # 设置图形属性
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # 设置图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:  # 确保有图例
            # 去重（因为可能有多个任务标记无效点）
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.legend(unique_handles, unique_labels, loc='best', frameon=True, shadow=True)
        
        # 设置网格
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置坐标轴范围
        valid_cumulative_values = []
        for data in all_cumulative_data:
            for val in data:
                if val is not None:
                    valid_cumulative_values.append(val)
        
        if valid_cumulative_values:
            x_max = max([len(d) for d in all_cumulative_data if d])
            ax.set_xlim(-0.5, x_max - 0.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            y_min, y_max = 0, np.max(valid_cumulative_values)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"\n✅ 图表已保存至: {save_path}")
        
        # 显示图形
        # plt.show()
        
        return fig, ax

# 主程序
if __name__ == "__main__":
    # 创建绘图器
    plotter = PerformancePlotter(figsize=(10, 6))

    # 定义任务
    tasks = ['loftune_tpch_600g']
    json_files = ['result/loftune_tpch600g_64u240n3.json']  # 替换为你的JSON文件路径
    colors = ['blue']

    # 绘制图形
    plotter.plot_cumulative_min_tasks(
        tasks=tasks,
        json_files=json_files,
        colors=colors,
        save_path="result/performance.png"
    )

    plt.close()