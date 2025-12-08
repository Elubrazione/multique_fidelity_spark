#!/bin/bash

# 遍历results目录下的所有子文件夹
for subdir in ./*/; do
    # 提取子文件夹名称（去除results/和后缀的/）
    subdir_name=$(basename "$subdir")
    echo "正在处理子文件夹: $subdir_name"
    
    # 遍历当前子文件夹中的所有JSON文件
    for json_file in "$subdir"*.json; do
        echo "找到JSON文件: $json_file"
        # 检查文件是否存在
        if [ -f "$json_file" ]; then
            # 提取JSON文件名（不含路径和扩展名）
            json_filename=$(basename "$json_file" .json)
            
            # 按第一个下划线分割文件名，获取第一部分
            # 如果文件名包含下划线，则取第一部分，否则使用整个文件名
            if [[ "$json_filename" == *"_"* ]]; then
                first_part=$(echo "$json_filename" | cut -d'_' -f1)
            else
                first_part="$json_filename"
            fi
            
            # 构建保存图片的路径
            save_path="./outputs/${subdir_name}_${first_part}.png"
            
            # 创建outputs目录（如果不存在）
            mkdir -p ./outputs
            
            # 调用draw_ratio.py脚本
            echo "处理文件: $json_file"
            echo "保存到: $save_path"
            python draw_ratio.py "$json_file" "$save_path"
            
            # 添加分隔线以便阅读
            echo "----------------------------------------"
        fi
    done
done

echo "所有文件处理完成！"