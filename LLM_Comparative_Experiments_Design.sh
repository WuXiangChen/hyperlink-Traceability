#!/bin/bash

# 设置根目录路径
root_path="../dataset/hyperlink_npz/"
max_jobs=1  # 最大并发任务数
source activate mmseg

# 定义类型数组
types=("LLM_FULLCONTENT")
# 遍历每种类型
for type_ in "${types[@]}"
do
  echo "Processing type: $type_"
  # 遍历目录下的所有文件
  for file in "$root_path"*
  do
    #cp BAAI_bge-m3_small/* model/BAAI_bge-m3_small
    # 获取文件名（不包括路径）
    filename=$(basename "$file")
    
    # 去掉文件扩展名
    filename_without_extension="${filename%.*}"
    
    # 构建日志文件名
    log_file="./logsAndResults/runlog/${type_}_${filename_without_extension}_results.log"
    
    # 执行 Python 脚本
    nohup bash -c "CUDA_VISIBLE_DEVICES=3 python main.py  -c 0 -n 5 -t 0.2 -type $type_" > "$log_file" 2>&1 &
    
    # 输出当前执行的参数组合
    echo "Running with $filename_without_extension. Log file: $log_file"
    wait

    done
done
wait
