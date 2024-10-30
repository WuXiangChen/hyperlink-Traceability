#!/bin/bash

# 设置根目录路径
root_path="dataset/hyperlink_npz/"
max_jobs=2  # 最大并发任务数
source activate mmseg
rm -r ./runlog/*
rm -r ./saved_results/*
# 定义类型数组
types=("hybird_LLM_GCN_ResNet_9E_6")
# 遍历每种类型
for type_ in "${types[@]}"
do
  echo "Processing type: $type_"
  # 遍历目录下的所有文件
  for file in "$root_path"*
  do
    # cp BAAI_bge-m3_small/* model/BAAI_bge-m3_small
    # 获取文件名（不包括路径）
    filename=$(basename "$file")
    
    # 去掉文件扩展名
    filename_without_extension="${filename%.*}"
    
    # 构建结果文件名
    result_file="saved_results/${type_}_${filename_without_extension}_results.csv"

    # 判断结果文件是否存在
    if [ ! -f "$result_file" ]; then
      # 执行命令并将输出重定向到日志文件
        # 打印文件名到控制台
      echo "Processing file: $filename_without_extension"
      # 生成 [0, 3] 之间的随机实数
      num_GPU=$(python -c "import random; print(random.randint(0, 3))")
      echo "GPU Used num: $num_GPU"
      # 使用生成的随机数作为参数执行命令
      nohup python main.py -r "$filename_without_extension" -c "$num_GPU" -n 5 -t 0.3 -type "$type_" > "runlog/${type_}_${filename_without_extension}.log" 2>&1 &
    else
      echo "Result file already exists: $result_file. Skipping..."
    fi

    # 检查当前运行的任务数量
    while (( $(pgrep -f "python main" | wc -l) >= $max_jobs )); do
      sleep 30  # 等待30秒再检查
    done
  done
done


# 等待所有后台进程完成
wait
