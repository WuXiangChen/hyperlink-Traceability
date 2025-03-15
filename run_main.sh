#!/bin/bash

# 设置根目录路径
root_path="../dataset/hyperlink_npz"
max_jobs=4 # 最大并发任务数
source activate mmseg
# 定义类型数组
types=("hybird_LLM_GCN_ResNet_9E_6")
# 初始化 GPU 编号
current_GPU=0
for type_ in "${types[@]}"
do
  echo "Processing type: $type_"
  # 遍历目录下的所有文件
  for file in "$root_path"/*.npz
  do
    echo "$file"
    filename=$(basename "$file")
    echo "$filename"

    filename_without_extension="${filename%.*}"
    # 构建结果文件名
    result_file="logsAndResults/saved_results/${filename_without_extension}_results.csv"
    if [ ! -f "$result_file" ]; then
      # 执行命令并将输出重定向到日志文件
      echo "Processing file: $filename_without_extension"
      # 逐次累加 GPU 编号并对4取模
      num_GPU=$((current_GPU % 4))

      echo "GPU Used num: $num_GPU"
      # 使用累加的 GPU 编号作为参数执行命令
      nohup bash -c "CUDA_VISIBLE_DEVICES=$num_GPU python -u main.py -r $filename_without_extension -tt 3" > "./runlog_training_main/${filename_without_extension}.log" 2>&1 &
      # 增加当前 GPU 编号
      current_GPU=$((current_GPU + 1))
    else
      echo "Result file already exists: $result_file. Skipping..."
    fi
    # 检查当前运行的任务数量
    while (( $(pgrep -f "python main" | wc -l) >= $max_jobs )); do
      sleep 300  # 等待30秒再检查
      current_GPU=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -F',' '$2 == " 0" {print $1}' | head -n 1)
    done
  done
done

# 等待所有后台进程完成
wait
