#!/bin/bash
source  activate mmseg
max_jobs=1  # 最大并发任务数
export CUDA_VISIBLE_DEVICES=2  # 指定使用的GPU编号
# 定义参数组合
for freeze in True False; do
    for with_knowledge in True False; do
        for cat in True False; do
            # 创建日志文件名
            log_file="logsAndResults/runlog/LLM_Comparative_Experiment_f${freeze}_k${with_knowledge}_c${cat}.log"
            
            # 执行 Python 脚本
            nohup python main.py --freeze $freeze --with_knowledge $with_knowledge --cat $cat -c 0 -n 5 -t 0.2 -type "LLM_Comparative_Experiment" > "$log_file" 2>&1 &
            
            # 输出当前执行的参数组合
            echo "Running with freeze=$freeze, with_knowledge=$with_knowledge, cat=$cat. Log file: $log_file"
            while (( $(pgrep -f "python main" | wc -l) >= max_jobs )); do
                sleep 30  # 等待30秒再检查
            done
        done
    done
done
