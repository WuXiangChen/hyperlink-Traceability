#!/bin/bash

# Find PIDs for "python main.py"
processes=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}')

# Find PIDs for "bash LLM_Comparative_Experiments_Design.sh"
bash_processes=$(ps aux | grep "LLM_Comparative_Experiments_Design.sh" | grep -v grep | awk '{print $2}')

# Combine both PIDs
all_processes="$processes $bash_processes"

if [ -z "$all_processes" ]; then
    echo "没有找到包含'python main.py'或'bash LLM_Comparative_Experiments_Design.sh'的进程。"
else
    echo "找到以下进程包含'python main.py'或'bash LLM_Comparative_Experiments_Design.sh'，正在杀死它们："
    echo "$all_processes"
    for pid in $all_processes; do
        kill -9 $pid
    done
    echo "已杀死包含'python main.py'或'bash LLM_Comparative_Experiments_Design.sh'的进程。"
fi