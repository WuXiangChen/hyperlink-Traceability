#!/bin/bash

processes=$(ps aux | grep main | grep -v grep | awk '{print $2}')

if [ -z "$processes" ]; then
    echo "没有找到包含'mian'的进程。"
else
    echo "找到以下进程包含'mian'，正在杀死它们："
    echo "$processes"
    for pid in $processes; do
        kill -9 $pid
    done
    echo "已杀死包含'mian'的进程。"
fi

