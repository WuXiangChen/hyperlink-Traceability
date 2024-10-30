#!/bin/bash
nohup python main.py -c 0 -n 5 -t 0.2 -type "repeat1" > "runlog/repeat1.log" 2>&1 &
nohup python main.py -c 1 -n 5 -t 0.2 -type "repeat2" > "runlog/repeat2.log" 2>&1 &
nohup python main.py -c 2 -n 5 -t 0.2 -type "repeat3" > "runlog/repeat3.log" 2>&1 &
wait