#!/bin/bash

cd /bigdata/users/liyuhao/mlbnc/src
source /bigdata/users/liyuhao/local/anaconda3/bin/activate mlbnc3.6cpu
echo -e "$$\t${BASH_SOURCE[0]}" >> pids.txt
python /bigdata/users/liyuhao/mlbnc/src/train_agent.py tsp subtour PriorCutEnv > /bigdata/users/liyuhao/mlbnc/logs/test1_log.txt