#!/bin/bash

cd /bigdata/users/liyuhao/Single_instance_test/mlbnc/src
source /bigdata/users/liyuhao/local/anaconda3/bin/activate mlbnc3.6cpu
echo -e "$$\t${BASH_SOURCE[0]}" >> pids.txt
python /bigdata/users/liyuhao/Single_instance_test/mlbnc/src/train_agent.py tsp subtour PriorCutEnv ../data/tsp_instances/$1 > /bigdata/users/liyuhao/Single_instance_test/mlbnc/logs/$1_test.txt