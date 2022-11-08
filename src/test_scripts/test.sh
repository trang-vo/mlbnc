#!/bin/bash

source /bigdata/users/liyuhao/local/anaconda3/bin/activate mlbnc3.6cpu
echo -e "$$\t${BASH_SOURCE[0]}" >> pids.txt
python ./train_agent.py tsp subtour PriorCutEnv ../data/tsp_instances/$1 > ../logs/$1_test.txt
