#!/bin/bash
#This script need an input, which indicate which folder under ../data/tsp_instances/ is used to train and eval
#For example, run from mlbnc/src/
#bash ./test_scripts/test.sh 200 

#source file path (to find train_agent.py)
source_file_dir="/bigdata/users/liyuhao/Single_instance_test/mlbnc/src"
#conda path and settings
conda_bin_activate_dir="/bigdata/users/liyuhao/local/anaconda3/bin/activate"
conda_env_name="mlbnc3.6cpu"
#Where to put the ouput log
stdout_log_path="/bigdata/users/liyuhao/Single_instance_test/mlbnc/logs"
cd "${source_file_dir}"
source "${conda_bin_activate_dir}" "${conda_env_name}"
echo -e "$$\t${BASH_SOURCE[0]}" >> "{stdout_log_path}/pids.txt"
python "${source_file_dir}/train_agent.py" tsp subtour PriorCutEnv "../data/tsp_instances/$1" > "${stdout_log_path}/$1_test.txt"