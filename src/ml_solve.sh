problem_type=tsp
cut_type=subtour
instance_path=../data/tsplib/kroA200.tsp
use_cut_detector=use-cut-detector
cut_detector_path=../models/subtour_detector/GNN_mincut.pt
use_rl_agent=no-use-rl-agent
agent_root=../logs
agent_folder=CutEnv_gap_12_1_661057
agent_name=best_model.pt
frequent=10
terminal_gap=0.01
result_path=""
display_log=display-log
log_path=""
device=cuda:0
python ml_solve.py $problem_type $cut_type $instance_path --$use_cut_detector --cut-detector-path=$cut_detector_path --$use_rl_agent --agent-root=$agent_root --agent-folder=$agent_folder --agent-name=$agent_name --frequent=$frequent --terminal-gap=$terminal_gap --result-path=$result_path --$display_log --log-path=$log_path --device=$device
