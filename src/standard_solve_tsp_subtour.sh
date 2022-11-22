problem_type=tsp
cut_type=subtour
instance_path=../data/tsplib/u574.tsp
use_callback=1
frequent=1
terminal_gap=0.01
result_path=../results/test.csv
python standard_solve.py $problem_type $cut_type $instance_path --use-callback=$use_callback --frequent=$frequent --terminal-gap=$terminal_gap --result-path=$result_path
