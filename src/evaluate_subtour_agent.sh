problem_type=tsp
cut_type=subtour
callback_type=RLUserCallback
instance_path=../data/tsplib/kroA100.tsp
model_dir=../logs
model_folder=CutEnv_gap_12_1_661057
model_name=best_model.pt
python evaluate_agent.py $problem_type $cut_type $callback_type $instance_path $model_dir $model_folder $model_name