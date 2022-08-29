problem_type=maxcut
cut_type=cycle
callback_type=RLCycleUserCallback
instance_path=../data/maxcut/rudy_all/pm1s_100.1
model_dir=../logs
model_folder=CycleEnv_8171628
model_name=model_748000_steps
python evaluate_agent.py $problem_type $cut_type $callback_type $instance_path $model_dir $model_folder $model_name
