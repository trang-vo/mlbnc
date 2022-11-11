from problem_generators.tsp_generator import generate_tsp_problems
if __name__ == "__main__":
    generate_tsp_problems(num_of_city=300,start_seed=1,end_seed=100,type_of_generator="procgen",output_folder="../data/tsp_instances/")
