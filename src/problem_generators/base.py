import os
from .tsp_generator import generate_tsp_problems
def ensure_tsp_folder(data_folder:str,instance_size:int):
    temp=os.path.split(os.path.abspath(data_folder))
    dir_name=temp[-1]
    dir_path=os.path.abspath(data_folder)
    parent_dir_path=os.path.dirname(os.path.abspath(data_folder))
    if str(instance_size) in os.listdir(dir_path):
        dir_list=os.listdir(os.path.join(dir_path,str(instance_size)))
        if "train" in dir_list and "eval" in dir_list:
            if os.listdir(os.path.join(dir_path,str(instance_size),"train")) and os.listdir(os.path.join(dir_path,str(instance_size),"eval")):
                return True
    elif dir_name==instance_size:
        generate_tsp_problems(instance_size,output_folder=parent_dir_path)
        return False
    generate_tsp_problems(instance_size,output_folder=dir_path)
    return False

