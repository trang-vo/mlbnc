from genericpath import isdir
import os
from .tsp_generator import generate_tsp_problems
def ensure_tsp_folder(data_folder:str,instance_size:int):
    temp=os.path.split(os.path.abspath(data_folder))
    dir_name=temp[-1]
    dir_path=os.path.abspath(data_folder)
    parent_dir_path=os.path.dirname(os.path.abspath(data_folder))

    if dir_name!=str(instance_size):
        if str(instance_size) in os.listdir(dir_path):
            if "train" in os.listdir(os.path.join(dir_path,str(instance_size))) and "eval" in os.listdir(os.path.join(dir_path,str(instance_size))):
                if os.listdir(os.path.join(dir_path,str(instance_size),"train")) and os.listdir(os.path.join(dir_path,str(instance_size),"eval")):
                    return True
        generate_tsp_problems(instance_size,output_folder=dir_path)
    else:
        if os.path.isdir(dir_path) and "train" in os.listdir(dir_path) and "eval" in os.listdir(dir_path):
            if os.listdir(os.path.join(dir_path,"train")) and os.listdir(os.path.join(dir_path,"eval")):
                return True
        generate_tsp_problems(instance_size,output_folder=parent_dir_path)

