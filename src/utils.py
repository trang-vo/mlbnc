import re
from lescode.namespace import asdict
from lescode.config import Config

def nodes2edge(u, v):
    return (min(u, v), max(u, v))

#Methods to modify config
def config_single_instance(ENV_CONFIG,instance_name):
    temp=asdict(ENV_CONFIG)
    cities_num=int(max(re.findall(r"\d*", instance_name), key=len))
    temp["instance_size"]=cities_num
    temp["ori_nEdges"]=cities_num*int(temp["k"])*2
    temp["sup_nNodes"]=cities_num
    temp["sup_nEdges"]=cities_num*3
    temp["data_folder"]=instance_name
    ENV_CONFIG=Config()
    ENV_CONFIG=temp
    return ENV_CONFIG
