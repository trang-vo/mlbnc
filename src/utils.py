from lescode.config import Config
from lescode.namespace import asdict
def nodes2edge(u, v):
    return (min(u, v), max(u, v))

def config_instance_size(ENV_CONFIG,cities_num):
    temp=asdict(ENV_CONFIG)
    temp["instance_size"]=cities_num
    temp["ori_nEdges"]=cities_num*int(temp["k"])*2
    temp["sup_nNodes"]=cities_num
    temp["sup_nEdges"]=cities_num*3
    temp["data_folder"]="../data/tsp_instances/"+str(cities_num)
    ENV_CONFIG=Config()
    ENV_CONFIG=temp
    print(ENV_CONFIG)
    return ENV_CONFIG