import json
from glob import glob

if __name__ == "__main__":
    with open("../data/tsp_instances/200/hard_instances.json", "r") as file:
        data = json.load(file)

    instances = []
    for path in data:
        instances.append(path.split("/")[-1])

    train_paths = glob("../data/tsp_instances/200/train/*.tsp")
    train_instances = []
    for path in train_paths:
        train_instances.append(path.split("/")[-1])

    for ins in train_instances:
        if ins not in instances:
            print(ins)

