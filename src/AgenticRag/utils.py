import os

import yaml
from pathlib import Path

def read_yaml(path:str)->dict:
    with open(path,'r') as file:
        return yaml.safe_load(file)
    
def create_directories(paths:list[str])->None:
    for path in paths:
        Path(path).mkdir(parents=True,exist_ok=True)


def get_size(path:str)->float:
    return round(os.path.getsize(path)/(1024,1024),2)