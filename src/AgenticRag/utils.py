import os

import yaml
from .logger import Logger
from pathlib import Path

def read_yaml(path:str)->dict:
    with open(path,'r') as file:
        config = yaml.safe_load(file)
    Logger.info("Configuration loaded from %s", path)
    return config
    
def create_directories(paths:list[str])->None:
    for path in paths:
        Path(path).mkdir(parents=True,exist_ok=True)
        Logger.info("Directory ready: %s", path)


def get_size(path:str)->float:
    return round(os.path.getsize(path)/(1024,1024),2)