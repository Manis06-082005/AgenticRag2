import os 
from pathlib import Path
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_groq import ChatGroq
from ..Exception import CustomException
from ..logger import Logger
import os

class Generator:
    def __init__(self,config_path:str|Path="config/config.yaml"):
        load_dotenv()
        if not os.getenv('GROQ_API_KEY'):
            raise ValueError(
                "GROQ_API_KEY is missing from your environment"
            )
        with Path(config_path).open("r",encoding="utf-8") as file:
            config=yaml.safe_load(file)
        settings=config['generator']
        self.llm=ChatGroq(
            model=config['model'],
            temperature=config['temprature'],
            max_tokens=config['max_tokens'],
        )
    # def generate(self,question:str,documents)->str:
        
        