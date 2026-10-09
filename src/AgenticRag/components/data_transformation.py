from src.AgenticRag.Exception import CustomException
from src.AgenticRag.logger import Logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from pathlib import Path

class DataTransformation:
    def __init__(self,config_path:str|Path="config/config.yaml"):
        with Path(config_path).open("r",encoding="utf-8") as file:
            config=yaml.safe_load(file)
        settings=config['chunking']
        chunk_size=config['chunk_size']
        chunk_overlap=config['chunk_overlap']
        if not isinstance(chunk_size,int) or chunk_size<=0:
            raise ValueError(" chunk_size must be a positive  integer.")
        if (
            not isinstance(chunk_overlap,int) or not 0<=chunk_overlap < chunk_size):
            
            raise ValueError(
            "chunk_overlap should be an integer between 0 and chunk_size"
        )
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def transform_documents(self,documents):
        try:
            if not documents:
                raise ValueError(
                    "NO documents are provided for transformation"
                )
            return self.splitter.split_documents(documents)
        except Exception as e:
            raise CustomException(e)