from src.AgenticRag.Exception import CustomException
from src.AgenticRag.logger import Logger
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader
)
from pathlib import Path
import sys

class DataIngestion:
    loaders={
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".csv": CSVLoader,
        ".docx": Docx2txtLoader,  
    }
    def __init__(self,source_path:str|Path):
        
        self.source_path=Path(source_path)
    def load_documents(self):
        try:
            if not  self.source_path.exists():
                raise FileNotFoundError(
                    f"source path does not exist:{self.source_path}"
                )
            if self.source_path.is_file():
                files=[self.source_path]
                if self.source_path.suffix.lower() not in self.loaders:
                    raise ValueError(
                        f"unsupported file type:{self.source_path.suffix}"
                    )
            else:
                files=sorted(
                    path 
                    for path in self.source_path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in self.loaders
                )
            
            if not files:
                raise ValueError(
                    "no supported files found."
                )
            documents=[]
            for file_path in files:
                suffix=file_path.suffix.lower()
                loader_class=self.loaders[suffix]
                if suffix in {".txt",".md",".csv"}:
                    loader=loader_class(
                        str(file_path),encoding="utf-8"
                    )
                else:
                    loader=loader_class(str(file_path))
                documents.extend(loader.load())
            return documents
        except Exception as e:
            raise CustomException(e)
            