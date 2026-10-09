from AgenticRag.Exception import CustomException
from AgenticRag.logger import Logger
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

class DataInjestion:
    def __init__(self):
        