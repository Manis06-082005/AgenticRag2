from src.AgenticRag.components.data_ingestion import DataIngestion
from src.AgenticRag.components.data_transformation import DataTransformation
ingestor = DataIngestion("data/raw")
documents = ingestor.load_documents()
from src.AgenticRag.logger import Logger
print(f"Loaded {len(documents)} documents")
print(documents[0].page_content[:500])
transformer = DataTransformation("config/config.yaml")
chunks = transformer.transform_documents(documents)
print(chunks)