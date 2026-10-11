import hashlib
import textwrap
from pathlib import Path
import time

from src.AgenticRag.components.data_ingestion import DataIngestion
from src.AgenticRag.components.data_transformation import DataTransformation
from src.AgenticRag.pipelines.ingestion_pipeline import ingest_chunks
from src.AgenticRag.utils import read_yaml


def main():
    config_path = "config/config.yaml"
    config = read_yaml(config_path)
    question = "What is RAG?"

    print("\nAGENTIC RAG - DOCUMENT RETRIEVAL")
    print("=" * 70)

    print("\n[1/4] Loading documents...", flush=True)
    raw_dir = Path(config["data"]["raw_dir"])
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")
    has_files = any(
        path.is_file() and path.suffix.lower() in DataIngestion.loaders
        for path in raw_dir.rglob("*")
    )
    documents = DataIngestion(raw_dir).load_documents() if has_files else []
    print(f"Loaded {len(documents)} document(s).")
    for source in sorted({str(doc.metadata.get("source", "Unknown")) for doc in documents}):
        print(f"  Source: {source}")

    print("\n[2/4] Splitting documents...", flush=True)
    chunks = DataTransformation(config_path).transform_documents(documents) if documents else []
    print(f"Created {len(chunks)} chunk(s).")

    print("\n[3/4] Connecting to Pinecone and uploading chunks...", flush=True)
    retriever = ingest_chunks(chunks, config_path)
    print(f"Uploaded {len(chunks)} chunk(s) to '{config['vectorstore']['index_name']}'.")

    print("\n[4/4] Retrieving relevant chunks...", flush=True)
    print(f"Question: {question}")

    # Newly uploaded vectors may take a short time to become searchable.
    results = []
    for attempt in range(6):
        results = retriever.retrieve(question)
        if results or not chunks:
            break
        if attempt < 5:
            print("  Waiting for uploaded vectors to become searchable...", flush=True)
            time.sleep(2)

    print(f"\nRetrieved {len(results)} chunk(s).")
    for number, document in enumerate(results, start=1):
        print("\n" + "-" * 70)
        print(f"RESULT {number}")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print("Text:")
        print(textwrap.fill(document.page_content, width=90))

    if not results:
        print("No matching chunks. If no source files remain, the active corpus is empty.")

    print("\n" + "=" * 70)
    print("Retrieval finished.")


if __name__ == "__main__":
    main()

