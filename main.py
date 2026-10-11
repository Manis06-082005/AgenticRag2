"""Ingest documents, retrieve context, and generate an answer."""

import time

from src.AgenticRag.components.generator import Generator
from src.AgenticRag.components.reranker import Reranker
from src.AgenticRag.pipelines.ingestion_pipeline import run_ingestion


CONFIG_PATH = "config/config.yaml"
QUESTION = "What is RAG?"


def main():
    print(f"\nQuestion: {QUESTION}")

    # Load and split all documents, then upload a new searchable corpus.
    print("\n[1/4] Ingesting documents...", flush=True)
    retriever = run_ingestion(CONFIG_PATH)

    # Allow newly uploaded Pinecone vectors a short time to become searchable.
    print("[2/4] Retrieving relevant chunks...", flush=True)
    documents = []
    for attempt in range(6):
        documents = retriever.retrieve(QUESTION)
        if documents or retriever.empty:
            break
        if attempt < 5:
            time.sleep(2)

    print(f"Found {len(documents)} relevant chunk(s).")
    if not documents:
        print("\nNo relevant information found in your documents.")
        return

    print("[3/4] Reranking retrieved chunks...", flush=True)
    documents = Reranker(CONFIG_PATH).rerank(QUESTION, documents)

    # Give the retrieved context to the LLM to answer the question.
    print("[4/4] Generating an answer...", flush=True)
    generator = Generator(CONFIG_PATH)
    answer = generator.generate(QUESTION, documents)
    print(f"\nAnswer:\n{answer}")

    print("\nSources:")
    for number, document in enumerate(documents, start=1):
        print(f"[{number}] {document.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()

