"""Update changed documents automatically, then answer using the active corpus."""

import argparse


CONFIG_PATH = "config/config.yaml"
QUESTION = "What is RAG?"


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documents.")
    parser.add_argument("--ingest", action="store_true", help="Rebuild the corpus from data/raw and exit.")
    parser.add_argument("--question", default=QUESTION, help="Question to answer using the active corpus.")
    args = parser.parse_args()

    if args.ingest:
        from src.AgenticRag.pipelines.ingestion_pipeline import run_ingestion

        print("Ingesting documents...", flush=True)
        run_ingestion(CONFIG_PATH)
        print("Ingestion complete. You can now ask questions without re-uploading documents.")
        return

    if not args.question.strip():
        parser.error("--question cannot be empty.")

    from src.AgenticRag.pipelines.ingestion_pipeline import run_ingestion

    # Hashes are saved in the active manifest; unchanged documents skip uploading.
    print("Checking documents for changes...", flush=True)
    run_ingestion(CONFIG_PATH)

    from src.AgenticRag.components.generator import Generator
    from src.AgenticRag.components.reranker import Reranker
    from src.AgenticRag.components.retriever import Retriever

    print(f"\nQuestion: {args.question}")
    print("\n[1/3] Retrieving relevant chunks...", flush=True)
    retriever = Retriever(CONFIG_PATH)
    documents = retriever.retrieve(args.question)
    print(f"Found {len(documents)} relevant chunk(s).")
    if not documents:
        print("\nNo relevant information found in your documents.")
        return

    print("[2/3] Reranking retrieved chunks...", flush=True)
    documents = Reranker(CONFIG_PATH).rerank(args.question, documents)

    print("[3/3] Generating an answer...", flush=True)
    generator = Generator(CONFIG_PATH)
    print("\nAnswer:", flush=True)
    for text in generator.stream(args.question, documents):
        print(text, end="", flush=True)
    print()

    print("\nSources:")
    for number, document in enumerate(documents, start=1):
        print(f"[{number}] {document.metadata.get('source', 'Unknown')}")


if __name__ == "__main__":
    main()
