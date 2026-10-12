"""Find relevant chunks using semantic embeddings and BM25 keyword vectors."""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

from ..Exception import CustomException
from ..utils import read_yaml
from ..logger import Logger


class Retriever:
    def __init__(self, config_path="config/config.yaml", *, snapshot=None):
        """Normally read the active corpus; ingestion supplies its pending version."""
        load_dotenv()
        config = read_yaml(config_path)
        embedding_settings = config["embedding"]
        store_settings = config["vectorstore"]
        search_settings = config["retriever"]

        self.alpha = float(search_settings["alpha"])
        self.top_k = search_settings["top_k"]
        if not 0 <= self.alpha <= 1:
            raise ValueError("retriever.alpha must be between 0 and 1.")
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("retriever.top_k must be a positive integer.")
        if store_settings["metric"] != "dotproduct":
            raise ValueError("Hybrid retrieval requires a dotproduct index.")

        # Select a namespace and its matching BM25 file together.
        if snapshot is None:
            snapshot = self.load_active_corpus(search_settings["manifest_path"])
        if (snapshot["index_name"] != store_settings["index_name"]
                or snapshot["model_name"] != embedding_settings["model_name"]):
            raise ValueError("Corpus configuration changed. Run ingestion again.")

        self.namespace = snapshot["namespace"]
        self.empty = snapshot["chunk_count"] == 0
        self.text_key = store_settings.get("text_key", "text")
        if self.empty:
            Logger.info("Retriever initialized with an empty corpus")
            return

        self.sparse_encoder = self.load_bm25(snapshot["bm25_path"])
        self.index = self.connect_index(store_settings, embedding_settings["dimension"])
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_settings["model_name"],
            encode_kwargs={"normalize_embeddings": True},
        )

        Logger.info("Retriever initialized for namespace %s", self.namespace)

    @staticmethod
    def load_active_corpus(manifest_path):
        """The manifest is a small JSON file identifying the active corpus."""
        path = Path(manifest_path)
        if not path.is_file():
            raise FileNotFoundError("No active corpus. Run ingestion first.")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def load_bm25(encoder_path):
        path = Path(encoder_path)
        if not path.is_file():
            raise FileNotFoundError(f"BM25 encoder not found: {path}. Run ingestion again.")
        from pinecone_text.sparse import BM25Encoder
        return BM25Encoder().load(str(path))

    @staticmethod
    def connect_index(settings, dimension):
        """Create the serverless index if needed, then wait for readiness."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing.")

        client = Pinecone(api_key=api_key)
        index_name = settings["index_name"]
        if not client.has_index(index_name):
            client.create_index(
                name=index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=ServerlessSpec(cloud=settings["cloud"], region=settings["region"]),
            )

        deadline = time.monotonic() + 120
        while True:
            description = client.describe_index(index_name)
            if description.dimension != dimension:
                raise ValueError("Index dimension does not match the embedding configuration.")
            if description.metric != "dotproduct":
                raise ValueError("Existing index must use dotproduct for hybrid retrieval.")
            if description.status["ready"]:
                Logger.info("Pinecone index %s is ready", index_name)
                return client.Index(host=description.host)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Index '{index_name}' is not ready after 120 seconds.")
            time.sleep(2)

    def retrieve(self, question: str) -> list[Document]:
        try:
            if not question.strip():
                raise ValueError("Question cannot be empty.")
            if self.empty:
                Logger.info("Retrieval skipped: active corpus is empty")
                return []

            query = self.build_hybrid_query(question)
            response = self.index.query(**query)
            documents = self.convert_matches_to_documents(response.matches)
            Logger.info("Retrieval returned %d chunks", len(documents))
            return documents
        except Exception as exc:
            Logger.exception("Retrieval failed")
            raise CustomException(exc) from exc

    def build_hybrid_query(self, question):
        """Alpha weights meaning; 1 - alpha weights keyword matching."""
        dense_vector = self.embeddings.embed_query(question)
        sparse_vector = self.sparse_encoder.encode_queries(question)

        query = {
            "vector": [self.alpha * value for value in dense_vector],
            "top_k": self.top_k,
            "namespace": self.namespace,
            "include_metadata": True,
        }
        if sparse_vector["indices"] and self.alpha < 1:
            query["sparse_vector"] = {
                "indices": sparse_vector["indices"],
                "values": [(1 - self.alpha) * value for value in sparse_vector["values"]],
            }
        return query

    def convert_matches_to_documents(self, matches):
        """Return the Document objects expected by the reranker and generator."""
        documents = []
        for match in matches:
            metadata = dict(match.metadata or {})
            text = metadata.pop(self.text_key, None)
            if not isinstance(text, str):
                raise ValueError(
                    f"Vector {match.id!r} is missing '{self.text_key}' text metadata. "
                    "Run ingestion again."
                )
            metadata["retrieval_score"] = float(match.score)
            documents.append(Document(page_content=text, metadata=metadata))
        return documents
