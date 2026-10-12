"""Rank retrieved chunks by how well they answer a question."""

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from ..utils import read_yaml
from ..logger import Logger


class Reranker:
    def __init__(self, config_path: str | Path = "config/config.yaml"):
        settings = read_yaml(config_path)["reranker"]
        self.enabled = settings["enabled"]
        self.top_k = settings["top_k"]
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("reranker.top_k must be a positive integer.")

        self.model = None
        if self.enabled:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(settings["model_name"])
            Logger.info("Reranker model loaded: %s", settings["model_name"])

    def rerank(self, question: str, documents: Sequence[Document]) -> list[Document]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if not documents:
            Logger.info("Reranking skipped: no documents")
            return []
        if not self.enabled:
            Logger.info("Reranking disabled; passing through %d chunks", len(documents))
            return list(documents)

        pairs = [(question, document.page_content) for document in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)

        # Copy each document so the original retrieval results remain available.
        results = [
            document.model_copy(
                update={"metadata": {**document.metadata, "rerank_score": float(score)}}
            )
            for document, score in ranked[:self.top_k]
        ]
        Logger.info("Reranked %d chunks; retained %d", len(documents), len(results))
        return results
