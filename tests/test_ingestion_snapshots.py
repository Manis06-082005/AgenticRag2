import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from src.AgenticRag.pipelines import ingestion_pipeline as pipeline


class FakeEncoder:
    def fit(self, texts):
        self.texts = texts
        return self

    def dump(self, path):
        Path(path).write_text(json.dumps(self.texts))

    def encode_documents(self, texts):
        return [{"indices": [1], "values": [1.0]} for _ in texts]


class FakeRetriever:
    fail = False

    def __init__(self, config_path, *, snapshot):
        self.snapshot = snapshot
        self.namespace = snapshot["namespace"]
        self.text_key = "text"
        self.embeddings = SimpleNamespace(embed_documents=lambda texts: [[1.0] for _ in texts])
        self.index = self
        self.records = []

    def upsert(self, *, vectors, namespace):
        if self.fail:
            raise RuntimeError("Upload failed")
        self.records.extend(vectors)

    def describe_index_stats(self):
        return SimpleNamespace(namespaces={self.namespace: SimpleNamespace(vector_count=len(self.records))})


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manifest = root / "active.json"
        self.config = {
            "vectorstore": {"namespace": "docs", "index_name": "test"},
            "embedding": {"model_name": "test-model"},
            "retriever": {"bm25_path": str(root / "bm25.json"), "manifest_path": str(self.manifest)},
        }
        self.old = {"namespace": "previous"}
        self.manifest.write_text(json.dumps(self.old))
        FakeRetriever.fail = False
        self.patches = [patch.object(pipeline, "read_yaml", return_value=self.config),
                        patch.object(pipeline, "BM25Encoder", FakeEncoder),
                        patch.object(pipeline, "Retriever", FakeRetriever)]
        for mocked in self.patches:
            mocked.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)

    def test_rebuild_uses_new_namespace_and_only_current_chunks(self):
        first = pipeline.ingest_chunks([Document(page_content="old")])
        second = pipeline.ingest_chunks([Document(page_content="new")])
        self.assertNotEqual(first.namespace, second.namespace)
        active = json.loads(self.manifest.read_text())
        self.assertEqual(active["namespace"], second.namespace)
        self.assertEqual(json.loads(Path(active["bm25_path"]).read_text()), ["new"])
        self.assertEqual([v["metadata"]["text"] for v in second.records], ["new"])

    def test_failed_upload_preserves_previous_manifest(self):
        FakeRetriever.fail = True
        with self.assertRaises(RuntimeError):
            pipeline.ingest_chunks([Document(page_content="new")])
        self.assertEqual(json.loads(self.manifest.read_text()), self.old)

    def test_empty_corpus_publishes_empty_snapshot(self):
        pipeline.ingest_chunks([])
        self.assertEqual(json.loads(self.manifest.read_text())["chunk_count"], 0)

    def test_incomplete_upload_never_publishes(self):
        with patch.object(FakeRetriever, "describe_index_stats", return_value=SimpleNamespace(namespaces={})), patch.object(pipeline.time, "monotonic", side_effect=[0, 121]):
            with self.assertRaises(TimeoutError):
                pipeline.ingest_chunks([Document(page_content="new")])
        self.assertEqual(json.loads(self.manifest.read_text()), self.old)


if __name__ == "__main__":
    unittest.main()
