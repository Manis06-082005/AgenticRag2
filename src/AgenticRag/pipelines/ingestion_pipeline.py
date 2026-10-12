"""Rebuild hybrid search from all current documents.

A corpus version pairs one Pinecone namespace with one fitted BM25 file.
The active_corpus.json file tells retrieval which pair to use.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from pinecone_text.sparse import BM25Encoder

from ..components.data_ingestion import DataIngestion
from ..components.data_transformation import DataTransformation
from ..components.retriever import Retriever
from ..utils import read_yaml
from ..logger import Logger

BATCH_SIZE = 64
UPLOAD_WAIT_SECONDS = 120


def ingest_chunks(chunks, config_path="config/config.yaml", *, fingerprint=None):
    """Pass ALL current chunks, including chunks from previously uploaded files."""
    config = read_yaml(config_path)

    # 1. Reserve a new namespace and BM25 filename for this rebuild.
    version = create_corpus_version(config, len(chunks))
    if fingerprint is not None:
        version["fingerprint"] = fingerprint

    # 2. Fit BM25 on the complete collection of chunks.
    if chunks:
        encoder = fit_and_save_bm25(chunks, version["bm25_path"])

    # The supplied version lets ingestion use the new namespace before activation.
    retriever = Retriever(config_path, snapshot=version)

    # 3. Upload both kinds of vectors and wait for the complete upload.
    if chunks:
        upload_chunks(chunks, encoder, retriever)
        wait_for_upload(retriever, expected_count=len(chunks))

    # 4. Only now switch future retrievers to the new corpus.
    # Any error above leaves the previous active corpus unchanged.
    activate_corpus(version, config["retriever"]["manifest_path"])
    return retriever


def create_corpus_version(config, chunk_count):
    """Give every rebuild its own namespace and matching encoder file."""
    version_id = uuid4().hex
    namespace_prefix = config["vectorstore"].get("namespace") or "corpus"
    encoder_base = Path(config["retriever"]["bm25_path"])
    encoder_filename = f"{encoder_base.stem}-{version_id}.json"

    return {
        "namespace": f"{namespace_prefix}-{version_id}",
        "bm25_path": str(encoder_base.with_name(encoder_filename)),
        "chunk_count": chunk_count,
        "index_name": config["vectorstore"]["index_name"],
        "model_name": config["embedding"]["model_name"],
    }


def fit_and_save_bm25(chunks, encoder_path):
    """Learn keyword statistics from all chunks, then save them."""
    texts = [chunk.page_content for chunk in chunks]
    encoder = BM25Encoder().fit(texts)

    encoder_path = Path(encoder_path)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(str(encoder_path))
    Logger.info("BM25 fitted on %d chunks and saved", len(chunks))
    return encoder


def upload_chunks(chunks, encoder, retriever):
    """Upload batches containing semantic vectors, keyword vectors, and text."""
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]

        dense_vectors = retriever.embeddings.embed_documents(texts)
        sparse_vectors = encoder.encode_documents(texts)
        vectors = []

        for offset, chunk in enumerate(batch):
            source = chunk.metadata.get("source", "")
            identity = f"{source}|{start + offset}|{chunk.page_content}"
            vector_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

            vectors.append({
                "id": vector_id,
                "values": dense_vectors[offset],
                "sparse_values": sparse_vectors[offset],
                "metadata": {
                    **chunk.metadata,
                    retriever.text_key: chunk.page_content,
                },
            })

        retriever.index.upsert(vectors=vectors, namespace=retriever.namespace)
        Logger.info("Uploaded batch of %d vectors to %s", len(vectors), retriever.namespace)


def wait_for_upload(retriever, expected_count):
    """Pinecone may take time to report newly uploaded vectors."""
    deadline = time.monotonic() + UPLOAD_WAIT_SECONDS

    while True:
        stats = retriever.index.describe_index_stats()
        namespace_stats = stats.namespaces.get(retriever.namespace)

        if namespace_stats is not None:
            if namespace_stats.vector_count == expected_count:
                Logger.info("Confirmed %d uploaded vectors", expected_count)
                return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Upload is not fully visible; previous corpus remains active."
            )
        time.sleep(2)


def activate_corpus(version, manifest_path):
    """Replace one small file to select the namespace and encoder together."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = f".{manifest_path.name}.{uuid4().hex}.tmp"
    temporary_path = manifest_path.with_name(temporary_name)

    try:
        temporary_path.write_text(json.dumps(version, indent=2), encoding="utf-8")
        os.replace(temporary_path, manifest_path)
        Logger.info("Activated corpus namespace %s", version["namespace"])
    finally:
        temporary_path.unlink(missing_ok=True)


def run_ingestion(config_path="config/config.yaml"):
    """Rebuild only when files or ingestion settings have changed."""
    config = read_yaml(config_path)
    raw_dir = Path(config["data"]["raw_dir"])
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw document directory not found: {raw_dir}")

    files = sorted(
        path for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DataIngestion.loaders
    )
    # Include settings so changing the chunk size or model also triggers ingestion.
    settings = {key: config[key] for key in ("chunking", "embedding", "vectorstore")}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in files:
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(json.dumps([path.relative_to(raw_dir).as_posix(), file_hash]).encode())
    fingerprint = digest.hexdigest()
    Logger.info("Checked fingerprints for %d source files", len(files))

    manifest = Path(config["retriever"]["manifest_path"])
    if manifest.is_file():
        active = json.loads(manifest.read_text(encoding="utf-8"))
        if active.get("fingerprint") == fingerprint:
            Logger.info("Documents unchanged; upload skipped")
            print("Documents unchanged. Skipping upload.", flush=True)
            return

    Logger.info("Corpus changed or missing; rebuilding")
    documents = DataIngestion(raw_dir).load_documents() if files else []
    chunks = DataTransformation(config_path).transform_documents(documents) if documents else []
    ingest_chunks(chunks, config_path, fingerprint=fingerprint)
    print("Active corpus updated.", flush=True)

