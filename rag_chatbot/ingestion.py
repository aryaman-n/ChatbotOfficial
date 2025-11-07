"""Document ingestion pipeline for Pinecone."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from openai import OpenAI
from pinecone import Pinecone

from .config import Settings
from .utils import batched, chunk_text, iter_text_files


def _init_pinecone(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _get_index(pc: Pinecone, settings: Settings):
    if settings.pinecone_host:
        return pc.Index(host=settings.pinecone_host)
    return pc.Index(settings.pinecone_index_name)


def ingest_path(path: str | Path, settings: Settings, batch_size: int = 64) -> None:
    """Ingest documents from ``path`` into Pinecone using OpenAI embeddings."""

    client = OpenAI(api_key=settings.openai_api_key)
    pc = _init_pinecone(settings)
    index = _get_index(pc, settings)

    files = list(iter_text_files(path))
    if not files:
        raise ValueError(
            "No supported documents found. Add .txt or .md files to ingest."
        )

    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        vectors = []
        for chunk in chunks:
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "metadata": {
                        "source": str(file),
                        "chunk": chunk,
                    },
                    "values": None,  # replaced later
                }
            )

        for batch in batched(vectors, batch_size):
            inputs = [item["metadata"]["chunk"] for item in batch]
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=inputs,
            )
            for idx, data in enumerate(response.data):
                batch[idx]["values"] = data.embedding

            index.upsert(
                vectors=[
                    {
                        "id": item["id"],
                        "values": item["values"],
                        "metadata": item["metadata"],
                    }
                    for item in batch
                ],
                namespace=settings.namespace,
            )


def export_ingested_metadata(output_file: str | Path, settings: Settings) -> None:
    """Export stored metadata for debugging or transparency."""

    pc = _init_pinecone(settings)
    index = _get_index(pc, settings)
    stats = index.describe_index_stats(namespace=settings.namespace)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)