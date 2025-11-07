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



def ingest_path(path: str | Path, settings: Settings, batch_size: int = 32) -> None:
    """Safely ingest documents into Pinecone using small, memory-efficient batches."""

    import time
    from tqdm import tqdm

    client = OpenAI(api_key=settings.openai_api_key)
    pc = _init_pinecone(settings)
    index = _get_index(pc, settings)

    files = list(iter_text_files(path))
    if not files:
        raise ValueError("No supported documents found. Add .txt or .md files to ingest.")

    typer_msg = f"Ingesting {len(files)} file(s) with batch_size={batch_size}..."
    print(typer_msg)

    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️  Skipping {file}: {e}")
            continue

        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        print(f"📄 {file.name}: {len(chunks)} chunks")

        # Process chunks in small batches to prevent OOM
        for batch_idx, batch_chunks in enumerate(batched(chunks, batch_size), start=1):
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model,
                    input=batch_chunks,
                )

                vectors = [
                    {
                        "id": str(uuid.uuid4()),
                        "values": data.embedding,
                        "metadata": {"source": str(file), "chunk": chunk},
                    }
                    for data, chunk in zip(response.data, batch_chunks)
                ]

                index.upsert(vectors=vectors, namespace=settings.namespace)
                print(f"✅  {file.name} batch {batch_idx} ({len(batch_chunks)} chunks) uploaded")

                # brief sleep to avoid API rate limit & memory spikes
                time.sleep(0.3)

            except Exception as e:
                print(f"❌  Failed batch {batch_idx} of {file.name}: {e}")
                time.sleep(1)
                continue



def export_ingested_metadata(output_file: str | Path, settings: Settings) -> None:
    """Export stored metadata for debugging or transparency."""

    pc = _init_pinecone(settings)
    index = _get_index(pc, settings)
    stats = index.describe_index_stats(namespace=settings.namespace)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)