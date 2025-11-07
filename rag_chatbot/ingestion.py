# ingestion.py
from __future__ import annotations
import gc, hashlib, json, os, time
from pathlib import Path
from typing import Iterator, List
from openai import OpenAI
from pinecone import Pinecone
from .config import Settings
from .utils import batched, iter_text_files  # NOTE: we won't call chunk_text

try:
    import psutil
except Exception:
    psutil = None

def _init_pinecone(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)

def _get_index(pc: Pinecone, settings: Settings):
    if getattr(settings, "pinecone_host", None):
        return pc.Index(host=settings.pinecone_host)
    return pc.Index(settings.pinecone_index_name)

def _deterministic_id(source: str, chunk_text: str, idx: int) -> str:
    h = hashlib.sha256()
    h.update(f"{source}\x1f{idx}\x1f{len(chunk_text)}".encode("utf-8", "ignore"))
    return h.hexdigest()

def _sleep_backoff(attempt: int, base: float = 0.4, cap: float = 6.0) -> None:
    time.sleep(min(cap, base * (2 ** attempt)))

def _embed_batch(client: OpenAI, model: str, inputs: List[str]) -> List[List[float]]:
    attempts = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            return [d.embedding for d in resp.data]
        except Exception:
            attempts += 1
            if attempts > 5:
                raise
            _sleep_backoff(attempts)

def _upsert_vectors(index, vectors: List[dict], namespace: str) -> None:
    attempts = 0
    while True:
        try:
            index.upsert(vectors=vectors, namespace=namespace)
            return
        except Exception:
            attempts += 1
            if attempts > 5:
                raise
            _sleep_backoff(attempts)

def _gen_chunks(text: str, size: int, overlap: int) -> Iterator[str]:
    # generator version of chunking (no big list)
    if size <= 0:
        size = 800
    if overlap < 0:
        overlap = 0
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        yield text[start : start + size]

def ingest_path(path: str | Path, settings: Settings, batch_size: int = 4) -> None:
    # keep BLAS single-threaded
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    client = OpenAI(api_key=settings.openai_api_key)
    pc = _init_pinecone(settings)
    index = _get_index(pc, settings)

    files = list(iter_text_files(path))
    if not files:
        raise ValueError("No supported documents found. Add .txt or .md files to ingest.")

    print(f"🧠 Starting ingestion of {len(files)} file(s) (batch={batch_size})")

    for file in files:
        p = Path(file)
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"⚠️  Skipping {p}: {e}")
            continue

        print(f"📄 {p.name}: streaming chunks…")
        source = str(p)
        chunk_iter = _gen_chunks(text, settings.chunk_size, settings.chunk_overlap)

        batch_index = 0
        while True:
            # pull a tiny batch from the generator
            batch = []
            for _ in range(batch_size):
                try:
                    batch.append(next(chunk_iter))
                except StopIteration:
                    break
            if not batch:
                break

            batch_index += 1
            try:
                embeddings = _embed_batch(client, settings.embedding_model, batch)
                vectors = [
                    {
                        "id": _deterministic_id(source, chunk_text, i + (batch_index * 10_000)),
                        "values": emb,
                        "metadata": {"source": source, "chunk": chunk_text},
                    }
                    for i, (chunk_text, emb) in enumerate(zip(batch, embeddings))
                ]
                _upsert_vectors(index, vectors, namespace=settings.namespace)

                # hard memory clean
                del embeddings, vectors, batch
                gc.collect()

                if psutil and (batch_index % 5 == 0):
                    print(f"✅ {p.name} batch {batch_index} | Mem: {psutil.virtual_memory().percent:.1f}%")

                time.sleep(0.12)
            except Exception as e:
                print(f"❌ {p.name} batch {batch_index} failed: {e}")
                gc.collect()
                _sleep_backoff(1)
                continue

        print(f"✅ Completed {p.name}")
        # drop big text string ASAP
        del text
        gc.collect()

def export_ingested_metadata(output_file: str | Path, settings: Settings) -> None:
    pc = _init_pinecone(settings)
    index = _get_index(pc, settings)
    stats = index.describe_index_stats(namespace=settings.namespace)
    Path(output_file).write_text(json.dumps(stats, indent=2), encoding="utf-8")
