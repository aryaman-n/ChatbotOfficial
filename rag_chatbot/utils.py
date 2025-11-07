"""Utility helpers for loading and chunking documents."""

from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Iterable, List


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}


def iter_text_files(path: str | os.PathLike[str]) -> Iterable[Path]:
    """Yield text documents within ``path`` matching supported extensions."""

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Document path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root
        return

    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield file


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split ``text`` into overlapping chunks of roughly ``chunk_size`` characters."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start == end:  # Prevent infinite loop when chunk_overlap == 0
            start += 1
    return chunks


def batched(iterable: Iterable, batch_size: int) -> Iterable[list]:
    """Yield ``iterable`` in lists of size ``batch_size``."""

    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch