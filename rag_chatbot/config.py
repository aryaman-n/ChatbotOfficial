"""Configuration utilities for the RAG chatbot project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_host: Optional[str] = None
    pinecone_environment: Optional[str] = None
    namespace: str = "default"
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 5
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        """Create a :class:`Settings` instance using environment variables."""

        missing = []
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            missing.append("OPENAI_API_KEY")

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            missing.append("PINECONE_API_KEY")

        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        if not pinecone_index_name:
            missing.append("PINECONE_INDEX_NAME")

        if missing:
            missing_vars = ", ".join(missing)
            raise EnvironmentError(
                f"Missing required environment variables: {missing_vars}"
            )

        return cls(
            openai_api_key=openai_api_key,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_name=pinecone_index_name,
            pinecone_host=os.getenv("PINECONE_HOST"),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
            namespace=os.getenv("PINECONE_NAMESPACE", "default"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "5")),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        )


def get_settings() -> Settings:
    """Return application settings from the environment."""

    return Settings.from_env()