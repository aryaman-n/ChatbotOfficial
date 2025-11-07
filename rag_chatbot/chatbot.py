"""Conversational Retrieval-Augmented Generation chatbot."""

from __future__ import annotations

from typing import List

from openai import OpenAI
from pinecone import Pinecone

from .config import Settings


class RAGChatbot:
    """A Pinecone + OpenAI powered chatbot."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_host:
            self.index = self.pc.Index(host=settings.pinecone_host)
        else:
            self.index = self.pc.Index(settings.pinecone_index_name)

    def _retrieve(self, query: str) -> List[str]:
        embedding = (
            self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=[query],
            )
            .data[0]
            .embedding
        )

        result = self.index.query(
            namespace=self.settings.namespace,
            vector=embedding,
            top_k=self.settings.top_k,
            include_metadata=True,
        )
        contexts: List[str] = []
        for match in result.get("matches", []):
            metadata = match.get("metadata", {})
            chunk = metadata.get("chunk")
            if chunk:
                contexts.append(chunk)
        return contexts

    def _build_prompt(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return query

        joined_context = "\n\n".join(contexts)
        prompt = (
            "Use the following context to answer the question.\n"
            "If the answer is not contained within the context, say you do not know.\n"
            "Context:\n"
            f"{joined_context}\n\n"
            f"Question: {query}"
        )
        return prompt

    def chat(self, query: str) -> str:
        contexts = self._retrieve(query)
        prompt = self._build_prompt(query, contexts)
        response = self.client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers using the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()