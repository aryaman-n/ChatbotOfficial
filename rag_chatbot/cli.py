"""Command line interface for working with the RAG chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .chatbot import RAGChatbot
from .config import get_settings
from .ingestion import export_ingested_metadata, ingest_path

app = typer.Typer(help="Utilities for the Pinecone + OpenAI RAG chatbot.")


@app.command()
def ingest(
    path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to a directory or file containing documents to ingest.",
    ),
    batch_size: int = typer.Option(64, help="Number of chunks per embedding batch."),
) -> None:
    """Ingest markdown or text documents into Pinecone."""

    settings = get_settings()
    typer.echo("Starting ingestion...")
    ingest_path(path, settings=settings, batch_size=batch_size)
    typer.echo("Ingestion complete.")


@app.command()
def chat(question: str = typer.Argument(..., help="Question to ask the chatbot.")) -> None:
    """Ask a question using retrieval-augmented generation."""

    settings = get_settings()
    bot = RAGChatbot(settings=settings)
    answer = bot.chat(question)
    typer.echo(answer)


@app.command()
def stats(output: Optional[Path] = typer.Option(None, help="Optional file to write stats.")) -> None:
    """Show Pinecone index statistics for the configured namespace."""

    settings = get_settings()
    temp_file = output or Path("pinecone_stats.json")
    export_ingested_metadata(temp_file, settings=settings)
    typer.echo(f"Stats written to {temp_file}")


def run() -> None:
    try:
        app()
    except EnvironmentError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    run()