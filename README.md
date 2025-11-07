# Company RAG Chatbot

This project provides a Retrieval-Augmented Generation (RAG) chatbot that augments OpenAI's
language models with your proprietary knowledge base stored in Pinecone. The toolkit includes
scripts for chunking source documents, embedding them with OpenAI, and performing similarity
search at query time to ground answers in your company's data.

## Features

- **Chunk-aware ingestion**: Splits Markdown and text files into overlapping chunks for
  high-quality embedding coverage.
- **Pinecone vector store**: Stores embeddings in Pinecone and supports configurable namespaces.
- **OpenAI-powered responses**: Uses OpenAI chat completions to generate grounded answers.
- **CLI utilities**: Provides Typer-based commands for ingesting data, querying the bot, and
  inspecting Pinecone index statistics.

## Project structure

```
.
├── .env.example              # Template for required environment variables
├── README.md                 # Project documentation
├── data/
│   └── source_documents/     # Add your Markdown/Text documents here before ingestion
├── rag_chatbot/
│   ├── __init__.py
│   ├── chatbot.py            # Chatbot runtime with retrieval + generation logic
│   ├── cli.py                # Typer CLI with ingest/chat/stats commands
│   ├── config.py             # Environment-driven configuration helpers
│   ├── ingestion.py          # Embedding + Pinecone upsert pipeline
│   └── utils.py              # File loading and text chunking helpers
└── requirements.txt          # Python dependencies
```

## Prerequisites

1. **Python 3.10+**
2. **Pinecone index** sized for your chosen embedding dimensionality (1536 for
   `text-embedding-3-small`). Create the index ahead of time using the Pinecone console or API.
3. **OpenAI API key** with access to the chat and embedding models you plan to use.

## Configuration

1. Copy the example environment file and fill in your secrets:

   ```bash
   cp .env.example .env
   ```

2. Populate the `.env` file with your OpenAI and Pinecone credentials. If you are using a
   serverless Pinecone deployment you may only need the `PINECONE_INDEX_NAME`. For dedicated
   deployments provide `PINECONE_HOST` or `PINECONE_ENVIRONMENT` as appropriate.

3. (Optional) Adjust chunking or retrieval parameters (`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`)
   to match your document characteristics and latency goals.

## Installation

Install dependencies into a virtual environment of your choice:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use a `.env` file, load it before running the CLI (e.g., via `source .env` or a tool like
`direnv`).

## Preparing documents

1. Place the Markdown (`.md`) or plain-text (`.txt`) files that contain your company knowledge
   inside the `data/source_documents/` directory. You can also point the CLI at another directory
   or a single file when ingesting.
2. Ensure the documents are UTF-8 encoded for consistent ingestion.

## Ingesting data into Pinecone

Run the ingestion command once your environment variables are available:

```bash
python -m rag_chatbot.cli ingest data/source_documents
```

The script will:

1. Load all supported documents from the provided path.
2. Split them into overlapping chunks (default 800 characters with 200 overlap).
3. Generate embeddings with the configured OpenAI embedding model.
4. Upsert vectors and metadata into the configured Pinecone namespace.

Use the `--batch-size` option to control the number of chunks sent per embedding request.

## Chatting with the bot

After ingestion you can issue questions to the chatbot:

```bash
python -m rag_chatbot.cli chat "What services does our consulting team offer?"
```

The CLI retrieves the most similar chunks from Pinecone, builds a context-aware prompt, and asks
OpenAI's chat completion API to produce an answer grounded in your knowledge base.

## Inspecting Pinecone stats

Retrieve a snapshot of index statistics (vector counts, dimensions, namespaces) for debugging:

```bash
python -m rag_chatbot.cli stats --output pinecone_stats.json
```

## Notes

- The code never attempts to create a Pinecone index automatically. Ensure the index exists and
  matches the embedding dimensionality before ingestion.
- If you encounter network or API errors while running commands in restricted environments, verify
  that the required endpoints are whitelisted.
- Extend `rag_chatbot/utils.py` if you need additional document loaders (PDF, HTML, etc.).

## Next steps

- Integrate the chatbot into your preferred UI (web, Slack, etc.) by importing `RAGChatbot` from
  `rag_chatbot.chatbot` and wiring it into your application.
- Add automated evaluation to monitor answer quality as your knowledge base evolves.
