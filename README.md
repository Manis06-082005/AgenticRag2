# AgenticRag

AgenticRag answers questions using documents in a local corpus. It chunks and
embeds supported files, indexes them in Pinecone with hybrid semantic and
keyword search, reranks retrieved passages, and generates answers with Groq.
Answers from the command-line workflow include source references.

The LangGraph workflow also supports query retries and a Tavily web-search
fallback when retrieved document context is insufficient. The `main.py`
command-line entry point currently runs the document retrieval and generation
pipeline directly.

## Requirements

- Python 3.13 or newer
- A [Pinecone](https://www.pinecone.io/) API key
- A [Groq](https://console.groq.com/) API key
- A Tavily API key to use the LangGraph workflow's web-search fallback

## Setup

From the repository root, create and activate a virtual environment, then
install the dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root and add your credentials:

```dotenv
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
```

`TAVILY_API_KEY` is only required when using web search. Keep `.env` private;
it is excluded from version control.

## Add documents

Place files in `data/raw/`. The ingestion pipeline supports PDF, TXT, Markdown,
CSV, and DOCX files, including files in nested folders. Chunking, embedding,
Pinecone index, retrieval, reranking, and generation settings are configured in
[`config/config.yaml`](config/config.yaml).

## Run

Rebuild the document index explicitly:

```powershell
python main.py --ingest
```

Ask a question:

```powershell
python main.py --question "What is RAG?"
```

If no question is supplied, the application uses its default question. Before
answering, it checks whether the source documents or relevant ingestion
settings have changed and updates the active corpus when needed.

The command-line entry point uses the local document corpus and does not invoke
the LangGraph web-search fallback.

## Tests

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```
