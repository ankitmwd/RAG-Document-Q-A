# RAG-Powered Document Q&A Assistant

A FastAPI application that lets users upload PDFs, indexes their text in Pinecone with Gemini embeddings, and answers questions using retrieved context with source citations.

## Tech Stack

- Python
- FastAPI
- LangChain
- Gemini API
- Pinecone
- pypdf

## Project Structure

```text
.
├── app/
│   ├── config.py
│   ├── main.py
│   └── schemas.py
├── services/
│   ├── pdf_loader.py
│   ├── qa.py
│   └── vector_store.py
├── static/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── data/uploads/
├── requirements.txt
└── .env.example
```

## Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your environment file:



Fill in:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-document-qa-gemini-3072
```

Run the app:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## API Endpoints

### Health Check

```http
GET /health
```

### Upload PDF

```http
POST /upload
Content-Type: multipart/form-data
```

Form field:

```text
file: PDF
```

### Ask Question

```http
POST /ask
Content-Type: application/json
```

```json
{
  "question": "What is this document about?",
  "document_id": "optional-document-id"
}
```

## How It Works

1. A PDF is uploaded through the FastAPI backend.
2. Text is extracted page by page using `pypdf`.
3. Text is split into overlapping chunks with LangChain.
4. Chunks are embedded using Gemini embeddings with `models/gemini-embedding-001`.
5. Embeddings and metadata are stored in Pinecone.
6. User questions are embedded and matched against the most relevant chunks.
7. Retrieved chunks are passed to the chat model.
8. The response returns an answer plus source previews.
