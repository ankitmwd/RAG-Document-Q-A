from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.schemas import AnswerResponse, QuestionRequest, UploadResponse
from services.pdf_loader import load_pdf_pages, save_upload, split_documents
from services.qa import answer_question
from services.vector_store import index_documents, similarity_search


app = FastAPI(title="RAG Document Q&A Assistant")

CONFIG_ERROR = (
    "Missing configuration. Add GEMINI_API_KEY and PINECONE_API_KEY "
    "to your local .env file, then restart the server."
)


def provider_error_message(action: str, error: Exception) -> str:
    message = str(error)
    normalized = message.lower()

    if "resourceexhausted" in normalized or "quota" in normalized or "429" in normalized:
        return (
            "Gemini quota or rate limit reached. Wait a minute and try again, "
            "or use a Gemini API key/project with more quota."
        )
    if "not found" in normalized and "model" in normalized:
        return (
            "The configured Gemini model is not available for this API key. "
            "Update EMBEDDING_MODEL or CHAT_MODEL in .env."
        )
    if "dimension" in normalized and "pinecone" in normalized:
        return message

    return f"{action} failed while contacting Gemini or Pinecone. Check provider access and try again."

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    try:
        file_bytes = await file.read()
        document_id, file_path = save_upload(file_bytes, file.filename or "document.pdf")
        pages = load_pdf_pages(file_path, document_id, file.filename or "document.pdf")

        if not pages:
            raise HTTPException(status_code=400, detail="No readable text found in this PDF.")

        chunks = split_documents(pages)
        chunks_indexed = index_documents(chunks)
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=CONFIG_ERROR) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=provider_error_message("Upload", exc),
        ) from exc

    return UploadResponse(
        document_id=document_id,
        filename=file.filename or "document.pdf",
        chunks_indexed=chunks_indexed,
    )


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest) -> AnswerResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not payload.document_id:
        raise HTTPException(status_code=400, detail="Upload and index a PDF before asking a question.")

    try:
        matches = similarity_search(question, payload.document_id)
        answer, sources = answer_question(question, matches)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=CONFIG_ERROR) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=provider_error_message("Question", exc),
        ) from exc

    return AnswerResponse(answer=answer, sources=sources)
