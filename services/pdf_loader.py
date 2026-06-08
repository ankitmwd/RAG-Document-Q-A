from pathlib import Path
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import get_settings


def save_upload(file_bytes: bytes, filename: str) -> tuple[str, Path]:
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    document_id = uuid4().hex
    safe_name = Path(filename).name
    file_path = upload_dir / f"{document_id}-{safe_name}"
    file_path.write_bytes(file_bytes)
    return document_id, file_path


def load_pdf_pages(file_path: Path, document_id: str, filename: str) -> list[Document]:
    reader = PdfReader(str(file_path))
    pages: list[Document] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "page": index,
                },
            )
        )

    return pages


def split_documents(documents: list[Document]) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['document_id']}-{index}"

    return chunks
