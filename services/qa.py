from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.schemas import Source


SYSTEM_PROMPT = """You answer questions using only the provided PDF context.
If the answer is not in the context, say you do not know.
Include concise citations in the answer using the page numbers when available."""


def format_context(documents: list[Document]) -> str:
    sections = []
    for doc in documents:
        page = doc.metadata.get("page", "unknown")
        filename = doc.metadata.get("filename", "document")
        sections.append(
            f"Source: {filename}, page {page}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(sections)


def build_sources(documents: list[Document]) -> list[Source]:
    sources: list[Source] = []
    seen = set()

    for doc in documents:
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        preview = " ".join(doc.page_content.split())[:240]
        sources.append(
            Source(
                document_id=doc.metadata.get("document_id", ""),
                filename=doc.metadata.get("filename", ""),
                page=doc.metadata.get("page"),
                chunk_id=chunk_id,
                preview=preview,
            )
        )

    return sources


def answer_question(question: str, documents: list[Document]) -> tuple[str, list[Source]]:
    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=0,
        google_api_key=settings.gemini_api_key,
    )

    if not documents:
        return "I do not know. I could not find relevant context in the indexed PDFs.", []

    context = format_context(documents)
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                f"Question: {question}\n\nPDF context:\n{context}",
            ),
        ]
    )

    return str(response.content), build_sources(documents)
