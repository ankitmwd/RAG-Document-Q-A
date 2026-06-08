import time

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from app.config import get_settings


def get_value(source, key: str):
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )


def ensure_index() -> None:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = [get_value(index, "name") for index in pc.list_indexes()]

    if settings.pinecone_index_name not in existing_indexes:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    for _ in range(30):
        description = pc.describe_index(settings.pinecone_index_name)
        status = get_value(description, "status") or {}
        if get_value(status, "ready"):
            break
        time.sleep(1)

    description = pc.describe_index(settings.pinecone_index_name)
    actual_dimension = get_value(description, "dimension")
    if actual_dimension != settings.embedding_dimension:
        raise ValueError(
            f"Pinecone index '{settings.pinecone_index_name}' has dimension "
            f"{actual_dimension}, but '{settings.embedding_model}' returns "
            f"{settings.embedding_dimension}-dimensional embeddings. Use a new "
            "PINECONE_INDEX_NAME or recreate the index with the correct dimension."
        )


def get_pinecone_index():
    settings = get_settings()
    ensure_index()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


def index_documents(chunks: list[Document]) -> int:
    index = get_pinecone_index()
    embeddings = get_embeddings()
    vectors = []

    chunk_embeddings = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    for chunk, values in zip(chunks, chunk_embeddings, strict=True):
        metadata = {
            **chunk.metadata,
            "text": chunk.page_content,
        }
        vectors.append(
            {
                "id": chunk.metadata["chunk_id"],
                "values": values,
                "metadata": metadata,
            }
        )

    index.upsert(vectors=vectors)
    return len(chunks)


def similarity_search(question: str, document_id: str | None = None) -> list[Document]:
    settings = get_settings()
    index = get_pinecone_index()
    embeddings = get_embeddings()
    filter_query = {"document_id": document_id} if document_id else None

    result = index.query(
        vector=embeddings.embed_query(question),
        top_k=settings.top_k,
        include_metadata=True,
        filter=filter_query,
    )

    documents: list[Document] = []
    for match in result.get("matches", []):
        metadata = dict(match.get("metadata", {}))
        text = metadata.pop("text", "")
        documents.append(Document(page_content=text, metadata=metadata))

    return documents
