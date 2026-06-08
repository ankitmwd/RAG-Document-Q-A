from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int


class QuestionRequest(BaseModel):
    question: str
    document_id: str | None = None


class Source(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    preview: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
