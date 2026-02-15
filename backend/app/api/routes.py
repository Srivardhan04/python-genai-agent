"""
FastAPI Routes for the Document Intelligence RAG System.

This module defines the HTTP endpoints for:
1. Document upload and indexing
2. Question submission (async)
3. Job status retrieval

Design Pattern:
- RESTful endpoints
- Async processing via Kafka
- Immediate response with job_id for question processing
"""

import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from app.rag.chunker import TextChunker, extract_text_from_file
from app.rag.vector_store import get_vector_store
from app.kafka.producer import publish_document_indexed, publish_question_received
from app.kafka.consumer import get_job_status, create_job, JobStatus
from app.utils.logger import get_api_logger, log_event

logger = get_api_logger()
router = APIRouter()


# --- Request/Response Models ---

class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str


class QuestionResponse(BaseModel):
    """Response model for question submission."""
    job_id: str
    message: str
    status: str


class JobStatusResponse(BaseModel):
    """Response model for job status check."""
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    documents_indexed: int
    chunks_indexed: int


# --- Endpoints ---

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and basic statistics about indexed documents.
    """
    vector_store = get_vector_store()
    
    return HealthResponse(
        status="healthy",
        documents_indexed=vector_store.get_document_count(),
        chunks_indexed=vector_store.get_chunk_count()
    )


@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
    """
    Upload and index a document.
    
    This endpoint:
    1. Accepts a text or PDF file
    2. Extracts text content
    3. Chunks the document
    4. Generates embeddings
    5. Stores chunks in the vector database
    6. Publishes a DOCUMENT_INDEXED event to Kafka
    
    Args:
        file: The uploaded file (text or PDF)
        document_id: Optional custom document ID
    
    Returns:
        DocumentUploadResponse with indexing details
    """
    log_event(logger, "DOCUMENT_UPLOAD_STARTED", {
        "filename": file.filename,
        "content_type": file.content_type
    })
    
    # Generate document ID if not provided
    if not document_id:
        document_id = str(uuid.uuid4())[:8]
    
    try:
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Extract text from file
        text = extract_text_from_file(content, file.filename)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from file"
            )
        
        log_event(logger, "TEXT_EXTRACTED", {
            "document_id": document_id,
            "text_length": len(text)
        })
        
        # Chunk the document
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(
            document_id=document_id,
            text=text,
            metadata={"filename": file.filename}
        )
        
        # Store chunks in vector database
        vector_store = get_vector_store()
        chunks_added = vector_store.add_chunks(chunks)
        
        # Publish Kafka event
        publish_document_indexed(document_id, chunks_added)
        
        log_event(logger, "DOCUMENT_INDEXED", {
            "document_id": document_id,
            "chunks": chunks_added
        })
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_created=chunks_added,
            message=f"Document indexed successfully with {chunks_added} chunks"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions so FastAPI handles them properly
        raise
        
    except ValueError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error uploading document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


@router.post("/ask-question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Submit a question for asynchronous processing.
    
    This endpoint:
    1. Validates the question
    2. Creates a job entry
    3. Publishes a QUESTION_RECEIVED event to Kafka
    4. Returns immediately with a job_id
    
    The actual question answering happens asynchronously via the Kafka consumer.
    Use GET /job-status/{job_id} to check for the answer.
    
    Args:
        request: QuestionRequest containing the question
    
    Returns:
        QuestionResponse with job_id for tracking
    """
    question = request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Maximum 1000 characters."
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())[:8]
    
    log_event(logger, "QUESTION_RECEIVED", {
        "job_id": job_id,
        "question_length": len(question)
    })
    
    # Create job entry
    create_job(job_id, question)
    
    # Publish to Kafka for async processing
    publish_question_received(job_id, question)
    
    return QuestionResponse(
        job_id=job_id,
        message="Question submitted for processing",
        status=JobStatus.PENDING
    )


@router.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status_endpoint(job_id: str):
    """
    Check the status of a question processing job.
    
    Args:
        job_id: The job ID returned from /ask-question
    
    Returns:
        JobStatusResponse with current status and result if completed
    """
    job = get_job_status(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    log_event(logger, "JOB_STATUS_CHECKED", {
        "job_id": job_id,
        "status": job.get("status")
    })
    
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", JobStatus.PENDING),
        result=job.get("result"),
        error=job.get("error")
    )


@router.get("/stats")
async def get_stats():
    """
    Get system statistics.
    
    Returns information about indexed documents and system state.
    """
    vector_store = get_vector_store()
    
    return {
        "total_documents": vector_store.get_document_count(),
        "total_chunks": vector_store.get_chunk_count(),
        "embedding_dimension": vector_store.embedding_dim
    }
