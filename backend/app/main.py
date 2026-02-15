"""
FastAPI Application Entry Point.

This is the main entry point for the Event-Driven Agentic RAG System.
It initializes the FastAPI application, sets up routes, and manages
the Kafka consumer lifecycle.

To run the application:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.api.routes import router
from app.kafka.consumer import start_consumer, stop_consumer
from app.utils.logger import setup_logger

# Initialize main logger
logger = setup_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize Kafka consumer
    - Shutdown: Gracefully stop Kafka consumer
    """
    # --- Startup ---
    logger.info("Application starting up...")
    
    # Start Kafka consumer in background thread
    start_consumer()
    logger.info("Kafka consumer started")
    
    yield  # Application runs here
    
    # --- Shutdown ---
    logger.info("Application shutting down...")
    
    # Stop Kafka consumer
    stop_consumer()
    logger.info("Kafka consumer stopped")
    
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Kafka Q&A automation",
    description="""
    A Kafka-driven document intelligence system that combines:
    - Retrieval-Augmented Generation (RAG) for accurate question answering
    - Event-driven architecture using Kafka for scalability
    - Agent-based workflow for structured processing
    
    ## Features
    
    - **Document Upload**: Upload text or PDF documents for indexing
    - **Question Answering**: Ask questions and get answers grounded in your documents
    - **Async Processing**: Questions are processed asynchronously via Kafka
    
    ## Architecture
    
    1. Documents are chunked and embedded into a FAISS vector store
    2. Questions trigger Kafka events for async processing
    3. RetrievalAgent fetches relevant context
    4. ReasoningAgent generates grounded answers using LLM
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["RAG API"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Kafka Q&A automation",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True  # Enable auto-reload for development
    )
