"""
Vector store module for document embeddings.

This module handles embedding generation and storage using FAISS (Facebook AI Similarity Search).
FAISS provides efficient similarity search for dense vectors, making it suitable for RAG retrieval.

Design Decisions:
- Uses OpenAI embeddings by default, with fallback to mock embeddings for testing
- FAISS IndexFlatL2 provides exact L2 distance search (brute force but accurate)
- In-memory storage is used for simplicity; production would use persistent storage
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from app.rag.chunker import DocumentChunk
from app.utils.logger import get_rag_logger, log_event

logger = get_rag_logger()


@dataclass
class RetrievalResult:
    """Represents a single retrieval result from the vector store."""
    chunk_id: str
    content: str
    score: float  # Lower is better for L2 distance
    metadata: dict


class EmbeddingGenerator:
    """
    Generates embeddings for text using OpenAI's embedding model.
    Falls back to mock embeddings if API key is not available.
    """
    
    def __init__(self):
        """Initialize the embedding generator."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_mock = not self.api_key
        
        if self.use_mock:
            logger.warning("OpenAI API key not found. Using mock embeddings for testing.")
            self.embedding_dim = 384  # Mock embedding dimension
        else:
            self.embedding_dim = 1536  # OpenAI text-embedding-ada-002 dimension
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI embedding client initialized successfully")
            except ImportError:
                logger.warning("OpenAI package not installed. Using mock embeddings.")
                self.use_mock = True
                self.embedding_dim = 384
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embedding vector
        """
        if self.use_mock:
            return self._mock_embedding(text)
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            log_event(logger, "EMBEDDING_GENERATED", {"text_length": len(text)})
            return embedding
        
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {str(e)}. Falling back to mock.")
            return self._mock_embedding(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if self.use_mock:
            return np.array([self._mock_embedding(t) for t in texts], dtype=np.float32)
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            embeddings = np.array(
                [item.embedding for item in response.data],
                dtype=np.float32
            )
            log_event(logger, "BATCH_EMBEDDINGS_GENERATED", {"count": len(texts)})
            return embeddings
        
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {str(e)}. Using mock.")
            return np.array([self._mock_embedding(t) for t in texts], dtype=np.float32)
    
    def _mock_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic mock embedding for testing.
        Uses hash of text to create reproducible vectors.
        """
        # Create deterministic embedding from text hash
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class VectorStore:
    """
    FAISS-based vector store for document chunks.
    
    Provides:
    - Storage of document chunks with embeddings
    - Similarity search for retrieval
    - Document management (add, clear)
    
    Note: This implementation uses in-memory FAISS. For production,
    consider using FAISS with disk persistence or a managed vector DB.
    """
    
    def __init__(self):
        """Initialize the vector store with FAISS index."""
        self.embedding_generator = EmbeddingGenerator()
        self.embedding_dim = self.embedding_generator.embedding_dim
        
        # Initialize FAISS index
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.faiss_available = True
            logger.info(f"FAISS index initialized with dimension {self.embedding_dim}")
        except ImportError:
            logger.warning("FAISS not available. Using naive numpy-based search.")
            self.faiss_available = False
            self.embeddings_matrix = np.array([]).reshape(0, self.embedding_dim)
        
        # Metadata storage - maps index position to chunk info
        self.chunk_metadata: List[Dict] = []
        self.document_ids: set = set()
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of DocumentChunk objects to add
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        # Generate embeddings for all chunks
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        
        # Add to FAISS index or numpy matrix
        if self.faiss_available:
            self.index.add(embeddings)
        else:
            if len(self.embeddings_matrix) == 0:
                self.embeddings_matrix = embeddings
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
        
        # Store metadata
        for chunk in chunks:
            self.chunk_metadata.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "metadata": chunk.metadata
            })
            self.document_ids.add(chunk.document_id)
        
        log_event(logger, "CHUNKS_INDEXED", {
            "count": len(chunks),
            "total_indexed": len(self.chunk_metadata)
        })
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Query text to search for
            top_k: Number of top results to return
        
        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        if len(self.chunk_metadata) == 0:
            logger.warning("Search attempted on empty vector store")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Adjust top_k if necessary
        actual_k = min(top_k, len(self.chunk_metadata))
        
        # Perform search
        if self.faiss_available:
            distances, indices = self.index.search(query_embedding, actual_k)
            distances = distances[0]
            indices = indices[0]
        else:
            # Naive numpy-based L2 search
            distances = np.linalg.norm(self.embeddings_matrix - query_embedding, axis=1)
            indices = np.argsort(distances)[:actual_k]
            distances = distances[indices]
        
        # Build results
        results = []
        for idx, (dist, i) in enumerate(zip(distances, indices)):
            if i < 0 or i >= len(self.chunk_metadata):
                continue
            
            chunk_info = self.chunk_metadata[i]
            results.append(RetrievalResult(
                chunk_id=chunk_info["chunk_id"],
                content=chunk_info["content"],
                score=float(dist),
                metadata=chunk_info["metadata"]
            ))
        
        log_event(logger, "VECTOR_SEARCH", {
            "query_length": len(query),
            "results_returned": len(results),
            "top_score": results[0].score if results else None
        })
        
        return results
    
    def get_document_count(self) -> int:
        """Return the number of unique documents in the store."""
        return len(self.document_ids)
    
    def get_chunk_count(self) -> int:
        """Return the total number of chunks in the store."""
        return len(self.chunk_metadata)
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        if self.faiss_available:
            import faiss
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.embeddings_matrix = np.array([]).reshape(0, self.embedding_dim)
        
        self.chunk_metadata = []
        self.document_ids = set()
        logger.info("Vector store cleared")


# Global instance - singleton pattern for the vector store
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create the global vector store instance.
    
    Returns:
        VectorStore instance (singleton)
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
