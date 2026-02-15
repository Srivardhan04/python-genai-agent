"""
Retrieval Agent for the RAG pipeline.

This agent is responsible for retrieving relevant document chunks from the vector store
based on a user's question. It acts as the first step in the agent workflow, providing
context for the ReasoningAgent to use.

Agent Pattern:
This implements a simple reactive agent that:
1. Receives a question
2. Queries the vector store for similar chunks
3. Returns the top-k most relevant chunks

No complex planning or multi-step reasoning - just focused retrieval.
"""

from typing import List
from dataclasses import dataclass

from app.rag.vector_store import get_vector_store, RetrievalResult
from app.utils.logger import get_agent_logger, log_event

logger = get_agent_logger()


@dataclass
class RetrievedContext:
    """Represents context retrieved for answering a question."""
    content: str
    source_chunk_id: str
    relevance_score: float
    metadata: dict


class RetrievalAgent:
    """
    Agent responsible for retrieving relevant context from the vector store.
    
    This is a simple, focused agent that does one thing well:
    - Takes a question
    - Finds the most relevant document chunks
    - Returns them for use by the ReasoningAgent
    
    Design Decisions:
    - top_k=5 provides good coverage without overwhelming context
    - Score threshold filtering removes low-relevance results
    - Simple interface allows easy testing and modification
    """
    
    def __init__(self, top_k: int = 5, score_threshold: float = None):
        """
        Initialize the retrieval agent.
        
        Args:
            top_k: Number of chunks to retrieve
            score_threshold: Optional maximum distance score (lower is more similar)
        """
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.vector_store = get_vector_store()
        logger.info(f"RetrievalAgent initialized with top_k={top_k}")
    
    def retrieve(self, question: str) -> List[RetrievedContext]:
        """
        Retrieve relevant context for a question.
        
        Args:
            question: The user's question
        
        Returns:
            List of RetrievedContext objects, ordered by relevance
        """
        log_event(logger, "RETRIEVAL_STARTED", {
            "question_length": len(question),
            "top_k": self.top_k
        })
        
        # Check if vector store has documents
        if self.vector_store.get_chunk_count() == 0:
            logger.warning("No documents in vector store. Cannot retrieve context.")
            return []
        
        # Perform vector similarity search
        results: List[RetrievalResult] = self.vector_store.search(
            query=question,
            top_k=self.top_k
        )
        
        # Filter by score threshold if specified
        if self.score_threshold is not None:
            results = [r for r in results if r.score <= self.score_threshold]
        
        # Convert to RetrievedContext objects
        contexts = []
        for result in results:
            context = RetrievedContext(
                content=result.content,
                source_chunk_id=result.chunk_id,
                relevance_score=result.score,
                metadata=result.metadata
            )
            contexts.append(context)
        
        log_event(logger, "RETRIEVAL_COMPLETED", {
            "chunks_retrieved": len(contexts),
            "best_score": contexts[0].relevance_score if contexts else None
        })
        
        return contexts
    
    def format_context_for_prompt(self, contexts: List[RetrievedContext]) -> str:
        """
        Format retrieved contexts into a string for the LLM prompt.
        
        Args:
            contexts: List of RetrievedContext objects
        
        Returns:
            Formatted string containing all context chunks
        """
        if not contexts:
            return "No relevant context found in the documents."
        
        formatted_parts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_parts.append(
                f"[Context {i}]\n{ctx.content}\n"
            )
        
        return "\n".join(formatted_parts)
    
    def get_stats(self) -> dict:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with vector store statistics
        """
        return {
            "total_documents": self.vector_store.get_document_count(),
            "total_chunks": self.vector_store.get_chunk_count(),
            "top_k_setting": self.top_k
        }
