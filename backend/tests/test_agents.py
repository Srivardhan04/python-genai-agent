"""
Unit tests for Agent components.

Tests:
- RetrievalAgent functionality
- ReasoningAgent functionality
- Agent workflow integration
"""

import pytest

from app.agents.retrieval_agent import RetrievalAgent, RetrievedContext
from app.agents.reasoning_agent import ReasoningAgent
from app.rag.vector_store import VectorStore, get_vector_store
from app.rag.chunker import DocumentChunk


class TestRetrievalAgent:
    """Tests for the RetrievalAgent class."""
    
    @pytest.fixture(autouse=True)
    def setup_vector_store(self):
        """Set up vector store with sample data for each test."""
        # Get the global vector store and clear it
        vector_store = get_vector_store()
        vector_store.clear()
        
        # Add sample documents
        chunks = [
            DocumentChunk(
                chunk_id="doc1_0",
                document_id="doc1",
                content="The quarterly financial report indicates strong performance in the technology sector.",
                metadata={"chunk_index": 0}
            ),
            DocumentChunk(
                chunk_id="doc1_1",
                document_id="doc1",
                content="Revenue grew by 20% year-over-year, driven by cloud services.",
                metadata={"chunk_index": 1}
            ),
            DocumentChunk(
                chunk_id="doc2_0",
                document_id="doc2",
                content="Risk management policies have been updated to address market volatility.",
                metadata={"chunk_index": 0}
            )
        ]
        vector_store.add_chunks(chunks)
        
        yield
        
        # Cleanup
        vector_store.clear()
    
    def test_retrieve_returns_contexts(self):
        """Agent should return RetrievedContext objects."""
        agent = RetrievalAgent(top_k=2)
        
        contexts = agent.retrieve("What is the financial performance?")
        
        assert len(contexts) > 0
        assert all(isinstance(c, RetrievedContext) for c in contexts)
    
    def test_retrieve_respects_top_k(self):
        """Agent should return at most top_k results."""
        agent = RetrievalAgent(top_k=1)
        
        contexts = agent.retrieve("company performance")
        
        assert len(contexts) <= 1
    
    def test_format_context_for_prompt(self):
        """Agent should format contexts into a string."""
        agent = RetrievalAgent()
        contexts = [
            RetrievedContext(
                content="Sample content 1",
                source_chunk_id="chunk1",
                relevance_score=0.5,
                metadata={}
            ),
            RetrievedContext(
                content="Sample content 2",
                source_chunk_id="chunk2",
                relevance_score=0.6,
                metadata={}
            )
        ]
        
        formatted = agent.format_context_for_prompt(contexts)
        
        assert "Sample content 1" in formatted
        assert "Sample content 2" in formatted
        assert "[Context 1]" in formatted
    
    def test_format_empty_context(self):
        """Formatting empty context should return appropriate message."""
        agent = RetrievalAgent()
        
        formatted = agent.format_context_for_prompt([])
        
        assert "No relevant context" in formatted
    
    def test_get_stats(self):
        """Agent should return correct statistics."""
        agent = RetrievalAgent(top_k=5)
        
        stats = agent.get_stats()
        
        assert stats["top_k_setting"] == 5
        assert "total_documents" in stats
        assert "total_chunks" in stats


class TestReasoningAgent:
    """Tests for the ReasoningAgent class."""
    
    def test_initialization(self):
        """Agent should initialize with default settings."""
        agent = ReasoningAgent()
        
        assert agent.model == "gpt-3.5-turbo"
        assert agent.temperature == 0.0
    
    def test_mock_response_generation(self):
        """Agent should generate mock responses when OpenAI unavailable."""
        agent = ReasoningAgent()
        contexts = [
            RetrievedContext(
                content="The company achieved record sales in 2024.",
                source_chunk_id="chunk1",
                relevance_score=0.5,
                metadata={}
            )
        ]
        
        answer = agent.reason("What were the sales achievements?", contexts)
        
        assert len(answer) > 0
        # Mock response should mention the mock mode or include context
        assert "MOCK" in answer or "record sales" in answer.lower() or "2024" in answer
    
    def test_empty_context_handling(self):
        """Agent should handle empty context gracefully."""
        agent = ReasoningAgent()
        
        answer = agent.reason("Any question?", [])
        
        assert "Information not found" in answer or "MOCK" in answer
    
    def test_validate_grounding_missing_info(self):
        """Grounding validation should pass for 'not found' responses."""
        agent = ReasoningAgent()
        
        is_grounded = agent.validate_grounding(
            "Information not found in documents.",
            []
        )
        
        assert is_grounded is True
    
    def test_validate_grounding_with_context(self):
        """Grounding validation should check word overlap."""
        agent = ReasoningAgent()
        contexts = [
            RetrievedContext(
                content="The revenue increased by 15% in the fourth quarter.",
                source_chunk_id="chunk1",
                relevance_score=0.5,
                metadata={}
            )
        ]
        
        # Answer that uses words from context
        is_grounded = agent.validate_grounding(
            "Revenue increased by 15%",
            contexts
        )
        
        assert is_grounded is True
    
    def test_format_context(self):
        """Agent should format context correctly."""
        agent = ReasoningAgent()
        contexts = [
            RetrievedContext(
                content="First source content.",
                source_chunk_id="chunk1",
                relevance_score=0.5,
                metadata={}
            )
        ]
        
        formatted = agent._format_context(contexts)
        
        assert "[Source 1]" in formatted
        assert "First source content." in formatted


class TestAgentIntegration:
    """Integration tests for the agent workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Set up test data."""
        vector_store = get_vector_store()
        vector_store.clear()
        
        chunks = [
            DocumentChunk(
                chunk_id="financial_0",
                document_id="financial",
                content="The annual report shows net income of $500 million for fiscal year 2024.",
                metadata={"chunk_index": 0, "source": "annual_report.pdf"}
            ),
            DocumentChunk(
                chunk_id="financial_1",
                document_id="financial",
                content="Operating margin improved to 25% from 22% in the previous year.",
                metadata={"chunk_index": 1, "source": "annual_report.pdf"}
            )
        ]
        vector_store.add_chunks(chunks)
        
        yield
        
        vector_store.clear()
    
    def test_full_workflow(self):
        """Test complete retrieval and reasoning workflow."""
        # Step 1: Retrieve context
        retrieval_agent = RetrievalAgent(top_k=3)
        contexts = retrieval_agent.retrieve("What is the net income?")
        
        assert len(contexts) > 0
        
        # Step 2: Generate answer
        reasoning_agent = ReasoningAgent()
        answer = reasoning_agent.reason("What is the net income?", contexts)
        
        assert len(answer) > 0
    
    def test_workflow_with_empty_store(self):
        """Workflow should handle empty vector store gracefully."""
        vector_store = get_vector_store()
        vector_store.clear()
        
        retrieval_agent = RetrievalAgent()
        contexts = retrieval_agent.retrieve("Any question?")
        
        assert contexts == []
        
        reasoning_agent = ReasoningAgent()
        answer = reasoning_agent.reason("Any question?", contexts)
        
        # Should indicate no information found
        assert "Information not found" in answer or "MOCK" in answer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
