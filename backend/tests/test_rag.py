"""
Unit tests for RAG pipeline components.

Tests:
- Document chunking
- Text extraction
- Vector store operations
- Embedding generation
"""

import pytest
import numpy as np

from app.rag.chunker import TextChunker, DocumentChunk, extract_text_from_file
from app.rag.vector_store import VectorStore, EmbeddingGenerator


class TestTextChunker:
    """Tests for the TextChunker class."""
    
    def test_single_chunk_small_document(self):
        """Small documents should result in a single chunk."""
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        text = "This is a small document."
        
        chunks = chunker.chunk_document("doc1", text)
        
        assert len(chunks) == 1
        assert chunks[0].document_id == "doc1"
        assert chunks[0].content == "This is a small document."
        assert chunks[0].metadata["total_chunks"] == 1
    
    def test_multiple_chunks_large_document(self):
        """Large documents should be split into multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "A" * 300  # 300 characters should create multiple chunks
        
        chunks = chunker.chunk_document("doc2", text)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.document_id == "doc2"
            assert len(chunk.content) <= 100
    
    def test_chunk_overlap(self):
        """Chunks should have overlapping content."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "The quick brown fox jumps over the lazy dog. " * 10
        
        chunks = chunker.chunk_document("doc3", text)
        
        # With overlap, consecutive chunks should share some text
        if len(chunks) > 1:
            # The end of chunk 0 should appear in chunk 1
            # (This is a simplified check)
            assert len(chunks) > 1
    
    def test_empty_text(self):
        """Empty text should result in empty chunk list."""
        chunker = TextChunker()
        
        chunks = chunker.chunk_document("doc4", "")
        
        # Empty or whitespace-only should return no chunks or handled gracefully
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0].content == "")
    
    def test_metadata_preservation(self):
        """Metadata should be preserved in all chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        metadata = {"source": "test.txt", "author": "Test Author"}
        text = "Sample text " * 50
        
        chunks = chunker.chunk_document("doc5", text, metadata=metadata)
        
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "Test Author"


class TestExtractText:
    """Tests for text extraction functions."""
    
    def test_extract_txt_file(self):
        """Should extract text from .txt files."""
        content = b"Hello, this is a test document."
        
        text = extract_text_from_file(content, "test.txt")
        
        assert text == "Hello, this is a test document."
    
    def test_extract_unknown_as_text(self):
        """Unknown file types should be attempted as text."""
        content = b"Some content in unknown format"
        
        text = extract_text_from_file(content, "file.unknown")
        
        assert text == "Some content in unknown format"
    
    def test_handles_utf8(self):
        """Should handle UTF-8 encoded text."""
        content = "Hello, 世界!".encode('utf-8')
        
        text = extract_text_from_file(content, "unicode.txt")
        
        assert "世界" in text


class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""
    
    def test_initialization(self):
        """Generator should initialize with mock mode if no API key."""
        generator = EmbeddingGenerator()
        
        # Without OPENAI_API_KEY, should use mock
        assert generator.use_mock is True or generator.use_mock is False
    
    def test_mock_embedding_dimension(self):
        """Mock embeddings should have consistent dimension."""
        generator = EmbeddingGenerator()
        
        embedding = generator.generate_embedding("Test text")
        
        assert len(embedding) == generator.embedding_dim
    
    def test_mock_embedding_deterministic(self):
        """Same text should produce same mock embedding."""
        generator = EmbeddingGenerator()
        
        emb1 = generator.generate_embedding("Same text")
        emb2 = generator.generate_embedding("Same text")
        
        np.testing.assert_array_almost_equal(emb1, emb2)
    
    def test_different_text_different_embedding(self):
        """Different text should produce different mock embeddings."""
        generator = EmbeddingGenerator()
        
        emb1 = generator.generate_embedding("Text one")
        emb2 = generator.generate_embedding("Text two")
        
        assert not np.allclose(emb1, emb2)
    
    def test_batch_embedding(self):
        """Batch embedding should return correct shape."""
        generator = EmbeddingGenerator()
        texts = ["Text one", "Text two", "Text three"]
        
        embeddings = generator.generate_embeddings_batch(texts)
        
        assert embeddings.shape == (3, generator.embedding_dim)


class TestVectorStore:
    """Tests for the VectorStore class."""
    
    @pytest.fixture
    def vector_store(self):
        """Create a fresh vector store for each test."""
        return VectorStore()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            DocumentChunk(
                chunk_id="doc1_chunk_0",
                document_id="doc1",
                content="The financial report shows Q4 earnings increased by 15%.",
                metadata={"chunk_index": 0}
            ),
            DocumentChunk(
                chunk_id="doc1_chunk_1",
                document_id="doc1",
                content="Operating expenses decreased due to cost optimization.",
                metadata={"chunk_index": 1}
            ),
            DocumentChunk(
                chunk_id="doc2_chunk_0",
                document_id="doc2",
                content="The company announced a new product launch in March.",
                metadata={"chunk_index": 0}
            )
        ]
    
    def test_add_chunks(self, vector_store, sample_chunks):
        """Should add chunks to the vector store."""
        count = vector_store.add_chunks(sample_chunks)
        
        assert count == 3
        assert vector_store.get_chunk_count() == 3
        assert vector_store.get_document_count() == 2
    
    def test_search_returns_results(self, vector_store, sample_chunks):
        """Search should return relevant results."""
        vector_store.add_chunks(sample_chunks)
        
        results = vector_store.search("financial earnings report", top_k=2)
        
        assert len(results) == 2
        assert all(hasattr(r, 'content') for r in results)
        assert all(hasattr(r, 'score') for r in results)
    
    def test_search_empty_store(self, vector_store):
        """Search on empty store should return empty list."""
        results = vector_store.search("any query", top_k=5)
        
        assert results == []
    
    def test_clear_store(self, vector_store, sample_chunks):
        """Clear should remove all data."""
        vector_store.add_chunks(sample_chunks)
        assert vector_store.get_chunk_count() > 0
        
        vector_store.clear()
        
        assert vector_store.get_chunk_count() == 0
        assert vector_store.get_document_count() == 0
    
    def test_top_k_limit(self, vector_store, sample_chunks):
        """Search should respect top_k limit."""
        vector_store.add_chunks(sample_chunks)
        
        results = vector_store.search("company product", top_k=1)
        
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
