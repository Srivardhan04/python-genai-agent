"""
Integration tests for FastAPI endpoints.

Tests:
- Document upload endpoint
- Question submission endpoint
- Job status endpoint
- Health check endpoint
"""

import pytest
from fastapi.testclient import TestClient
import time

from app.main import app
from app.rag.vector_store import get_vector_store
from app.kafka.consumer import job_store


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up before and after each test."""
    # Clear vector store
    vector_store = get_vector_store()
    vector_store.clear()
    
    # Clear job store
    job_store.clear()
    
    yield
    
    # Cleanup after test
    vector_store.clear()
    job_store.clear()


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "documents_indexed" in data
        assert "chunks_indexed" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestDocumentUpload:
    """Tests for the document upload endpoint."""
    
    def test_upload_text_file(self, client):
        """Should successfully upload a text file."""
        content = b"This is a test document about financial reporting and quarterly earnings."
        
        response = client.post(
            "/api/v1/upload-document",
            files={"file": ("test.txt", content, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        assert data["chunks_created"] >= 1
    
    def test_upload_with_custom_id(self, client):
        """Should accept custom document ID."""
        content = b"Document content for testing."
        
        response = client.post(
            "/api/v1/upload-document",
            files={"file": ("doc.txt", content, "text/plain")},
            data={"document_id": "custom123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "custom123"
    
    def test_upload_empty_file(self, client):
        """Should reject empty files."""
        content = b""
        
        response = client.post(
            "/api/v1/upload-document",
            files={"file": ("empty.txt", content, "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_upload_large_document(self, client):
        """Should handle large documents with multiple chunks."""
        # Create a document large enough to require chunking
        content = ("This is a sentence about financial services. " * 100).encode()
        
        response = client.post(
            "/api/v1/upload-document",
            files={"file": ("large.txt", content, "text/plain")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] > 1


class TestAskQuestion:
    """Tests for the question submission endpoint."""
    
    def test_submit_question(self, client):
        """Should submit a question and return job_id."""
        response = client.post(
            "/api/v1/ask-question",
            json={"question": "What is the quarterly revenue?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "PENDING"
    
    def test_empty_question_rejected(self, client):
        """Should reject empty questions."""
        response = client.post(
            "/api/v1/ask-question",
            json={"question": ""}
        )
        
        assert response.status_code == 400
    
    def test_whitespace_question_rejected(self, client):
        """Should reject whitespace-only questions."""
        response = client.post(
            "/api/v1/ask-question",
            json={"question": "   "}
        )
        
        assert response.status_code == 400
    
    def test_long_question_rejected(self, client):
        """Should reject questions exceeding length limit."""
        long_question = "What is " + "a" * 1000
        
        response = client.post(
            "/api/v1/ask-question",
            json={"question": long_question}
        )
        
        assert response.status_code == 400


class TestJobStatus:
    """Tests for the job status endpoint."""
    
    def test_get_pending_job_status(self, client):
        """Should return status for a submitted job."""
        # First, submit a question
        submit_response = client.post(
            "/api/v1/ask-question",
            json={"question": "Test question?"}
        )
        job_id = submit_response.json()["job_id"]
        
        # Check status
        response = client.get(f"/api/v1/job-status/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    
    def test_nonexistent_job(self, client):
        """Should return 404 for non-existent job."""
        response = client.get("/api/v1/job-status/nonexistent123")
        
        assert response.status_code == 404


class TestStatsEndpoint:
    """Tests for the stats endpoint."""
    
    def test_get_stats(self, client):
        """Should return system statistics."""
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "total_chunks" in data
        assert "embedding_dimension" in data


class TestEndToEndFlow:
    """End-to-end integration tests."""
    
    def test_upload_then_query(self, client):
        """Should be able to upload a document and submit a question."""
        # Upload document
        content = b"The company reported revenue of $10 billion in Q4 2024."
        upload_response = client.post(
            "/api/v1/upload-document",
            files={"file": ("report.txt", content, "text/plain")}
        )
        assert upload_response.status_code == 200
        
        # Submit question
        question_response = client.post(
            "/api/v1/ask-question",
            json={"question": "What was the Q4 2024 revenue?"}
        )
        assert question_response.status_code == 200
        job_id = question_response.json()["job_id"]
        
        # Check status is valid
        status_response = client.get(f"/api/v1/job-status/{job_id}")
        assert status_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
