"""
Document chunking module for the RAG pipeline.

This module handles splitting documents into smaller, semantically meaningful chunks
for embedding generation and vector storage. Proper chunking is essential for
effective retrieval - chunks must be small enough to be specific but large enough
to retain context.
"""

from typing import List
from dataclasses import dataclass

from app.utils.logger import get_rag_logger, log_event

logger = get_rag_logger()


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document."""
    chunk_id: str
    document_id: str
    content: str
    metadata: dict


class TextChunker:
    """
    Simple text chunker using fixed-size windows with overlap.
    
    Design Decision:
    - Fixed chunk size of 500 characters provides balance between specificity and context
    - 50 character overlap ensures sentences aren't cut off mid-thought
    - This is a simple approach; production systems may use semantic chunking
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"TextChunker initialized with size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(self, document_id: str, text: str, metadata: dict = None) -> List[DocumentChunk]:
        """
        Split document text into chunks.
        
        Args:
            document_id: Unique identifier for the source document
            text: Full text content to chunk
            metadata: Optional metadata to attach to each chunk
        
        Returns:
            List of DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Clean the text - normalize whitespace
        text = " ".join(text.split())
        
        if len(text) <= self.chunk_size:
            # Document is small enough to be a single chunk
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_0",
                document_id=document_id,
                content=text,
                metadata={**metadata, "chunk_index": 0, "total_chunks": 1}
            )
            log_event(logger, "DOCUMENT_CHUNKED", {
                "document_id": document_id,
                "total_chunks": 1,
                "avg_chunk_size": len(text)
            })
            return [chunk]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                # Look for sentence-ending punctuation within the last 100 chars
                search_region = text[max(start, end - 100):end]
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = search_region.rfind(punct)
                    if last_punct != -1:
                        end = max(start, end - 100) + last_punct + len(punct)
                        break
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=chunk_content,
                    metadata={**metadata, "chunk_index": chunk_index}
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
        
        # Update metadata with total chunk count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total_chunks
        
        avg_size = sum(len(c.content) for c in chunks) / total_chunks if total_chunks > 0 else 0
        
        log_event(logger, "DOCUMENT_CHUNKED", {
            "document_id": document_id,
            "total_chunks": total_chunks,
            "avg_chunk_size": int(avg_size)
        })
        
        return chunks


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """
    Extract text content from uploaded file.
    
    Supports:
    - Plain text files (.txt)
    - PDF files (.pdf) - requires PyPDF2
    
    Args:
        file_content: Raw file bytes
        filename: Original filename for type detection
    
    Returns:
        Extracted text content
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.txt'):
        return file_content.decode('utf-8', errors='ignore')
    
    elif filename_lower.endswith('.pdf'):
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n".join(text_parts)
        
        except ImportError:
            logger.warning("PyPDF2 not installed. PDF parsing unavailable.")
            raise ValueError("PDF support requires PyPDF2 package")
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    else:
        # Attempt to read as plain text
        try:
            return file_content.decode('utf-8', errors='ignore')
        except Exception:
            raise ValueError(f"Unsupported file format: {filename}")
