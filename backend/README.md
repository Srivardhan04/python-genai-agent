# Kafka Q&A automation for Document Intelligence

## Project Overview

This project implements a production-style prototype for a Kafka-driven Retrieval-Augmented Generation (RAG) system designed for document intelligence. The system allows users to upload documents, index them for semantic search, and ask questions that are answered using only the information contained within the uploaded documents.

The architecture demonstrates key concepts relevant to building AI-powered document processing systems in enterprise environments, particularly in financial services where accuracy, traceability, and grounding are critical.

## Key Features

- Document upload and automatic text extraction (TXT, PDF)
- Semantic chunking and embedding generation
- Vector similarity search using FAISS
- LLM-powered question answering with strict grounding
- Asynchronous job processing via Kafka
- Comprehensive logging for audit trails

## Architecture

```
                                    +------------------+
                                    |   FastAPI        |
                                    |   REST API       |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
            +-------v-------+       +--------v--------+      +--------v--------+
            | POST          |       | POST            |      | GET             |
            | /upload-doc   |       | /ask-question   |      | /job-status     |
            +-------+-------+       +--------+--------+      +--------+--------+
                    |                        |                        |
                    v                        v                        |
            +---------------+       +----------------+                |
            | Text Chunker  |       | Kafka Producer |                |
            | + Embeddings  |       +--------+-------+                |
            +-------+-------+                |                        |
                    |                        v                        |
                    v               +----------------+                |
            +---------------+       | Kafka Topic    |                |
            | FAISS Vector  |       | (ai_jobs)      |                |
            | Store         |       +--------+-------+                |
            +---------------+                |                        |
                    ^                        v                        |
                    |               +----------------+                |
                    |               | Kafka Consumer |                |
                    |               +--------+-------+                |
                    |                        |                        |
                    |           +------------+------------+           |
                    |           |                         |           |
                    |   +-------v--------+   +-----------v------+    |
                    +---+ Retrieval      |   | Reasoning        |    |
                        | Agent          |   | Agent            |    |
                        +----------------+   +--------+---------+    |
                                                      |              |
                                             +--------v--------+     |
                                             | Job Store       +-----+
                                             | (In-Memory)     |
                                             +-----------------+
```

## Agent Workflow

The system implements a simple but effective two-agent workflow:

### 1. RetrievalAgent

The RetrievalAgent is responsible for finding relevant document chunks based on the user's question.

**Process:**
- Receives the user question
- Generates an embedding for the question
- Performs vector similarity search in FAISS
- Returns top-k most relevant document chunks

**Key Design Decisions:**
- Uses L2 distance for similarity matching
- Default top_k=5 balances coverage and precision
- Simple interface allows easy testing and modification

### 2. ReasoningAgent

The ReasoningAgent generates answers using the retrieved context and an LLM.

**Process:**
- Receives the question and retrieved context chunks
- Formats context into a grounded prompt
- Calls OpenAI API (or mock for testing)
- Returns the generated answer

**Key Design Decisions:**
- Temperature=0 for deterministic, consistent answers
- System prompt enforces grounding behavior
- Includes validation for grounding check

### Agent Loop Flow

```
Input Question
    |
    v
[RetrievalAgent] -- Query Vector Store --> Retrieve top-k chunks
    |
    v
[ReasoningAgent] -- Format Prompt --> Call LLM --> Validate Grounding
    |
    v
Return Response
```

This is intentionally simple. The system does not use AutoGPT-style planners or complex multi-step reasoning. Each agent has a single responsibility, making the system predictable and debuggable.

## Why Kafka is Used

Kafka serves as the message broker that decouples the API layer from the AI processing layer. This architecture choice provides several benefits:

### 1. Asynchronous Processing
- The API returns immediately with a job_id
- Heavy LLM processing happens in the background
- Users can poll for results without blocking

### 2. Fault Tolerance
- If the consumer crashes, messages remain in Kafka
- Jobs can be reprocessed after recovery
- No question is lost due to transient failures

### 3. Scalability Path
- Multiple consumers can process jobs in parallel
- Easy to add more processing capacity
- Natural load balancing across consumers

### 4. Audit Trail
- All events are logged with timestamps
- Message flow is traceable
- Useful for compliance in financial services

### Implementation Notes
- Single topic (ai_jobs) for simplicity
- Fallback to in-memory queue if Kafka unavailable
- Synchronous message confirmation for reliability

## How RAG Reduces Hallucinations

Retrieval-Augmented Generation addresses the hallucination problem in LLMs through several mechanisms:

### 1. Grounded Context
- The LLM only sees relevant document excerpts
- No reliance on training data knowledge
- Answers must come from provided context

### 2. Explicit Prompt Instructions
The system prompt includes strict instructions:
```
Answer ONLY using the provided context.
If information is missing, say 'Information not found in documents.'
```

### 3. Limited Scope
- Each question uses only top-k relevant chunks
- Reduces noise from irrelevant information
- Focused context leads to focused answers

### 4. Validation
- The ReasoningAgent includes a grounding validation step
- Checks that answer terms appear in context
- Flags potentially ungrounded responses

### Limitations
- Simple word overlap validation is not foolproof
- Sophisticated paraphrasing might pass validation
- Production systems should use NLI models for better validation

## Running the Project Locally

### Prerequisites

1. Python 3.10 or higher
2. Apache Kafka (optional - system works with fallback)
3. OpenAI API key (optional - mock responses available)

### Installation Steps

1. Clone the repository and navigate to the project directory:
```bash
cd project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables (optional):
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"
$env:KAFKA_BOOTSTRAP_SERVERS="localhost:9092"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
```

5. Start the application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

6. Access the API documentation:
- OpenAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Running with Kafka (Optional)

If you want to run with actual Kafka:

1. Download and start Kafka:
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties
```

2. Create the topic:
```bash
bin/kafka-topics.sh --create --topic ai_jobs --bootstrap-server localhost:9092
```

The system will automatically detect and use Kafka if available.

### Testing the API

1. Upload a document:
```bash
curl -X POST "http://localhost:8000/api/v1/upload-document" \
  -F "file=@sample.txt"
```

2. Ask a question:
```bash
curl -X POST "http://localhost:8000/api/v1/ask-question" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

3. Check job status:
```bash
curl "http://localhost:8000/api/v1/job-status/{job_id}"
```

## Project Structure

```
project/
|-- app/
|   |-- main.py              # FastAPI application entry point
|   |-- api/
|   |   |-- routes.py        # HTTP endpoint definitions
|   |-- agents/
|   |   |-- retrieval_agent.py   # Context retrieval logic
|   |   |-- reasoning_agent.py   # LLM-based answer generation
|   |-- rag/
|   |   |-- vector_store.py  # FAISS vector database wrapper
|   |   |-- chunker.py       # Document chunking utilities
|   |-- kafka/
|   |   |-- producer.py      # Kafka message producer
|   |   |-- consumer.py      # Kafka message consumer
|   |-- prompts/
|   |   |-- qa_prompt.txt    # QA prompt template
|   |-- utils/
|   |   |-- logger.py        # Centralized logging configuration
|-- tests/
|   |-- test_api.py          # API endpoint tests
|   |-- test_agents.py       # Agent unit tests
|   |-- test_rag.py          # RAG pipeline tests
|-- requirements.txt         # Python dependencies
|-- README.md               # This file
```

## Limitations

This section documents known limitations and design trade-offs.

### Scope Limitations

1. **In-Memory Storage**: The vector store and job store use in-memory storage. Data is lost on restart. Production systems would use persistent storage (PostgreSQL, Redis, managed vector DB).

2. **Single Consumer**: The Kafka consumer runs in a single thread. Production systems would deploy multiple consumer instances for parallel processing.

3. **No Authentication**: The API has no authentication or authorization. Production systems require proper security controls.

4. **Mock Fallbacks**: Without OpenAI API key or Kafka, the system uses mock implementations. These are for testing only.

### Technical Limitations

1. **Embedding Quality**: Mock embeddings use hash-based vectors. They enable functional testing but do not provide semantic similarity.

2. **PDF Extraction**: PDF text extraction is basic. Complex PDFs with tables, images, or non-standard formatting may not extract correctly.

3. **Chunk Boundaries**: Fixed-size chunking with overlap may split important content. Semantic chunking would be better for production.

4. **Grounding Validation**: The word overlap grounding check is simplistic. Production systems should use Natural Language Inference models.

### Operational Limitations

1. **No Monitoring**: The system lacks operational monitoring dashboards. Production systems need proper observability.

2. **No Rate Limiting**: API endpoints have no rate limiting. Production systems need protection against abuse.

3. **No Retry Logic**: Failed LLM calls are not retried. Production systems need retry with exponential backoff.

## Technology Choices Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| API Framework | FastAPI | Async support, automatic OpenAPI docs, Pydantic validation |
| Vector Store | FAISS | Fast, well-documented, works locally without infrastructure |
| Message Queue | Kafka | Industry standard, demonstrates async patterns, audit trail |
| LLM | OpenAI | Widely available, good documentation, easy to mock |
| Embeddings | text-embedding-ada-002 | Cost-effective, good quality for general use |

## Author

This project was created as a demonstration of backend engineering skills for a Python + GenAI role, focusing on clean architecture, proper error handling, and production-ready patterns without over-engineering.
