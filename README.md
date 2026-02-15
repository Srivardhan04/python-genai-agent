# Kafka Q&A automation - Complete Solution

This project implements a full-stack production-style prototype for a Kafka-driven Document Intelligence system. It features an event-driven RAG pipeline with an Agentic AI workflow and a modern web interface.

## Project Structure

```
project/
├── backend/                # FastAPI Backend
│   ├── app/                # Main application logic
│   ├── tests/              # Unit and integration tests
│   ├── .env.example        # Environment configuration
│   └── requirements.txt    # Python dependencies
├── frontend/               # Modern JS Frontend
│   ├── index.html          # Main UI
│   ├── styles.css          # Styling
│   └── app.js              # Frontend logic and API integration
└── run_system.py           # Multi-service launcher script
```

## Features

### Backend (Agentic RAG)
- **Retrieval-Augmented Generation**: Grounded answers using document context.
- **Agentic Workflow**: Two-agent system (Retrieval & Reasoning) with validation.
- **Event-Driven**: Kafka integration (with in-memory fallback) for async processing.
- **Vector Database**: FAISS for high-performance semantic search.

### Frontend (Modern UI)
- **Responsive Dashboard**: Built with Tailwind CSS.
- **Async Interactions**: Real-time status polling for AI jobs.
- **Document Management**: Drag-and-drop upload for TXT and PDF.
- **Live Monitoring**: API health and system statistics tracking.

## System Integration

The system follows a decoupled architecture:
1. **Frontend** sends requests to the **FastAPI Backend**.
2. **Backend** triggers **Kafka Events** (ai_jobs topic).
3. **Kafka Consumer** picks up jobs and executes the **Agent Workflow**.
4. **Agents** retrieve context from **FAISS** and reason via **OpenAI LLM**.
5. **Frontend** polls the status endpoint until the answer is complete.

## Quick Start

### 1. Requirements
- Python 3.10+
- (Optional) Kafka installed locally
- (Optional) OpenAI API Key

### 2. Setup
```bash
# Navigate to backend
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 3. Run Everything
From the root directory:
```bash
python run_system.py
```
This script will:
1. Start the FastAPI backend on http://localhost:8000
2. Start a simple frontend server on http://localhost:3000
3. Automatically open your browser to the UI.

## API Documentation
Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing
```bash
cd backend
# With venv activated
pytest
```
