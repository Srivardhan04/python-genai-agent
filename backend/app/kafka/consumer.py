"""
Kafka Consumer for processing AI jobs asynchronously.

This module handles consuming messages from Kafka and triggering the appropriate
agent workflow. The consumer runs in a separate thread to enable asynchronous
processing while the API remains responsive.

Design Decision:
- Consumer runs in a background thread
- Job status is tracked in a shared dictionary (would use Redis in production)
- Graceful shutdown handling for clean termination
"""

import os
import json
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from app.utils.logger import get_kafka_logger, log_event

logger = get_kafka_logger()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "ai_jobs"
KAFKA_GROUP_ID = "rag_agent_group"


class JobStatus:
    """Enumeration of job status values."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# In-memory job store (would use Redis or database in production)
job_store: Dict[str, Dict[str, Any]] = {}


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a job.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Job status dictionary or None if not found
    """
    return job_store.get(job_id)


def update_job_status(job_id: str, status: str, result: Any = None, error: str = None):
    """
    Update the status of a job.
    
    Args:
        job_id: Job identifier
        status: New status value
        result: Optional result data
        error: Optional error message
    """
    if job_id not in job_store:
        job_store[job_id] = {
            "created_at": datetime.utcnow().isoformat()
        }
    
    job_store[job_id].update({
        "status": status,
        "updated_at": datetime.utcnow().isoformat()
    })
    
    if result is not None:
        job_store[job_id]["result"] = result
    
    if error is not None:
        job_store[job_id]["error"] = error
    
    log_event(logger, "JOB_STATUS_UPDATED", {
        "job_id": job_id,
        "status": status
    })


def create_job(job_id: str, question: str) -> None:
    """
    Create a new job entry.
    
    Args:
        job_id: Job identifier
        question: The question to process
    """
    job_store[job_id] = {
        "status": JobStatus.PENDING,
        "question": question,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "result": None,
        "error": None
    }
    log_event(logger, "JOB_CREATED", {"job_id": job_id})


class KafkaMessageConsumer:
    """
    Kafka consumer for processing AI job events.
    
    Supports:
    - Background thread processing
    - Fallback mode for testing without Kafka
    - Graceful shutdown
    """
    
    def __init__(self, message_handler: Callable[[Dict[str, Any]], None]):
        """
        Initialize the consumer.
        
        Args:
            message_handler: Function to call for each message
        """
        self.message_handler = message_handler
        self.consumer = None
        self.kafka_available = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        try:
            from kafka import KafkaConsumer
            
            self.consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=KAFKA_GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000  # Poll timeout for shutdown checking
            )
            self.kafka_available = True
            logger.info(f"Kafka consumer connected to {KAFKA_BOOTSTRAP_SERVERS}")
            
        except Exception as e:
            logger.warning(f"Kafka not available: {str(e)}. Running in fallback mode.")
            self.kafka_available = False
    
    def start(self) -> None:
        """Start the consumer in a background thread."""
        if self._running:
            logger.warning("Consumer already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("Kafka consumer started in background thread")
    
    def _consume_loop(self) -> None:
        """Main consumption loop - runs in background thread."""
        logger.info("Consumer loop started")
        
        while self._running:
            if self.kafka_available:
                try:
                    # Poll for messages
                    for message in self.consumer:
                        if not self._running:
                            break
                        
                        log_event(logger, "MESSAGE_RECEIVED", {
                            "topic": message.topic,
                            "partition": message.partition,
                            "offset": message.offset
                        })
                        
                        try:
                            self.message_handler(message.value)
                        except Exception as e:
                            logger.error(f"Error handling message: {str(e)}")
                
                except Exception as e:
                    if self._running:
                        logger.error(f"Consumer error: {str(e)}")
            else:
                # In fallback mode, check the producer's fallback queue
                import time
                time.sleep(0.5)  # Polling interval
                
                # Process any messages in the fallback queue
                from app.kafka.producer import get_producer
                producer = get_producer()
                messages = producer.get_fallback_queue()
                
                if messages:
                    producer.clear_fallback_queue()
                    for msg in messages:
                        if not self._running:
                            break
                        try:
                            self.message_handler(msg)
                        except Exception as e:
                            logger.error(f"Error handling fallback message: {str(e)}")
    
    def stop(self) -> None:
        """Stop the consumer gracefully."""
        logger.info("Stopping consumer...")
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        if self.consumer:
            self.consumer.close()
        
        logger.info("Consumer stopped")


def process_question_event(event: Dict[str, Any]) -> None:
    """
    Process a QUESTION_RECEIVED event.
    
    This function is called by the consumer when a question event is received.
    It triggers the agent workflow to process the question.
    
    Args:
        event: The event message containing job_id and question
    """
    event_type = event.get("event_type")
    
    if event_type != "QUESTION_RECEIVED":
        # Ignore other event types
        logger.debug(f"Ignoring event type: {event_type}")
        return
    
    payload = event.get("payload", {})
    job_id = payload.get("job_id")
    question = payload.get("question")
    
    if not job_id or not question:
        logger.error(f"Invalid event payload: {payload}")
        return
    
    log_event(logger, "PROCESSING_QUESTION", {
        "job_id": job_id,
        "question_length": len(question)
    })
    
    # Update job status to processing
    update_job_status(job_id, JobStatus.PROCESSING)
    
    try:
        # Import here to avoid circular imports
        from app.agents.retrieval_agent import RetrievalAgent
        from app.agents.reasoning_agent import ReasoningAgent
        
        # Execute the agent workflow
        # Step 1: Retrieve relevant context
        retrieval_agent = RetrievalAgent()
        context_chunks = retrieval_agent.retrieve(question)
        
        # Step 2: Reason with LLM
        reasoning_agent = ReasoningAgent()
        answer = reasoning_agent.reason(question, context_chunks)
        
        # Update job with result
        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result={
                "answer": answer,
                "sources_used": len(context_chunks)
            }
        )
        
        log_event(logger, "QUESTION_PROCESSED", {
            "job_id": job_id,
            "status": "completed"
        })
        
    except Exception as e:
        logger.error(f"Error processing question {job_id}: {str(e)}")
        update_job_status(job_id, JobStatus.FAILED, error=str(e))


# Global consumer instance
_consumer_instance: Optional[KafkaMessageConsumer] = None


def get_consumer() -> KafkaMessageConsumer:
    """
    Get or create the global consumer instance.
    
    Returns:
        KafkaMessageConsumer instance (singleton)
    """
    global _consumer_instance
    if _consumer_instance is None:
        _consumer_instance = KafkaMessageConsumer(message_handler=process_question_event)
    return _consumer_instance


def start_consumer() -> None:
    """Start the global consumer."""
    consumer = get_consumer()
    consumer.start()


def stop_consumer() -> None:
    """Stop the global consumer."""
    global _consumer_instance
    if _consumer_instance:
        _consumer_instance.stop()
        _consumer_instance = None
