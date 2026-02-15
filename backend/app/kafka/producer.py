"""
Kafka Producer for publishing events to the message queue.

This module handles producing messages to Kafka topics. In this system, we use Kafka
to decouple the API layer from the AI processing layer, enabling:
1. Asynchronous processing of questions
2. Better fault tolerance
3. Scalability (multiple consumers can process jobs)

Design Decision:
- Single topic 'ai_jobs' for simplicity
- JSON serialization for message format
- Synchronous message confirmation for reliability
"""

import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

from app.utils.logger import get_kafka_logger, log_event

logger = get_kafka_logger()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "ai_jobs"


class KafkaMessageProducer:
    """
    Kafka producer wrapper for publishing events.
    
    Falls back to an in-memory queue if Kafka is not available,
    enabling development and testing without Kafka infrastructure.
    """
    
    def __init__(self):
        """Initialize the Kafka producer or fallback mechanism."""
        self.producer = None
        self.kafka_available = False
        self._fallback_queue = []
        
        try:
            from kafka import KafkaProducer
            
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3
            )
            self.kafka_available = True
            logger.info(f"Kafka producer connected to {KAFKA_BOOTSTRAP_SERVERS}")
            
        except Exception as e:
            logger.warning(f"Kafka not available: {str(e)}. Using in-memory fallback.")
            self.kafka_available = False
    
    def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        key: Optional[str] = None
    ) -> bool:
        """
        Publish an event to the Kafka topic.
        
        Args:
            event_type: Type of event (e.g., DOCUMENT_INDEXED, QUESTION_RECEIVED)
            payload: Event data to publish
            key: Optional message key for partitioning
        
        Returns:
            True if message was published successfully
        """
        message = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "payload": payload
        }
        
        if self.kafka_available:
            try:
                future = self.producer.send(
                    KAFKA_TOPIC,
                    value=message,
                    key=key or event_type
                )
                # Wait for confirmation (synchronous for reliability)
                record_metadata = future.get(timeout=10)
                
                log_event(logger, "EVENT_PUBLISHED", {
                    "event_type": event_type,
                    "topic": record_metadata.topic,
                    "partition": record_metadata.partition,
                    "offset": record_metadata.offset
                })
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish event: {str(e)}")
                # Fall back to in-memory queue
                self._fallback_queue.append(message)
                return False
        else:
            # Use fallback queue for testing without Kafka
            self._fallback_queue.append(message)
            log_event(logger, "EVENT_QUEUED_FALLBACK", {
                "event_type": event_type,
                "queue_size": len(self._fallback_queue)
            })
            return True
    
    def get_fallback_queue(self) -> list:
        """Get messages from the fallback queue (for testing)."""
        return self._fallback_queue.copy()
    
    def clear_fallback_queue(self) -> None:
        """Clear the fallback queue."""
        self._fallback_queue = []
    
    def close(self) -> None:
        """Close the producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


# Global producer instance
_producer_instance: Optional[KafkaMessageProducer] = None


def get_producer() -> KafkaMessageProducer:
    """
    Get or create the global producer instance.
    
    Returns:
        KafkaMessageProducer instance (singleton)
    """
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = KafkaMessageProducer()
    return _producer_instance


def publish_document_indexed(document_id: str, chunk_count: int) -> bool:
    """
    Publish a DOCUMENT_INDEXED event.
    
    Args:
        document_id: ID of the indexed document
        chunk_count: Number of chunks created
    
    Returns:
        True if published successfully
    """
    producer = get_producer()
    return producer.publish_event(
        event_type="DOCUMENT_INDEXED",
        payload={
            "document_id": document_id,
            "chunk_count": chunk_count,
            "status": "indexed"
        },
        key=document_id
    )


def publish_question_received(job_id: str, question: str) -> bool:
    """
    Publish a QUESTION_RECEIVED event.
    
    Args:
        job_id: Unique job identifier
        question: The question text
    
    Returns:
        True if published successfully
    """
    producer = get_producer()
    return producer.publish_event(
        event_type="QUESTION_RECEIVED",
        payload={
            "job_id": job_id,
            "question": question,
            "status": "pending"
        },
        key=job_id
    )
