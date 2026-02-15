"""
Centralized logging configuration for the RAG system.

This module provides a consistent logging setup across all components,
enabling traceability of document processing, agent execution, and Kafka events.
"""

import logging
import sys
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance with consistent formatting.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with detailed formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Pre-configured loggers for different components
def get_api_logger() -> logging.Logger:
    """Logger for API endpoints and request handling."""
    return setup_logger("api.routes")


def get_agent_logger() -> logging.Logger:
    """Logger for agent execution and workflow."""
    return setup_logger("agents.workflow")


def get_rag_logger() -> logging.Logger:
    """Logger for RAG pipeline operations."""
    return setup_logger("rag.pipeline")


def get_kafka_logger() -> logging.Logger:
    """Logger for Kafka producer and consumer events."""
    return setup_logger("kafka.events")


def log_event(logger: logging.Logger, event_type: str, details: dict) -> None:
    """
    Structured event logging for audit trail.
    
    Args:
        logger: Logger instance to use
        event_type: Type of event (e.g., DOCUMENT_INDEXED, LLM_CALL)
        details: Dictionary of event details
    """
    detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f"[{event_type}] {detail_str}")
