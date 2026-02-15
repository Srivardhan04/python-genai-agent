"""
Pytest configuration and fixtures.

This module provides shared fixtures for all test modules.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def test_document_content():
    """Sample document content for testing."""
    return """
    Annual Financial Report 2024
    
    Executive Summary
    
    The fiscal year 2024 demonstrated strong performance across all business segments.
    Total revenue reached $15 billion, representing a 12% increase from the previous year.
    Net income improved to $2.3 billion, driven by operational efficiency and market expansion.
    
    Key Highlights:
    - Cloud services revenue grew by 35%
    - Operating margin improved to 24%
    - Customer base expanded by 2 million new accounts
    - International revenue now represents 40% of total revenue
    
    Risk Factors
    
    The company faces several risk factors including market volatility, regulatory changes,
    and competitive pressures. Management has implemented comprehensive risk management
    policies to address these challenges.
    
    Outlook
    
    For fiscal year 2025, management expects continued growth with projected revenue
    of $17 billion and further margin improvements.
    """


@pytest.fixture(scope="session")
def sample_questions():
    """Sample questions for testing."""
    return [
        "What was the total revenue in 2024?",
        "How much did net income improve?",
        "What percentage did cloud services grow?",
        "What are the key risk factors?",
        "What is the revenue outlook for 2025?"
    ]
