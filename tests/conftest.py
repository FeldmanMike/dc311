"""
Configure pytest
"""

from config.logging_config import setup_logging


def pytest_configure():
    """Configure pytest logging"""
    setup_logging()
