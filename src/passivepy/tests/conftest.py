"""Test configuration and fixtures for PassivePy."""

import pytest
from passivepy import PassiveDetector, CustomPassiveDetector, SimplePassiveDetector


@pytest.fixture
def passive_detector():
    """Fixture providing a PassiveDetector instance."""
    return PassiveDetector()


@pytest.fixture
def custom_detector():
    """Fixture providing a CustomPassiveDetector instance."""
    return CustomPassiveDetector()


@pytest.fixture
def simple_detector():
    """Fixture providing a SimplePassiveDetector instance."""
    return SimplePassiveDetector()


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing."""
    return {
        "passive": "The book was written by the author.",
        "active": "The author wrote the book.",
        "mixed": "The book was written by the author, who published it last year.",
        "complex": "The project was completed on time, and the results were presented to the committee.",
    } 