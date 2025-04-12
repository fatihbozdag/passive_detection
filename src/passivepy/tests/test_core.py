"""Tests for core passive voice detection functionality."""

import pytest
from passivepy import PassiveDetector


def test_passive_detection_basic(passive_detector, sample_texts):
    """Test basic passive voice detection."""
    # Test passive sentence
    result = passive_detector.detect(sample_texts["passive"])
    assert result["is_passive"] is True
    assert len(result["passive_phrases"]) > 0

    # Test active sentence
    result = passive_detector.detect(sample_texts["active"])
    assert result["is_passive"] is False
    assert len(result["passive_phrases"]) == 0


def test_passive_detection_mixed(passive_detector, sample_texts):
    """Test detection in mixed active/passive sentences."""
    result = passive_detector.detect(sample_texts["mixed"])
    assert result["is_passive"] is True
    assert len(result["passive_phrases"]) > 0


def test_passive_detection_complex(passive_detector, sample_texts):
    """Test detection in complex sentences with multiple clauses."""
    result = passive_detector.detect(sample_texts["complex"])
    assert result["is_passive"] is True
    assert len(result["passive_phrases"]) > 1


def test_detector_initialization():
    """Test detector initialization with different parameters."""
    # Test default initialization
    detector = PassiveDetector()
    assert detector is not None

    # Test initialization with custom parameters
    detector = PassiveDetector(threshold=0.8, use_spacy=True)
    assert detector.threshold == 0.8
    assert detector.use_spacy is True


def test_false_positives(passive_detector):
    """Test detection of false positive passive constructions."""
    # Test "used to" constructions
    texts = [
        "I am used to reading books for long hours.",
        "We got used to smoking that much.",
        "They are used to working late.",
        "He became used to the noise.",
        "She grew used to the routine.",
    ]
    
    for text in texts:
        result = passive_detector.detect(text)
        assert result["is_passive"] is False
        assert len(result["passive_phrases"]) == 0


def test_real_passives_with_similar_patterns(passive_detector):
    """Test real passive constructions that might look similar to false positives."""
    texts = [
        "The book was used to teach students.",
        "The tools were used to build the house.",
        "The money was used to fund the project.",
    ]
    
    for text in texts:
        result = passive_detector.detect(text)
        assert result["is_passive"] is True
        assert len(result["passive_phrases"]) > 0


def test_edge_cases(passive_detector):
    """Test edge cases and special constructions."""
    texts = [
        "I am used to it.",  # Short form
        "Getting used to new things takes time.",  # Gerund form
        "They are accustomed to the changes.",  # Alternative form
        "The system was used to process data.",  # Real passive
        "We got accustomed to the new schedule.",  # False positive
    ]
    
    expected_results = [False, False, False, True, False]
    
    for text, expected in zip(texts, expected_results):
        result = passive_detector.detect(text)
        assert result["is_passive"] == expected 