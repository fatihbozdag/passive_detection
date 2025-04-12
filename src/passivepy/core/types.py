"""Type definitions for the PassivePy core module."""

from typing import Dict, List, Optional, TypedDict, Union


class DetectionResult(TypedDict):
    """Type definition for passive voice detection results."""
    is_passive: bool
    passive_phrases: List[str]
    confidence: float
    metadata: Dict[str, Union[str, int, float]]


class DetectorConfig(TypedDict, total=False):
    """Type definition for detector configuration."""
    threshold: float
    use_spacy: bool
    language: str
    max_length: Optional[int]
    batch_size: Optional[int] 