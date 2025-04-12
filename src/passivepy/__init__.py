"""
PassivePy - A Python package for passive voice detection and analysis.
"""

__version__ = "0.1.0"

from passivepy.core.passive_voice_detector import PassiveDetector
from passivepy.core.my_passive_detector import CustomPassiveDetector
from passivepy.core.simple_passivepy import SimplePassiveDetector

__all__ = [
    "PassiveDetector",
    "CustomPassiveDetector",
    "SimplePassiveDetector",
]
