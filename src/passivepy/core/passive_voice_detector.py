"""Core passive voice detection implementation."""

from typing import Dict, List, Optional, Union, Set

import spacy
from spacy.tokens import Doc, Token

from passivepy.core.types import DetectionResult, DetectorConfig
from passivepy.utils.logging import setup_logging

logger = setup_logging()


class PassiveDetector:
    """Main class for passive voice detection."""

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """Initialize the passive voice detector.
        
        Args:
            config: Optional configuration for the detector
        """
        self.config = config or DetectorConfig(
            threshold=0.7,
            use_spacy=True,
            language="en",
        )
        
        # Load the transformer model
        try:
            self.nlp = spacy.load("en_core_web_trf") if self.config["use_spacy"] else None
            logger.info("Loaded transformer model: en_core_web_trf")
        except OSError:
            logger.warning("Transformer model not found, downloading...")
            spacy.cli.download("en_core_web_trf")
            self.nlp = spacy.load("en_core_web_trf")
            logger.info("Downloaded and loaded transformer model")
        
        # Define false positive patterns
        self.false_positives: Set[str] = {
            "used to",  # "I am used to reading"
            "get used to",  # "We got used to smoking"
            "be used to",  # "They are used to working"
            "become used to",  # "He became used to the noise"
            "grow used to",  # "She grew used to the routine"
            "get accustomed to",  # "They got accustomed to the changes"
            "be accustomed to",  # "We are accustomed to the schedule"
        }
        
        # Define auxiliary verbs that can indicate passive voice
        self.auxiliary_verbs: Set[str] = {
            "am", "is", "are", "was", "were", "be", "been", "being",
            "get", "got", "gotten", "getting",
            "become", "became", "becoming",
        }
        
        logger.info("PassiveDetector initialized with config: %s", self.config)

    def detect(self, text: str) -> DetectionResult:
        """Detect passive voice in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult containing passive voice information
        """
        if not text.strip():
            return DetectionResult(
                is_passive=False,
                passive_phrases=[],
                confidence=0.0,
                metadata={"error": "Empty text"},
            )

        if self.config["use_spacy"] and self.nlp:
            doc = self.nlp(text)
            return self._detect_with_spacy(doc)
        else:
            return self._detect_with_regex(text)

    def _detect_with_spacy(self, doc: Doc) -> DetectionResult:
        """Detect passive voice using SpaCy's dependency parsing.
        
        Args:
            doc: SpaCy document object
            
        Returns:
            DetectionResult containing passive voice information
        """
        passive_phrases = []
        for token in doc:
            if token.dep_ == "nsubjpass":
                # Check for false positives before adding the phrase
                if not self._is_false_positive(token):
                    phrase = self._extract_passive_phrase(token)
                    if phrase:
                        passive_phrases.append(phrase)

        return DetectionResult(
            is_passive=len(passive_phrases) > 0,
            passive_phrases=passive_phrases,
            confidence=min(1.0, len(passive_phrases) * 0.2),
            metadata={"method": "spacy", "num_phrases": len(passive_phrases)},
        )

    def _is_false_positive(self, token: Token) -> bool:
        """Check if a potential passive construction is a false positive.
        
        Args:
            token: The token to check
            
        Returns:
            True if the construction is a false positive, False otherwise
        """
        # Get the verb token
        verb = token.head
        
        # Check for "used to" constructions
        if verb.text.lower() == "used":
            # Look for "to" in the children of "used"
            for child in verb.children:
                if child.text.lower() == "to":
                    return True
        
        # Check for "get used to" constructions
        if verb.text.lower() in {"get", "got", "getting"}:
            for child in verb.children:
                if child.text.lower() == "used":
                    for grandchild in child.children:
                        if grandchild.text.lower() == "to":
                            return True
        
        # Check for other false positive patterns
        phrase = self._get_phrase_text(token)
        for pattern in self.false_positives:
            if pattern in phrase.lower():
                return True
        
        return False

    def _get_phrase_text(self, token: Token) -> str:
        """Get the text of the phrase containing the token.
        
        Args:
            token: The token to get the phrase for
            
        Returns:
            The text of the phrase
        """
        # Get the sentence containing the token
        sent = token.sent
        
        # Get the start and end indices of the phrase
        start = min(t.i for t in sent if t.dep_ in {"nsubjpass", "auxpass"})
        end = max(t.i for t in sent if t.dep_ in {"nsubjpass", "auxpass", "pobj"})
        
        # Return the text of the phrase
        return sent[start:end + 1].text

    def _extract_passive_phrase(self, token: Token) -> Optional[str]:
        """Extract the full passive phrase starting from a passive subject token.
        
        Args:
            token: The passive subject token
            
        Returns:
            The extracted passive phrase or None if not found
        """
        # Get the sentence containing the token
        sent = token.sent
        
        # Find the passive verb
        verb = token.head
        
        # Find the auxiliary verb if present
        aux = None
        for child in verb.children:
            if child.dep_ == "auxpass":
                aux = child
                break
        
        # Get the start and end indices of the phrase
        start = min(t.i for t in sent if t.dep_ in {"nsubjpass", "auxpass"})
        end = max(t.i for t in sent if t.dep_ in {"nsubjpass", "auxpass", "pobj"})
        
        # Return the text of the phrase
        return sent[start:end + 1].text

    def _detect_with_regex(self, text: str) -> DetectionResult:
        """Detect passive voice using regex patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult containing passive voice information
        """
        # Implementation of regex-based detection
        # This is a placeholder - implement your regex patterns here
        passive_phrases = []
        confidence = 0.0

        return DetectionResult(
            is_passive=len(passive_phrases) > 0,
            passive_phrases=passive_phrases,
            confidence=confidence,
            metadata={"method": "regex"},
        ) 