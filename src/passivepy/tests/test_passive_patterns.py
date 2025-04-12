#!/usr/bin/env python3
"""
Test script for passive voice detection using patterns.py
"""

import spacy
import re
from src.passive_detector.core.patterns import (
    PASSIVE_PATTERNS,
    PASSIVE_EXPRESSIONS,
    NON_PASSIVE_COMBOS,
    PERSONAL_MENTAL_STATES,
    ADJECTIVAL_PARTICIPLES,
    COMMON_PASSIVE_PATTERNS,
    SPECIAL_PASSIVE_PATTERNS,
    FALSE_POSITIVE_PATTERNS,
    COMMON_PASSIVE_VERBS,
    CONFIDENCE_SCORES
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Test sentences
test_sentences = [
    "The report was written by John.",
    "The ceremony was held in the town square.",
    "The suspect was arrested by the police.",
    "The building was constructed in 1920.",
    "I am excited about the new project.",
    "The milk has gone bad.",
    "He is considered the best candidate for the job.",
    "The house is located on a hill.",
    "The president was elected by a narrow margin.",
    "The game was played at the new stadium.",
    "He is able to solve the problem.",
    "The meeting will be scheduled next week.",
    "The company was acquired by a larger firm.",
    "He is going to start a new job.",
    "The letter was delivered yesterday.",
    "The research is conducted by our team.",
    "The document was signed by both parties."
]

def apply_patterns(text):
    """Apply all passive patterns to text and report results"""
    matches = []
    doc = nlp(text)
    
    # Check basic patterns
    for pattern_type, pattern in PASSIVE_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            phrase = match.group(0)
            
            # Skip non-passive combinations
            if any(non_passive.lower() in phrase.lower() for non_passive in NON_PASSIVE_COMBOS):
                continue
                
            # Skip personal mental states
            if any(mental_state.lower() in phrase.lower() for mental_state in PERSONAL_MENTAL_STATES):
                continue
                
            matches.append({
                'pattern_type': pattern_type,
                'match': phrase,
                'confidence': CONFIDENCE_SCORES.get(pattern_type, 0.7)
            })
    
    # Check common passive patterns
    for pattern in COMMON_PASSIVE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            phrase = match.group(0)
            matches.append({
                'pattern_type': 'common_passive',
                'match': phrase,
                'confidence': CONFIDENCE_SCORES.get('high_confidence', 0.85)
            })
    
    # Check special patterns
    for special_type, pattern in SPECIAL_PASSIVE_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            phrase = match.group(0)
            matches.append({
                'pattern_type': special_type,
                'match': phrase,
                'confidence': CONFIDENCE_SCORES.get(special_type, 0.85)
            })
    
    # Check for false positives
    filtered_matches = []
    for match_info in matches:
        phrase = match_info['match'].lower()
        is_false_positive = False
        
        for pattern in FALSE_POSITIVE_PATTERNS:
            if re.search(pattern, phrase, re.IGNORECASE):
                is_false_positive = True
                break
        
        if not is_false_positive:
            filtered_matches.append(match_info)
    
    return filtered_matches

def main():
    print("Testing passive voice detection with patterns.py\n")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Sentence {i}: {sentence}")
        
        matches = apply_patterns(sentence)
        
        if matches:
            print("  Passive patterns detected:")
            for match in matches:
                print(f"  - Type: {match['pattern_type']}, Match: '{match['match']}', Confidence: {match['confidence']:.2f}")
        else:
            print("  No passive patterns detected")
        
        print()

if __name__ == "__main__":
    main() 