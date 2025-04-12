#!/usr/bin/env python3
"""
Fix Custom Passive Detector

This script modifies my_passive_detector.py to fix the issues with 'passive_ratio'
and other errors that occurred during previous runs.
"""

import os
import spacy
import re
import sys
from pathlib import Path

# Import the patterns for passive voice detection
try:
    from src.passive_detector.core.patterns import (
        PASSIVE_PATTERNS,
        PASSIVE_EXPRESSIONS,
        NON_PASSIVE_COMBOS,
        PERSONAL_MENTAL_STATES,
        ADJECTIVAL_PARTICIPLES
    )
    patterns_imported = True
except ImportError:
    patterns_imported = False
    print("Warning: Could not import patterns from src.passive_detector.core.patterns")
    print("Will use default patterns")

# Default patterns in case the import fails
DEFAULT_PASSIVE_PATTERNS = [
    r'\b(?:am|are|is|was|were|be|been|being)\s+(\w+ed)\b(?!\s+by)',
    r'\b(?:am|are|is|was|were|be|been|being)\s+(\w+en)\b(?!\s+by)',
    r'\b(?:am|are|is|was|were|be|been|being)\s+(\w+ed)\s+by\b',
    r'\b(?:am|are|is|was|were|be|been|being)\s+(\w+en)\s+by\b',
    r'\b(?:has|have|had|having)\s+been\s+(\w+ed)\b',
    r'\b(?:has|have|had|having)\s+been\s+(\w+en)\b',
    r'\bgot\s+(\w+ed)\b',
    r'\bgot\s+(\w+en)\b',
    r'\b(?:can|could|shall|should|will|would|may|might|must)\s+be\s+(\w+ed)\b',
    r'\b(?:can|could|shall|should|will|would|may|might|must)\s+be\s+(\w+en)\b'
]

DEFAULT_PASSIVE_EXPRESSIONS = [
    'supposed to',
    'meant to',
    'required to',
    'asked to',
    'forced to',
    'expected to',
    'allowed to'
]

DEFAULT_NON_PASSIVE_COMBOS = [
    'is able', 'was able',
    'is about', 'was about',
    'is also', 'was also',
    'is apparent', 'was apparent',
    'is available', 'was available',
    'is aware', 'was aware',
    'is bound', 'was bound',
    'is certain', 'was certain',
    'is clear', 'was clear',
    'is common', 'was common',
    'is due', 'was due',
    'is essential', 'was essential',
    'is evident', 'was evident',
    'is likely', 'was likely',
    'is necessary', 'was necessary',
    'is obvious', 'was obvious',
    'is possible', 'was possible',
    'is responsible', 'was responsible',
    'is subject', 'was subject',
    'is sure', 'was sure',
    'is unlikely', 'was unlikely',
    'is useful', 'was useful',
    'is valid', 'was valid'
]

DEFAULT_PERSONAL_MENTAL_STATES = [
    'excited', 'worried', 'interested', 'concerned', 'pleased',
    'surprised', 'confused', 'satisfied', 'delighted', 'frightened',
    'shocked', 'disappointed', 'thrilled', 'horrified', 'amazed'
]

DEFAULT_ADJECTIVAL_PARTICIPLES = [
    'educated', 'experienced', 'qualified', 'sophisticated', 'talented',
    'established', 'settled', 'renowned', 'acclaimed', 'accomplished',
    'cultured', 'trained', 'skilled', 'gifted', 'prepared',
    'developed', 'advanced', 'polished', 'refined', 'improved'
]

def fix_process_text_function():
    """Fix the process_text function in my_passive_detector.py"""
    
    # Path to the detector file
    detector_path = Path("my_passive_detector.py")
    
    if not detector_path.exists():
        print(f"Error: Could not find {detector_path}")
        return False
    
    # Read the file
    with open(detector_path, 'r') as file:
        content = file.read()
    
    # Define the new improved process_text function
    new_process_text = '''
def process_text(text, nlp):
    """Process text with spaCy and extract passive phrases."""
    if text is None or text.strip() == "":
        return {
            "text": text,
            "is_passive": False,
            "passive_count": 0,
            "passive_phrases": [],
            "passive_ratio": 0.0
        }
    
    doc = nlp(text)
    
    # Find passive voice phrases
    passive_phrases = extract_passive_phrases(doc)
    
    # Calculate passive ratio: number of sentences with passives / total number of sentences
    sentences = list(doc.sents)
    sentence_count = max(1, len(sentences))  # Avoid division by zero
    sentences_with_passive = set()
    
    for phrase in passive_phrases:
        phrase_start = phrase.get('start_char', 0)
        for i, sent in enumerate(sentences):
            if sent.start_char <= phrase_start < sent.end_char:
                sentences_with_passive.add(i)
                break
    
    passive_ratio = len(sentences_with_passive) / sentence_count
    
    # Return results
    return {
        "text": text,
        "is_passive": bool(passive_phrases),
        "passive_count": len(passive_phrases),
        "passive_phrases": passive_phrases,
        "passive_ratio": passive_ratio
    }
'''
    
    # Replace the existing process_text function
    pattern = r'def process_text\(text, nlp\):[^}]*?\}\s*\n'
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, new_process_text, content, flags=re.DOTALL)
        
        # Write the updated content back to the file
        with open(detector_path, 'w') as file:
            file.write(new_content)
        
        print(f"Successfully updated process_text function in {detector_path}")
        return True
    else:
        print(f"Error: Could not find process_text function in {detector_path}")
        return False

def fix_extract_passive_phrases_function():
    """Fix the extract_passive_phrases function in my_passive_detector.py"""
    
    # Path to the detector file
    detector_path = Path("my_passive_detector.py")
    
    if not detector_path.exists():
        print(f"Error: Could not find {detector_path}")
        return False
    
    # Read the file
    with open(detector_path, 'r') as file:
        content = file.read()
    
    # Choose the patterns to use
    passive_patterns = PASSIVE_PATTERNS if patterns_imported else DEFAULT_PASSIVE_PATTERNS
    passive_expressions = PASSIVE_EXPRESSIONS if patterns_imported else DEFAULT_PASSIVE_EXPRESSIONS
    non_passive_combos = NON_PASSIVE_COMBOS if patterns_imported else DEFAULT_NON_PASSIVE_COMBOS
    personal_mental_states = PERSONAL_MENTAL_STATES if patterns_imported else DEFAULT_PERSONAL_MENTAL_STATES
    adjectival_participles = ADJECTIVAL_PARTICIPLES if patterns_imported else DEFAULT_ADJECTIVAL_PARTICIPLES
    
    # Define the improved extract_passive_phrases function
    new_extract_passive_phrases = f'''
def extract_passive_phrases(doc):
    """Extract passive voice phrases from a spaCy Doc object."""
    passive_phrases = []
    text = doc.text.lower()
    
    # Pattern-based detection for common passive voice constructions
    passive_patterns = {passive_patterns}
    
    # Common passive expressions that might not match standard patterns
    passive_expressions = {passive_expressions}
    
    # Combinations that look like passive voice but are actually not
    non_passive_combos = {non_passive_combos}
    
    # Personal mental states often used in non-passive constructions
    personal_mental_states = {personal_mental_states}
    
    # Common adjectival participles that should not be counted as passive
    adjectival_participles = {adjectival_participles}
    
    # Check for pattern matches
    for pattern in passive_patterns:
        for match in re.finditer(pattern, text):
            # Get the match span
            start, end = match.span()
            passive_phrase = text[start:end]
            
            # Check if this is actually a non-passive combination
            is_non_passive = False
            for combo in non_passive_combos:
                if combo in passive_phrase:
                    is_non_passive = True
                    break
            
            # Check if the participle is actually an adjectival form
            participle = match.group(1) if match.groups() else None
            if participle and participle in adjectival_participles:
                is_non_passive = True
            
            # Check if the participle is a personal mental state
            if participle and participle in personal_mental_states:
                # Check if it's being used as an adjective or in passive voice
                if "by" not in passive_phrase:
                    is_non_passive = True
            
            if not is_non_passive:
                # Find the main verb (participle)
                main_verb = participle if participle else ""
                
                # Find the auxiliary verb
                aux_verb = passive_phrase.split()[0]
                
                # Determine if there's an agent (introduced by "by")
                agent = ""
                if "by" in passive_phrase:
                    agent_part = passive_phrase.split("by")[1].strip()
                    agent = agent_part
                
                passive_phrases.append({
                    "passive_phrase": passive_phrase,
                    "start_char": start,
                    "end_char": end,
                    "main_verb": main_verb,
                    "auxiliary_verb": aux_verb,
                    "agent": agent,
                    "lemmatized_main_verb": main_verb
                })
    
    # Check for common passive expressions
    for expression in passive_expressions:
        for match in re.finditer(r'\b' + re.escape(expression) + r'\b', text):
            start, end = match.span()
            
            # Get context (5 words before and after)
            words = text.split()
            for i, word in enumerate(words):
                if expression in word and i > 0:
                    context_start = max(0, i - 5)
                    context_end = min(len(words), i + 6)
                    context = ' '.join(words[context_start:context_end])
                    
                    passive_phrases.append({
                        "passive_phrase": context,
                        "start_char": text.find(context),
                        "end_char": text.find(context) + len(context),
                        "main_verb": expression,
                        "auxiliary_verb": "",
                        "agent": "",
                        "lemmatized_main_verb": expression
                    })
                    break
    
    # Remove duplicates based on start_char and end_char
    unique_phrases = []
    seen_spans = set()
    
    for phrase in passive_phrases:
        span = (phrase["start_char"], phrase["end_char"])
        if span not in seen_spans:
            seen_spans.add(span)
            unique_phrases.append(phrase)
    
    return unique_phrases
'''
    
    # Replace the existing extract_passive_phrases function
    pattern = r'def extract_passive_phrases\(doc\):[^}]*?\}\s*\n'
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, new_extract_passive_phrases, content, flags=re.DOTALL)
        
        # Write the updated content back to the file
        with open(detector_path, 'w') as file:
            file.write(new_content)
        
        print(f"Successfully updated extract_passive_phrases function in {detector_path}")
        return True
    else:
        print(f"Error: Could not find extract_passive_phrases function in {detector_path}")
        return False

def main():
    """Main function to fix the custom passive detector."""
    
    print("Fixing the custom passive detector...")
    
    # Fix the process_text function
    process_text_fixed = fix_process_text_function()
    
    # Fix the extract_passive_phrases function
    extract_passive_phrases_fixed = fix_extract_passive_phrases_function()
    
    if process_text_fixed and extract_passive_phrases_fixed:
        print("Successfully fixed the custom passive detector.")
    else:
        print("Failed to fix one or more functions in the custom passive detector.")

if __name__ == "__main__":
    main() 