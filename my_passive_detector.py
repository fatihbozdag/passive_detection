#!/usr/bin/env python3
"""
Custom Passive Voice Detector

This module implements a custom passive voice detector based on spaCy's dependency parsing,
combined with advanced regex patterns for identifying passive constructions.
"""

import pandas as pd
import spacy
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import json
from tqdm import tqdm  # Import tqdm for progress bars
from typing import List, Dict, Any, Optional, Tuple
import os

try:
    from spellchecker import SpellChecker  # pyspellchecker package
    SPELL_CHECK_AVAILABLE = True
except ImportError:
    print("Warning: Spell checking disabled. To enable, install pyspellchecker: pip install pyspellchecker")
    SPELL_CHECK_AVAILABLE = False

# Configure settings
if torch.cuda.is_available():
    torch.device('cuda')
elif torch.backends.mps.is_available():
    torch.device('mps')
else:
    torch.device('cpu')
    print("Warning: Using CPU. For better performance, use a CUDA-enabled GPU if available.")

spacy.prefer_gpu()
pd.set_option('display.max_colwidth', None)

# Initialize models and tools
spell = SpellChecker() if SPELL_CHECK_AVAILABLE else None
pattern = r'ICLE\-\w+\-\w+\-\d+\.\d+'
pattern_ = r'[^\w\s]'

# Import patterns if file exists, otherwise define them here
try:
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
except ImportError:
    # Define fallback patterns
    print("Warning: Could not import patterns from src.passive_detector.core.patterns.")
    print("Using fallback pattern definitions.")
    # The fallback patterns would be defined here, but they're omitted since the import should work
    # ... existing code ...

def correct_spelling(text):
    """Correct spelling in text"""
    if not SPELL_CHECK_AVAILABLE:
        return text
    try:
        return spell(text)
    except:
        return text  # Return original text if spell checking fails

def is_adjectival_participle(token):
    """
    Check if a past participle is being used as an adjective.
    Returns True if it's likely an adjective, False if likely a verb.
    Enhanced to better distinguish adjectival participles from passives.
    """
    # Quick check for common adjectival participles
    if token.lemma_.lower() in ADJECTIVAL_PARTICIPLES:
        # But check for by-agent which suggests passive voice
        has_by_agent = False
        for i in range(1, 5):  # Look ahead up to 5 tokens
            if i + token.i < len(token.doc) and token.doc[token.i + i].text.lower() == 'by':
                has_by_agent = True
                break
        
        # If it has a by-agent, it's more likely a passive
        if has_by_agent:
            return False
        return True
    
    # Check dependency tag - if it's a modifier, it's likely adjectival
    if token.dep_ == 'amod':  # Adjectival modifier
        return True
    
    # Check if token is used as a complement to a linking verb
    if token.dep_ == 'acomp' and token.head.lemma_ in ['be', 'seem', 'appear', 'look', 'sound', 'feel', 'taste', 'smell']:
        # But check for by-agent which suggests passive voice
        has_by_agent = False
        for i in range(1, 5):  # Look ahead up to 5 tokens
            if i + token.i < len(token.doc) and token.doc[token.i + i].text.lower() == 'by':
                has_by_agent = True
                break
        
        # If it has a by-agent, it's likely a passive
        if has_by_agent:
            return False
            
        # Check if it's preceded by an auxiliary that suggests passive
        has_passive_aux = False
        for child in token.head.children:
            if child.dep_ == 'auxpass':
                has_passive_aux = True
                break
        
        if has_passive_aux:
            return False
            
        # Check if it's modified by adverbs typical for adjectives
        has_adj_adverbs = False
        for child in token.children:
            if child.dep_ == 'advmod' and child.text.lower() in ['very', 'quite', 'extremely', 'really', 'so', 'too']:
                has_adj_adverbs = True
                break
        
        if has_adj_adverbs:
            return True
            
        # Check if it denotes a state rather than an action
        state_participles = {
            'tired', 'bored', 'excited', 'interested', 'worried', 'concerned',
            'satisfied', 'pleased', 'disappointed', 'confused', 'annoyed',
            'frustrated', 'relaxed', 'prepared', 'qualified', 'educated'
        }
        
        if token.lemma_.lower() in state_participles:
            return True
        
        # Use head verb to disambiguate
        if token.head.lemma_ == 'be':
            # Basic copular constructions are often adjectival
            return True
    
    # Check for passive auxiliary dependents
    has_auxpass = False
    for child in token.children:
        if child.dep_ == 'auxpass':
            has_auxpass = True
            break
    
    if has_auxpass:
        return False  # Has passive auxiliary = not adjectival
    
    # Check for specific verbs that are commonly adjectival in certain contexts
    common_state_participles = {
        'interested', 'worried', 'concerned', 'excited', 'tired',
        'bored', 'pleased', 'satisfied', 'disappointed', 'confused',
        'prepared', 'qualified', 'located', 'situated', 'based'
    }
    
    # For specific common adjectival participles, check context more carefully
    if token.lemma_.lower() in common_state_participles:
        # If modified by degree adverbs like "very", "quite", it's likely adjectival
        for child in token.children:
            if child.dep_ == 'advmod' and child.text.lower() in ['very', 'quite', 'extremely', 'really', 'so', 'too']:
                return True
                
        # Check if it's used with "in" or "about" which suggests adjectival state
        for i in range(1, 4):  # Look ahead up to 4 tokens
            if i + token.i < len(token.doc) and token.doc[token.i + i].text.lower() in ['in', 'about', 'with']:
                return True
    
    # Check if part of a compound adjective
    if token.dep_ == 'compound' and token.head.pos_ == 'ADJ':
        return True
        
    # Check if has comparative/superlative forms (suggesting adjectival nature)
    if token.tag_ in ['JJR', 'JJS']:
        return True
        
    # Check for predicative complement
    if token.dep_ == 'xcomp' and token.head.lemma_ in ['be', 'seem', 'appear', 'look', 'sound', 'feel', 'smell', 'taste']:
        return True
    
    # Special handling for "filled" which is often misclassified
    if token.lemma_.lower() == 'filled' and not has_auxpass:
        # Check if followed by "with" which suggests adjectival use
        for i in range(1, 4):  # Look ahead up to 4 tokens
            if i + token.i < len(token.doc) and token.doc[token.i + i].text.lower() == 'with':
                return True
    
    # Check for common patterns where a participle is used adjectivally
    if token.i > 0 and token.doc[token.i-1].text.lower() in ['get', 'gets', 'got', 'gotten']:
        # "Get excited", "got bored" - these are adjectival states, not passives
        if token.lemma_.lower() in common_state_participles:
            return True
    
    # If none of the above, check the lexical category
    return token.pos_ == 'ADJ'

def is_valid_subject(token):
    """Check if token is a valid passive subject"""
    if token.pos_ not in ['NOUN', 'PROPN', 'PRON']:
        return False
    
    if token.lemma_.lower() in ['it', 'there'] and token.dep_ == 'expl':
        return False
    
    return token.dep_ == 'nsubjpass'

def is_valid_passive_construction(verb_token, aux_chain, subject, adverbs):
    """Validate if construction is truly passive"""
    if is_adjectival_participle(verb_token):
        return False
    
    if not any(aux.lemma_ == 'be' for aux in aux_chain):
        return False
    
    infinitive_be = any(aux.lemma_ == 'be' and aux.tag_ == 'VB' for aux in aux_chain)
    if infinitive_be:
        has_modal = any(aux.dep_ == 'aux' and aux.tag_ == 'MD' for aux in aux_chain)
        has_to = any(aux.dep_ == 'aux' and aux.text == 'to' for aux in aux_chain)
        if not (has_modal or has_to):
            return False
    elif not subject:
        return False
    
    if adverbs and not all(adv.dep_ in ['advmod', 'npadvmod'] for adv in adverbs):
        return False
    
    return True

def is_passive_token(token) -> bool:
    """Check if a token is part of a passive construction based on dependency parsing"""
    # Check for passivized clauses using spaCy's dependency parsing
    if token.dep_ == "auxpass" or token.dep_ == "agent":
        return True
    
    # Check for nsubjpass dependency
    if token.dep_ == "nsubjpass":
        return True
    
    # Check for aux + auxpass pattern
    if token.dep_ == "aux" and any(child.dep_ == "auxpass" for child in token.head.children):
        return True
    
    return False

def check_regex_patterns(text: str) -> List[Dict[str, Any]]:
    """Check text against regex patterns for passive voice"""
    passive_phrases = []
    
    # Check the special common passive patterns first (these are higher confidence)
    for pattern in COMMON_PASSIVE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            phrase = match.group(0)
            
            # Extract verb
            verb_parts = phrase.split()
            if len(verb_parts) >= 2:
                participle = verb_parts[-1].lower()
                
                # Skip if it's in adjectival participles list
                if participle in ADJECTIVAL_PARTICIPLES:
                    continue
                
                # Check for 'by' agent which reinforces passive
                has_by_agent = False
                rest_of_sentence = text[match.end():].strip()
                by_match = re.match(r'(\s+by\s+|\s+from\s+|\s+through\s+)', rest_of_sentence)
                if by_match:
                    has_by_agent = True
                
                # Use confidence scores from our patterns
                confidence = CONFIDENCE_SCORES.get('high_confidence', 0.85)
                if has_by_agent:
                    confidence = CONFIDENCE_SCORES.get('by_agent', 0.95)
                
                # Special case for election-related passives - these are almost always true passives
                if participle in ['elected', 'chosen', 'selected', 'appointed', 'voted']:
                    confidence = CONFIDENCE_SCORES.get('election_passive', 0.9)
                    if has_by_agent:
                        confidence = 0.98
                
                # Special case for event-related passives - these are very likely true passives
                if participle in ['held', 'scheduled', 'organized', 'canceled', 'cancelled', 'postponed']:
                    confidence = CONFIDENCE_SCORES.get('event_passive', 0.88)
                    if has_by_agent:
                        confidence = 0.96
                
                # Special case for discussion-related passives - these are very likely true passives
                if participle in ['discussed', 'analyzed', 'examined', 'reviewed']:
                    confidence = 0.87
                    if has_by_agent:
                        confidence = 0.95
                
                passive_phrases.append({
                    'passive_phrase': phrase,
                    'pattern_type': 'high_confidence',
                    'verb': participle,
                    'span': (match.start(), match.end()),
                    'has_by_agent': has_by_agent,
                    'confidence': confidence
                })
    
    # Check each standard pattern type
    for pattern_type, pattern in PASSIVE_PATTERNS.items():
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            phrase = match.group(0)
            
            # Skip if phrase contains a non-passive combination
            if any(non_passive.lower() in phrase.lower() for non_passive in NON_PASSIVE_COMBOS):
                continue
            
            # Skip if the phrase has a personal mental state
            if any(mental_state.lower() in phrase.lower() for mental_state in PERSONAL_MENTAL_STATES):
                continue
            
            # Skip adjectival participles
            participle = match.group(2).lower() if len(match.groups()) > 1 else ""
            if participle in ADJECTIVAL_PARTICIPLES:
                continue
                
            # Check for 'by' preposition which is a strong indicator of passive voice
            has_by_agent = False
            rest_of_sentence = text[match.end():].strip()
            by_match = re.match(r'(\s*by\s+|\s*from\s+|\s*through\s+)', rest_of_sentence)
            if by_match:
                has_by_agent = True
                
            # Skip common false positives unless they have a 'by' agent
            if not has_by_agent:
                # Skip "is topic", "is subject", etc.
                if re.search(r'\b(topic|subject|issue|career|part|benefit|exercise|advantage|example|scene)\b', 
                            text[match.end():match.end() + 30], re.IGNORECASE):
                    continue
                    
                # Skip certain subject forms like "Politics is/are"
                subject_match = re.search(r'\b(politics|science|business|sports|culture)\s+(is|are|was|were)\b', 
                                        text[max(0, match.start() - 20):match.start() + 5], re.IGNORECASE)
                if subject_match:
                    continue
                
                # Skip adjectival uses like "is heated", "is paramount", etc.
                if re.match(r'\b(is|are|was|were)\s+(heated|excited|easy|fun|paramount|full|ready)\b', phrase.lower()):
                    continue
                
                # Skip "is having" constructions (active voice)
                if re.search(r'\bis\s+having\b', phrase.lower()):
                    continue
            
            # Extract base verb (removing -ed, -en, -t endings)
            verb = participle
            if verb.endswith('ed'):
                base_verb = verb[:-2]
            elif verb.endswith('en') or verb.endswith('t'):
                base_verb = verb[:-2]
            else:
                base_verb = verb
            
            # Calculate confidence for this pattern
            confidence = 0.7  # Base confidence
            
            # Increase confidence for by-agent constructions
            if has_by_agent:
                confidence += 0.2
            
            # Perfect passive forms are very reliable
            if pattern_type == 'perfect':
                confidence += 0.1
            
            # Basic forms can be ambiguous (adjectival uses)
            if pattern_type == 'basic' and not has_by_agent:
                confidence -= 0.1
                
            # Get passives are often ambiguous
            if pattern_type == 'get':
                confidence -= 0.1
            
            # Record the passive construction
            passive_phrases.append({
                'passive_phrase': phrase,
                'pattern_type': pattern_type,
                'verb': base_verb,
                'span': (match.start(), match.end()),
                'has_by_agent': has_by_agent,
                'confidence': confidence
            })
    
    # Check for passive expressions
    for expression in PASSIVE_EXPRESSIONS:
        for match in re.finditer(r'\b' + re.escape(expression) + r'\b', text, re.IGNORECASE):
            phrase = match.group(0)
            
            # Record the passive construction
            passive_phrases.append({
                'passive_phrase': phrase,
                'pattern_type': 'expression',
                'verb': expression.split()[0],  # Use first word as verb
                'span': (match.start(), match.end()),
                'has_by_agent': False,
                'confidence': 0.75  # Lower base confidence for expressions
            })
    
    return passive_phrases

def extract_passive_phrases(doc) -> List[Dict[str, Any]]:
    """Extract passive phrases from a spaCy doc"""
    if not doc:
        return []
    
    passive_phrases = []
    text = doc.text
    
    # Use spaCy's dependency parsing to identify passive voice
    for sent in doc.sents:
        for token in sent:
            if is_passive_token(token):
                # Extract the passive phrase
                phrase = extract_passive_phrase(token)
                
                # Skip empty phrases
                if not phrase:
                    continue
                    
                # Check if main verb token is an adjectival participle
                main_verb = token.head if token.dep_ == "auxpass" else token
                if is_adjectival_participle(main_verb):
                    continue
                
                # Record the passive phrase
                passive_phrases.append({
                    'passive_phrase': phrase,
                    'pattern_type': 'dependency',
                    'verb': token.head.lemma_ if token.dep_ == "auxpass" else token.lemma_,
                    'span': (token.idx, token.idx + len(token.text))
                })
    
    # Also check regex patterns
    regex_passives = check_regex_patterns(text)
    
    # Combine results, ensuring no duplicates
    seen_spans = set((p['span'][0], p['span'][1]) for p in passive_phrases)
    
    for regex_passive in regex_passives:
        span = regex_passive['span']
        if (span[0], span[1]) not in seen_spans:
            passive_phrases.append(regex_passive)
            seen_spans.add((span[0], span[1]))
    
    # Additional filtering for common false positives
    filtered_phrases = []
    for phrase_info in passive_phrases:
        phrase = phrase_info.get('passive_phrase', '').lower()
        
        # Skip certain copular constructions with "is" followed by non-passive states
        if re.match(r'\b(is|are|was|were)\s+(just|still|also|now|here|there|about|almost|nearly|already|not|never)\b', phrase):
            continue
            
        # Skip noun phrases that look like passives but aren't (e.g., "politics is topic")
        if re.match(r'\b(politics|science|culture|business|sports)\s+(is|are|was|were)\s+\w+\b', phrase):
            continue
            
        # Skip constructions like "is hot topic", "is great convenience"
        if re.match(r'\b(is|are|was|were)\s+\w+\s+(topic|issue|career|subject|convenience|exercise|benefit|advantage)\b', phrase):
            continue
        
        filtered_phrases.append(phrase_info)
    
    return filtered_phrases

def extract_passive_phrase(token) -> str:
    """Extract the full passive phrase based on a passive token"""
    # If token is auxpass, get the phrase from the token and its head
    if token.dep_ == "auxpass":
        # Get the main verb (head)
        main_verb = token.head
        
        # Get all auxiliary verbs
        aux_verbs = [t for t in main_verb.children if t.dep_ in ("aux", "auxpass")]
        aux_verbs.sort(key=lambda t: t.i)  # Sort by token index
        
        # Get the subject
        subjects = [t for t in main_verb.children if t.dep_ in ("nsubj", "nsubjpass")]
        
        # Get the agent (by-phrase)
        agents = []
        for child in main_verb.children:
            if child.dep_ == "agent":
                for agent_child in child.children:
                    if agent_child.dep_ in ("pobj", "nmod"):
                        agents.append(agent_child)
        
        # Construct the passive phrase
        phrase_parts = []
        
        # Add subject if available
        if subjects:
            subject = min(subjects, key=lambda t: t.i)  # Get leftmost subject
            phrase_parts.append(subject.text)
        
        # Add auxiliary verbs
        phrase_parts.extend([aux.text for aux in aux_verbs])
        
        # Add main verb
        phrase_parts.append(main_verb.text)
        
        # Add agent if available
        if agents:
            agent = min(agents, key=lambda t: t.i)  # Get leftmost agent
            phrase_parts.append("by")
            phrase_parts.append(agent.text)
        
        return " ".join(phrase_parts)
    
    # If token is nsubjpass, get its head and proceeed similarly
    elif token.dep_ == "nsubjpass":
        # Get the head (should be the main verb)
        main_verb = token.head
        
        # Get all auxiliary verbs
        aux_verbs = [t for t in main_verb.children if t.dep_ in ("aux", "auxpass")]
        aux_verbs.sort(key=lambda t: t.i)  # Sort by token index
        
        # Construct the passive phrase
        phrase_parts = [token.text]  # Start with the subject
        
        # Add auxiliary verbs
        phrase_parts.extend([aux.text for aux in aux_verbs])
        
        # Add main verb
        phrase_parts.append(main_verb.text)
        
        return " ".join(phrase_parts)
    
    # For agent tokens, return nothing as we handle them with auxpass
    elif token.dep_ == "agent":
        return ""
    
    # For any other passive-related token, just return the token text
    return token.text

def process_text(text, nlp):
    """Process text with spaCy and extract passive phrases
    
    Args:
        text (str): The text to analyze
        nlp (spacy.Language): spaCy language model
        
    Returns:
        dict: Dictionary containing text, passive detection results, count and phrases
    """
    if not text or not text.strip():
        return {
            "text": text,
            "is_passive": False,
            "passive_count": 0,
            "passive_phrases": [],
            "passive_ratio": 0.0
        }
    
    # Apply spelling correction
    corrected_text = correct_spelling(text)
    
    # Process with spaCy
    doc = nlp(corrected_text)
    
    # Extract passive phrases using our existing function
    passive_phrases_info = extract_passive_phrases(doc)
    
    # Sort phrases by confidence (phrases with 'by' agents first)
    passive_phrases_info.sort(key=lambda x: x.get('has_by_agent', False), reverse=True)
    
    # Calculate confidence score for each phrase
    for phrase_info in passive_phrases_info:
        # Skip if confidence is already set (e.g., by high_confidence patterns)
        if 'confidence' in phrase_info:
            continue
            
        # Start with base confidence
        confidence = 0.7
        
        # Phrases with "by" agents are more likely to be true passives
        if phrase_info.get('has_by_agent', False):
            confidence += 0.2
        
        # Dependency-parsed phrases are generally more reliable than regex
        if phrase_info.get('pattern_type') == 'dependency':
            confidence += 0.1
            
        # Perfect passives (have been + participle) are strong indicators
        if phrase_info.get('pattern_type') == 'perfect':
            confidence += 0.1
            
        # Modal passives can sometimes be ambiguous
        if phrase_info.get('pattern_type') == 'modal':
            confidence -= 0.05
            
        # Get passives can be ambiguous
        if phrase_info.get('pattern_type') == 'get':
            confidence -= 0.1
            
        # Cap confidence at 1.0
        phrase_info['confidence'] = min(1.0, confidence)
    
    # Additional filtering for high precision
    high_confidence_phrases = [p for p in passive_phrases_info if p.get('confidence', 0) >= 0.75]
    
    # SPECIAL CASE HANDLING for commonly missed true passives
    text_lower = text.lower()
    
    # 1. Check for election-related passives (very common misses)
    election_match = re.search(SPECIAL_PASSIVE_PATTERNS['election'], text, re.IGNORECASE)
    if election_match and not any(p['passive_phrase'].lower() == election_match.group(0).lower() for p in high_confidence_phrases):
        high_confidence_phrases.append({
            'passive_phrase': election_match.group(0),
            'pattern_type': 'election_passive',
            'verb': election_match.group(2),
            'span': (election_match.start(), election_match.end()),
            'has_by_agent': 'by' in text_lower[election_match.end():election_match.end()+15],
            'confidence': CONFIDENCE_SCORES.get('election_passive', 0.95)
        })
    
    # 2. Check for event-related passives
    event_match = re.search(SPECIAL_PASSIVE_PATTERNS['event'], text, re.IGNORECASE)
    if event_match and not any(p['passive_phrase'].lower() == event_match.group(0).lower() for p in high_confidence_phrases):
        high_confidence_phrases.append({
            'passive_phrase': event_match.group(0),
            'pattern_type': 'event_passive',
            'verb': event_match.group(2),
            'span': (event_match.start(), event_match.end()),
            'has_by_agent': 'by' in text_lower[event_match.end():event_match.end()+15],
            'confidence': CONFIDENCE_SCORES.get('event_passive', 0.90)
        })
    
    # 3. Check for ANY past participle + by agent (extremely reliable passive indicators)
    by_agent_match = re.search(SPECIAL_PASSIVE_PATTERNS['by_agent'], text, re.IGNORECASE)
    if by_agent_match and not any(p['passive_phrase'].lower() == by_agent_match.group(0).lower() for p in high_confidence_phrases):
        high_confidence_phrases.append({
            'passive_phrase': by_agent_match.group(0),
            'pattern_type': 'by_agent_passive',
            'verb': by_agent_match.group(2),
            'span': (by_agent_match.start(), by_agent_match.end()),
            'has_by_agent': True,
            'confidence': CONFIDENCE_SCORES.get('by_agent', 0.95)
        })
        
    # 4. FIXED: Check for negated passives (isn't/aren't/wasn't/weren't + participle)
    negated_match = re.search(SPECIAL_PASSIVE_PATTERNS['negated'], text, re.IGNORECASE)
    if negated_match and not any(p['passive_phrase'].lower() == negated_match.group(0).lower() for p in high_confidence_phrases):
        verb = negated_match.group(2)
        # Skip if it's an adjectival participle
        if verb.lower() not in ADJECTIVAL_PARTICIPLES:
            # Check for 'by' agent
            has_by_agent = 'by' in text_lower[negated_match.end():negated_match.end()+15]
            confidence = 0.85
            if has_by_agent:
                confidence = CONFIDENCE_SCORES.get('by_agent', 0.95)
                
            high_confidence_phrases.append({
                'passive_phrase': negated_match.group(0),
                'pattern_type': 'negated_passive',
                'verb': verb,
                'span': (negated_match.start(), negated_match.end()),
                'has_by_agent': has_by_agent,
                'confidence': confidence
            })
    
    # 5. FIXED: Check for "left + participle" which is often a passive marker
    left_match = re.search(SPECIAL_PASSIVE_PATTERNS['left'], text, re.IGNORECASE)
    if left_match and not any(p['passive_phrase'].lower() == left_match.group(0).lower() for p in high_confidence_phrases):
        high_confidence_phrases.append({
            'passive_phrase': left_match.group(0),
            'pattern_type': 'left_passive',
            'verb': left_match.group(2),
            'span': (left_match.start(), left_match.end()),
            'has_by_agent': 'by' in text_lower[left_match.end():left_match.end()+15],
            'confidence': 0.90
        })
    
    # 6. FIXED: Handle simple passives without auxiliaries like "Sports are played."
    simple_passive_match = re.search(SPECIAL_PASSIVE_PATTERNS['simple'], text, re.IGNORECASE)
    if simple_passive_match and not any(p['passive_phrase'].lower() == simple_passive_match.group(0).lower() for p in high_confidence_phrases):
        verb = simple_passive_match.group(3)
        # Skip if it's an adjectival participle
        if verb.lower() not in ADJECTIVAL_PARTICIPLES:
            high_confidence_phrases.append({
                'passive_phrase': simple_passive_match.group(0),
                'pattern_type': 'simple_passive',
                'verb': verb,
                'span': (simple_passive_match.start(), simple_passive_match.end()),
                'has_by_agent': 'by' in text_lower[simple_passive_match.end():simple_passive_match.end()+15],
                'confidence': 0.85
            })
    
    # ENHANCED: Remove common false positives
    filtered_phrases = []
    for phrase_info in high_confidence_phrases:
        phrase = phrase_info.get('passive_phrase', '').lower()
        
        # Skip if it matches any of our false positive patterns (unless it has a by-agent)
        if not phrase_info.get('has_by_agent', False):
            skip = False
            for pattern in FALSE_POSITIVE_PATTERNS:
                if re.search(pattern, phrase, re.IGNORECASE):
                    skip = True
                    break
            if skip:
                continue
        
        # Otherwise, keep the phrase
        filtered_phrases.append(phrase_info)
    
    # Calculate passive ratio (passive phrases / total tokens)
    total_tokens = len(doc)
    passive_ratio = len(filtered_phrases) / total_tokens if total_tokens > 0 else 0
    
    # Process the passive phrases to include additional details
    processed_phrases = []
    for phrase_info in filtered_phrases:
        # Extract main verb and lemmatize it if available
        main_verb = phrase_info.get('verb', '')
        lemmatized_verb = main_verb
        
        # Check if there's an explicit subject
        has_subject = False
        if phrase_info.get('pattern_type') == 'dependency':
            # For dependency-parsed phrases, check for subject in the phrase
            phrase_parts = phrase_info.get('passive_phrase', '').split()
            has_subject = len(phrase_parts) > 2
        
        # Check if it's an infinitive passive
        is_infinitive = 'to be' in phrase_info.get('passive_phrase', '').lower()
        
        # Check if it has a modal
        has_modal = any(modal in phrase_info.get('passive_phrase', '').lower() 
                       for modal in ['will', 'would', 'shall', 'should', 'may', 'might', 'must', 'can', 'could'])
        
        # Get part-of-speech tags for the phrase if it's from dependency parsing
        pos_tagged = ''
        simplified_pos = ''
        
        if phrase_info.get('pattern_type') == 'dependency':
            # Try to find the phrase in the original doc to get POS tags
            phrase_text = phrase_info.get('passive_phrase', '')
            for sent in doc.sents:
                if phrase_text in sent.text:
                    # Get POS tags for this phrase
                    phrase_doc = nlp(phrase_text)
                    pos_tagged = ' '.join([f"{token.text}/{token.tag_}" for token in phrase_doc])
                    simplified_pos = ' '.join([token.pos_ for token in phrase_doc])
                    break
        
        # Add processed information
        processed_phrase = {
            'passive_phrase': phrase_info.get('passive_phrase', ''),
            'pattern_type': phrase_info.get('pattern_type', ''),
            'Lemmatized_Main_Verb': lemmatized_verb,
            'POS_Tagged_Phrase': pos_tagged,
            'Simplified_POS': simplified_pos,
            'Has_Subject': has_subject,
            'Is_Infinitive': is_infinitive,
            'Has_Modal': has_modal,
            'Has_By_Agent': phrase_info.get('has_by_agent', False),
            'confidence': phrase_info.get('confidence', 0.0),
            'lemmatized_phrase': phrase_info.get('passive_phrase', '').lower(),  # Simplified lemmatization
            'span': phrase_info.get('span', (0, 0))
        }
        processed_phrases.append(processed_phrase)
    
    # For better compatibility with different use cases
    is_passive = len(processed_phrases) > 0
    
    return {
        "text": text,
        "is_passive": is_passive,
        "passive_count": len(processed_phrases),
        "passive_phrases": processed_phrases,
        "passive_ratio": passive_ratio
    }

def process_texts(texts_with_context, nlp):
    """Process texts with contexts using nlp.pipe with a progress bar"""
    corrected_texts_with_context = [
        (correct_spelling(text), context) for text, context in texts_with_context
    ]
    
    results = []
    # Wrap the nlp.pipe generator with tqdm to display a progress bar
    for doc, context in tqdm(nlp.pipe(corrected_texts_with_context, as_tuples=True),
                             total=len(corrected_texts_with_context),
                             desc="Processing texts"):
        native_language = context['Native_Language']
        doc_id = context['file_name']
        passive_phrases = extract_passive_phrases(doc)
        for entry in passive_phrases:
            results.append({
                "DocID": doc_id,
                "Native_Language": native_language,
                "Passive_Phrase": entry['passive_phrase'],
                "Lemmatized_Phrase": entry['lemmatized_phrase'],
                "POS_Tagged_Phrase": entry['POS_Tagged_Phrase'],
                "Simplified_POS": entry['Simplified_POS'],
                "Lemmatized_Main_Verb": entry['Lemmatized_Main_Verb'],
                "Has_Subject": entry['Has_Subject'],
                "Is_Infinitive": entry['Is_Infinitive'],
                "Has_Modal": entry['Has_Modal']
            })
    
    return pd.DataFrame(results)

def convert_to_serializable(obj):
    """Convert numpy types to Python native types"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
        np.int16, np.int32, np.int64, np.uint8,
        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float16, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, Counter):
        return {str(k): int(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def analyze_passive_patterns(df):
    """Analyze patterns in passive voice usage"""
    # Analyze by native language
    language_stats = df.groupby('Native_Language').agg({
        'DocID': 'count',
        'Passive_Phrase': 'count',
        'Has_Subject': 'mean',
        'Is_Infinitive': 'mean',
        'Has_Modal': 'mean'
    }).round(3)
    
    # Analyze POS patterns
    pos_patterns = Counter(df['POS_Tagged_Phrase'])
    common_patterns = pd.DataFrame([
        {'Pattern': k, 'Count': v, 'Percentage': v/len(df)*100}
        for k, v in pos_patterns.most_common()
    ])
    
    # Analyze verb usage
    verb_usage = Counter(df['Lemmatized_Main_Verb'])
    common_verbs = pd.DataFrame([
        {'Verb': k, 'Count': v, 'Percentage': v/len(df)*100}
        for k, v in verb_usage.most_common()
    ])
    
    return {
        'language_stats': language_stats,
        'pos_patterns': common_patterns,
        'verb_usage': common_verbs
    }

def visualize_patterns(analysis_results, output_prefix='passive_analysis'):
    """Create visualizations of passive voice patterns"""
    # 1. Language-specific patterns
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Passive phrases by language
    plt.subplot(2, 2, 1)
    analysis_results['language_stats']['Passive_Phrase'].plot(kind='bar')
    plt.title('Passive Phrases by Native Language')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Plot 2: Top POS patterns
    plt.subplot(2, 2, 2)
    top_patterns = analysis_results['pos_patterns'].head(10)
    sns.barplot(data=top_patterns, x='Count', y='Pattern')
    plt.title('Top 10 POS Patterns')
    
    # Plot 3: Modal vs Non-modal usage by language
    plt.subplot(2, 2, 3)
    analysis_results['language_stats']['Has_Modal'].plot(kind='bar')
    plt.title('Modal Usage in Passives by Language')
    plt.xticks(rotation=45)
    
    # Plot 4: Top verbs in passive constructions
    plt.subplot(2, 2, 4)
    top_verbs = analysis_results['verb_usage'].head(10)
    sns.barplot(data=top_verbs, x='Count', y='Verb')
    plt.title('Top 10 Verbs in Passive Constructions')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_patterns.png')
    plt.close()

def main():
    """Main execution function"""
    try:
        print("Loading spaCy transformer model...")
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        print("Error: spaCy model 'en_core_web_trf' not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_trf"])
        nlp = spacy.load("en_core_web_trf")
    
    print("\nLoading dataset...")
    try:
        df = pd.read_csv('/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/metadata_with_text.csv')
    except FileNotFoundError:
        print("Error: sampled_metadata.csv not found in current directory")
        return
    
    # Prepare texts with context for batch processing
    texts_with_context = [
        (row['text_field'], {'Native_Language': row['Native_Language'], 'file_name': row['file_name']})
        for _, row in df.iterrows()
        if pd.notna(row['text_field']) and row['text_field'].strip()
    ]
    
    if not texts_with_context:
        print("Error: No valid texts found in the dataset")
        return
    
    print(f"Processing {len(texts_with_context)} texts...")
    results_df = process_texts(texts_with_context, nlp)
    
    if len(results_df) == 0:
        print("Warning: No passive phrases detected in the dataset")
        return
    
    # Save detailed results
    try:
        results_df.to_csv('passive_detection_results.csv', index=False)
        print("\nDetailed results saved to 'passive_detection_results.csv'")
    except Exception as e:
        print(f"Warning: Could not save results to CSV: {str(e)}")
    
    # Analyze patterns
    print("\nAnalyzing passive voice patterns...")
    analysis_results = analyze_passive_patterns(results_df)
    
    # Create visualizations
    print("Generating visualizations...")
    try:
        visualize_patterns(analysis_results)
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {str(e)}")
    
    # Generate summary statistics
    stats = {
        'total_texts': len(texts_with_context),
        'total_passive_phrases': len(results_df),
        'avg_phrases_per_text': len(results_df) / len(texts_with_context),
        'language_stats': analysis_results['language_stats'].to_dict(),
        'top_patterns': analysis_results['pos_patterns'].head(10).to_dict('records'),
        'top_verbs': analysis_results['verb_usage'].head(10).to_dict('records')
    }
    
    try:
        with open('passive_detection_summary.json', 'w') as f:
            json.dump(convert_to_serializable(stats), f, indent=2)
        print("Summary statistics saved to 'passive_detection_summary.json'")
    except Exception as e:
        print(f"Warning: Could not save summary statistics: {str(e)}")

if __name__ == "__main__":
    main() 