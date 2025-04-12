#!/usr/bin/env python3
"""
Simple PassivePy Detection

A simplified implementation that mimics PassivePy's detection logic 
but works reliably with the ICLE dataset.
"""

import spacy
import re
import torch
from typing import List, Dict, Any, Optional
import pandas as pd

# Configure GPU settings
print("Setting up GPU acceleration for Simple PassivePy...")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device('cpu')
    print("Warning: Using CPU. No GPU acceleration available.")

# Enable GPU for spaCy if available
spacy.prefer_gpu()

class SimplePassivePyDetector:
    """A simplified version of PassivePy that detects basic passive voice patterns"""
    
    def __init__(self, spacy_model: str = 'en_core_web_sm'):
        """Initialize the detector with a spaCy model"""
        self.nlp = spacy.load(spacy_model)
        
        # Basic patterns for passive voice
        self.patterns = {
            'basic': r'\b(am|is|are|was|were|be|been|being)\s+(\w+ed|\w+en|\w+t)\b',
            'perfect': r'\b(have|has|had)\s+been\s+(\w+ed|\w+en|\w+t)\b',
            'modal': r'\b(will|would|shall|should|may|might|must|can|could)\s+be\s+(\w+ed|\w+en|\w+t)\b',
            'get': r'\b(get|gets|got|gotten)\s+(\w+ed|\w+en|\w+t)\b',
        }
        
        # Non-passive combinations (adjectival uses)
        self.non_passive_combos = [
            'is able', 'was able', 'are able', 'were able',
            'is about', 'was about', 'are about', 'were about',
            'is available', 'was available', 'are available', 'were available',
            'is possible', 'was possible', 'are possible', 'were possible',
        ]
        
        # Common adjectival participles (not passive)
        self.adjectival_participles = [
            'advanced', 'animated', 'balanced', 'calculated', 
            'committed', 'complicated', 'concentrated', 'controlled',
            'dedicated', 'educated', 'experienced', 'motivated',
            'organized', 'qualified', 'sophisticated', 'structured'
        ]
    
    def detect_passive(self, text: str) -> Dict[str, Any]:
        """
        Detect passive voice in text using both dependency parsing and regex
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict with passive voice analysis
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        passive_phrases = []
        
        # Method 1: Use dependency parsing for detection
        for sent in doc.sents:
            for token in sent:
                # Check for passive auxiliary
                if token.dep_ == "auxpass":
                    # Find the main verb (head)
                    verb = token.head
                    
                    # Skip adjectival uses
                    if any(t.dep_ == "amod" for t in verb.children):
                        continue
                    
                    # Extract the full passive construction
                    span_tokens = []
                    
                    # Add auxiliaries
                    for aux in verb.children:
                        if aux.dep_ in ["aux", "auxpass"]:
                            span_tokens.append(aux)
                    
                    # Add the main verb
                    span_tokens.append(verb)
                    
                    # Sort by position in sentence
                    span_tokens.sort(key=lambda t: t.i)
                    
                    if span_tokens:
                        start_idx = span_tokens[0].i
                        end_idx = span_tokens[-1].i + 1
                        span_text = doc[start_idx:end_idx].text
                        
                        # Check if it's not a common adjectival use
                        if not any(combo in span_text.lower() for combo in self.non_passive_combos):
                            # Check for 'by' agent
                            has_by_agent = False
                            for token in verb.children:
                                if token.lower_ == "by" and token.dep_ == "agent":
                                    has_by_agent = True
                                    break
                            
                            passive_phrases.append({
                                "span": span_text,
                                "pattern_type": "dependency",
                                "start": start_idx,
                                "end": end_idx,
                                "has_by_agent": has_by_agent,
                                "confidence": 0.9
                            })
        
        # Method 2: Use regex patterns for detection
        for pattern_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                span_text = match.group(0)
                
                # Skip if it's a common non-passive combination
                if any(combo in span_text.lower() for combo in self.non_passive_combos):
                    continue
                
                # Skip if the participle is a common adjectival participle
                participle = match.group(2).lower()
                if participle in self.adjectival_participles:
                    continue
                
                # Check for 'by' agent within 15 characters after the match
                has_by_agent = ' by ' in text[match.end():match.end()+15].lower()
                
                # Add the match to our results
                passive_phrases.append({
                    "span": span_text,
                    "pattern_type": pattern_type,
                    "start": match.start(),
                    "end": match.end(),
                    "has_by_agent": has_by_agent,
                    "confidence": 0.8
                })
        
        # Remove duplicates and create the result
        unique_phrases = []
        seen_spans = set()
        
        for phrase in passive_phrases:
            span = phrase["span"]
            if span not in seen_spans:
                seen_spans.add(span)
                unique_phrases.append(phrase)
        
        return {
            "text": text,
            "passive": unique_phrases,
            "is_passive": len(unique_phrases) > 0,
            "passive_count": len(unique_phrases)
        }
    
    def process_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts and return passive analysis for each"""
        return [self.detect_passive(text) for text in texts]
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Process a dataframe with text data and add passive detection results
        
        Args:
            df: DataFrame containing texts to analyze
            text_column: Name of the column containing the text
            
        Returns:
            DataFrame with passive detection results added
        """
        # Create a copy of the input dataframe
        result_df = df.copy()
        
        # Process each text and collect the results
        passive_results = []
        for text in df[text_column]:
            passive_results.append(self.detect_passive(text))
        
        # Add the results to the dataframe
        result_df['passive_analysis'] = passive_results
        result_df['is_passive'] = [r['is_passive'] for r in passive_results]
        result_df['passive_count'] = [r['passive_count'] for r in passive_results]
        result_df['passive_phrases'] = [r['passive'] for r in passive_results]
        
        return result_df


def main():
    """Demo of SimplePassivePyDetector"""
    # Sample sentences from ICLE
    samples = [
        "The process is carried out in three major stages.",
        "Such confused graduates have not been thought of in Bulgaria.",
        "If smoking is forbidden in restaurant, these innocent people can have a better health status.",
        "By the way, if the countryside is being developed, there will be nowhere for Hong Kong citizens to go.",
        "I would recommend that stricter laws should be introduced to protect the right of local people.",
        "The room is blue and very large.",  # Not passive
    ]
    
    # Initialize the detector
    detector = SimplePassivePyDetector()
    
    # Process the samples
    for sample in samples:
        result = detector.detect_passive(sample)
        print(f"\nText: {sample}")
        print(f"Is Passive: {result['is_passive']}")
        print(f"Passive Count: {result['passive_count']}")
        
        if result['passive']:
            print("Passive Phrases:")
            for i, phrase in enumerate(result['passive']):
                print(f"  {i+1}. {phrase['span']} (Type: {phrase['pattern_type']}, " + 
                      f"Has By Agent: {phrase['has_by_agent']})")
    
    print("\nProcessing with DataFrame API:")
    df = pd.DataFrame({'text': samples})
    results_df = detector.process_dataframe(df, 'text')
    print(results_df[['text', 'is_passive', 'passive_count']])

if __name__ == "__main__":
    main() 