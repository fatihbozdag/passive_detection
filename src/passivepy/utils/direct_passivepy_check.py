#!/usr/bin/env python3
"""
Direct PassivePy check on ICLE dataset

This script directly uses PassivePy on a small sample of ICLE sentences
to debug the passive voice detection issue.
"""

import sys
import os
import pandas as pd
import spacy
from tqdm import tqdm

# Add PassivePy to path and import
sys.path.append('PassivePyCode/PassivePySrc')
from PassivePy import PassivePyAnalyzer

def main():
    print("Loading ICLE data...")
    # Read the CSV file
    df = pd.read_csv('icle_concord.csv')
    
    # Combine Left, Center, Right to form complete sentences
    df['sentence'] = df['Left'].fillna('') + ' ' + df['Center'].fillna('') + ' ' + df['Right'].fillna('')
    df['sentence'] = df['sentence'].str.strip()
    
    # Take a small sample for testing
    sample_df = df.sample(n=20, random_state=42)
    
    print(f"Loaded {len(sample_df)} sample sentences from ICLE")
    
    # Initialize PassivePy
    print("Initializing PassivePy...")
    passivepy = PassivePyAnalyzer(spacy_model='en_core_web_sm')
    
    # Process each sentence individually
    results = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing with PassivePy"):
        text = row['sentence']
        
        print(f"\nSentence {idx}: {text}")
        
        try:
            # Try using match_text directly
            matches = passivepy.match_text(text)
            if 'passive' in matches:
                passive_matches = matches['passive']
                print(f"  - match_text found {len(passive_matches)} passives")
                for i, p in enumerate(passive_matches):
                    if 'span' in p:
                        print(f"    {i+1}. {p['span']}")
                    else:
                        print(f"    {i+1}. {p}")
            else:
                print("  - match_text found no passives")
                
            # Create a small dataframe for this sentence
            mini_df = pd.DataFrame([{'text': text}])
            
            # Try using match_sentence_level
            try:
                sentence_results = passivepy.match_sentence_level(mini_df, 'text')
                if 'passive' in sentence_results.columns:
                    passive_list = sentence_results.iloc[0]['passive']
                    if isinstance(passive_list, list):
                        print(f"  - match_sentence_level found {len(passive_list)} passives")
                        for i, p in enumerate(passive_list):
                            if isinstance(p, dict) and 'span' in p:
                                print(f"    {i+1}. {p['span']}")
                            else:
                                print(f"    {i+1}. {p}")
                    else:
                        print(f"  - match_sentence_level found no passives (passive_list is {type(passive_list)})")
                else:
                    print("  - match_sentence_level found no passives (no 'passive' column)")
            except Exception as e:
                print(f"  - match_sentence_level error: {e}")
                
            # Try manual passive detection
            print("  - Manual check for passive patterns:")
            doc = passivepy.nlp(text)
            
            # Log all verbs
            verbs = [token for token in doc if token.pos_ == "VERB" or token.pos_ == "AUX"]
            print(f"    Found {len(verbs)} verbs/auxiliaries: {', '.join([v.text for v in verbs])}")
            
            # Check for be + past participle
            for token in doc:
                if token.lemma_ == "be" and token.i + 1 < len(doc):
                    next_token = doc[token.i + 1]
                    if next_token.tag_ == "VBN":
                        print(f"    PASSIVE FOUND: '{token.text} {next_token.text}'")
                
            # Add to results
            results.append({
                'text': text,
                'match_text_passives': len(matches.get('passive', [])),
                'manual_check': "See console output"
            })
                
        except Exception as e:
            print(f"  - Error: {e}")
            results.append({
                'text': text,
                'match_text_passives': 0,
                'error': str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('passivepy_direct_check_results.csv', index=False)
    print("\nResults saved to passivepy_direct_check_results.csv")
    
    # Summary
    print(f"\nProcessed {len(results)} sentences")
    print(f"Found passives in {sum(results_df['match_text_passives'] > 0)} sentences using match_text")

if __name__ == "__main__":
    main() 