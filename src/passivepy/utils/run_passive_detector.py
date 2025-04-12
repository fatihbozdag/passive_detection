#!/usr/bin/env python3
"""
Passive Voice Detector - Command Line Tool

This script allows users to analyze text for passive voice constructions
using the custom passive voice detector.

Usage:
  python run_passive_detector.py --text "The book was written by the author."
  python run_passive_detector.py --file input.txt
  python run_passive_detector.py --file input.txt --output results.csv
"""

import argparse
import spacy
import pandas as pd
import sys
import os
from tqdm import tqdm

import my_passive_detector as detector

def setup_argparse():
    """Configure argument parser"""
    parser = argparse.ArgumentParser(description="Analyze text for passive voice constructions")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Text to analyze directly')
    input_group.add_argument('--file', type=str, help='Path to text file to analyze')
    
    parser.add_argument('--output', type=str, help='Path to save CSV output (optional)')
    parser.add_argument('--model', type=str, default='en_core_web_sm', 
                       help='SpaCy model to use (default: en_core_web_sm)')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed information about passive phrases')
    
    return parser

def load_spacy_model(model_name):
    """Load spaCy model, handling errors gracefully"""
    try:
        print(f"Loading {model_name} model...")
        return spacy.load(model_name)
    except OSError:
        print(f"Error: Model '{model_name}' not found. Trying to download...")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            return spacy.load(model_name)
        except Exception as e:
            print(f"Failed to download model: {str(e)}")
            print("Falling back to en_core_web_sm...")
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                return spacy.load("en_core_web_sm")
            except Exception as e2:
                print(f"Failed to load fallback model: {str(e2)}")
                sys.exit(1)

def process_text(text, nlp):
    """Process text with spaCy and extract passive phrases"""
    doc = nlp(text)
    passive_phrases = detector.extract_passive_phrases(doc)
    
    return {
        "text": text,
        "is_passive": bool(passive_phrases),
        "passive_count": len(passive_phrases),
        "passive_phrases": passive_phrases
    }

def analyze_file(file_path, nlp):
    """Process a text file line by line"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        results = []
        for line in tqdm(lines, desc="Processing sentences"):
            result = process_text(line, nlp)
            results.append(result)
        
        return results
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []

def format_output(results, detailed=False):
    """Format results for display and CSV output"""
    # Create a DataFrame for the results
    df = pd.DataFrame({
        'Text': [r['text'] for r in results],
        'Is_Passive': [r['is_passive'] for r in results],
        'Passive_Count': [r['passive_count'] for r in results]
    })
    
    # Add detailed columns if requested
    if detailed:
        df['Passive_Phrases'] = ['; '.join([p['passive_phrase'] for p in r['passive_phrases']]) 
                               if r['passive_phrases'] else '' for r in results]
        df['Main_Verbs'] = ['; '.join([p['Lemmatized_Main_Verb'] for p in r['passive_phrases']]) 
                          if r['passive_phrases'] else '' for r in results]
    
    return df

def display_results(results, detailed=False):
    """Display results in the console"""
    df = format_output(results, detailed)
    
    # Print summary
    total = len(results)
    passive_count = sum(1 for r in results if r['is_passive'])
    passive_percent = (passive_count / total * 100) if total > 0 else 0
    
    print("\n===== PASSIVE VOICE ANALYSIS SUMMARY =====")
    print(f"Total sentences analyzed: {total}")
    print(f"Sentences with passive voice: {passive_count} ({passive_percent:.1f}%)")
    
    # Print table
    print("\n===== DETAILED RESULTS =====")
    if detailed:
        pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))
    
    return df

def save_results(df, output_path):
    """Save results to CSV file"""
    try:
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    """Main execution function"""
    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load spaCy model
    nlp = load_spacy_model(args.model)
    
    # Process input
    if args.text:
        # Parse the input text and split it into sentences
        doc = nlp(args.text)
        results = []
        for sent in doc.sents:
            results.append(process_text(sent.text, nlp))
    else:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        results = analyze_file(args.file, nlp)
    
    # Display results
    df = display_results(results, args.detailed)
    
    # Save to CSV if requested
    if args.output:
        save_results(df, args.output)

if __name__ == "__main__":
    main() 