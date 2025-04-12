#!/usr/bin/env python3
"""
Compare PassivePy and Custom Passive Detector on ICLE Dataset

This script processes the ICLE dataset with both PassivePy and the custom
passive detector, then compares their results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
import os
import json
import torch
import spacy
from collections import Counter

# Configure GPU settings
print("Setting up GPU acceleration...")
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

# Try to import PassivePy
try:
    sys.path.append('PassivePyCode/PassivePySrc')
    from PassivePy import PassivePyAnalyzer
except ImportError:
    print("Error: PassivePy module not found. Make sure it's installed correctly.")
    sys.exit(1)

# Import custom detector
from my_passive_detector import process_text

# Initialize PassivePy with the correct parameters
passivepy = PassivePyAnalyzer(spacy_model='en_core_web_sm')

def analyze_with_passivepy(df):
    """Analyze text with PassivePy"""
    print("Analyzing with PassivePy...")
    
    # Create a dataframe with just the text column for PassivePy
    text_df = pd.DataFrame({'text': df['sentence']})
    
    # Process with batch_size=1 to avoid memory issues
    try:
        # Use the correct API to process at sentence level
        results_df = passivepy.match_sentence_level(text_df, 'text', batch_size=1, n_process=1)
        print(f"PassivePy processed {len(results_df)} sentences")
        
        # Join with original dataframe to preserve metadata
        results_df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
        
        # Extract is_passive flag and count from 'passive' column
        passivepy_results = []
        
        for idx, row in results_df.iterrows():
            text = row['sentence']
            file_name = row['File name'] if 'File name' in row else f"doc_{idx}"
            native_language = row['Native language'] if 'Native language' in row else "Unknown"
            
            # Check for passive voice
            is_passive = False
            passive_count = 0
            passive_phrases = []
            
            if 'passive' in row and isinstance(row['passive'], list):
                passive_list = row['passive']
                is_passive = len(passive_list) > 0
                passive_count = len(passive_list)
                passive_phrases = [p.get('text', '') for p in passive_list]
            
            # Calculate passive ratio
            total_words = len(text.split())
            passive_ratio = passive_count / total_words if total_words > 0 else 0
            
            # Create new row with passivepy results
            passivepy_results.append({
                'text': text,
                'file_name': file_name,
                'Native_Language': native_language,
                'is_passive_passivepy': is_passive,
                'passive_count_passivepy': passive_count,
                'passive_ratio_passivepy': passive_ratio,
                'passive_phrases_passivepy': passive_phrases
            })
        
        return pd.DataFrame(passivepy_results)
        
    except Exception as e:
        print(f"Error using PassivePy.match_sentence_level: {e}")
        print("Falling back to manual text processing...")
        
        # If the batch processing failed, process each sentence individually
        passivepy_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing with PassivePy"):
            text = row['sentence']
            file_name = row['File name'] if 'File name' in row else f"doc_{idx}"
            native_language = row['Native language'] if 'Native language' in row else "Unknown"
            
            try:
                # Process with PassivePy's direct API
                passive_analysis = passivepy.match_text(text)
                
                # Extract passive information
                is_passive = False
                passive_count = 0
                passive_phrases = []
                
                if isinstance(passive_analysis, dict) and 'passive' in passive_analysis:
                    passive_list = passive_analysis['passive']
                    is_passive = len(passive_list) > 0
                    passive_count = len(passive_list)
                    passive_phrases = [p.get('text', '') for p in passive_list]
                
                # Calculate passive ratio
                total_words = len(text.split())
                passive_ratio = passive_count / total_words if total_words > 0 else 0
                
                passivepy_results.append({
                    'text': text,
                    'file_name': file_name,
                    'Native_Language': native_language,
                    'is_passive_passivepy': is_passive,
                    'passive_count_passivepy': passive_count,
                    'passive_ratio_passivepy': passive_ratio,
                    'passive_phrases_passivepy': passive_phrases
                })
            except Exception as e:
                print(f"Error processing with PassivePy: {e}")
                # Add empty results
                passivepy_results.append({
                    'text': text,
                    'file_name': file_name,
                    'Native_Language': native_language,
                    'is_passive_passivepy': False,
                    'passive_count_passivepy': 0,
                    'passive_ratio_passivepy': 0,
                    'passive_phrases_passivepy': []
                })
        
        return pd.DataFrame(passivepy_results)

def load_icle_data():
    """Load ICLE concordance data"""
    print("Loading ICLE data...")
    try:
        # Read the CSV file
        df = pd.read_csv('icle_concord.csv')
        
        # We need to combine Left, Center, Right to form complete sentences
        df['sentence'] = df['Left'].fillna('') + ' ' + df['Center'].fillna('') + ' ' + df['Right'].fillna('')
        df['sentence'] = df['sentence'].str.strip()
        
        # Take a sample for faster processing if needed
        # df = df.sample(n=1000, random_state=42)
        
        print(f"Loaded {len(df)} sentences from ICLE data")
        return df
    except Exception as e:
        print(f"Error loading ICLE data: {e}")
        return None

def analyze_with_custom(df):
    """Analyze text with custom passive detector"""
    print("Analyzing with custom detector...")
    
    # Import spaCy for custom detector
    import spacy
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Initialize results list
    results = []
    
    # Process each sentence
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing with custom detector"):
        text = row['text']  # Use text from PassivePy results
        file_name = row['file_name']
        native_language = row['Native_Language']
        
        # Process with custom detector
        try:
            custom_results = process_text(text, nlp)
            
            # Extract results
            is_passive = custom_results.get('is_passive', False)
            passive_count = custom_results.get('passive_count', 0)
            passive_ratio = custom_results.get('passive_ratio', 0)
            passive_phrases = [p.get('passive_phrase', '') for p in custom_results.get('passive_phrases', [])]
            
            # Check if any passive has by-agent
            has_by_agents = any(p.get('Has_By_Agent', False) for p in custom_results.get('passive_phrases', []))
            
            # Get pattern types
            pattern_types = [p.get('pattern_type', '') for p in custom_results.get('passive_phrases', [])]
            pattern_types_str = ';'.join(pattern_types) if pattern_types else ''
            
            row_results = {
                'is_passive_custom': is_passive,
                'passive_count_custom': passive_count,
                'passive_ratio_custom': passive_ratio,
                'passive_phrases_custom': passive_phrases,
                'has_by_agents': has_by_agents,
                'pattern_types': pattern_types_str
            }
            
            # Update the row with custom results
            results.append({**row, **row_results})
        except Exception as e:
            print(f"Error processing with custom detector: {e}")
            # Add empty results
            results.append({
                **row,
                'is_passive_custom': False,
                'passive_count_custom': 0,
                'passive_ratio_custom': 0,
                'passive_phrases_custom': [],
                'has_by_agents': False,
                'pattern_types': ''
            })
    
    return pd.DataFrame(results)

def calculate_metrics(df):
    """Calculate comparison metrics between PassivePy and custom detector"""
    print("Calculating comparison metrics...")
    
    total_sentences = len(df)
    
    # PassivePy metrics
    passivepy_passive_sentences = sum(df['is_passive_passivepy'])
    passivepy_passive_percentage = (passivepy_passive_sentences / total_sentences) * 100
    passivepy_passive_count_avg = df[df['is_passive_passivepy']]['passive_count_passivepy'].mean() if passivepy_passive_sentences > 0 else 0
    passivepy_passive_ratio_avg = df[df['is_passive_passivepy']]['passive_ratio_passivepy'].mean() if passivepy_passive_sentences > 0 else 0
    
    # Custom detector metrics
    custom_passive_sentences = sum(df['is_passive_custom'])
    custom_passive_percentage = (custom_passive_sentences / total_sentences) * 100
    custom_passive_count_avg = df[df['is_passive_custom']]['passive_count_custom'].mean() if custom_passive_sentences > 0 else 0
    custom_passive_ratio_avg = df[df['is_passive_custom']]['passive_ratio_custom'].mean() if custom_passive_sentences > 0 else 0
    custom_by_agent_count = sum(df['has_by_agents'])
    custom_by_agent_percentage = (custom_by_agent_count / custom_passive_sentences) * 100 if custom_passive_sentences > 0 else 0
    
    # Agreement metrics
    agreement = sum((df['is_passive_passivepy'] & df['is_passive_custom']) | 
                   (~df['is_passive_passivepy'] & ~df['is_passive_custom']))
    agreement_percentage = (agreement / total_sentences) * 100
    
    # Disagreement details
    passivepy_only = sum(df['is_passive_passivepy'] & ~df['is_passive_custom'])
    custom_only = sum(~df['is_passive_passivepy'] & df['is_passive_custom'])
    
    metrics = {
        'total_sentences': total_sentences,
        'passivepy': {
            'passive_sentences': int(passivepy_passive_sentences),
            'passive_percentage': float(passivepy_passive_percentage),
            'avg_passive_count': float(passivepy_passive_count_avg),
            'avg_passive_ratio': float(passivepy_passive_ratio_avg)
        },
        'custom': {
            'passive_sentences': int(custom_passive_sentences),
            'passive_percentage': float(custom_passive_percentage),
            'avg_passive_count': float(custom_passive_count_avg),
            'avg_passive_ratio': float(custom_passive_ratio_avg),
            'by_agent_count': int(custom_by_agent_count),
            'by_agent_percentage': float(custom_by_agent_percentage)
        },
        'comparison': {
            'agreement': int(agreement),
            'agreement_percentage': float(agreement_percentage),
            'passivepy_only': int(passivepy_only),
            'custom_only': int(custom_only)
        }
    }
    
    # Calculate by language
    language_metrics = {}
    for language in df['Native_Language'].unique():
        lang_df = df[df['Native_Language'] == language]
        lang_total = len(lang_df)
        
        # Skip languages with too few samples
        if lang_total < 10:
            continue
        
        passivepy_lang_passive = sum(lang_df['is_passive_passivepy'])
        passivepy_lang_percentage = (passivepy_lang_passive / lang_total) * 100
        
        custom_lang_passive = sum(lang_df['is_passive_custom'])
        custom_lang_percentage = (custom_lang_passive / lang_total) * 100
        
        language_metrics[language] = {
            'total': lang_total,
            'passivepy_percentage': float(passivepy_lang_percentage),
            'custom_percentage': float(custom_lang_percentage),
            'difference': float(custom_lang_percentage - passivepy_lang_percentage)
        }
    
    metrics['by_language'] = language_metrics
    
    # Pattern type distribution (custom detector only)
    pattern_types = []
    for patterns in df['pattern_types'].dropna():
        if isinstance(patterns, str) and patterns:
            pattern_types.extend([p.strip() for p in patterns.split(';')])
    
    pattern_counts = Counter(pattern_types)
    total_patterns = sum(pattern_counts.values())
    
    pattern_distribution = {
        pattern: {
            'count': count,
            'percentage': float((count / total_patterns) * 100) if total_patterns > 0 else 0
        }
        for pattern, count in pattern_counts.most_common()
    }
    
    metrics['pattern_distribution'] = pattern_distribution
    
    return metrics

def save_metrics(metrics, filename='passivepy_custom_icle_comparison.json'):
    """Save metrics to JSON file"""
    print(f"Saving metrics to {filename}...")
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    # Load ICLE data
    icle_df = load_icle_data()
    if icle_df is None:
        print("Failed to load ICLE data. Exiting.")
        return
    
    # Take a sample for faster processing
    sample_size = 500  # Smaller sample due to the complexity of processing
    sample_df = icle_df.sample(n=sample_size, random_state=42) if len(icle_df) > sample_size else icle_df
    
    # Analyze with PassivePy
    passivepy_results = analyze_with_passivepy(sample_df)
    
    # Analyze with custom detector
    comparison_results = analyze_with_custom(passivepy_results)
    
    # Calculate metrics
    metrics = calculate_metrics(comparison_results)
    
    # Save metrics
    save_metrics(metrics)
    
    # Print summary
    print("\nComparison Summary:")
    print(f"Total sentences analyzed: {metrics['total_sentences']}")
    print(f"PassivePy passive sentences: {metrics['passivepy']['passive_sentences']} ({metrics['passivepy']['passive_percentage']:.1f}%)")
    print(f"Custom passive sentences: {metrics['custom']['passive_sentences']} ({metrics['custom']['passive_percentage']:.1f}%)")
    print(f"Agreement: {metrics['comparison']['agreement']} ({metrics['comparison']['agreement_percentage']:.1f}%)")
    print(f"PassivePy only: {metrics['comparison']['passivepy_only']}")
    print(f"Custom only: {metrics['comparison']['custom_only']}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 