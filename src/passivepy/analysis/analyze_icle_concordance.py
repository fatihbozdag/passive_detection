#!/usr/bin/env python3
"""
ICLE Concordance Analysis Script

This script processes the ICLE concordance dataset, merges the Left, Center, and Right columns 
to create complete sentences, then analyzes them with our custom passive voice detector implementation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
import os
import sys
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import custom implementation
import my_passive_detector as custom_detector

def load_icle_concordance(file_path):
    """
    Load the ICLE concordance dataset and create a unified sentence column
    
    Args:
        file_path (str): Path to the ICLE concordance CSV file
    
    Returns:
        pandas.DataFrame: DataFrame with merged sentence column
    """
    print(f"Loading ICLE concordance data from {file_path}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Create a unified sentence column by concatenating Left, Center, and Right
        df['sentence'] = df['Left'] + ' ' + df['Center'] + ' ' + df['Right']
        
        # Clean the sentences (remove extra spaces, newlines, etc.)
        df['sentence'] = df['sentence'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
        
        print(f"Loaded {len(df)} sentences from ICLE concordance data")
        return df
    
    except Exception as e:
        print(f"Error loading ICLE concordance data: {e}")
        return None

def analyze_with_custom_detector(df):
    """
    Analyze sentences with our custom passive detector
    
    Args:
        df (pandas.DataFrame): DataFrame with 'sentence' column
    
    Returns:
        pandas.DataFrame: Original DataFrame with custom detector predictions and details
    """
    print("Analyzing sentences with custom passive detector...")
    
    try:
        # Load SpaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Process sentences in batches to avoid memory issues
        batch_size = 500
        num_batches = len(df) // batch_size + 1
        
        all_results = []
        for i in tqdm(range(num_batches), desc="Processing with custom detector"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            
            if start_idx >= len(df):
                break
                
            batch = df.iloc[start_idx:end_idx]
            if batch.empty:
                continue
            
            # Analyze each sentence in the batch
            batch_results = []
            for text in batch['sentence']:
                result = custom_detector.process_text(text, nlp)
                passive_details = {
                    'is_passive': result['is_passive'],
                    'passive_count': result['passive_count'],
                    'passive_ratio': result['passive_ratio']
                }
                
                # Extract passive phrases if available
                if result['passive_phrases']:
                    phrases = [p['passive_phrase'] for p in result['passive_phrases']]
                    phrase_types = [p['pattern_type'] for p in result['passive_phrases']]
                    has_by_agents = [p['Has_By_Agent'] for p in result['passive_phrases']]
                    
                    passive_details['passive_phrases'] = '; '.join(phrases)
                    passive_details['pattern_types'] = '; '.join(phrase_types)
                    passive_details['has_by_agents'] = '; '.join([str(x) for x in has_by_agents])
                else:
                    passive_details['passive_phrases'] = ''
                    passive_details['pattern_types'] = ''
                    passive_details['has_by_agents'] = ''
                
                batch_results.append(passive_details)
            
            all_results.extend(batch_results)
        
        # Convert all results to a DataFrame and merge with original
        results_df = pd.DataFrame(all_results)
        
        # Add custom detector results to original DataFrame
        for col in results_df.columns:
            df[col] = results_df[col].values
        
        passive_count = results_df['is_passive'].sum()
        print(f"Custom detector analysis complete. Found {passive_count} passive sentences ({passive_count/len(df)*100:.1f}%).")
        return df
    
    except Exception as e:
        print(f"Error analyzing with custom detector: {e}")
        return df

def analyze_passive_patterns(df):
    """
    Analyze patterns in passive voice usage
    
    Args:
        df (pandas.DataFrame): DataFrame with passive detection results
    
    Returns:
        dict: Dictionary with various statistics
    """
    print("Analyzing passive patterns...")
    
    # Filter to only passive sentences
    passive_sentences = df[df['is_passive'] == True]
    
    if passive_sentences.empty:
        print("No passive sentences found for analysis")
        return {}
    
    # 1. Count pattern types
    all_pattern_types = []
    for pattern_types in passive_sentences['pattern_types'].dropna():
        if isinstance(pattern_types, str) and pattern_types:
            all_pattern_types.extend([p.strip() for p in pattern_types.split(';')])
    
    pattern_counts = Counter(all_pattern_types)
    
    # 2. Count sentences with by-agents
    has_by_agent_count = 0
    for by_agents in passive_sentences['has_by_agents'].dropna():
        if isinstance(by_agents, str) and by_agents:
            if 'True' in by_agents:
                has_by_agent_count += 1
    
    # 3. Analyze by native language
    language_stats = df.groupby('Native language').agg({
        'is_passive': 'mean',
        'passive_count': 'mean',
        'passive_ratio': 'mean'
    }).sort_values('is_passive', ascending=False)
    
    # 4. Distribution of passive counts per sentence
    passive_count_dist = Counter(passive_sentences['passive_count'])
    
    # 5. Calculate core statistics
    total_sentences = len(df)
    passive_sentences_count = len(passive_sentences)
    passive_percentage = (passive_sentences_count / total_sentences) * 100
    avg_passive_per_sentence = passive_sentences['passive_count'].mean()
    by_agent_percentage = (has_by_agent_count / passive_sentences_count) * 100
    
    print(f"Analysis complete:")
    print(f"  - Total sentences: {total_sentences}")
    print(f"  - Passive sentences: {passive_sentences_count} ({passive_percentage:.1f}%)")
    print(f"  - Average passives per sentence: {avg_passive_per_sentence:.2f}")
    print(f"  - Sentences with by-agents: {has_by_agent_count} ({by_agent_percentage:.1f}% of passive sentences)")
    
    return {
        'total_sentences': total_sentences,
        'passive_sentences': passive_sentences_count,
        'passive_percentage': passive_percentage,
        'avg_passive_per_sentence': avg_passive_per_sentence,
        'pattern_counts': pattern_counts,
        'by_agent_count': has_by_agent_count,
        'language_stats': language_stats,
        'passive_count_dist': passive_count_dist
    }

def visualize_results(stats):
    """
    Create visualizations of passive analysis results
    
    Args:
        stats (dict): Dictionary with analysis statistics
    
    Returns:
        None
    """
    if not stats:
        print("No statistics available for visualization")
        return
    
    print("Generating visualizations...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Passive vs Non-passive pie chart
    passive_pct = stats['passive_percentage']
    non_passive_pct = 100 - passive_pct
    
    axes[0, 0].pie([passive_pct, non_passive_pct], 
                  labels=['Passive', 'Non-Passive'],
                  colors=['#ff9999', '#99ff99'],
                  autopct='%1.1f%%',
                  startangle=90)
    axes[0, 0].set_title('Passive vs Non-Passive Sentences')
    axes[0, 0].axis('equal')
    
    # 2. Top pattern types bar chart
    if 'pattern_counts' in stats:
        pattern_df = pd.DataFrame([
            {'Pattern': k, 'Count': v}
            for k, v in stats['pattern_counts'].most_common(10)
        ])
        
        if not pattern_df.empty:
            sns.barplot(data=pattern_df, x='Count', y='Pattern', ax=axes[0, 1])
            axes[0, 1].set_title('Top 10 Passive Pattern Types')
    
    # 3. Passive ratio by native language
    if 'language_stats' in stats and not stats['language_stats'].empty:
        language_df = stats['language_stats'].head(10).copy()
        language_df['is_passive'] = language_df['is_passive'] * 100  # Convert to percentage
        
        language_df['is_passive'].plot(kind='bar', ax=axes[1, 0], color='#66b3ff')
        axes[1, 0].set_title('Passive Sentence Percentage by Native Language (Top 10)')
        axes[1, 0].set_xlabel('Native Language')
        axes[1, 0].set_ylabel('Passive Sentences (%)')
        axes[1, 0].set_ylim([0, 100])
    
    # 4. Distribution of passives per sentence
    if 'passive_count_dist' in stats:
        counts = []
        freqs = []
        
        for count, freq in sorted(stats['passive_count_dist'].items()):
            if count <= 5:  # Limit to 5 passives per sentence for readability
                counts.append(str(count))
                freqs.append(freq)
        
        if counts and freqs:
            axes[1, 1].bar(counts, freqs, color='#99ccff')
            axes[1, 1].set_title('Passive Phrases per Sentence')
            axes[1, 1].set_xlabel('Number of Passive Phrases')
            axes[1, 1].set_ylabel('Frequency')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('icle_passive_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved to 'icle_passive_analysis.png'")
    
    # Additional visualization for language statistics
    if 'language_stats' in stats and len(stats['language_stats']) > 10:
        plt.figure(figsize=(14, 8))
        language_df = stats['language_stats'].copy()
        language_df['is_passive'] = language_df['is_passive'] * 100  # Convert to percentage
        
        language_df['is_passive'].plot(kind='bar', color='#66b3ff')
        plt.title('Passive Sentence Percentage by Native Language')
        plt.xlabel('Native Language')
        plt.ylabel('Passive Sentences (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('icle_language_stats.png', dpi=300, bbox_inches='tight')
        print("Language statistics visualization saved to 'icle_language_stats.png'")

def main():
    """Main execution function"""
    # Load ICLE concordance data
    icle_file = 'icle_concord.csv'
    df = load_icle_concordance(icle_file)
    
    if df is None:
        print("Exiting due to error loading data.")
        return
    
    # Sample data for initial display
    print("\nSample data:")
    print(df[['Left', 'Center', 'Right', 'sentence']].head())
    
    # Analyze with custom detector
    df = analyze_with_custom_detector(df)
    
    # Save annotated dataset
    df.to_csv('icle_annotated.csv', index=False)
    print("Annotated dataset saved to 'icle_annotated.csv'")
    
    # Analyze passive patterns
    stats = analyze_passive_patterns(df)
    
    # Visualize results
    visualize_results(stats)
    
    # Save key statistics to CSV files for further analysis
    if stats and 'language_stats' in stats:
        stats['language_stats'].to_csv('icle_language_stats.csv')
        print("Language statistics saved to 'icle_language_stats.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 