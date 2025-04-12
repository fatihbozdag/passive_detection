#!/usr/bin/env python3
"""
Process Merged ICLE Dataset with GPU Acceleration

This script merges ICLE concordance datasets, processes them with both PassivePy and
the custom detector using GPU acceleration, and provides comparison metrics.
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
import time

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

def merge_icle_datasets():
    """Merge ICLE concordance datasets and create a unified dataset with sentences"""
    print("Merging ICLE datasets...")
    
    # Define file paths
    file1 = '/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/icle_concord.csv'
    file2 = '/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/icle_concord_anywords.csv'
    
    # Read the CSV files
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        print(f"Read {len(df1)} sentences from {file1}")
        print(f"Read {len(df2)} sentences from {file2}")
        
        # Concatenate the dataframes
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Drop duplicates if any
        original_len = len(merged_df)
        merged_df.drop_duplicates(inplace=True)
        print(f"Dropped {original_len - len(merged_df)} duplicate rows")
        
        # Create sentence column by combining Left, Center, and Right
        merged_df['sentence'] = merged_df['Left'].fillna('') + ' ' + merged_df['Center'].fillna('') + ' ' + merged_df['Right'].fillna('')
        merged_df['sentence'] = merged_df['sentence'].str.strip()
        
        # Save the merged dataset
        output_file = 'merged_icle_concordance.csv'
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged dataset with {len(merged_df)} sentences to {output_file}")
        
        return merged_df
        
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return None

def analyze_with_passivepy(df):
    """Analyze text with PassivePy"""
    print("Analyzing with PassivePy...")
    
    # Initialize PassivePy with the correct parameters
    passivepy = PassivePyAnalyzer(spacy_model='en_core_web_sm')  # Use smaller model for stability
    
    # Initialize results list
    passivepy_results = []
    start_time = time.time()
    
    # Process each sentence individually to ensure reliable results
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing with PassivePy"):
        text = row['sentence']
        file_name = row['File name'] if 'File name' in row else f"doc_{idx}"
        native_language = row['Native language'] if 'Native language' in row else "Unknown"
        
        try:
            # Process with PassivePy's direct API for individual sentences
            passive_analysis = passivepy.match_text(text)
            
            # Extract passive information
            is_passive = False
            passive_count = 0
            passive_phrases = []
            
            if isinstance(passive_analysis, dict) and 'passive' in passive_analysis:
                passive_list = passive_analysis['passive']
                is_passive = len(passive_list) > 0
                passive_count = len(passive_list)
                
                # Extract phrases from the passive list
                for p in passive_list:
                    if isinstance(p, dict) and 'text' in p:
                        passive_phrases.append(p['text'])
                    elif isinstance(p, str):
                        passive_phrases.append(p)
            
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
            print(f"Error processing with PassivePy at index {idx}: {e}")
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
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"PassivePy processed {len(passivepy_results)} sentences in {processing_time:.2f} seconds ({len(passivepy_results)/processing_time:.2f} sentences/sec)")
    
    # Create a DataFrame from the results
    result_df = pd.DataFrame(passivepy_results)
    
    # Log PassivePy detection summary
    passive_count = sum(result_df['is_passive_passivepy'])
    print(f"PassivePy detected passive voice in {passive_count} out of {len(result_df)} sentences ({passive_count/len(result_df)*100:.1f}%)")
    
    return result_df

def analyze_with_custom(df):
    """Analyze text with custom passive detector"""
    print("Analyzing with custom detector...")
    
    # Import spaCy for custom detector
    # Load spaCy model - use transformer model for better accuracy
    nlp = spacy.load('en_core_web_trf')
    
    # Initialize results list
    results = []
    start_time = time.time()
    
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
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Custom detector processed {len(results)} sentences in {processing_time:.2f} seconds ({len(results)/processing_time:.2f} sentences/sec)")
    
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
    
    # Cohen's Kappa
    true_pos = sum(df['is_passive_passivepy'] & df['is_passive_custom'])
    true_neg = sum(~df['is_passive_passivepy'] & ~df['is_passive_custom'])
    false_pos = sum(~df['is_passive_passivepy'] & df['is_passive_custom'])
    false_neg = sum(df['is_passive_passivepy'] & ~df['is_passive_custom'])
    
    p_observed = (true_pos + true_neg) / total_sentences
    p_expected = (((true_pos + false_neg) * (true_pos + false_pos)) + 
                 ((false_pos + true_neg) * (false_neg + true_neg))) / (total_sentences ** 2)
    
    if p_expected == 1:
        kappa = 1.0
    else:
        kappa = (p_observed - p_expected) / (1 - p_expected)
    
    # Precision, Recall, F1 (treating custom as ground truth)
    custom_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    custom_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    custom_f1 = 2 * (custom_precision * custom_recall) / (custom_precision + custom_recall) if (custom_precision + custom_recall) > 0 else 0
    
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
            'custom_only': int(custom_only),
            'cohen_kappa': float(kappa),
            'precision': float(custom_precision),
            'recall': float(custom_recall),
            'f1_score': float(custom_f1)
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

def visualize_metrics(metrics, output_file='passive_comparison_visualization.png'):
    """Create visualizations of the comparison metrics"""
    print(f"Creating visualizations to {output_file}...")
    
    plt.figure(figsize=(20, 15))
    
    # 1. Passive sentences comparison
    plt.subplot(2, 3, 1)
    labels = ['PassivePy', 'Custom Detector']
    values = [metrics['passivepy']['passive_percentage'], metrics['custom']['passive_percentage']]
    plt.bar(labels, values, color=['#1f77b4', '#ff7f0e'])
    plt.title('Percentage of Passive Sentences Detected')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    for i, v in enumerate(values):
        plt.text(i, v + 2, f"{v:.1f}%", ha='center')
    
    # 2. Agreement pie chart
    plt.subplot(2, 3, 2)
    agreement_labels = ['Both Passive', 'Both Not Passive', 'PassivePy Only', 'Custom Only']
    true_pos = (metrics['comparison']['agreement'] - metrics['total_sentences'] + 
                metrics['passivepy']['passive_sentences'] + metrics['custom']['passive_sentences']) // 2
    true_neg = metrics['comparison']['agreement'] - true_pos
    agreement_values = [true_pos, true_neg, metrics['comparison']['passivepy_only'], metrics['comparison']['custom_only']]
    plt.pie(agreement_values, labels=agreement_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Agreement Between Detectors')
    
    # 3. Performance metrics
    plt.subplot(2, 3, 3)
    perf_labels = ['Agreement', 'Cohen\'s Kappa', 'Precision', 'Recall', 'F1 Score']
    perf_values = [
        metrics['comparison']['agreement_percentage'] / 100,
        metrics['comparison']['cohen_kappa'],
        metrics['comparison']['precision'],
        metrics['comparison']['recall'],
        metrics['comparison']['f1_score']
    ]
    plt.bar(perf_labels, perf_values, color='#2ca02c')
    plt.title('Performance Metrics')
    plt.ylabel('Score (0-1)')
    plt.ylim(0, 1)
    for i, v in enumerate(perf_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # 4. By-language comparison
    plt.subplot(2, 3, 4)
    languages = list(metrics['by_language'].keys())
    passivepy_lang_values = [metrics['by_language'][lang]['passivepy_percentage'] for lang in languages]
    custom_lang_values = [metrics['by_language'][lang]['custom_percentage'] for lang in languages]
    
    x = np.arange(len(languages))
    width = 0.35
    
    plt.bar(x - width/2, passivepy_lang_values, width, label='PassivePy')
    plt.bar(x + width/2, custom_lang_values, width, label='Custom')
    plt.title('Passive Usage by Native Language')
    plt.ylabel('Percentage')
    plt.xticks(x, languages, rotation=45, ha='right')
    plt.legend()
    
    # 5. Pattern distribution
    plt.subplot(2, 3, 5)
    patterns = list(metrics['pattern_distribution'].keys())[:10]  # Top 10 patterns
    pattern_values = [metrics['pattern_distribution'][p]['percentage'] for p in patterns]
    plt.barh(patterns, pattern_values, color='#d62728')
    plt.title('Top 10 Pattern Types (Custom Detector)')
    plt.xlabel('Percentage')
    
    # 6. By-agent percentage
    plt.subplot(2, 3, 6)
    plt.pie([metrics['custom']['by_agent_count'], 
             metrics['custom']['passive_sentences'] - metrics['custom']['by_agent_count']], 
            labels=['With By-Agent', 'Without By-Agent'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Passive Sentences with By-Agent')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Visualizations saved to {output_file}")

def save_metrics(metrics, filename='merged_icle_comparison.json'):
    """Save metrics to JSON file"""
    print(f"Saving metrics to {filename}...")
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filename}")

def update_progress_md(metrics):
    """Update the progress.md file with new findings"""
    print("Updating progress.md with new findings...")
    
    progress_file = '/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/progress.md'
    
    # Read existing content
    with open(progress_file, 'r') as f:
        content = f.read()
    
    # Create new content to append
    new_content = f"""
## Merged ICLE Dataset Analysis Results (GPU Accelerated)
- {metrics['total_sentences']} sentences from the merged ICLE concordance dataset were analyzed
- PassivePy detected passive voice in {metrics['passivepy']['passive_percentage']:.1f}% of sentences
- Custom detector identified passive voice in {metrics['custom']['passive_percentage']:.1f}% of sentences
- Agreement between implementations: {metrics['comparison']['agreement_percentage']:.1f}%
- Cohen's Kappa: {metrics['comparison']['cohen_kappa']:.2f}
- Precision: {metrics['comparison']['precision']:.2f}
- Recall: {metrics['comparison']['recall']:.2f}
- F1 Score: {metrics['comparison']['f1_score']:.2f}
- {metrics['custom']['by_agent_percentage']:.1f}% of passive sentences detected by custom implementation contain by-agents

The most common pattern types detected were:
"""
    
    # Add top 5 pattern types
    pattern_types = list(metrics['pattern_distribution'].keys())[:5]
    for pattern in pattern_types:
        pattern_info = metrics['pattern_distribution'][pattern]
        new_content += f"- {pattern}: {pattern_info['count']} occurrences ({pattern_info['percentage']:.1f}%)\n"
    
    # Add language information
    new_content += "\nPassive voice usage by native language:\n"
    languages = sorted(
        metrics['by_language'].keys(),
        key=lambda x: metrics['by_language'][x]['custom_percentage'],
        reverse=True
    )[:5]
    
    for language in languages:
        lang_info = metrics['by_language'][language]
        new_content += f"- {language}: PassivePy {lang_info['passivepy_percentage']:.1f}%, Custom {lang_info['custom_percentage']:.1f}%\n"
    
    # Update the file
    with open(progress_file, 'w') as f:
        f.write(content + new_content)
    
    print(f"Updated {progress_file}")

def main():
    # Set sample size (adjust based on available resources)
    sample_size = 2000  # Adjust as needed
    
    # Merge the datasets
    merged_df = merge_icle_datasets()
    if merged_df is None:
        print("Failed to merge datasets. Exiting.")
        return
    
    # Take a sample for faster processing
    sample_df = merged_df.sample(n=sample_size, random_state=42) if len(merged_df) > sample_size else merged_df
    print(f"Using a sample of {len(sample_df)} sentences for analysis")
    
    # Analyze with PassivePy
    passivepy_results = analyze_with_passivepy(sample_df)
    
    # Save intermediate results
    passivepy_results.to_csv('icle_passivepy_results.csv', index=False)
    print("Saved intermediate PassivePy results to icle_passivepy_results.csv")
    
    # Analyze with custom detector
    comparison_results = analyze_with_custom(passivepy_results)
    
    # Save annotated dataset
    comparison_results.to_csv('icle_annotated_comparison.csv', index=False)
    print("Saved annotated dataset to icle_annotated_comparison.csv")
    
    # Calculate metrics
    metrics = calculate_metrics(comparison_results)
    
    # Save metrics
    save_metrics(metrics)
    
    # Create visualizations
    visualize_metrics(metrics)
    
    # Update progress.md
    update_progress_md(metrics)
    
    # Print summary
    print("\nComparison Summary:")
    print(f"Total sentences analyzed: {metrics['total_sentences']}")
    print(f"PassivePy passive sentences: {metrics['passivepy']['passive_sentences']} ({metrics['passivepy']['passive_percentage']:.1f}%)")
    print(f"Custom passive sentences: {metrics['custom']['passive_sentences']} ({metrics['custom']['passive_percentage']:.1f}%)")
    print(f"Agreement: {metrics['comparison']['agreement']} ({metrics['comparison']['agreement_percentage']:.1f}%)")
    print(f"Cohen's Kappa: {metrics['comparison']['cohen_kappa']:.2f}")
    print(f"Precision: {metrics['comparison']['precision']:.2f}")
    print(f"Recall: {metrics['comparison']['recall']:.2f}")
    print(f"F1 Score: {metrics['comparison']['f1_score']:.2f}")
    print(f"Analysis complete!")

if __name__ == "__main__":
    main() 