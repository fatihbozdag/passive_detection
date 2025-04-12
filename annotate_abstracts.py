#!/usr/bin/env python3
"""
Paper Abstracts Annotation Script

This script annotates the paper abstracts dataset with both PassivePy and 
a custom passive voice detector, and compares the results.
"""

import os
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import time
import json
from datetime import datetime

# Import PassivePy
from PassivePySrc.PassivePy import PassivePyAnalyzer

# Import custom passive detector
from run_passive_detector import process_text as my_process_text
from my_passive_detector import convert_to_serializable

# Constants
ABSTRACTS_PATH = "/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/Data/paper_abstract_50_samples.csv"
CROWD_SOURCE_PATH = "/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/Data/crowd_source_dataset.csv"
OUTPUT_DIR = "analysis_results"

def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create a progress log file
    progress_file = os.path.join(output_dir, "progress.md")
    if not os.path.exists(progress_file):
        with open(progress_file, "w") as f:
            f.write("# Data Analysis Progress\n\n")
            f.write("| Dataset | Step | Status | Time | Details |\n")
            f.write("|--------|------|--------|------|--------|\n")
    
    return progress_file

def load_dataset(file_path, text_column):
    """Load the dataset."""
    try:
        df = pd.read_csv(file_path)
        
        print(f"Loaded {len(df)} examples from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def process_abstracts_with_passivepy(df, output_dir, progress_file):
    """Annotate the dataset with PassivePy."""
    update_progress(progress_file, "abstracts", "PassivePy Annotation", "Started", "")
    start_time = time.time()
    
    try:
        # Initialize PassivePy
        passivepy = PassivePyAnalyzer(spacy_model="en_core_web_sm")
        
        # Process texts sentence-by-sentence
        print("Processing abstracts with PassivePy...")
        
        # Create a temporary dataframe with one row per abstract
        temp_df = pd.DataFrame({
            'abstract_text': df['abstract_text'],
            'title': df['title'],
            'url': df['url']
        })
        
        # Use PassivePy to process the abstracts
        # This will split the abstracts into sentences and process each one
        results_df = passivepy.match_sentence_level(
            df=temp_df,
            column_name="abstract_text",
            n_process=1,
            batch_size=10,
            add_other_columns=True
        )
        
        # Print the columns to debug
        print(f"PassivePy results columns: {results_df.columns.tolist()}")
        
        # Save results
        output_path = os.path.join(output_dir, "abstract_passivepy_results.csv")
        results_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "abstracts", "PassivePy Annotation", "Completed", 
                     f"Time: {elapsed_time:.2f}s, Processed {len(results_df)} sentences, Saved to {output_path}")
        
        return results_df
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "abstracts", "PassivePy Annotation", "Failed", 
                     f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error annotating with PassivePy: {e}")
        return None

def process_abstracts_with_custom(df, output_dir, progress_file):
    """Annotate the dataset with custom passive detector."""
    update_progress(progress_file, "abstracts", "Custom Detector Annotation", "Started", "")
    start_time = time.time()
    
    try:
        # Load spaCy model
        print("Loading spaCy model for custom detector...")
        nlp = spacy.load("en_core_web_sm")
        
        # Process abstracts
        print("Processing abstracts with custom detector...")
        
        results = []
        # Process each abstract
        for i, row in df.iterrows():
            abstract = row['abstract_text']
            title = row['title']
            url = row['url']
            
            # Process the abstract with the custom detector
            try:
                # Process the entire abstract
                custom_result = my_process_text(abstract, nlp)
                
                # Add metadata
                custom_result['original_text'] = abstract
                custom_result['title'] = title
                custom_result['url'] = url
                custom_result['index'] = i
                
                # Manually add passive_ratio if it's missing 
                if 'passive_ratio' not in custom_result:
                    # Calculate as passive count / total words (or tokens)
                    doc = nlp(abstract)
                    total_tokens = len(doc)
                    passive_count = custom_result.get('passive_count', 0)
                    if total_tokens > 0:
                        custom_result['passive_ratio'] = passive_count / total_tokens
                    else:
                        custom_result['passive_ratio'] = 0.0
                
                # Convert passive_phrases to a string for CSV storage
                if 'passive_phrases' in custom_result:
                    custom_result['passive_phrases_str'] = '; '.join([p.get('passive_phrase', '') for p in custom_result['passive_phrases']])
                
                results.append(custom_result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(df)} abstracts with custom detector")
                
            except Exception as e:
                print(f"Error processing abstract {i} with custom detector: {e}")
                results.append({
                    'original_text': abstract,
                    'title': title,
                    'url': url,
                    'index': i,
                    'is_passive': False,
                    'passive_count': 0,
                    'passive_ratio': 0.0,
                    'error': str(e)
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.join(output_dir, "abstract_custom_results.csv")
        results_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "abstracts", "Custom Detector Annotation", "Completed", 
                     f"Time: {elapsed_time:.2f}s, Processed {len(results_df)} abstracts, Saved to {output_path}")
        
        return results_df
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "abstracts", "Custom Detector Annotation", "Failed", 
                     f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error with custom detector: {e}")
        return None

def compare_abstract_annotations(passivepy_results, custom_results, output_dir, progress_file):
    """Compare the annotations from PassivePy and the custom detector on abstracts."""
    update_progress(progress_file, "abstracts", "Annotation Comparison", "Started", "")
    start_time = time.time()
    
    try:
        # Ensure we have both results
        if passivepy_results is None or custom_results is None:
            raise ValueError("Missing results from one or both detectors")
        
        # Debug: print the column names
        print("PassivePy results columns:", passivepy_results.columns.tolist())
        print("Custom results columns:", custom_results.columns.tolist())
        
        # Calculate summary statistics for PassivePy results
        passivepy_summary = {
            'total_sentences': len(passivepy_results),
            'passive_sentences': int(passivepy_results['binary'].sum()),
            'passive_ratio': float(passivepy_results['binary'].mean() * 100)
        }
        
        # Calculate summary statistics for custom detector results - with safer column checking
        custom_summary = {'total_abstracts': len(custom_results)}
        
        if 'passive_count' in custom_results.columns:
            custom_summary['abstracts_with_passive'] = int((custom_results['passive_count'] > 0).sum())
            custom_summary['average_passive_count'] = float(custom_results['passive_count'].mean())
        else:
            print("Warning: 'passive_count' column not found in custom results")
            custom_summary['abstracts_with_passive'] = 0
            custom_summary['average_passive_count'] = 0.0
            
        # Debug: check if 'passive_ratio' is in the columns
        print("Has 'passive_ratio':", 'passive_ratio' in custom_results.columns)
        
        try:
            if 'passive_ratio' in custom_results.columns:
                # Debug: check the values
                print("First few passive_ratio values:", custom_results['passive_ratio'].head())
                custom_summary['average_passive_ratio'] = float(custom_results['passive_ratio'].mean() * 100)
            else:
                custom_summary['average_passive_ratio'] = 0.0
        except Exception as e:
            print(f"Error processing passive_ratio: {e}")
            custom_summary['average_passive_ratio'] = 0.0
        
        # Save summary statistics
        summary = {
            'passivepy': passivepy_summary,
            'custom': custom_summary
        }
        
        # Debug: print the summary before serialization
        print("Summary before serialization:", summary)
        
        summary_path = os.path.join(output_dir, "abstract_comparison_summary.json")
        
        try:
            with open(summary_path, "w") as f:
                json.dump(convert_to_serializable(summary), f, indent=2)
            print("Successfully wrote summary to", summary_path)
        except Exception as e:
            print(f"Error writing summary to JSON: {e}")
            # Try a simpler approach
            with open(summary_path, "w") as f:
                json.dump({
                    'passivepy': {
                        'total_sentences': passivepy_summary['total_sentences'],
                        'passive_sentences': passivepy_summary['passive_sentences'],
                        'passive_ratio': passivepy_summary['passive_ratio']
                    },
                    'custom': {
                        'total_abstracts': custom_summary['total_abstracts'],
                        'abstracts_with_passive': custom_summary.get('abstracts_with_passive', 0),
                        'average_passive_count': custom_summary.get('average_passive_count', 0.0),
                        'average_passive_ratio': custom_summary.get('average_passive_ratio', 0.0)
                    }
                }, f, indent=2)
        
        # Create visualizations
        
        # PassivePy: Distribution of passive sentences
        plt.figure(figsize=(10, 6))
        sns.countplot(x='binary', data=passivepy_results)
        plt.title('PassivePy: Distribution of Passive vs. Non-Passive Sentences')
        plt.xlabel('Passive (1) vs. Non-Passive (0)')
        plt.ylabel('Count')
        passivepy_dist_path = os.path.join(output_dir, "abstract_passivepy_distribution.png")
        plt.savefig(passivepy_dist_path)
        plt.close()
        
        # Custom: Distribution of passive counts per abstract
        plt.figure(figsize=(10, 6))
        sns.histplot(custom_results['passive_count'], kde=True, bins=10)
        plt.title('Custom Detector: Distribution of Passive Counts per Abstract')
        plt.xlabel('Number of Passive Constructions')
        plt.ylabel('Count')
        custom_dist_path = os.path.join(output_dir, "abstract_custom_distribution.png")
        plt.savefig(custom_dist_path)
        plt.close()
        
        # Custom: Distribution of passive ratio per abstract
        plt.figure(figsize=(10, 6))
        sns.histplot(custom_results['passive_ratio'], kde=True, bins=10)
        plt.title('Custom Detector: Distribution of Passive Ratio per Abstract')
        plt.xlabel('Passive Ratio')
        plt.ylabel('Count')
        custom_ratio_path = os.path.join(output_dir, "abstract_custom_ratio_distribution.png")
        plt.savefig(custom_ratio_path)
        plt.close()
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "abstracts", "Annotation Comparison", "Completed", 
                     f"Time: {elapsed_time:.2f}s, " +
                     f"Saved to {summary_path}, {passivepy_dist_path}, {custom_dist_path}, {custom_ratio_path}")
        
        return summary
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "abstracts", "Annotation Comparison", "Failed", 
                     f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error comparing annotations: {e}")
        return None

def process_crowd_source_with_passivepy(df, output_dir, progress_file):
    """Process the crowd source dataset with PassivePy."""
    update_progress(progress_file, "crowd_source", "PassivePy Processing", "Started", "")
    start_time = time.time()
    
    try:
        # Initialize PassivePy
        passivepy = PassivePyAnalyzer(spacy_model="en_core_web_sm")
        
        # Process sentences
        print("Processing crowd source data with PassivePy...")
        
        # Create a DataFrame with the sentences
        temp_df = pd.DataFrame({
            'text_field': df['Sentence'],
            'ID': df['ID'],
            'Test': df['Test'],
            'Cond': df['Cond'],
            'human_coding': df['human_coding']
        })
        
        # Use PassivePy to process the sentences
        results_df = passivepy.match_sentence_level(
            df=temp_df,
            column_name="text_field",
            n_process=1,
            batch_size=50,
            add_other_columns=True
        )
        
        # Print the columns to debug
        print(f"PassivePy results columns: {results_df.columns.tolist()}")
        
        # Save results
        output_path = os.path.join(output_dir, "crowd_source_passivepy_results.csv")
        results_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "crowd_source", "PassivePy Processing", "Completed", 
                     f"Time: {elapsed_time:.2f}s, Processed {len(results_df)} sentences, Saved to {output_path}")
        
        return results_df
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "crowd_source", "PassivePy Processing", "Failed", 
                     f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error processing with PassivePy: {e}")
        return None

def process_crowd_source_with_custom(df, output_dir, progress_file):
    """Process the crowd source dataset with custom detector."""
    update_progress(progress_file, "crowd_source", "Custom Detector Processing", "Started", "")
    start_time = time.time()
    
    try:
        # Load spaCy model
        print("Loading spaCy model for custom detector...")
        nlp = spacy.load("en_core_web_sm")
        
        # Process sentences
        print("Processing crowd source data with custom detector...")
        
        results = []
        # Process each sentence
        for i, row in df.iterrows():
            sentence = row['Sentence']
            
            # Process the sentence with the custom detector
            try:
                # Process the sentence
                custom_result = my_process_text(sentence, nlp)
                
                # Add metadata
                custom_result['text_field'] = sentence
                custom_result['ID'] = row['ID']
                custom_result['Test'] = row['Test']
                custom_result['Cond'] = row['Cond']
                custom_result['human_coding'] = row['human_coding']
                custom_result['custom_prediction'] = 1 if custom_result['passive_count'] > 0 else 0
                
                results.append(custom_result)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(df)} sentences with custom detector")
                
            except Exception as e:
                print(f"Error processing sentence {i} with custom detector: {e}")
                results.append({
                    'text_field': sentence,
                    'ID': row['ID'],
                    'Test': row['Test'],
                    'Cond': row['Cond'],
                    'human_coding': row['human_coding'],
                    'is_passive': False,
                    'passive_count': 0,
                    'passive_ratio': 0.0,
                    'custom_prediction': 0,
                    'error': str(e)
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.join(output_dir, "crowd_source_custom_results.csv")
        results_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "crowd_source", "Custom Detector Processing", "Completed", 
                     f"Time: {elapsed_time:.2f}s, Processed {len(results_df)} sentences, Saved to {output_path}")
        
        return results_df
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "crowd_source", "Custom Detector Processing", "Failed", 
                     f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error with custom detector: {e}")
        return None

def analyze_crowd_source_results(passivepy_results, custom_results, output_dir, progress_file):
    """Analyze the results from both implementations on the crowd source dataset."""
    update_progress(progress_file, "crowd_source", "Results Analysis", "Started", "")
    start_time = time.time()
    
    try:
        # Ensure we have both results
        if passivepy_results is None or custom_results is None:
            raise ValueError("Missing results from one or both detectors")
        
        # Prepare combined dataframe for analysis
        # Get binary predictions from PassivePy
        if 'text_field' in passivepy_results.columns:
            passivepy_df = passivepy_results[['text_field', 'binary', 'human_coding']].copy()
        else:
            # Use 'sentences' column if 'text_field' is not available
            passivepy_df = passivepy_results[['sentences', 'binary', 'human_coding']].copy()
            passivepy_df.rename(columns={'sentences': 'text_field'}, inplace=True)
            
        passivepy_df.rename(columns={'binary': 'passivepy_prediction'}, inplace=True)
        
        # Convert all values to numeric to ensure consistency
        passivepy_df['passivepy_prediction'] = pd.to_numeric(passivepy_df['passivepy_prediction'], errors='coerce').fillna(0).astype(int)
        passivepy_df['human_coding'] = pd.to_numeric(passivepy_df['human_coding'], errors='coerce').fillna(0).astype(int)
        
        # Get binary predictions from custom detector
        custom_df = custom_results[['text_field', 'custom_prediction', 'human_coding']].copy()
        custom_df['custom_prediction'] = pd.to_numeric(custom_df['custom_prediction'], errors='coerce').fillna(0).astype(int)
        custom_df['human_coding'] = pd.to_numeric(custom_df['human_coding'], errors='coerce').fillna(0).astype(int)
        
        # Merge the two dataframes
        merged_df = pd.merge(passivepy_df, custom_df, on=['text_field', 'human_coding'], how='inner')
        
        # Calculate metrics
        # 1. Human vs PassivePy
        human_passivepy_cm = confusion_matrix(merged_df['human_coding'], merged_df['passivepy_prediction'])
        human_passivepy_report = classification_report(
            merged_df['human_coding'], 
            merged_df['passivepy_prediction'], 
            output_dict=True
        )
        human_passivepy_kappa = cohen_kappa_score(merged_df['human_coding'], merged_df['passivepy_prediction'])
        
        # 2. Human vs Custom
        human_custom_cm = confusion_matrix(merged_df['human_coding'], merged_df['custom_prediction'])
        human_custom_report = classification_report(
            merged_df['human_coding'], 
            merged_df['custom_prediction'],
            output_dict=True
        )
        human_custom_kappa = cohen_kappa_score(merged_df['human_coding'], merged_df['custom_prediction'])
        
        # 3. PassivePy vs Custom
        passivepy_custom_cm = confusion_matrix(merged_df['passivepy_prediction'], merged_df['custom_prediction'])
        passivepy_custom_report = classification_report(
            merged_df['passivepy_prediction'],
            merged_df['custom_prediction'],
            output_dict=True
        )
        passivepy_custom_kappa = cohen_kappa_score(merged_df['passivepy_prediction'], merged_df['custom_prediction'])
        
        # Compile results
        results = {
            'human_vs_passivepy': {
                'confusion_matrix': human_passivepy_cm.tolist(),
                'classification_report': human_passivepy_report,
                'cohen_kappa': human_passivepy_kappa
            },
            'human_vs_custom': {
                'confusion_matrix': human_custom_cm.tolist(),
                'classification_report': human_custom_report,
                'cohen_kappa': human_custom_kappa
            },
            'passivepy_vs_custom': {
                'confusion_matrix': passivepy_custom_cm.tolist(),
                'classification_report': passivepy_custom_report,
                'cohen_kappa': passivepy_custom_kappa
            }
        }
        
        # Save results
        results_path = os.path.join(output_dir, "crowd_source_analysis_results.json")
        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        # Create visualizations
        
        # 1. Human vs PassivePy confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(human_passivepy_cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Non-Passive', 'Passive'],
                  yticklabels=['Non-Passive', 'Passive'])
        plt.title('Human vs PassivePy Predictions')
        plt.xlabel('PassivePy Prediction')
        plt.ylabel('Human Label')
        human_passivepy_cm_path = os.path.join(output_dir, "human_vs_passivepy_cm.png")
        plt.savefig(human_passivepy_cm_path)
        plt.close()
        
        # 2. Human vs Custom confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(human_custom_cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Non-Passive', 'Passive'],
                  yticklabels=['Non-Passive', 'Passive'])
        plt.title('Human vs Custom Detector Predictions')
        plt.xlabel('Custom Detector Prediction')
        plt.ylabel('Human Label')
        human_custom_cm_path = os.path.join(output_dir, "human_vs_custom_cm.png")
        plt.savefig(human_custom_cm_path)
        plt.close()
        
        # 3. PassivePy vs Custom confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(passivepy_custom_cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Non-Passive', 'Passive'],
                  yticklabels=['Non-Passive', 'Passive'])
        plt.title('PassivePy vs Custom Detector Predictions')
        plt.xlabel('Custom Detector Prediction')
        plt.ylabel('PassivePy Prediction')
        passivepy_custom_cm_path = os.path.join(output_dir, "passivepy_vs_custom_cm.png")
        plt.savefig(passivepy_custom_cm_path)
        plt.close()
        
        # 4. Find disagreements between all methods
        disagreements = merged_df[
            (merged_df['human_coding'] != merged_df['passivepy_prediction']) |
            (merged_df['human_coding'] != merged_df['custom_prediction']) |
            (merged_df['passivepy_prediction'] != merged_df['custom_prediction'])
        ].copy()
        
        # Save disagreements to CSV
        disagreements_path = os.path.join(output_dir, "crowd_source_disagreements.csv")
        disagreements.to_csv(disagreements_path, index=False)
        
        # Count of disagreements
        disagreement_counts = {
            'total_samples': len(merged_df),
            'total_disagreements': len(disagreements),
            'human_passivepy_disagreements': (merged_df['human_coding'] != merged_df['passivepy_prediction']).sum(),
            'human_custom_disagreements': (merged_df['human_coding'] != merged_df['custom_prediction']).sum(),
            'passivepy_custom_disagreements': (merged_df['passivepy_prediction'] != merged_df['custom_prediction']).sum()
        }
        
        # Add disagreement counts to results
        results['disagreement_counts'] = disagreement_counts
        
        # Update the JSON file
        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "crowd_source", "Results Analysis", "Completed", 
                     f"Time: {elapsed_time:.2f}s, " +
                     f"Saved to {results_path}, {human_passivepy_cm_path}, {human_custom_cm_path}, {passivepy_custom_cm_path}, {disagreements_path}")
        
        # Print a summary of the results
        print("\n--- Human annotations vs PassivePy ---")
        print(f"Accuracy: {human_passivepy_report['accuracy']:.2f}")
        print(f"F1-score (passive): {human_passivepy_report['1']['f1-score']:.2f}")
        print(f"Cohen's Kappa: {human_passivepy_kappa:.4f}")
        
        print("\n--- Human annotations vs Custom detector ---")
        print(f"Accuracy: {human_custom_report['accuracy']:.2f}")
        print(f"F1-score (passive): {human_custom_report['1']['f1-score']:.2f}")
        print(f"Cohen's Kappa: {human_custom_kappa:.4f}")
        
        print("\n--- PassivePy vs Custom detector ---")
        print(f"Accuracy: {passivepy_custom_report['accuracy']:.2f}")
        print(f"F1-score (passive): {passivepy_custom_report['1']['f1-score']:.2f}")
        print(f"Cohen's Kappa: {passivepy_custom_kappa:.4f}")
        
        print(f"\nFound {disagreement_counts['total_disagreements']} sentences where at least one method disagrees.")
        print(f"Details saved to {disagreements_path}")
        
        return results
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "crowd_source", "Results Analysis", "Failed", 
                     f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error analyzing results: {e}")
        return None

def update_progress(progress_file, dataset, step, status, details):
    """Update the progress in the markdown file."""
    with open(progress_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"| {dataset} | {step} | {status} | {timestamp} | {details} |\n")

def main():
    """Main function to run the annotation."""
    # Ensure output directory exists
    progress_file = ensure_output_dir(OUTPUT_DIR)
    
    # Process the paper abstracts dataset
    print("\n=== Processing Paper Abstracts Dataset ===\n")
    
    # Load dataset
    abstracts_df = load_dataset(ABSTRACTS_PATH, "abstract_text")
    if abstracts_df is None:
        update_progress(progress_file, "abstracts", "Dataset Loading", "Failed", "Could not load dataset")
        return
    
    update_progress(progress_file, "abstracts", "Dataset Loading", "Completed", f"Loaded {len(abstracts_df)} examples")
    
    # Process with PassivePy
    abstracts_passivepy_results = process_abstracts_with_passivepy(abstracts_df, OUTPUT_DIR, progress_file)
    
    # Process with custom detector
    abstracts_custom_results = process_abstracts_with_custom(abstracts_df, OUTPUT_DIR, progress_file)
    
    # Compare annotations
    abstracts_comparison = compare_abstract_annotations(
        abstracts_passivepy_results, 
        abstracts_custom_results, 
        OUTPUT_DIR, 
        progress_file
    )
    
    # Process the crowd source dataset
    print("\n=== Processing Crowd Source Dataset ===\n")
    
    # Load dataset
    crowd_source_df = load_dataset(CROWD_SOURCE_PATH, "Sentence")
    if crowd_source_df is None:
        update_progress(progress_file, "crowd_source", "Dataset Loading", "Failed", "Could not load dataset")
        return
    
    update_progress(progress_file, "crowd_source", "Dataset Loading", "Completed", f"Loaded {len(crowd_source_df)} examples")
    
    # Process with PassivePy
    crowd_source_passivepy_results = process_crowd_source_with_passivepy(crowd_source_df, OUTPUT_DIR, progress_file)
    
    # Process with custom detector
    crowd_source_custom_results = process_crowd_source_with_custom(crowd_source_df, OUTPUT_DIR, progress_file)
    
    # Analyze results
    crowd_source_analysis = analyze_crowd_source_results(
        crowd_source_passivepy_results, 
        crowd_source_custom_results, 
        OUTPUT_DIR, 
        progress_file
    )
    
    # Final update
    update_progress(
        progress_file, 
        "combined", 
        "Analysis Complete", 
        "Completed", 
        "Processed both datasets and performed comparative analysis"
    )

if __name__ == "__main__":
    main() 