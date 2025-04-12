import os
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
from typing import List, Dict
import logging
import sys
from pathlib import Path
from PassivePy import PassivePyAnalyzer as PassivePy
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Import custom detector
import my_passive_detector as custom_detector

# Import local PassivePy implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.passive_detector.core.detector import PassivePyAnalyzer as PassivePy
from my_passive_detector import process_text

# Create output directory
output_dir = Path('comparison_results')
output_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dir / 'comparison.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_crowd_source_dataset():
    """Load and prepare the crowd source dataset with human annotations"""
    print("Loading crowd source dataset...")
    try:
        df = pd.read_csv("Data/crowd_source_dataset.csv")
        # Ensure text column is properly named for processing
        df = df.rename(columns={"Sentence": "text_field"})
        # Add metadata columns needed by the custom detector
        df['Native_Language'] = 'English'  # We don't have this info, so we'll assume English
        df['file_name'] = df['ID'].astype(str)
        return df
    except FileNotFoundError:
        print("Error: crowd_source_dataset.csv not found in the Data directory")
        return None

def load_icle_dataset():
    """Load and prepare the ICLE dataset (without human annotations)"""
    print("Loading ICLE dataset...")
    try:
        # Load the merged_icle_concordance.csv file from the root directory
        df = pd.read_csv("merged_icle_concordance.csv")
        
        # Ensure text column is properly named for processing
        if "Sentence" in df.columns:
            df = df.rename(columns={"Sentence": "text_field"})
        elif "sentence" in df.columns:
            df = df.rename(columns={"sentence": "text_field"})
        elif "Text" in df.columns:
            df = df.rename(columns={"Text": "text_field"})
        elif "text" in df.columns:
            df = df.rename(columns={"text": "text_field"})
        
        # If there's no text_field yet, attempt to create one by merging Left, Center and Right
        if "text_field" not in df.columns and all(col in df.columns for col in ["Left", "Center", "Right"]):
            # Merge the Left, Center, and Right context into a single text field
            df["text_field"] = df["Left"] + " " + df["Center"] + " " + df["Right"]
        
        # If we still don't have text_field, raise an error
        if "text_field" not in df.columns:
            raise ValueError(f"Could not find or create a text field. Available columns: {df.columns.tolist()}")
            
        # If there's no Native_Language column, add it based on metadata or set to default
        if "Native_Language" not in df.columns and "L1" in df.columns:
            df['Native_Language'] = df['L1']
        elif "Native_Language" not in df.columns and "Native language" in df.columns:
            df['Native_Language'] = df['Native language']
        elif "Native_Language" not in df.columns:
            df['Native_Language'] = "Unknown"
            
        # Create a file_name column if it doesn't exist
        if "file_name" not in df.columns and "ID" in df.columns:
            df['file_name'] = df['ID'].astype(str)
        elif "file_name" not in df.columns and "File name" in df.columns:
            df['file_name'] = df['File name'].astype(str)
        elif "file_name" not in df.columns:
            df['file_name'] = df.index.astype(str)
        
        # Process the full dataset
        print(f"Loaded {len(df)} sentences from ICLE dataset")
        
        # The script will now process the full dataset and save intermediate results along the way
        print("Processing full dataset - this may take some time.")
        print("Intermediate results will be saved periodically.")
        
        return df
    except FileNotFoundError:
        print("Error: ICLE dataset not found at the specified path")
        return None
    except Exception as e:
        print(f"Error loading ICLE dataset: {str(e)}")
        return None

def process_with_passivepy(df):
    """Process the dataset with PassivePy"""
    print("Processing with PassivePy...")
    try:
        # Initialize PassivePy with spaCy model
        print("Initializing PassivePy with spaCy model and GPU acceleration if available")
        # Enable GPU if available
        spacy.prefer_gpu()
        passive_py = PassivePyAnalyzer(spacy_model="en_core_web_sm")
        
        # Use multiple CPU cores for processing
        import multiprocessing
        n_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        print(f"Using {n_cores} CPU cores for PassivePy processing")
        
        # Match at sentence level with progress tracking
        print(f"Starting PassivePy processing on {len(df)} sentences")
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 2000  # Process 2000 sentences at a time
        num_chunks = (len(df) + chunk_size - 1) // chunk_size
        all_results = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx].copy()
            
            print(f"Processing chunk {i+1}/{num_chunks} (rows {start_idx} to {end_idx})")
            
            try:
                chunk_results = passive_py.match_sentence_level(
                    df=chunk, 
                    column_name="text_field",
                    n_process=n_cores,
                    batch_size=200,
                    add_other_columns=False  # Set to False to reduce memory usage
                )
                
                all_results.append(chunk_results)
                
                # Save intermediate results
                chunk_results.to_csv(f"passivepy_results_chunk_{i+1}.csv", index=False)
                print(f"Saved PassivePy results for chunk {i+1}")
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
                # Continue with the next chunk instead of failing completely
                continue
        
        # Combine all chunks
        if not all_results:
            raise ValueError("No chunks were successfully processed")
            
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Save complete results
        results_df.to_csv("passivepy_full_results.csv", index=False)
        print("Saved complete PassivePy results")
        
        # Print the columns to debug
        print(f"PassivePy result columns: {results_df.columns.tolist()}")
        
        # Now use the 'sentences' column as text and 'binary' as the passive indicator
        if 'sentences' in results_df.columns and 'binary' in results_df.columns:
            passivepy_results = pd.DataFrame({
                'text_field': results_df['sentences'],
                'passivepy_prediction': results_df['binary'].astype(int)
            })
            
            # Save the final processed results
            passivepy_results.to_csv("passivepy_processed_results.csv", index=False)
            print("Saved processed PassivePy results")
            
            return passivepy_results
        else:
            raise ValueError(f"Required columns not found. Available columns: {results_df.columns}")
        
    except Exception as e:
        print(f"Error processing with PassivePy: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_with_custom_detector(df, nlp=None):
    """Process the dataset with custom passive detector"""
    print("Processing with custom passive detector...")
    try:
        # Load spaCy model if not provided
        if nlp is None:
            try:
                # Attempt to load the transformer model with GPU acceleration
                print("Attempting to load transformer model with GPU acceleration...")
                spacy.prefer_gpu()
                nlp = spacy.load("en_core_web_trf")
                print("Successfully loaded transformer model with GPU acceleration")
            except Exception as e:
                print(f"Could not load transformer model: {str(e)}")
                print("Falling back to en_core_web_sm model...")
                try:
                    nlp = spacy.load("en_core_web_sm")
                    print("Successfully loaded en_core_web_sm model")
                except OSError:
                    print("Error: No spaCy model available. Installing en_core_web_sm...")
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    nlp = spacy.load("en_core_web_sm")
        
        # Set a larger max_length to handle longer texts
        nlp.max_length = 2000000
        
        # Prepare texts with context
        texts_with_context = [
            (row['text_field'], {'Native_Language': row['Native_Language'], 'file_name': row['file_name']})
            for _, row in df.iterrows()
            if pd.notna(row['text_field']) and row['text_field'].strip()
        ]
        
        print(f"Total texts to process: {len(texts_with_context)}")
        
        # Process texts and detect passive voice
        results = []
        
        # Process in batches for memory efficiency
        batch_size = 50  # Smaller batch size for GPU memory conservation
        total_batches = (len(texts_with_context) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts_with_context))
            batch = texts_with_context[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} texts)")
            
            try:
                batch_results = []
                # Process each batch
                for doc, context in tqdm(nlp.pipe(batch, as_tuples=True),
                                        total=len(batch),
                                        desc=f"Batch {batch_idx + 1}"):
                    doc_id = context['file_name']
                    passive_phrases = custom_detector.extract_passive_phrases(doc)
                    
                    # A sentence is passive if at least one passive phrase is found
                    batch_results.append({
                        "DocID": doc_id,
                        "text_field": doc.text,
                        "custom_prediction": 1 if passive_phrases else 0
                    })
                
                results.extend(batch_results)
                print(f"Completed batch {batch_idx + 1}/{total_batches}")
                
                # Save intermediate results every 5 batches
                if batch_idx % 5 == 0 and batch_idx > 0:
                    interim_df = pd.DataFrame(results)
                    interim_df.to_csv(f"custom_detector_results_batch_{batch_idx}.csv", index=False)
                    print(f"Saved interim results up to batch {batch_idx}")
            
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                # Continue with the next batch instead of failing completely
                continue
        
        # Create final results DataFrame
        if not results:
            raise ValueError("No batches were successfully processed")
            
        results_df = pd.DataFrame(results)
        
        # Save final results
        results_df.to_csv("custom_detector_full_results.csv", index=False)
        print("Saved complete custom detector results")
        
        return results_df
    
    except Exception as e:
        print(f"Error processing with custom detector: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(df, has_human_annotations=True, output_dir="."):
    """Calculate and display metrics comparing implementations, and human labels if available"""
    print("\nCalculating metrics...")
    
    # Check if dataframe is empty
    if df.empty:
        print("Error: Dataframe is empty, cannot calculate metrics.")
        return None
    
    # Ensure we have the basic required columns
    required_cols = ['text_field', 'passivepy_prediction', 'custom_prediction']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing columns: {missing}")
        return None
    
    # If we have human annotations, check for human_coding column and compare
    if has_human_annotations:
        if 'human_coding' not in df.columns:
            print("Warning: has_human_annotations=True but no 'human_coding' column found.")
            has_human_annotations = False
        else:
            print("\n--- Human annotations vs PassivePy ---")
            print(classification_report(df['human_coding'], df['passivepy_prediction']))
            
            print("\n--- Human annotations vs Custom detector ---")
            print(classification_report(df['human_coding'], df['custom_prediction']))
            
            # Cohen's Kappa for human comparisons
            kappa_human_passivepy = cohen_kappa_score(df['human_coding'], df['passivepy_prediction'])
            kappa_human_custom = cohen_kappa_score(df['human_coding'], df['custom_prediction'])
            
            print(f"\nCohen's Kappa (Human-PassivePy): {kappa_human_passivepy:.4f}")
            print(f"Cohen's Kappa (Human-Custom): {kappa_human_custom:.4f}")
    
    # Always compare PassivePy vs Custom detector
    print("\n--- PassivePy vs Custom detector ---")
    print(classification_report(df['passivepy_prediction'], df['custom_prediction']))
    
    kappa_passivepy_custom = cohen_kappa_score(df['passivepy_prediction'], df['custom_prediction'])
    print(f"Cohen's Kappa (PassivePy-Custom): {kappa_passivepy_custom:.4f}")
    
    # Create confusion matrices
    plt.figure(figsize=(18, 6 if has_human_annotations else 3))
    
    if has_human_annotations:
        plt.subplot(1, 3, 1)
        cm_human_passivepy = confusion_matrix(df['human_coding'], df['passivepy_prediction'])
        sns.heatmap(cm_human_passivepy, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Passive', 'Passive'],
                    yticklabels=['Not Passive', 'Passive'])
        plt.title('Human vs PassivePy')
        plt.ylabel('Human Label')
        plt.xlabel('PassivePy Prediction')
        
        plt.subplot(1, 3, 2)
        cm_human_custom = confusion_matrix(df['human_coding'], df['custom_prediction'])
        sns.heatmap(cm_human_custom, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Passive', 'Passive'],
                    yticklabels=['Not Passive', 'Passive'])
        plt.title('Human vs Custom Detector')
        plt.ylabel('Human Label')
        plt.xlabel('Custom Detector Prediction')
        
        subplot_position = 3
    else:
        subplot_position = 1
        cm_human_passivepy = None
        cm_human_custom = None
    
    plt.subplot(1, 3 if has_human_annotations else 1, subplot_position)
    cm_passivepy_custom = confusion_matrix(df['passivepy_prediction'], df['custom_prediction'])
    sns.heatmap(cm_passivepy_custom, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Passive', 'Passive'],
                yticklabels=['Not Passive', 'Passive'])
    plt.title('PassivePy vs Custom Detector')
    plt.ylabel('PassivePy Prediction')
    plt.xlabel('Custom Detector Prediction')
    
    plt.tight_layout()
    
    # Generate a filename based on whether we're working with annotated data
    viz_filename = os.path.join(output_dir, f"implementation_comparison_{'annotated' if has_human_annotations else 'unannotated'}.png")
    plt.savefig(viz_filename)
    print(f"Comparison visualizations saved to '{viz_filename}'")
    
    # Create a detailed comparison dataframe for analysis
    disagreements = df[df['passivepy_prediction'] != df['custom_prediction']].copy()
    
    # Check if there are any disagreements
    if disagreements.empty:
        print("No disagreements found between PassivePy and Custom detector.")
        return {
            'kappa_human_passivepy': kappa_human_passivepy if has_human_annotations else None,
            'kappa_human_custom': kappa_human_custom if has_human_annotations else None,
            'kappa_passivepy_custom': kappa_passivepy_custom,
            'confusion_matrices': {
                'human_passivepy': cm_human_passivepy if has_human_annotations else None,
                'human_custom': cm_human_custom if has_human_annotations else None,
                'passivepy_custom': cm_passivepy_custom
            },
            'disagreements': None,
            'output_dir': output_dir
        }
    
    # Initialize the human_agrees_with column if we have human annotations
    if has_human_annotations:
        disagreements['human_agrees_with'] = 'Neither'
        
        # Use element-wise comparison for each row
        for idx, row in disagreements.iterrows():
            if row['human_coding'] == row['passivepy_prediction']:
                disagreements.at[idx, 'human_agrees_with'] = 'PassivePy'
            elif row['human_coding'] == row['custom_prediction']:
                disagreements.at[idx, 'human_agrees_with'] = 'Custom'
    
    # Save disagreements to CSV for manual review
    disagreement_filename = os.path.join(output_dir, f"implementation_disagreements_{'annotated' if has_human_annotations else 'unannotated'}.csv")
    disagreements.to_csv(disagreement_filename, index=False)
    print(f"Found {len(disagreements)} sentences where implementations disagree. Details saved to '{disagreement_filename}'")
    
    return {
        'kappa_human_passivepy': kappa_human_passivepy if has_human_annotations else None,
        'kappa_human_custom': kappa_human_custom if has_human_annotations else None,
        'kappa_passivepy_custom': kappa_passivepy_custom,
        'confusion_matrices': {
            'human_passivepy': cm_human_passivepy if has_human_annotations else None,
            'human_custom': cm_human_custom if has_human_annotations else None,
            'passivepy_custom': cm_passivepy_custom
        },
        'disagreements': disagreements,
        'output_dir': output_dir
    }

def analyze_disagreements(disagreements, has_human_annotations=True, output_dir="."):
    """Analyze patterns in disagreements between implementations"""
    if disagreements is None or (isinstance(disagreements, pd.DataFrame) and disagreements.empty):
        print("No disagreements found.")
        return
    
    plt.figure(figsize=(10, 8 if has_human_annotations else 4))
    
    if has_human_annotations:
        # Get counts of each agreement category
        agreement_counts = disagreements['human_agrees_with'].value_counts()
        
        plt.subplot(2, 1, 1)
        sns.barplot(x=agreement_counts.index, y=agreement_counts.values)
        plt.title('Human Agreement in Implementation Disagreements')
        plt.ylabel('Count')
        plt.xlabel('Human Agrees With')
        
        # Plot sentence length distribution by agreement category
        plt.subplot(2, 1, 2)
        disagreements['sentence_length'] = disagreements['text_field'].apply(len)
        sns.boxplot(x='human_agrees_with', y='sentence_length', data=disagreements)
        plt.title('Sentence Length by Agreement Category')
        plt.ylabel('Sentence Length (chars)')
        plt.xlabel('Human Agrees With')
    else:
        # Only plot sentence length distribution by detector prediction
        disagreements['sentence_length'] = disagreements['text_field'].apply(len)
        sns.boxplot(x='passivepy_prediction', y='sentence_length', data=disagreements)
        plt.title('Sentence Length by PassivePy Prediction')
        plt.ylabel('Sentence Length (chars)')
        plt.xlabel('PassivePy Prediction')
    
    plt.tight_layout()
    analysis_filename = os.path.join(output_dir, f"disagreement_analysis_{'annotated' if has_human_annotations else 'unannotated'}.png")
    plt.savefig(analysis_filename)
    print(f"Disagreement analysis saved to '{analysis_filename}'")
    
    # Additional analysis: Most common constructions in disagreements
    print("\nAnalyzing common patterns in disagreements...")
    try:
        # For computational efficiency, limit to a sample of disagreements if very large
        sample_size = min(500, len(disagreements))
        sample = disagreements.sample(sample_size) if len(disagreements) > sample_size else disagreements
        
        # Extract key phrases around disagreements (for example, 10 characters before and after)
        import re
        
        # Function to extract the verb phrase from the text
        def extract_verb_phrase(text):
            # Simple regex to find potential verb phrases (is/was/are/were + participle)
            matches = re.findall(r'\b(is|are|was|were|be|been|being)\s+(\w+ed|\w+en|\w+t)', text)
            return matches if matches else []
        
        # Extract verb phrases 
        sample['verb_phrases'] = sample['text_field'].apply(extract_verb_phrase)
        
        # Flatten the list of verb phrases
        all_phrases = [phrase for phrases in sample['verb_phrases'] for phrase in phrases]
        
        # Count occurrences
        from collections import Counter
        phrase_counts = Counter(all_phrases)
        
        # Get the top 20 most common phrases
        top_phrases = phrase_counts.most_common(20)
        
        # Create a dataframe for visualization
        if top_phrases:
            phrases_df = pd.DataFrame(top_phrases, columns=['Phrase', 'Count'])
            
            # Save to CSV
            phrases_file = os.path.join(output_dir, 'common_disagreement_phrases.csv')
            phrases_df.to_csv(phrases_file, index=False)
            print(f"Common disagreement phrases saved to {phrases_file}")
            
            # Print the top phrases
            print("\nMost common verb phrases in disagreements:")
            for phrase, count in top_phrases:
                print(f"  {phrase}: {count} occurrences")
    except Exception as e:
        print(f"Error in pattern analysis: {e}")

def load_annotated_samples():
    """Load the previously annotated samples."""
    try:
        df = pd.read_csv('annotation_results/annotated_samples.csv')
        logging.info(f"Loaded {len(df)} annotated samples")
        return df
    except Exception as e:
        logging.error(f"Error loading annotated samples: {str(e)}")
        raise

def run_passivepy(text: str, detector: PassivePy) -> List[Dict]:
    """Run PassivePy on a text and return results in compatible format."""
    try:
        results = detector.match_core_passives(text)
        return results
    except Exception as e:
        logging.error(f"Error running PassivePy: {str(e)}")
        return []

def compare_implementations(df: pd.DataFrame, n_samples: int = 100):
    """Compare PassivePy and custom implementation on a subset of samples."""
    try:
        # Initialize PassivePy
        detector = PassivePy()
        
        # Initialize spaCy for custom implementation
        nlp = spacy.load('en_core_web_sm')
        
        # Sample texts if needed
        if n_samples < len(df):
            df = df.sample(n=n_samples, random_state=42)
        
        results = []
        processing_times = {'passivepy': [], 'custom': []}
        
        logging.info(f"Starting comparison on {len(df)} texts...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = row['text']
            
            # Run PassivePy
            start_time = time.time()
            passivepy_results = run_passivepy(text, detector)
            processing_times['passivepy'].append(time.time() - start_time)
            
            # Run custom implementation
            start_time = time.time()
            custom_results = process_text(text, nlp)
            processing_times['custom'].append(time.time() - start_time)
            
            # Compare results
            passivepy_passives = len(passivepy_results)
            custom_passives = len(custom_results)
            
            results.append({
                'text_id': idx,
                'text': text,
                'passivepy_passives': passivepy_passives,
                'custom_passives': custom_passives,
                'passivepy_results': passivepy_results,
                'custom_results': custom_results
            })
        
        # Calculate metrics
        avg_time_passivepy = np.mean(processing_times['passivepy'])
        avg_time_custom = np.mean(processing_times['custom'])
        
        # Calculate agreement rate
        agreement = sum(1 for r in results if r['passivepy_passives'] == r['custom_passives']) / len(results)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'comparison_results.csv', index=False)
        
        # Save metrics
        metrics = {
            'total_texts': len(results),
            'average_time_passivepy': avg_time_passivepy,
            'average_time_custom': avg_time_custom,
            'agreement_rate': agreement,
            'total_passivepy_passives': sum(r['passivepy_passives'] for r in results),
            'total_custom_passives': sum(r['custom_passives'] for r in results)
        }
        
        with open(output_dir / 'metrics.txt', 'w') as f:
            f.write("Comparison Metrics\n")
            f.write("================\n\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        
        logging.info("Comparison completed successfully!")
        return metrics
        
    except Exception as e:
        logging.error(f"Error in comparison process: {str(e)}")
        raise

def main():
    try:
        # Load annotated samples
        df = load_annotated_samples()
        
        # Run comparison
        metrics = compare_implementations(df)
        
        # Print summary
        print("\nComparison Summary:")
        print("==================")
        print(f"Total texts compared: {metrics['total_texts']}")
        print(f"Average processing time - PassivePy: {metrics['average_time_passivepy']:.4f} seconds")
        print(f"Average processing time - Custom: {metrics['average_time_custom']:.4f} seconds")
        print(f"Agreement rate: {metrics['agreement_rate']:.2%}")
        print(f"Total passives detected - PassivePy: {metrics['total_passivepy_passives']}")
        print(f"Total passives detected - Custom: {metrics['total_custom_passives']}")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 