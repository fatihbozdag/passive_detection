#!/usr/bin/env python3
"""
ICLE Concordance Dataset Annotation Script

This script annotates the ICLE concordance dataset with both PassivePy and 
a custom passive voice detector, and compares the results.
"""

import os
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import argparse
from datetime import datetime
import json

# Import PassivePy from the correct module path
from PassivePySrc.PassivePy import PassivePyAnalyzer

# Import our custom passive detector
from run_passive_detector import process_text as my_process_text

# Constants
ICLE_PATH = "/Users/fatihbozdag/Documents/Cursor-Projects/PassivePy/icle_concord.csv"
OUTPUT_DIR = "icle_analysis"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Annotate ICLE concordance dataset with PassivePy and custom detector")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to process")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for results")
    return parser.parse_args()

def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Create a progress log file
    progress_file = os.path.join(output_dir, "progress.md")
    if not os.path.exists(progress_file):
        with open(progress_file, "w") as f:
            f.write("# ICLE Annotation Progress\n\n")
            f.write("| Step | Status | Time | Details |\n")
            f.write("|------|--------|------|--------|\n")
    
    return progress_file

def load_dataset(icle_path, limit=None):
    """Load the ICLE concordance dataset."""
    try:
        df = pd.read_csv(icle_path)
        
        if limit is not None:
            df = df.head(limit)
        
        # Create a sentence column by combining Left, Center, and Right
        df['sentence'] = df['Left'] + ' ' + df['Center'] + ' ' + df['Right']
        df['sentence'] = df['sentence'].str.strip()
        
        print(f"Loaded {len(df)} examples from {icle_path}")
        print(f"Sample of combined sentences: {df['sentence'].head().tolist()}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def annotate_with_passivepy(df, output_dir, progress_file):
    """Annotate the dataset with PassivePy."""
    update_progress(progress_file, "PassivePy Annotation", "Started", "")
    start_time = time.time()
    
    try:
        # Initialize PassivePy
        passivepy = PassivePyAnalyzer(spacy_model="en_core_web_sm")
        
        # Process texts
        print("Annotating with PassivePy...")
        
        # Match at sentence level
        results_df = passivepy.match_sentence_level(
            df=df,
            column_name="sentence",
            n_process=1,
            batch_size=100,
            add_other_columns=True
        )
        
        # Print the columns to debug
        print(f"PassivePy results columns: {results_df.columns.tolist()}")
        
        # Save results
        output_path = os.path.join(output_dir, "passivepy_results.csv")
        results_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "PassivePy Annotation", "Completed", f"Time: {elapsed_time:.2f}s, Saved to {output_path}")
        
        return results_df
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "PassivePy Annotation", "Failed", f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error annotating with PassivePy: {e}")
        return None

def annotate_with_custom_detector(df, output_dir, progress_file):
    """Annotate the dataset with our custom passive voice detector."""
    update_progress(progress_file, "Custom Detector Annotation", "Started", "")
    start_time = time.time()
    
    try:
        # Load spaCy model
        print("Loading spaCy model for custom detector...")
        nlp = spacy.load("en_core_web_sm")
        
        # Process texts
        print("Annotating with custom detector...")
        texts = df["sentence"].tolist()
        
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing {i}/{len(texts)} with custom detector...")
            
            # Call our custom detector on the text
            try:
                passive_result = my_process_text(text, nlp)
                
                # Add the original text and index
                passive_result["original_text"] = text
                passive_result["index"] = i
                
                # Add a binary passive flag similar to PassivePy
                passive_result["binary"] = 1 if passive_result["passive_ratio"] > 0 else 0
                
                results.append(passive_result)
            except Exception as e:
                print(f"Error processing text {i} with custom detector: {e}")
                results.append({
                    "original_text": text,
                    "index": i,
                    "error": str(e),
                    "binary": 0
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        output_path = os.path.join(output_dir, "custom_detector_results.csv")
        results_df.to_csv(output_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "Custom Detector Annotation", "Completed", f"Time: {elapsed_time:.2f}s, Saved to {output_path}")
        
        return results_df
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "Custom Detector Annotation", "Failed", f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error annotating with custom detector: {e}")
        return None

def compare_annotations(passivepy_results, custom_results, output_dir, progress_file):
    """Compare the annotations from PassivePy and the custom detector."""
    update_progress(progress_file, "Annotation Comparison", "Started", "")
    start_time = time.time()
    
    try:
        # Ensure we have both results
        if passivepy_results is None or custom_results is None:
            raise ValueError("Missing results from one or both detectors")
        
        # Merge the results on the index
        passivepy_results = passivepy_results.rename(columns={"binary": "passivepy_binary"})
        custom_results = custom_results.rename(columns={"binary": "custom_binary"})
        
        # Select only necessary columns to avoid duplicates
        passivepy_results = passivepy_results[["index", "passivepy_binary"]]
        
        # Prepare for merge
        if "index" not in custom_results.columns:
            custom_results["index"] = custom_results.index
        
        # Select columns for merge
        custom_select = ["index", "custom_binary", "original_text"]
        
        # Ensure all required columns exist
        for col in custom_select:
            if col not in custom_results.columns:
                if col == "custom_binary":
                    # Try to derive it from other fields
                    if "passive_ratio" in custom_results.columns:
                        custom_results["custom_binary"] = custom_results["passive_ratio"].apply(lambda x: 1 if x > 0 else 0)
                    else:
                        custom_results["custom_binary"] = 0  # Default
                elif col == "original_text" and "original_text" not in custom_results.columns:
                    if "text" in custom_results.columns:
                        custom_results["original_text"] = custom_results["text"]
                    else:
                        custom_results["original_text"] = ""  # Default
        
        # Merge results
        comparison_df = pd.merge(custom_results[custom_select], passivepy_results, on="index", how="inner")
        
        # Calculate agreement metrics
        agreement = (comparison_df["passivepy_binary"] == comparison_df["custom_binary"]).mean()
        print(f"Agreement between PassivePy and custom detector: {agreement:.2%}")
        
        # Create confusion matrix
        cm = confusion_matrix(comparison_df["passivepy_binary"], comparison_df["custom_binary"])
        
        # Create classification report
        report = classification_report(comparison_df["passivepy_binary"], comparison_df["custom_binary"], 
                                      target_names=["Non-Passive", "Passive"], output_dict=True)
        
        # Save results
        comparison_path = os.path.join(output_dir, "comparison_results.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        # Save metrics
        metrics = {
            "agreement": agreement,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        metrics_path = os.path.join(output_dir, "comparison_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualizations
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["Non-Passive (Custom)", "Passive (Custom)"],
                   yticklabels=["Non-Passive (PassivePy)", "Passive (PassivePy)"])
        plt.title("Confusion Matrix: PassivePy vs Custom Detector")
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        
        # Create a more detailed comparison for disagreements
        disagreements = comparison_df[comparison_df["passivepy_binary"] != comparison_df["custom_binary"]]
        disagreements_path = os.path.join(output_dir, "disagreements.csv")
        disagreements.to_csv(disagreements_path, index=False)
        
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "Annotation Comparison", "Completed", 
                      f"Time: {elapsed_time:.2f}s, Agreement: {agreement:.2%}, " +
                      f"Saved to {comparison_path}, {metrics_path}, {cm_path}, {disagreements_path}")
        
        return comparison_df, metrics
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        update_progress(progress_file, "Annotation Comparison", "Failed", f"Time: {elapsed_time:.2f}s, Error: {str(e)}")
        print(f"Error comparing annotations: {e}")
        return None, None

def update_progress(progress_file, step, status, details):
    """Update the progress in the markdown file."""
    with open(progress_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"| {step} | {status} | {timestamp} | {details} |\n")

def main():
    """Main function to run the annotation."""
    # Parse arguments
    args = parse_args()
    
    # Ensure output directory exists
    progress_file = ensure_output_dir(args.output_dir)
    
    # Update progress
    update_progress(progress_file, "Initialization", "Completed", f"Output directory: {args.output_dir}")
    
    # Load dataset
    df = load_dataset(ICLE_PATH, args.limit)
    if df is None:
        update_progress(progress_file, "Dataset Loading", "Failed", "Could not load dataset")
        return
    
    update_progress(progress_file, "Dataset Loading", "Completed", f"Loaded {len(df)} examples")
    
    # Annotate with PassivePy
    passivepy_results = annotate_with_passivepy(df, args.output_dir, progress_file)
    
    # Annotate with custom detector
    custom_results = annotate_with_custom_detector(df, args.output_dir, progress_file)
    
    # Compare annotations
    comparison_df, metrics = compare_annotations(passivepy_results, custom_results, args.output_dir, progress_file)
    
    # Final update
    if comparison_df is not None:
        update_progress(progress_file, "Full Annotation Process", "Completed", 
                      f"Processed {len(df)} examples, agreement: {metrics['agreement']:.2%}")
    else:
        update_progress(progress_file, "Full Annotation Process", "Failed", "Could not complete comparison")

if __name__ == "__main__":
    main() 