import pandas as pd
import numpy as np
from passive_voice_detector import PassiveVoiceDetector
import spacy
import torch
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Set up GPU acceleration
device = torch.device('mps')
spacy.prefer_gpu()

def create_balanced_sample(df, n_samples=500, group_column='Native_Language'):
    """Create a balanced sample from the dataset."""
    # Get the minimum number of samples per group
    min_samples = min(df[group_column].value_counts())
    samples_per_group = min(min_samples, n_samples // len(df[group_column].unique()))
    
    # Sample from each group
    balanced_sample = pd.DataFrame()
    for group in df[group_column].unique():
        group_df = df[df[group_column] == group]
        if len(group_df) > samples_per_group:
            sampled = group_df.sample(n=samples_per_group, random_state=42)
        else:
            sampled = group_df
        balanced_sample = pd.concat([balanced_sample, sampled])
    
    return balanced_sample

def detect_passive_spacy(text, nlp):
    """Detect passive voice using spaCy's dependency parser."""
    doc = nlp(text)
    is_passive = False
    
    for token in doc:
        if token.dep_ == "nsubjpass" or token.dep_ == "auxpass":
            is_passive = True
            break
    
    return {'is_passive': is_passive}

def annotate_texts(df, text_column='text_field', use_sample=False):
    """Annotate texts using both spaCy and our implementation."""
    # Initialize detectors
    print("Loading spaCy transformer model...")
    nlp = spacy.load("en_core_web_trf")
    our_detector = PassiveVoiceDetector()
    
    # Create results DataFrame
    results = df.copy()
    
    # Annotate with spaCy
    print("Annotating with spaCy...")
    spacy_results = []
    for text in tqdm(df[text_column], desc="Processing with spaCy"):
        result = detect_passive_spacy(text, nlp)
        spacy_results.append(result['is_passive'])
    results['spacy_annotation'] = spacy_results
    
    # Annotate with our implementation
    print("Annotating with our implementation...")
    our_results = []
    for text in tqdm(df[text_column], desc="Processing with our detector"):
        result = our_detector.detect_passive_voice(text)
        our_results.append(result['is_passive'])
    results['our_annotation'] = our_results
    
    return results

def calculate_metrics(results):
    """Calculate comparison metrics between the two annotation methods."""
    metrics = {}
    
    # Basic agreement metrics
    agreement = (results['spacy_annotation'] == results['our_annotation']).mean()
    metrics['agreement_rate'] = agreement
    
    # Classification report
    report = classification_report(
        results['spacy_annotation'],
        results['our_annotation'],
        output_dict=True
    )
    metrics['classification_report'] = report
    
    # Confusion matrix
    cm = confusion_matrix(
        results['spacy_annotation'],
        results['our_annotation']
    )
    metrics['confusion_matrix'] = cm
    
    return metrics

def save_results(results, metrics, output_dir='comparison_results'):
    """Save the results and metrics to files."""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save annotated dataset
    results.to_csv(f'{output_dir}/annotated_dataset.csv', index=False)
    
    # Save metrics
    with open(f'{output_dir}/metrics.txt', 'w') as f:
        f.write(f"Agreement Rate: {metrics['agreement_rate']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(str(metrics['classification_report']))
        
        # Add dataset size information
        f.write(f"\n\nDataset Size: {len(results)} texts\n")
        f.write(f"Passive Voice (spaCy): {results['spacy_annotation'].sum()} texts\n")
        f.write(f"Passive Voice (Our Implementation): {results['our_annotation'].sum()} texts\n")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix: spaCy vs Our Implementation')
    plt.xlabel('Our Implementation')
    plt.ylabel('spaCy')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

def main():
    # Remove previous results
    if os.path.exists('comparison_results'):
        for file in os.listdir('comparison_results'):
            os.remove(os.path.join('comparison_results', file))
    
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv('metadata_with_text.csv')
    
    # Process the entire dataset
    print(f"Processing {len(df)} texts...")
    results = annotate_texts(df, use_sample=False)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Save results
    print("Saving results...")
    save_results(results, metrics)
    
    print("Analysis complete! Results saved in 'comparison_results' directory.")

if __name__ == "__main__":
    main() 