import pandas as pd
import numpy as np
import re
from typing import List, Dict
import random
import logging
import sys
import os
from pathlib import Path
import spacy

# Import local PassivePy implementation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from my_passive_detector import process_text

# Create output directory
output_dir = Path('annotation_results')
output_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(output_dir / 'annotation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def sample_balanced_texts(df: pd.DataFrame, samples_per_language: int = 20) -> pd.DataFrame:
    """Sample texts while maintaining balance across native languages."""
    try:
        sampled_texts = []
        for lang in df['Native_Language'].unique():
            lang_texts = df[df['Native_Language'] == lang]
            if len(lang_texts) >= samples_per_language:
                sampled = lang_texts.sample(n=samples_per_language, random_state=42)
                sampled_texts.append(sampled)
                logging.info(f"Sampled {samples_per_language} texts for language: {lang}")
        
        return pd.concat(sampled_texts)
    except Exception as e:
        logging.error(f"Error in sampling texts: {str(e)}")
        raise

def custom_passive_detection(text: str, nlp) -> List[Dict]:
    """Your custom implementation of passive voice detection."""
    try:
        return process_text(text, nlp)
    except Exception as e:
        logging.error(f"Error in custom passive detection: {str(e)}")
        return []

def main():
    try:
        # Read the dataset
        logging.info("Reading dataset...")
        df = pd.read_csv('metadata_with_text.csv')
        logging.info(f"Dataset loaded with {len(df)} rows")
        
        # Sample texts
        logging.info("Sampling texts...")
        sampled_df = sample_balanced_texts(df)
        logging.info(f"Sampled {len(sampled_df)} texts")
        
        # Save sampled texts
        sampled_df.to_csv(output_dir / 'sampled_texts.csv', index=False)
        logging.info(f"Saved sampled texts to {output_dir / 'sampled_texts.csv'}")
        
        # Initialize spaCy
        logging.info("Loading spaCy model...")
        nlp = spacy.load('en_core_web_sm')
        
        # Create results DataFrame
        results = []
        
        logging.info("Starting annotation process...")
        for idx, row in sampled_df.iterrows():
            try:
                text = row['text_field']
                logging.info(f"Processing text {idx + 1}/{len(sampled_df)}")
                
                # Get custom implementation annotations
                custom_results = custom_passive_detection(text, nlp)
                
                results.append({
                    'file_name': row['file_name'],
                    'native_language': row['Native_Language'],
                    'text': text,
                    'custom_annotations': custom_results
                })
            except Exception as e:
                logging.error(f"Error processing text {idx}: {str(e)}")
                continue
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'annotated_samples.csv', index=False)
        logging.info(f"Successfully saved {len(results_df)} annotated samples to {output_dir / 'annotated_samples.csv'}")
        
        # Save statistics
        stats = {
            'total_texts': len(results_df),
            'languages_represented': len(results_df['native_language'].unique()),
            'texts_per_language': results_df['native_language'].value_counts().to_dict()
        }
        
        with open(output_dir / 'statistics.txt', 'w') as f:
            f.write("Annotation Statistics\n")
            f.write("===================\n\n")
            f.write(f"Total texts annotated: {stats['total_texts']}\n")
            f.write(f"Number of languages represented: {stats['languages_represented']}\n\n")
            f.write("Texts per language:\n")
            for lang, count in stats['texts_per_language'].items():
                f.write(f"{lang}: {count}\n")
        
        logging.info("Annotation process completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 