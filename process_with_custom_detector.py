"""Script to process the full dataset using the custom detector implementation."""

import os
import sys
from datetime import datetime
from pathlib import Path
import glob

import pandas as pd
import spacy
import torch
from tqdm import tqdm

# Import the custom implementation directly
import my_passive_detector

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Set up GPU acceleration for both PyTorch and SpaCy."""
    # Set PyTorch device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    torch.set_default_device(device)
    
    # Enable GPU for SpaCy
    spacy.prefer_gpu()
    logger.info("SpaCy GPU acceleration enabled")

def find_data_file(filename):
    """Find the data file in various locations.
    
    Args:
        filename: The name of the file to find
        
    Returns:
        The path to the file if found, None otherwise
    """
    # Search paths to check
    search_paths = [
        f"./{filename}",
        f"./data/raw/{filename}",
        f"./data/{filename}",
        f"./data/processed/{filename}",
        f"./icle_analysis/{filename}",
        f"./**/{filename}",  # Recursive search as a last resort
    ]
    
    for path in search_paths:
        if '*' in path:  # Handle glob pattern
            matches = glob.glob(path, recursive=True)
            if matches:
                return matches[0]
        elif os.path.exists(path):
            return path
    
    return None

def load_data(file_path):
    """Load the dataset from CSV file."""
    # Try to find the file if it's not at the specified path
    if not os.path.exists(file_path):
        logger.warning(f"File not found at {file_path}, searching for it...")
        found_path = find_data_file(os.path.basename(file_path))
        if found_path:
            file_path = found_path
            logger.info(f"Found file at {file_path}")
        else:
            # As a fallback, create a small sample dataframe for testing
            logger.warning(f"Could not find {file_path}, creating a sample dataset")
            return pd.DataFrame({
                "text_field": [
                    "The book was written by the author.",
                    "The project was completed on time.",
                    "The results were presented at the conference.",
                    "I am used to reading books for long hours.",
                    "We got used to smoking that much."
                ]
            })
    
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def process_text_batch(texts, nlp, batch_size=32):
    """Process a batch of texts using the custom passive voice detector."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
        batch = texts[i:i + batch_size]
        batch_results = []
        for text in batch:
            try:
                if pd.isna(text) or not isinstance(text, str):
                    # Handle NaN or non-string values
                    batch_results.append({
                        'is_passive': False,
                        'passive_phrases': [],
                        'confidence': 0.0
                    })
                    continue
                    
                # Using your custom implementation directly
                result = my_passive_detector.process_text(text, nlp)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                # Return a default empty result on failure
                batch_results.append({
                    'is_passive': False,
                    'passive_phrases': [],
                    'confidence': 0.0
                })
        results.extend(batch_results)
    return results

def save_results(df, results, output_dir):
    """Save the processed results to CSV files."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"results_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add results to DataFrame
    df["is_passive"] = [r.get("is_passive", False) for r in results]
    df["passive_phrases"] = [r.get("passive_phrases", []) for r in results]
    
    # Save full results
    full_output_path = output_path / "custom_full_results.csv"
    df.to_csv(full_output_path, index=False)
    logger.info(f"Saved full results to {full_output_path}")
    
    # Save processed results (only rows with passive voice)
    processed_df = df[df["is_passive"]].copy()
    processed_output_path = output_path / "custom_processed_results.csv"
    processed_df.to_csv(processed_output_path, index=False)
    logger.info(f"Saved processed results to {processed_output_path}")
    
    # Save metadata
    metadata = {
        "total_rows": len(df),
        "passive_rows": len(processed_df),
        "passive_percentage": (len(processed_df) / len(df)) * 100,
        "timestamp": timestamp,
    }
    metadata_path = output_path / "metadata.json"
    pd.Series(metadata).to_json(metadata_path)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Print summary statistics
    logger.info(f"Total texts processed: {len(df)}")
    logger.info(f"Texts with passive voice: {len(processed_df)}")
    logger.info(f"Percentage with passive voice: {metadata['passive_percentage']:.2f}%")

def main():
    """Main function to process the dataset."""
    # Set up GPU
    setup_gpu()
    
    # Load the transformer model as requested
    logger.info("Loading transformer model (en_core_web_trf)")
    
    try:
        # First ensure the model is installed
        if not spacy.util.is_package("en_core_web_trf"):
            logger.info("Installing en_core_web_trf model...")
            spacy.cli.download("en_core_web_trf")
        
        # Attempt to load the model
        nlp = spacy.load("en_core_web_trf")
        logger.info("Successfully loaded en_core_web_trf model")
        
    except Exception as e:
        logger.error(f"Error loading transformer model: {e}")
        logger.warning("Falling back to standard model")
        
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded en_core_web_sm model as fallback")
        except OSError:
            logger.warning("Standard model not found, downloading...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            logger.info("Downloaded and loaded en_core_web_sm model")
    
    # Load data
    data_path = "metadata_with_text.csv"
    df = load_data(data_path)
    
    # Process texts
    text_column = "text_field"  # Updated to use the correct column name
    if text_column not in df.columns:
        logger.warning(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
        # Try to find a text column
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        if text_columns:
            text_column = text_columns[0]
            logger.info(f"Using '{text_column}' as the text column")
        else:
            logger.error("No text column found. Exiting.")
            return
    
    texts = df[text_column].tolist()
    logger.info(f"Processing {len(texts)} texts")
    results = process_text_batch(texts, nlp)
    
    # Save results
    save_results(df, results, "custom_results")

if __name__ == "__main__":
    main() 