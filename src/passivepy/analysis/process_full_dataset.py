"""Script to process the full dataset with GPU acceleration."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import spacy
import torch
from tqdm import tqdm

from passivepy import PassiveDetector
from passivepy.core.types import DetectorConfig
from passivepy.utils.logging import setup_logging

logger = setup_logging()


def setup_gpu() -> None:
    """Set up GPU acceleration for both PyTorch and SpaCy."""
    # Set PyTorch device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Enable GPU for SpaCy
    spacy.prefer_gpu()
    logger.info("SpaCy GPU acceleration enabled")


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    logger.info("Loading data from %s", file_path)
    return pd.read_csv(file_path)


def process_text_batch(
    texts: List[str],
    detector: PassiveDetector,
    batch_size: int = 32
) -> List[Dict]:
    """Process a batch of texts using the passive voice detector.
    
    Args:
        texts: List of texts to process
        detector: Passive voice detector instance
        batch_size: Size of each batch
        
    Returns:
        List of detection results
    """
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
        batch = texts[i:i + batch_size]
        batch_results = [detector.detect(text) for text in batch]
        results.extend(batch_results)
    return results


def save_results(
    df: pd.DataFrame,
    results: List[Dict],
    output_dir: str
) -> None:
    """Save the processed results to CSV files.
    
    Args:
        df: Original DataFrame
        results: List of detection results
        output_dir: Directory to save results
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"results_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add results to DataFrame
    df["is_passive"] = [r["is_passive"] for r in results]
    df["passive_phrases"] = [r["passive_phrases"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    
    # Save full results
    full_output_path = output_path / "passivepy_full_results.csv"
    df.to_csv(full_output_path, index=False)
    logger.info("Saved full results to %s", full_output_path)
    
    # Save processed results (only rows with passive voice)
    processed_df = df[df["is_passive"]].copy()
    processed_output_path = output_path / "passivepy_processed_results.csv"
    processed_df.to_csv(processed_output_path, index=False)
    logger.info("Saved processed results to %s", processed_output_path)
    
    # Save metadata
    metadata = {
        "total_rows": len(df),
        "passive_rows": len(processed_df),
        "passive_percentage": (len(processed_df) / len(df)) * 100,
        "timestamp": timestamp,
    }
    metadata_path = output_path / "metadata.json"
    pd.Series(metadata).to_json(metadata_path)
    logger.info("Saved metadata to %s", metadata_path)


def main():
    """Main function to process the dataset."""
    # Set up GPU
    setup_gpu()
    
    # Initialize detector with transformer model
    config = DetectorConfig(
        threshold=0.7,
        use_spacy=True,
        language="en",
    )
    detector = PassiveDetector(config)
    
    # Load data
    data_path = "data/raw/metadata_with_text.csv"
    df = load_data(data_path)
    
    # Process texts
    texts = df["text"].tolist()
    results = process_text_batch(texts, detector)
    
    # Save results
    save_results(df, results, "data/results")


if __name__ == "__main__":
    main() 