# Advanced Passive Voice Detector

An enhanced passive voice detection system with improved accuracy over existing tools.

## Overview

This passive voice detector uses advanced natural language processing techniques to accurately identify passive voice constructions in English text. It addresses common challenges such as distinguishing adjectival participles from true passive forms, handling infinitive passives, and recognizing special verb forms.

## Features

- **High Accuracy**: Outperforms PassivePy on benchmark tests with 97% accuracy and 0.90 Cohen's Kappa
- **Detailed Analysis**: Provides comprehensive information about each passive construction
- **Adjectival Participle Handling**: Sophisticated detection of adjectives that look like passives
- **Infinitive Passive Support**: Correctly identifies passive infinitives (e.g., "to be completed")
- **Command-line Interface**: Easy-to-use CLI for analyzing text or files

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/passive-detector.git
cd passive-detector

# Install dependencies
pip install -r requirements.txt

# Install SpaCy model
python -m spacy download en_core_web_sm
```

## Usage

### Command-line Interface

```bash
# Analyze a text string
python run_passive_detector.py --text "The book was written by the author."

# Analyze a file
python run_passive_detector.py --file input.txt

# Save results to CSV
python run_passive_detector.py --file input.txt --output results.csv

# Show detailed information
python run_passive_detector.py --text "The book was written by the author." --detailed
```

### As a Python Module

```python
import spacy
import my_passive_detector as detector

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Process text
doc = nlp("The book was written by the author.")
passive_phrases = detector.extract_passive_phrases(doc)

# Check if text contains passive voice
is_passive = bool(passive_phrases)

# Get details about passive phrases
for phrase in passive_phrases:
    print(f"Passive phrase: {phrase['passive_phrase']}")
    print(f"Main verb: {phrase['Lemmatized_Main_Verb']}")
```

## How It Works

The detector follows these steps to identify passive voice:

1. **Part-of-Speech Tagging**: Identifies past participles (VBN tags)
2. **Adjectival Filtering**: Determines if the participle is being used as an adjective
3. **Auxiliary Verification**: Checks for "be" verbs in the auxiliary chain
4. **Subject Validation**: Ensures the construction has a valid passive subject
5. **Phrase Extraction**: Captures the full passive phrase with context

## Performance

In a comparison with PassivePy using a dataset of 1,163 human-annotated sentences:

| Implementation | Precision | Recall | F1-Score | Accuracy | Cohen's Kappa |
|----------------|-----------|--------|----------|----------|---------------|
| PassivePy      | 0.95      | 0.77   | 0.85     | 0.95     | 0.83          |
| Custom Detector| 0.92      | 0.92   | 0.92     | 0.97     | 0.90          |

## Limitations

- Currently only supports English text
- Requires SpaCy language model
- May have reduced accuracy on domain-specific text

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PassivePy project for the benchmark comparison
- SpaCy for the underlying NLP functionality 