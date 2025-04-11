# Passive Voice Detection

A Python library for detecting passive voice in text, building upon and extending the capabilities of PassivePy. This implementation offers enhanced features and improved accuracy in specific cases while maintaining comparable overall performance.

## Project Overview

The project focuses on improving passive voice detection through:

1. Advanced regex pattern matching
2. Dependency parsing with SpaCy
3. Special case handling for commonly missed passive constructions
4. Enhanced filtering to reduce false positives

## Repository Structure

- `src/`: Core implementation of passive voice detection
- `my_passive_detector.py`: Custom passive voice detection implementation
- `run_passive_detector.py`: Script for running passive detection on text data
- `test_passivepy.py`: Test suite for passive detection
- `progress.md`: Documentation of project progress and findings

## Key Features

### Enhanced Passive Detection

Our implementation offers improved detection for:

- Election-related passives: "was elected", "was chosen", "was selected"
- Event-related passives: "was held", "was organized", "was cancelled"
- Negated passives: "isn't considered", "wasn't seen"
- "Left + participle" constructions: "was left saddened"
- Simple subject-verb passives: "Sports are played"
- By-agent passives across different tenses

### Reduced False Positives

Special filtering for:

- Adjectival states that look like passives: "is heated", "is excited"
- Emotional/mental states that aren't passive: "am stunned", "is surprised"
- Active voice constructions that resemble passives: "is having"
- Non-passive copular constructions: "is topic", "is subject"

## Performance

Our implementation:

- Achieves an F1-score of 0.85 on the crowdsource dataset
- Provides better recall (0.84) compared to baseline
- Shows strong agreement with human annotations (Cohen's Kappa: 0.8186)

## Getting Started

### Prerequisites

- Python 3.6+
- SpaCy with English model: `pip install spacy && python -m spacy download en_core_web_sm`
- Required libraries: Install from requirements.txt

### Installation

```bash
# Clone the repository
git clone https://github.com/fatihbozdag/passive_detection.git
cd passive_detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
import my_passive_detector as detector
import spacy

nlp = spacy.load("en_core_web_sm")
text = "The bill was approved by the committee."
result = detector.process_text(text, nlp)
print(f"Is passive: {result['is_passive']}")
print(f"Passive phrases: {result['passive_phrases']}")
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project builds upon the work of PassivePy by Sepehri, A., Mirshafiee, M. S., & Markowitz, D. M. (2022). PassivePy: A tool to automatically identify passive voice in big text data. Journal of Consumer Psychology.
