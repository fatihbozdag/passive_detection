# PassivePy

A Python package for passive voice detection and analysis in text.

## Features

- Passive voice detection in text
- Analysis of passive voice patterns
- Comparison of different detection methods
- Visualization of results
- Support for multiple languages

## Installation

```bash
# Using pip
pip install passivepy

# Using rye
rye sync
```

## Usage

```python
from passivepy import PassiveDetector

# Initialize detector
detector = PassiveDetector()

# Detect passive voice in text
text = "The book was written by the author."
results = detector.detect(text)

# Analyze results
analysis = detector.analyze(results)
```

## Project Structure

```
passivepy/
├── src/
│   └── passivepy/
│       ├── core/          # Core detection logic
│       ├── utils/         # Utility functions
│       ├── analysis/      # Analysis tools
│       └── tests/         # Test suite
├── docs/                  # Documentation
├── scripts/              # Utility scripts
└── data/                 # Data files
    ├── raw/             # Raw data
    ├── processed/       # Processed data
    └── results/         # Analysis results
```

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/passivepy.git
cd passivepy
```

2. Install development dependencies:
```bash
rye sync
```

3. Run tests:
```bash
pytest
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
