# PassivePy Project Progress

This document tracks the progress and improvements made to the PassivePy project, a Python library for passive voice detection and analysis.

## Project Structure and Organization

### Initial Setup
- Restructured the project to follow modern Python packaging conventions
- Created `src/passivepy` package structure with clear module organization
- Implemented proper packaging with `pyproject.toml` and dependency management
- Added comprehensive documentation including README, CONTRIBUTING, and CHANGELOG files

### Code Structure
- Organized code into logical modules:
  - `core/`: Core passive voice detection algorithms
  - `analysis/`: Tools for analyzing passive voice usage patterns
  - `utils/`: Helper functions and utilities
  - `tests/`: Comprehensive test suite

### Development Environment
- Added `Makefile` for common development tasks
- Implemented pre-commit hooks for code quality
- Set up logging configuration for better debugging
- Created virtual environment management with `rye`

## Continuous Integration & Quality Assurance

### CI/CD Pipeline
- Implemented GitHub Actions workflow for automated testing
- Added quality checks including linting, formatting, and type checking
- Set up test coverage reporting

### Code Quality Tools
- Integrated Ruff for linting and code quality
- Added mypy for static type checking
- Implemented pytest for test automation
- Created pre-commit hooks for enforcing standards

## Documentation Improvements

### User Documentation
- Created comprehensive README with installation and usage examples
- Added API documentation with detailed function descriptions
- Included examples of common use cases

### Developer Documentation
- Created CONTRIBUTING.md with guidelines for contributors
- Added detailed inline docstrings for all public functions
- Documented internal architecture and design decisions

## Passive Voice Detection Enhancements

### Algorithm Improvements
- Enhanced passive voice detection to reduce false positives
- Implemented more sophisticated pattern matching for better accuracy
- Added support for complex passive constructions
- Improved handling of edge cases

### Custom Implementation
- Created a custom passive detector implementation with:
  - More accurate detection patterns
  - Better handling of complex sentences
  - Reduced false positives with improved heuristics
  - Support for more passive voice constructions

## Dataset Processing

### Processing Pipeline
- Created script for processing large datasets with GPU acceleration
- Implemented batch processing to handle memory constraints
- Added progress tracking and error handling
- Optimized for performance with PyTorch and SpaCy GPU acceleration

### Custom Processing
- Developed `process_with_custom_detector.py` for processing datasets using the custom detector
- Implemented robust file handling and path searching
- Enhanced logging for better visibility into processing status
- Added fallback mechanisms for handling missing files

## Formulaicity Analysis

### Analysis Framework
- Developed comprehensive analyzer for measuring formulaicity in passive constructions
- Implemented various formulaicity metrics:
  - N-gram frequency analysis (bigrams, trigrams, quadgrams)
  - Variant-to-phrase-frame ratio (VPR)
  - Simpson's diversity index and Balance
  - Hapaxity and Haprate measures

### Analysis Approaches
- POS-Tag based analysis of syntactic patterns
- Word-based analysis of lexical patterns
- Main verb lemma-based analysis for deeper pattern recognition

### Visualization
- Created visualization tools for analysis results:
  - Bar charts of top n-grams
  - Scatter plots of VPR values
  - Histograms of diversity indices
  - Visualizations of hapaxity measures
  - Word pattern diversity by length

### Performance Optimization
- Implemented GPU acceleration for faster processing
- Used batched processing to manage memory usage
- Added memory optimization techniques for large datasets
- Integrated PyTorch MPS for Mac GPU acceleration

## Results and Findings

### Passive Voice Statistics
- Processed 9,529 texts with 97.55% containing passive voice
- Identified 25,556 passive frames with 1,125 unique main verbs
- Discovered 62 unique POS patterns with an average length of 2.07 words

### Formulaicity Insights
- Identified common formulaic patterns:
  - Most frequent bigrams: "has been", "have been", "allowed to"
  - Most frequent trigrams: "is dominated by", "has been argued", "is caused by"
  - Most frequent lemma patterns: "have be", "be give", "be make"

- Discovered verb-specific formulaicity:
  - Some verbs (dominate, symbolize) show high formulaicity
  - Others (love, respect) show more variability
  - Strong auxiliary preferences for specific verbs

- Analyzed pattern diversity:
  - 1-word patterns: Very low diversity (0.018)
  - 2-word patterns: Moderate diversity (0.165)
  - 3-word patterns: Higher diversity (0.452)

## Future Work

### Planned Improvements
- Fine-tune passive detection with machine learning approaches
- Expand formulaicity analysis with more sophisticated metrics
- Add support for cross-linguistic passive detection
- Develop interactive visualization tools for exploration

### Research Directions
- Investigate the relationship between formulaicity and language proficiency
- Examine cross-genre differences in passive voice usage
- Study developmental patterns in passive voice acquisition
- Explore pedagogical applications of formulaicity analysis

## Timeline

1. Project restructuring and modern Python setup
2. Continuous integration and quality assurance implementation
3. Documentation improvements and standardization
4. Enhancement of passive voice detection algorithms
5. Development of custom detector implementation
6. Dataset processing with GPU acceleration
7. Formulaicity analysis framework development
8. Comprehensive analysis of passive voice patterns
9. Visualization and result interpretation
