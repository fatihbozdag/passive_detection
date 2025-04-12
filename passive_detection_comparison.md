# Passive Voice Detection Comparison: Custom Detector vs. PassivePy

## Overview

This document presents a comparison between a custom passive voice detector implementation and the established PassivePy package. Both implementations were evaluated against a human-annotated dataset of 1,163 sentences.

## Performance Metrics

### Accuracy Comparison

| Implementation | Precision | Recall | F1-Score | Accuracy | Cohen's Kappa |
|----------------|-----------|--------|----------|----------|---------------|
| PassivePy      | 0.95      | 0.77   | 0.85     | 0.95     | 0.83          |
| Custom Detector| 0.92      | 0.92   | 0.92     | 0.97     | 0.90          |

The custom detector shows better overall performance with higher recall and F1-score for passive sentences, and a higher Cohen's Kappa score when compared to human annotations.

### Confusion Matrices

The confusion matrices show that:

1. PassivePy has a tendency to miss some passive sentences (more false negatives)
2. The custom detector is more balanced with similar rates of false positives and false negatives
3. Both implementations have very high precision for the "not passive" class

## Key Differences in Implementations

### Custom Detector Strengths:

1. More sophisticated handling of adjectival participles with an extensive dictionary of common exceptions
2. Better detection of subject validity in passive constructions
3. More accurate recognition of infinitive passive forms
4. Detailed analysis of aux chain structure to avoid false positives

### PassivePy Strengths:

1. Rule-based pattern matching using spaCy's matcher for efficiency
2. Good handling of common special cases (e.g., specific verbs that are often confused)

## Disagreement Analysis

The implementations disagreed on 56 sentences (4.8% of the dataset). In cases of disagreement:

- Humans agreed with the custom detector more often than with PassivePy
- Most disagreements occurred on shorter sentences
- Common points of disagreement included:
  - Adjectival participles vs. passive verb forms
  - Infinitive passives
  - Special verb forms

## Conclusion

The custom passive voice detector outperforms PassivePy on this dataset, particularly in terms of recall and overall agreement with human annotations. The key improvements come from more sophisticated handling of edge cases, especially adjectival participles and special verb forms.

For applications requiring high accuracy in passive voice detection, the custom implementation offers advantages over PassivePy, though both perform well for general use cases.

## Future Work

Potential improvements for passive voice detection:

1. Incorporating more contextual information to better distinguish adjectival participles
2. Adding language-specific rules for non-English texts
3. Training a machine learning model that combines rule-based features with contextual embeddings
4. Expanding the test dataset to include more domain-specific texts 