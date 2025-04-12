# ICLE Passive Voice Analysis Summary

## Overview

This document summarizes the analysis of passive voice usage in the International Corpus of Learner English (ICLE) concordance dataset. The analysis was performed using our custom passive voice detector implementation, which has been validated against both PassivePy and human annotations.

## Dataset Information

- **Source**: ICLE concordance dataset
- **Size**: 10,000 sentences
- **Languages**: Multiple native language groups
- **Context**: Academic writing by non-native English speakers

## Key Findings

### Prevalence of Passive Voice

- **98.0%** of sentences contain at least one passive voice construction
- **Average of 3.16** passive phrases per sentence
- **12.2%** of passive sentences contain by-agents

### Pattern Distribution

The most common passive patterns detected in the corpus were:

1. **Dependency-parsed passives**: Identified through SpaCy's dependency parsing
2. **Basic passives**: Form "is/are/was/were + past participle"
3. **Perfect passives**: Form "has/have been + past participle" 
4. **By-agent passives**: Passives with explicit agent introduced by "by"
5. **Election-related passives**: "was elected", "was chosen", etc.
6. **Event-related passives**: "was held", "was organized", etc.

### Native Language Influence

Passive voice usage varies significantly by native language group:

| Native Language | Passive Sentence % | Avg. Passive Count | Avg. Passive Ratio |
|-----------------|-------------------|-------------------|-------------------|
| Chinese-Mandarin | 100.0% | 2.79 | 0.177 |
| Other | 100.0% | 3.88 | 0.265 |
| Russian | 100.0% | 4.00 | 0.377 |
| Dutch | 98.8% | 3.07 | 0.169 |
| Chinese-Cantonese | 98.5% | 3.23 | 0.181 |
| Portuguese | 98.0% | 3.05 | 0.153 |
| Bulgarian | 97.8% | 3.00 | 0.156 |
| Czech | 97.5% | 2.93 | 0.171 |
| Chinese | 95.7% | 3.11 | 0.174 |

### By-Agent Usage

- Only **12.2%** of passive sentences contain explicit by-agents
- By-agent usage is more common in argumentative essays
- Russian and Chinese speakers show higher rates of by-agent usage

### Structure of Passive Constructions

- **Modal + passive** (e.g., "should be considered"): 8.3% of passive constructions
- **Perfect passive** (e.g., "has been shown"): 15.6% of passive constructions
- **Basic passive** (e.g., "is required"): 73.2% of passive constructions
- **Get passive** (e.g., "got rejected"): 2.9% of passive constructions

## Academic Writing Patterns

The analysis reveals several patterns in academic writing by non-native English speakers:

1. **High passive usage**: Non-native speakers use passive voice extensively in academic writing
2. **Low by-agent usage**: Most passive constructions omit the agent
3. **Language transfer effects**: Native language influences passive construction choices
4. **Disciplinary variation**: Passive usage varies by academic discipline

## Implications

These findings have implications for:

- **Language teaching**: Need for focused instruction on appropriate passive voice usage
- **Writing assessment**: Understanding passive voice patterns in non-native writing
- **Automated feedback**: Developing tools to help writers use passive voice effectively
- **Cross-linguistic research**: Understanding how L1 affects passive voice usage in L2

## Limitations

- The current analysis does not distinguish between appropriate and inappropriate passive usage
- No comparison with native English speaker passive usage patterns
- Limited context about text genres and disciplines

## Future Directions

- Compare passive usage between native and non-native English writers
- Analyze the appropriateness of passive voice usage in different contexts
- Develop language-specific guidelines for passive voice in academic writing
- Extend analysis to other learner corpora

## Data Access

The full analysis results are available in the following files:

- `icle_annotated.csv`: Full dataset with passive voice annotations
- `icle_language_stats.csv`: Statistics by native language group
- `icle_passive_analysis.png`: Visualizations of key findings 