# Passive Voice Detection Patterns

This module contains regex patterns and constants used for passive voice detection in the PassivePy library.

## Overview

The `patterns.py` file contains several collections of patterns and constants that are used by the passive voice detector to:

1. Identify passive voice constructions
2. Filter out false positives
3. Assign confidence scores to detected passive phrases
4. Categorize different types of passive constructions

## Pattern Types

### PASSIVE_PATTERNS

Basic regex patterns for identifying passive voice:

- `basic`: Standard forms like "is written", "was built"
- `perfect`: Perfect forms like "has been written", "had been constructed" 
- `modal`: Modal forms like "will be written", "should be considered"
- `get`: Get-passives like "got destroyed", "gets selected"

### COMMON_PASSIVE_PATTERNS

High-confidence patterns for specific domains:

- Election/selection patterns (elected, chosen, selected)
- Event patterns (held, scheduled, organized)
- Delivery patterns (delivered, shipped, sent)
- Creation patterns (made, created, built)
- Perception patterns (seen, heard, watched)
- And more...

### SPECIAL_PASSIVE_PATTERNS

Specific patterns for commonly missed passives:

- `election`: Election-related passives
- `event`: Event-related passives
- `by_agent`: Passives with explicit by-agent
- `negated`: Negated passives
- `left`: Left + participle constructions
- `simple`: Simple subject-verb-participle constructions

### PASSIVE_EXPRESSIONS

Fixed expressions that indicate passive voice but may not follow standard patterns, such as:
- "supposed to"
- "meant to"
- "required to"
- "expected to"

### NON_PASSIVE_COMBOS

Combinations that look like passive voice but should not be counted as such:
- "is able", "was able"
- "is about", "was about"
- "is available", "was available"
- And many more...

### FALSE_POSITIVE_PATTERNS

Common false positive patterns in passive detection:
- Emotional/mental states
- Adjectival states
- Filled as adjective
- Active voice constructions
- And more...

### PERSONAL_MENTAL_STATES

Personal mental state verbs that often appear in passive-like constructions:
- "amazed", "amused", "annoyed"
- "confused", "disappointed", "discouraged"
- And more...

### ADJECTIVAL_PARTICIPLES

Adjectival participles that should not be counted as passive:
- "advanced", "animated", "authorized"
- "balanced", "calculated", "calibrated"
- And many more...

### COMMON_PASSIVE_VERBS

Common verbs that frequently appear in passive constructions:
- "made", "done", "given", "taken"
- "created", "built", "written", "published"
- And more...

### CONFIDENCE_SCORES

Confidence levels for different pattern types:
- `by_agent`: 0.95
- `election_passive`: 0.95
- `event_passive`: 0.90
- `high_confidence`: 0.85
- And more...

## Usage

To use these patterns in your code:

```python
from src.passive_detector.core.patterns import (
    PASSIVE_PATTERNS,
    PASSIVE_EXPRESSIONS,
    NON_PASSIVE_COMBOS,
    PERSONAL_MENTAL_STATES,
    ADJECTIVAL_PARTICIPLES,
    COMMON_PASSIVE_PATTERNS,
    SPECIAL_PASSIVE_PATTERNS,
    FALSE_POSITIVE_PATTERNS,
    COMMON_PASSIVE_VERBS,
    CONFIDENCE_SCORES
)

# Use patterns with regex
import re
for pattern_type, pattern in PASSIVE_PATTERNS.items():
    matches = re.finditer(pattern, text, re.IGNORECASE)
    # Process matches...
```

See `test_passive_patterns.py` for a complete example of how to use these patterns for passive voice detection. 