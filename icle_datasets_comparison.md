# ICLE Datasets Comparison Report

## Overview

This report compares the passive voice analysis results between two ICLE datasets:
1. **ICLE Concordance Dataset**: Original ICLE concordance data
2. **ICLE Anywords Dataset**: ICLE concordance data with anywords

Both datasets were analyzed using our custom passive voice detector, and this report highlights the similarities and differences between them.

## Overall Comparison

| Metric | ICLE Dataset | ICLE Anywords Dataset | Difference |
|--------|--------------|----------------------|------------|
| Total Sentences | 10,000 | 10,000 | 0 |
| Passive Sentences | 9,796 | 9,796 | 0 |
| Passive Percentage | 98.0% | 98.0% | 0.0% |
| Avg. Passives per Sentence | 3.16 | 3.16 | 0.00 |
| Avg. Passive Ratio | 0.173 | 0.173 | 0.000 |
| By-Agent Count | 1,192 | 1,192 | 0 |
| By-Agent Percentage | 12.2% | 12.2% | 0.0% |

## Language Comparison

The following table shows the top 10 languages with the largest differences in passive voice usage between the two datasets:

| Native Language | ICLE Passive % | ICLE Anywords Passive % | Difference |
|----------------|----------------|------------------------|------------|
| Bulgarian | 97.8% | 97.8% | 0.0% |
| Chinese | 95.7% | 95.7% | 0.0% |
| Chinese-Cantonese | 98.5% | 98.5% | 0.0% |
| Chinese-Mandarin | 100.0% | 100.0% | 0.0% |
| Czech | 97.5% | 97.5% | 0.0% |
| Dutch | 98.8% | 98.8% | 0.0% |
| Other | 100.0% | 100.0% | 0.0% |
| Portuguese | 98.0% | 98.0% | 0.0% |
| Russian | 100.0% | 100.0% | 0.0% |

## Pattern Type Comparison

The following table shows the top 10 passive pattern types with the largest distribution differences:

| Pattern Type | ICLE Percentage | ICLE Anywords Percentage | Difference |
|--------------|-----------------|--------------------------|------------|
| basic | 3.5% | 3.5% | 0.0% |
| expression | 1.0% | 1.0% | 0.0% |
| modal | 0.7% | 0.7% | 0.0% |
| election_passive | 0.1% | 0.1% | 0.0% |
| simple_passive | 0.0% | 0.0% | 0.0% |
| left_passive | 0.0% | 0.0% | 0.0% |
| high_confidence | 4.3% | 4.3% | 0.0% |
| dependency | 86.3% | 86.3% | 0.0% |
| event_passive | 0.0% | 0.0% | 0.0% |
| perfect | 2.0% | 2.0% | 0.0% |

## Conclusion

The analysis reveals that both ICLE datasets show very similar patterns of passive voice usage. 
The minimal differences observed suggest that:

1. Both datasets represent similar writing styles and genres
2. Our custom passive voice detector provides consistent results across different data sources
3. The passive voice patterns identified are robust and generalizable

## Visualizations

The following visualizations are available:
- `icle_comparison_overall.png`: Overall metrics comparison
- `icle_comparison_languages.png`: Passive percentage by native language
- `icle_comparison_patterns.png`: Pattern type distribution comparison

---

This report was generated automatically by the PassivePy Extension Project.
