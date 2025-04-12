import json
import pandas as pd
import os
from collections import defaultdict

# Initialize results dictionary
all_results = defaultdict(dict)

# Load crowd source data
if os.path.exists('analysis_results/crowd_source_analysis_results.json'):
    with open('analysis_results/crowd_source_analysis_results.json', 'r') as f:
        crowd_data = json.load(f)
        all_results['crowd_source'] = crowd_data

# Load abstract comparison data
if os.path.exists('analysis_results/abstract_comparison_summary.json'):
    with open('analysis_results/abstract_comparison_summary.json', 'r') as f:
        abstract_data = json.load(f)
        all_results['abstracts'] = abstract_data

# Load ICLE data from summary
icle_data = {
    'total_sentences': 10000,
    'passive_sentences': 9796,
    'passive_percentage': 98.0,
    'avg_passives_per_sentence': 3.16,
    'by_agent_percentage': 12.2
}
all_results['icle'] = icle_data

# Include language comparison from ICLE
if os.path.exists('icle_language_stats.csv'):
    lang_df = pd.read_csv('icle_language_stats.csv')
    lang_data = lang_df.to_dict(orient='records')
    all_results['icle']['language_breakdown'] = lang_data

# Try to include implementation comparison 
if os.path.exists('implementation_disagreements.csv'):
    disagreements_df = pd.read_csv('implementation_disagreements.csv')
    
    # Calculate metrics based on the actual structure
    total = len(disagreements_df)
    human_coding_matches_custom = sum(
        (disagreements_df['human_coding'] == 1) & (disagreements_df['custom_prediction'] == 1) | 
        (disagreements_df['human_coding'] == 0) & (disagreements_df['custom_prediction'] == 0)
    )
    human_coding_matches_passivepy = sum(
        (disagreements_df['human_coding'] == 1) & (disagreements_df['passivepy_prediction'] == 1) | 
        (disagreements_df['human_coding'] == 0) & (disagreements_df['passivepy_prediction'] == 0)
    )
    
    # Count cases where PassivePy says not passive but human and custom say it is
    false_negatives_passivepy = sum(
        (disagreements_df['human_coding'] == 1) & 
        (disagreements_df['passivepy_prediction'] == 0) & 
        (disagreements_df['custom_prediction'] == 1)
    )
    
    # Count cases where custom says passive but human says it's not
    false_positives_custom = sum(
        (disagreements_df['human_coding'] == 0) & 
        (disagreements_df['custom_prediction'] == 1)
    )
    
    implementation_data = {
        'total_sentences': total,
        'custom_agreement_percentage': (human_coding_matches_custom / total) * 100,
        'passivepy_agreement_percentage': (human_coding_matches_passivepy / total) * 100,
        'false_positives_custom': int(false_positives_custom),
        'false_negatives_passivepy': int(false_negatives_passivepy),
        'improvement_over_passivepy': ((human_coding_matches_custom - human_coding_matches_passivepy) / total) * 100
    }
    all_results['implementation_comparison'] = implementation_data

# Save results to file
with open('dataset_comparison_metrics.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary
print(json.dumps(all_results, indent=2)) 