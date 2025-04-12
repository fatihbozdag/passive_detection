#!/usr/bin/env python3
"""
ICLE Datasets Comparison Script

This script compares the results of passive voice analysis between the 
ICLE concordance dataset and the ICLE anywords concordance dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def load_annotated_datasets():
    """
    Load both annotated datasets
    
    Returns:
        tuple: (icle_df, icle_anywords_df) - Loaded DataFrames
    """
    print("Loading annotated datasets...")
    
    try:
        # Load the annotated datasets
        icle_df = pd.read_csv('icle_annotated.csv')
        icle_anywords_df = pd.read_csv('icle_anywords_annotated.csv')
        
        print(f"Loaded {len(icle_df)} sentences from ICLE dataset")
        print(f"Loaded {len(icle_anywords_df)} sentences from ICLE Anywords dataset")
        
        return icle_df, icle_anywords_df
    
    except Exception as e:
        print(f"Error loading annotated datasets: {e}")
        return None, None

def load_language_stats():
    """
    Load language statistics for both datasets
    
    Returns:
        tuple: (icle_language_stats, icle_anywords_language_stats) - Loaded DataFrames
    """
    print("Loading language statistics...")
    
    try:
        # Load the language statistics
        icle_language_stats = pd.read_csv('icle_language_stats.csv')
        icle_anywords_language_stats = pd.read_csv('icle_anywords_language_stats.csv')
        
        print(f"Loaded language statistics for {len(icle_language_stats)} languages from ICLE dataset")
        print(f"Loaded language statistics for {len(icle_anywords_language_stats)} languages from ICLE Anywords dataset")
        
        return icle_language_stats, icle_anywords_language_stats
    
    except Exception as e:
        print(f"Error loading language statistics: {e}")
        return None, None

def calculate_overall_metrics(icle_df, icle_anywords_df):
    """
    Calculate overall comparison metrics for both datasets
    
    Args:
        icle_df (pandas.DataFrame): ICLE dataset
        icle_anywords_df (pandas.DataFrame): ICLE Anywords dataset
    
    Returns:
        dict: Dictionary with comparison metrics
    """
    print("Calculating overall metrics...")
    
    # Check for required columns
    required_cols = ['is_passive', 'passive_count', 'passive_ratio', 'has_by_agents']
    
    if not all(col in icle_df.columns for col in required_cols) or \
       not all(col in icle_anywords_df.columns for col in required_cols):
        print("Error: Required columns missing from datasets")
        return None
    
    # Calculate overall metrics for ICLE dataset
    icle_passive_sentences = icle_df[icle_df['is_passive'] == True]
    icle_metrics = {
        'total_sentences': len(icle_df),
        'passive_sentences': len(icle_passive_sentences),
        'passive_percentage': (len(icle_passive_sentences) / len(icle_df)) * 100,
        'avg_passive_per_sentence': icle_passive_sentences['passive_count'].mean(),
        'avg_passive_ratio': icle_passive_sentences['passive_ratio'].mean(),
        'by_agent_count': sum('True' in str(x) for x in icle_passive_sentences['has_by_agents']),
    }
    icle_metrics['by_agent_percentage'] = (icle_metrics['by_agent_count'] / icle_metrics['passive_sentences']) * 100
    
    # Calculate overall metrics for ICLE Anywords dataset
    anywords_passive_sentences = icle_anywords_df[icle_anywords_df['is_passive'] == True]
    anywords_metrics = {
        'total_sentences': len(icle_anywords_df),
        'passive_sentences': len(anywords_passive_sentences),
        'passive_percentage': (len(anywords_passive_sentences) / len(icle_anywords_df)) * 100,
        'avg_passive_per_sentence': anywords_passive_sentences['passive_count'].mean(),
        'avg_passive_ratio': anywords_passive_sentences['passive_ratio'].mean(),
        'by_agent_count': sum('True' in str(x) for x in anywords_passive_sentences['has_by_agents']),
    }
    anywords_metrics['by_agent_percentage'] = (anywords_metrics['by_agent_count'] / anywords_metrics['passive_sentences']) * 100
    
    # Calculate differences
    diff_metrics = {
        'total_sentences_diff': icle_metrics['total_sentences'] - anywords_metrics['total_sentences'],
        'passive_sentences_diff': icle_metrics['passive_sentences'] - anywords_metrics['passive_sentences'],
        'passive_percentage_diff': icle_metrics['passive_percentage'] - anywords_metrics['passive_percentage'],
        'avg_passive_per_sentence_diff': icle_metrics['avg_passive_per_sentence'] - anywords_metrics['avg_passive_per_sentence'],
        'avg_passive_ratio_diff': icle_metrics['avg_passive_ratio'] - anywords_metrics['avg_passive_ratio'],
        'by_agent_count_diff': icle_metrics['by_agent_count'] - anywords_metrics['by_agent_count'],
        'by_agent_percentage_diff': icle_metrics['by_agent_percentage'] - anywords_metrics['by_agent_percentage'],
    }
    
    print("Overall metrics calculated")
    
    return {
        'icle': icle_metrics,
        'anywords': anywords_metrics,
        'diff': diff_metrics
    }

def compare_language_stats(icle_language_stats, icle_anywords_language_stats):
    """
    Compare language statistics between the two datasets
    
    Args:
        icle_language_stats (pandas.DataFrame): ICLE language statistics
        icle_anywords_language_stats (pandas.DataFrame): ICLE Anywords language statistics
    
    Returns:
        pandas.DataFrame: Combined statistics with differences
    """
    print("Comparing language statistics...")
    
    # Rename columns for clarity
    icle_language_stats = icle_language_stats.rename(columns={
        'is_passive': 'icle_passive_percentage',
        'passive_count': 'icle_passive_count',
        'passive_ratio': 'icle_passive_ratio'
    })
    
    icle_anywords_language_stats = icle_anywords_language_stats.rename(columns={
        'is_passive': 'anywords_passive_percentage',
        'passive_count': 'anywords_passive_count',
        'passive_ratio': 'anywords_passive_ratio'
    })
    
    # Merge datasets on Native language
    merged_stats = pd.merge(
        icle_language_stats, 
        icle_anywords_language_stats,
        on='Native language', 
        how='outer'
    )
    
    # Fill NaN values with 0
    merged_stats = merged_stats.fillna(0)
    
    # Calculate differences
    merged_stats['passive_percentage_diff'] = merged_stats['icle_passive_percentage'] - merged_stats['anywords_passive_percentage']
    merged_stats['passive_count_diff'] = merged_stats['icle_passive_count'] - merged_stats['anywords_passive_count']
    merged_stats['passive_ratio_diff'] = merged_stats['icle_passive_ratio'] - merged_stats['anywords_passive_ratio']
    
    # Sort by absolute difference in passive percentage
    merged_stats = merged_stats.sort_values(by='passive_percentage_diff', key=abs, ascending=False)
    
    print(f"Language statistics compared for {len(merged_stats)} languages")
    
    return merged_stats

def compare_pattern_distributions(icle_df, icle_anywords_df):
    """
    Compare pattern type distributions between datasets
    
    Args:
        icle_df (pandas.DataFrame): ICLE dataset
        icle_anywords_df (pandas.DataFrame): ICLE Anywords dataset
    
    Returns:
        dict: Dictionary with pattern counts for both datasets
    """
    print("Comparing pattern distributions...")
    
    # Extract pattern types from ICLE dataset
    icle_passive_sentences = icle_df[icle_df['is_passive'] == True]
    icle_pattern_types = []
    
    for pattern_types in icle_passive_sentences['pattern_types'].dropna():
        if isinstance(pattern_types, str) and pattern_types:
            icle_pattern_types.extend([p.strip() for p in pattern_types.split(';')])
    
    icle_pattern_counts = Counter(icle_pattern_types)
    
    # Extract pattern types from ICLE Anywords dataset
    anywords_passive_sentences = icle_anywords_df[icle_anywords_df['is_passive'] == True]
    anywords_pattern_types = []
    
    for pattern_types in anywords_passive_sentences['pattern_types'].dropna():
        if isinstance(pattern_types, str) and pattern_types:
            anywords_pattern_types.extend([p.strip() for p in pattern_types.split(';')])
    
    anywords_pattern_counts = Counter(anywords_pattern_types)
    
    # Create DataFrames for each dataset
    icle_patterns_df = pd.DataFrame([
        {'Pattern': k, 'Count': v, 'Percentage': (v / sum(icle_pattern_counts.values())) * 100}
        for k, v in icle_pattern_counts.most_common()
    ])
    
    anywords_patterns_df = pd.DataFrame([
        {'Pattern': k, 'Count': v, 'Percentage': (v / sum(anywords_pattern_counts.values())) * 100}
        for k, v in anywords_pattern_counts.most_common()
    ])
    
    print("Pattern distributions compared")
    
    return {
        'icle': icle_patterns_df,
        'anywords': anywords_patterns_df,
        'icle_counts': icle_pattern_counts,
        'anywords_counts': anywords_pattern_counts
    }

def visualize_comparisons(overall_metrics, language_comparison, pattern_comparison):
    """
    Create visualizations comparing the two datasets
    
    Args:
        overall_metrics (dict): Overall comparison metrics
        language_comparison (pandas.DataFrame): Language comparison statistics
        pattern_comparison (dict): Pattern comparison data
    
    Returns:
        None
    """
    print("Generating comparison visualizations...")
    
    # 1. Overall metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Passive percentage comparison
    datasets = ['ICLE', 'ICLE Anywords']
    passive_percentages = [
        overall_metrics['icle']['passive_percentage'],
        overall_metrics['anywords']['passive_percentage']
    ]
    
    axes[0, 0].bar(datasets, passive_percentages, color=['#ff9999', '#66b3ff'])
    axes[0, 0].set_title('Passive Sentence Percentage')
    axes[0, 0].set_ylim([0, 100])
    for i, v in enumerate(passive_percentages):
        axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Average passives per sentence
    avg_passives = [
        overall_metrics['icle']['avg_passive_per_sentence'],
        overall_metrics['anywords']['avg_passive_per_sentence']
    ]
    
    axes[0, 1].bar(datasets, avg_passives, color=['#ff9999', '#66b3ff'])
    axes[0, 1].set_title('Average Passive Phrases per Sentence')
    for i, v in enumerate(avg_passives):
        axes[0, 1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # By-agent percentage
    by_agent_percentages = [
        overall_metrics['icle']['by_agent_percentage'],
        overall_metrics['anywords']['by_agent_percentage']
    ]
    
    axes[1, 0].bar(datasets, by_agent_percentages, color=['#ff9999', '#66b3ff'])
    axes[1, 0].set_title('Percentage of Passive Sentences with By-Agents')
    axes[1, 0].set_ylim([0, 100])
    for i, v in enumerate(by_agent_percentages):
        axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Passive ratio
    passive_ratios = [
        overall_metrics['icle']['avg_passive_ratio'],
        overall_metrics['anywords']['avg_passive_ratio']
    ]
    
    axes[1, 1].bar(datasets, passive_ratios, color=['#ff9999', '#66b3ff'])
    axes[1, 1].set_title('Average Passive Ratio')
    for i, v in enumerate(passive_ratios):
        axes[1, 1].text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('icle_comparison_overall.png', dpi=300, bbox_inches='tight')
    
    # 2. Language comparison visualization
    if len(language_comparison) > 0:
        plt.figure(figsize=(14, 10))
        
        # Get top 10 languages with largest differences
        top_diff_languages = language_comparison.head(10)
        
        # Create grouped bar chart
        x = np.arange(len(top_diff_languages))
        width = 0.35
        
        plt.bar(x - width/2, top_diff_languages['icle_passive_percentage'] * 100, width, label='ICLE')
        plt.bar(x + width/2, top_diff_languages['anywords_passive_percentage'] * 100, width, label='ICLE Anywords')
        
        plt.xlabel('Native Language')
        plt.ylabel('Passive Sentences (%)')
        plt.title('Passive Sentence Percentage by Native Language (Top 10 Differences)')
        plt.xticks(x, top_diff_languages['Native language'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('icle_comparison_languages.png', dpi=300, bbox_inches='tight')
    
    # 3. Pattern type comparison
    if pattern_comparison and 'icle' in pattern_comparison and 'anywords' in pattern_comparison:
        plt.figure(figsize=(14, 10))
        
        # Get top 8 patterns from each dataset
        icle_top_patterns = pattern_comparison['icle'].head(8)
        anywords_top_patterns = pattern_comparison['anywords'].head(8)
        
        # Combine and get unique patterns
        all_patterns = pd.concat([icle_top_patterns, anywords_top_patterns])
        unique_patterns = all_patterns['Pattern'].unique()
        
        # Create data for comparison
        patterns_data = []
        for pattern in unique_patterns:
            icle_percentage = 0
            icle_row = pattern_comparison['icle'][pattern_comparison['icle']['Pattern'] == pattern]
            if not icle_row.empty:
                icle_percentage = icle_row.iloc[0]['Percentage']
                
            anywords_percentage = 0
            anywords_row = pattern_comparison['anywords'][pattern_comparison['anywords']['Pattern'] == pattern]
            if not anywords_row.empty:
                anywords_percentage = anywords_row.iloc[0]['Percentage']
                
            patterns_data.append({
                'Pattern': pattern,
                'ICLE': icle_percentage,
                'ICLE Anywords': anywords_percentage,
                'Difference': icle_percentage - anywords_percentage
            })
        
        # Convert to DataFrame and sort by absolute difference
        patterns_df = pd.DataFrame(patterns_data)
        patterns_df = patterns_df.sort_values(by='Difference', key=abs, ascending=False)
        
        # Create grouped bar chart for top 10 patterns
        patterns_df = patterns_df.head(10)
        
        plt.figure(figsize=(14, 8))
        patterns_df.plot(x='Pattern', y=['ICLE', 'ICLE Anywords'], kind='bar', ax=plt.gca())
        plt.title('Pattern Type Distribution Comparison (Top 10 Differences)')
        plt.ylabel('Percentage of Passive Phrases')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('icle_comparison_patterns.png', dpi=300, bbox_inches='tight')
    
    print("Comparison visualizations saved")

def generate_comparison_report(overall_metrics, language_comparison, pattern_comparison):
    """
    Generate a comprehensive comparison report
    
    Args:
        overall_metrics (dict): Overall comparison metrics
        language_comparison (pandas.DataFrame): Language comparison statistics
        pattern_comparison (dict): Pattern comparison data
    
    Returns:
        str: Markdown report text
    """
    print("Generating comparison report...")
    
    report = """# ICLE Datasets Comparison Report

## Overview

This report compares the passive voice analysis results between two ICLE datasets:
1. **ICLE Concordance Dataset**: Original ICLE concordance data
2. **ICLE Anywords Dataset**: ICLE concordance data with anywords

Both datasets were analyzed using our custom passive voice detector, and this report highlights the similarities and differences between them.

## Overall Comparison

| Metric | ICLE Dataset | ICLE Anywords Dataset | Difference |
|--------|--------------|----------------------|------------|
"""
    
    # Add overall metrics to the report
    metrics = {
        'Total Sentences': ('total_sentences', 0),
        'Passive Sentences': ('passive_sentences', 0),
        'Passive Percentage': ('passive_percentage', 1),
        'Avg. Passives per Sentence': ('avg_passive_per_sentence', 2),
        'Avg. Passive Ratio': ('avg_passive_ratio', 3),
        'By-Agent Count': ('by_agent_count', 0),
        'By-Agent Percentage': ('by_agent_percentage', 1)
    }
    
    for metric_name, (metric_key, decimals) in metrics.items():
        icle_value = overall_metrics['icle'][metric_key]
        anywords_value = overall_metrics['anywords'][metric_key]
        diff_value = overall_metrics['diff'][f"{metric_key}_diff"]
        
        format_str = '{:,}' if decimals == 0 else '{:,.' + str(decimals) + 'f}'
        
        if metric_name.endswith('Percentage'):
            icle_str = format_str.format(icle_value) + '%'
            anywords_str = format_str.format(anywords_value) + '%'
            diff_str = format_str.format(diff_value) + '%'
        else:
            icle_str = format_str.format(icle_value)
            anywords_str = format_str.format(anywords_value)
            diff_str = format_str.format(diff_value)
        
        report += f"| {metric_name} | {icle_str} | {anywords_str} | {diff_str} |\n"
    
    # Add language comparison section
    report += """
## Language Comparison

The following table shows the top 10 languages with the largest differences in passive voice usage between the two datasets:

| Native Language | ICLE Passive % | ICLE Anywords Passive % | Difference |
|----------------|----------------|------------------------|------------|
"""
    
    # Add top 10 languages with largest differences
    for i, row in language_comparison.head(10).iterrows():
        lang = row['Native language']
        icle_pct = row['icle_passive_percentage'] * 100
        anywords_pct = row['anywords_passive_percentage'] * 100
        diff = row['passive_percentage_diff'] * 100
        
        report += f"| {lang} | {icle_pct:.1f}% | {anywords_pct:.1f}% | {diff:.1f}% |\n"
    
    # Add pattern comparison section
    report += """
## Pattern Type Comparison

The following table shows the top 10 passive pattern types with the largest distribution differences:

| Pattern Type | ICLE Percentage | ICLE Anywords Percentage | Difference |
|--------------|-----------------|--------------------------|------------|
"""
    
    # Combine pattern data
    if pattern_comparison and 'icle' in pattern_comparison and 'anywords' in pattern_comparison:
        icle_patterns = pattern_comparison['icle'].set_index('Pattern')
        anywords_patterns = pattern_comparison['anywords'].set_index('Pattern')
        
        # Get all unique patterns
        all_patterns = set(icle_patterns.index) | set(anywords_patterns.index)
        
        # Create comparison data
        pattern_comparison_data = []
        for pattern in all_patterns:
            icle_pct = 0
            if pattern in icle_patterns.index:
                icle_pct = icle_patterns.loc[pattern, 'Percentage']
                
            anywords_pct = 0
            if pattern in anywords_patterns.index:
                anywords_pct = anywords_patterns.loc[pattern, 'Percentage']
                
            pattern_comparison_data.append({
                'Pattern': pattern,
                'ICLE_Percentage': icle_pct,
                'Anywords_Percentage': anywords_pct,
                'Difference': icle_pct - anywords_pct
            })
        
        # Convert to DataFrame and sort
        pattern_df = pd.DataFrame(pattern_comparison_data)
        pattern_df = pattern_df.sort_values(by='Difference', key=abs, ascending=False)
        
        # Add top 10 patterns to report
        for i, row in pattern_df.head(10).iterrows():
            pattern = row['Pattern']
            icle_pct = row['ICLE_Percentage']
            anywords_pct = row['Anywords_Percentage']
            diff = row['Difference']
            
            report += f"| {pattern} | {icle_pct:.1f}% | {anywords_pct:.1f}% | {diff:.1f}% |\n"
    
    # Add conclusion section
    report += """
## Conclusion

"""
    
    # Determine if datasets are significantly different
    if abs(overall_metrics['diff']['passive_percentage_diff']) < 1.0 and \
       abs(overall_metrics['diff']['avg_passive_per_sentence_diff']) < 0.1:
        report += """The analysis reveals that both ICLE datasets show very similar patterns of passive voice usage. 
The minimal differences observed suggest that:

1. Both datasets represent similar writing styles and genres
2. Our custom passive voice detector provides consistent results across different data sources
3. The passive voice patterns identified are robust and generalizable"""
    else:
        report += """The analysis reveals notable differences in passive voice usage between the two ICLE datasets.
These differences could be attributed to:

1. Variations in text sources or domains
2. Different distributions of native languages
3. Different contexts or writing prompts
4. Potential biases in the sampling or collection process"""
    
    report += """

## Visualizations

The following visualizations are available:
- `icle_comparison_overall.png`: Overall metrics comparison
- `icle_comparison_languages.png`: Passive percentage by native language
- `icle_comparison_patterns.png`: Pattern type distribution comparison

---

This report was generated automatically by the PassivePy Extension Project.
"""
    
    # Save report to file
    with open('icle_datasets_comparison.md', 'w') as f:
        f.write(report)
    
    print("Comparison report saved to 'icle_datasets_comparison.md'")
    
    return report

def main():
    """Main execution function"""
    # Load annotated datasets
    icle_df, icle_anywords_df = load_annotated_datasets()
    
    if icle_df is None or icle_anywords_df is None:
        print("Exiting due to error loading datasets.")
        return
    
    # Load language statistics
    icle_language_stats, icle_anywords_language_stats = load_language_stats()
    
    if icle_language_stats is None or icle_anywords_language_stats is None:
        print("Warning: Language statistics could not be loaded.")
        # Continue with other comparisons
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(icle_df, icle_anywords_df)
    
    if overall_metrics is None:
        print("Exiting due to error calculating overall metrics.")
        return
    
    # Compare language statistics if available
    language_comparison = pd.DataFrame()
    if icle_language_stats is not None and icle_anywords_language_stats is not None:
        language_comparison = compare_language_stats(icle_language_stats, icle_anywords_language_stats)
    
    # Compare pattern distributions
    pattern_comparison = compare_pattern_distributions(icle_df, icle_anywords_df)
    
    # Create visualizations
    visualize_comparisons(overall_metrics, language_comparison, pattern_comparison)
    
    # Generate comparison report
    generate_comparison_report(overall_metrics, language_comparison, pattern_comparison)
    
    print("\nComparison analysis complete!")

if __name__ == "__main__":
    main() 