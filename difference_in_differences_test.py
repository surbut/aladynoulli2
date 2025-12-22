#!/usr/bin/env python3
"""
Difference-in-Differences Test

Tests whether the effect of Pyr3 on Ctrl is significantly different 
from the effect of Pyr3 on VAT.

Within each group, measurements are paired:
- diff_Ctrl = (Ctrl + Pyr3) - Ctrl  [one delta per sample in Ctrl group]
- diff_VAT = (VAT + Pyr3) - VAT    [one delta per sample in VAT group]

Between groups, samples are independent. We test:
- DID = mean(diff_VAT) - mean(diff_Ctrl)
- Null hypothesis: DID = 0 (no interaction effect)
- This is equivalent to testing if mean(diff_VAT) = mean(diff_Ctrl)

Uses a two-sample t-test comparing the two independent groups of differences.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def difference_in_differences_test(data, ctrl_col='Ctrl', ctrl_pyr3_col='Ctrl + Pyr3', 
                                   vat_col='VAT EVs', vat_pyr3_col='VAT EVs + Pyr3'):
    """
    Perform difference-in-differences test.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns for the 4 conditions
    ctrl_col : str
        Name of control column
    ctrl_pyr3_col : str
        Name of control + Pyr3 column
    vat_col : str
        Name of VAT column
    vat_pyr3_col : str
        Name of VAT + Pyr3 column
    
    Returns:
    --------
    dict : Results dictionary with statistics
    """
    
    # Calculate differences for each group independently
    # Within each group, measurements are paired (Ctrl vs Ctrl+Pyr3, VAT vs VAT+Pyr3)
    # But between groups, samples are independent
    
    # Ctrl group: need both Ctrl and Ctrl+Pyr3
    ctrl_pairs = data[[ctrl_col, ctrl_pyr3_col]].dropna()
    diff_ctrl = ctrl_pairs[ctrl_pyr3_col] - ctrl_pairs[ctrl_col]
    
    # VAT group: need both VAT and VAT+Pyr3
    vat_pairs = data[[vat_col, vat_pyr3_col]].dropna()
    diff_vat = vat_pairs[vat_pyr3_col] - vat_pairs[vat_col]
    
    if len(diff_ctrl) == 0:
        raise ValueError(f"No complete pairs found for {ctrl_col} and {ctrl_pyr3_col}")
    if len(diff_vat) == 0:
        raise ValueError(f"No complete pairs found for {vat_col} and {vat_pyr3_col}")
    
    print(f"Ctrl group: {len(diff_ctrl)} complete pairs")
    print(f"VAT group: {len(diff_vat)} complete pairs")
    
    # Calculate difference-in-differences (difference of means)
    # DID = mean(diff_VAT) - mean(diff_Ctrl)
    # This is a linear combination: we're testing if DID = 0
    did_mean = diff_vat.mean() - diff_ctrl.mean()
    
    # Perform two-sample t-test on the differences (independent samples)
    # H0: mean(diff_ctrl) = mean(diff_vat)  [i.e., DID = 0]
    # H1: mean(diff_ctrl) ≠ mean(diff_vat)  [i.e., DID ≠ 0]
    # Note: Each group has multiple delta values (one per sample), 
    # and we're comparing the distributions of these deltas
    t_stat, p_value = stats.ttest_ind(diff_vat, diff_ctrl)
    
    # Also calculate descriptive statistics
    results = {
        'n_ctrl': len(diff_ctrl),
        'n_vat': len(diff_vat),
        'diff_ctrl_mean': diff_ctrl.mean(),
        'diff_ctrl_std': diff_ctrl.std(),
        'diff_ctrl_sem': stats.sem(diff_ctrl),
        'diff_vat_mean': diff_vat.mean(),
        'diff_vat_std': diff_vat.std(),
        'diff_vat_sem': stats.sem(diff_vat),
        'did_mean': did_mean,
        'did_se': np.sqrt(diff_ctrl.var()/len(diff_ctrl) + diff_vat.var()/len(diff_vat)),
        't_statistic': t_stat,
        'p_value': p_value,
        'diff_ctrl': diff_ctrl.values,
        'diff_vat': diff_vat.values
    }
    
    return results


def print_results(results):
    """Print formatted results."""
    print("\n" + "="*60)
    print("DIFFERENCE-IN-DIFFERENCES TEST RESULTS")
    print("="*60)
    print(f"\nSample sizes:")
    print(f"  Ctrl group: n = {results['n_ctrl']}")
    print(f"  VAT group:  n = {results['n_vat']}")
    print(f"\nEffect of Pyr3 on Ctrl:")
    print(f"  Mean difference: {results['diff_ctrl_mean']:.4f}")
    print(f"  Std deviation:   {results['diff_ctrl_std']:.4f}")
    print(f"  Std error:       {results['diff_ctrl_sem']:.4f}")
    
    print(f"\nEffect of Pyr3 on VAT:")
    print(f"  Mean difference: {results['diff_vat_mean']:.4f}")
    print(f"  Std deviation:   {results['diff_vat_std']:.4f}")
    print(f"  Std error:       {results['diff_vat_sem']:.4f}")
    
    print(f"\nDifference-in-Differences (VAT effect - Ctrl effect):")
    print(f"  Mean:            {results['did_mean']:.4f}")
    print(f"  Std error:       {results['did_se']:.4f}")
    
    print(f"\nTwo-sample t-test (independent samples):")
    print(f"  t-statistic:     {results['t_statistic']:.4f}")
    print(f"  p-value:         {results['p_value']:.6f}")
    
    if results['p_value'] < 0.001:
        sig = "***"
    elif results['p_value'] < 0.01:
        sig = "**"
    elif results['p_value'] < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    print(f"  Significance:    {sig}")
    print("="*60)


def plot_results(results, output_path=None):
    """Create visualization of results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Individual differences (boxplot)
    ax1 = axes[0]
    data_to_plot = pd.DataFrame({
        'Ctrl': results['diff_ctrl'],
        'VAT': results['diff_vat']
    })
    data_to_plot.boxplot(ax=ax1)
    ax1.set_ylabel('Effect of Pyr3', fontsize=12)
    ax1.set_title('Effect of Pyr3 by Condition', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histograms of differences
    ax2 = axes[1]
    ax2.hist(results['diff_ctrl'], bins=10, alpha=0.6, label='Ctrl', edgecolor='black')
    ax2.hist(results['diff_vat'], bins=10, alpha=0.6, label='VAT', edgecolor='black')
    ax2.axvline(results['diff_ctrl_mean'], color='blue', linestyle='--', linewidth=2, 
                label=f'Ctrl mean = {results["diff_ctrl_mean"]:.2f}')
    ax2.axvline(results['diff_vat_mean'], color='orange', linestyle='--', linewidth=2,
                label=f'VAT mean = {results["diff_vat_mean"]:.2f}')
    ax2.set_xlabel('Effect of Pyr3', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Effects', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage
if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Load data from CSV file
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Try to find APD_delta_delta.csv in current directory
        csv_path = Path(__file__).parent / 'APD_delta_delta.csv'
        if not csv_path.exists():
            csv_path = 'APD_delta_delta.csv'
    
    if Path(csv_path).exists():
        print(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        # Convert empty strings to NaN
        df = df.replace('', np.nan)
        print(f"Columns found: {list(df.columns)}")
    else:
        print(f"CSV file not found: {csv_path}")
        print("Using example data instead...")
        example_data = {
            'Ctrl': [328.4211, 294.7368, 303.1579, 338.9474, 378.9474, 351.5789, 374.7368, 372.6316, 372.6316],
            'Ctrl + Pyr3': [286.8421, 355.7895, 263.1579, 302.6316, 294.7368, 263.1579, np.nan, np.nan, np.nan],
            'VAT EVs': [404.2105, 400, 368.4211, 378.9474, 418.9474, 406.3158, np.nan, np.nan, np.nan],
            'VAT EVs + Pyr3': [332.6316, 290.5263, 298.9474, 275.7895, 341.0526, 269.4737, np.nan, np.nan, np.nan]
        }
        df = pd.DataFrame(example_data)
    
    print("Input data:")
    print(df)
    
    # Perform test
    results = difference_in_differences_test(df)
    
    # Print results
    print_results(results)
    
    # Create plot
    plot_results(results, output_path='difference_in_differences_plot.png')
    
    # Save results to CSV
    summary_df = pd.DataFrame({
        'Metric': ['n_Ctrl', 'n_VAT', 'Ctrl Effect Mean', 'Ctrl Effect Std', 
                   'VAT Effect Mean', 'VAT Effect Std', 'DID Mean', 'DID SE', 
                   't-statistic', 'p-value'],
        'Value': [results['n_ctrl'], results['n_vat'], results['diff_ctrl_mean'], 
                 results['diff_ctrl_std'], results['diff_vat_mean'], results['diff_vat_std'], 
                 results['did_mean'], results['did_se'], results['t_statistic'], results['p_value']]
    })
    summary_df.to_csv('difference_in_differences_results.csv', index=False)
    print("\n✓ Saved results to: difference_in_differences_results.csv")

