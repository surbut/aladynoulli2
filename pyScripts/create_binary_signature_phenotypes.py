#!/usr/bin/env python3
"""
Create binary signature phenotypes from continuous AUC values.

This script:
1. Reads the signature AUC phenotypes file
2. Generates histograms showing the distribution of each signature
3. Creates binary (0/1) phenotypes based on a specified threshold method
4. Outputs files suitable for GWAS (PLINK/REGENIE format)

Threshold methods:
- percentile: top X% are cases (e.g., top 25% = percentile 75)
- mean_sd: mean + X*SD are cases
- absolute: values above X are cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_signature_data(filepath):
    """Load signature AUC phenotypes file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep='\t')
    print(f"Loaded {len(df)} individuals with {len(df.columns)-2} signatures")
    return df


def get_signature_columns(df):
    """Get list of signature AUC columns."""
    return [col for col in df.columns if col.endswith('_AUC')]


def compute_summary_statistics(df, sig_cols):
    """Compute summary statistics for each signature."""
    stats = []
    for col in sig_cols:
        values = df[col]
        stats.append({
            'Signature': col.replace('_AUC', ''),
            'Mean': values.mean(),
            'Std': values.std(),
            'Min': values.min(),
            'P25': values.quantile(0.25),
            'Median': values.quantile(0.50),
            'P75': values.quantile(0.75),
            'P90': values.quantile(0.90),
            'P95': values.quantile(0.95),
            'Max': values.max()
        })
    return pd.DataFrame(stats)


def plot_histograms(df, sig_cols, output_dir, threshold_method='percentile', 
                    threshold_value=75, show_threshold=True):
    """
    Create histograms for all signatures with threshold lines.
    
    Parameters:
    - threshold_method: 'percentile', 'mean_sd', or 'absolute'
    - threshold_value: percentile (0-100), number of SDs, or absolute value
    """
    n_sigs = len(sig_cols)
    n_cols = 4
    n_rows = (n_sigs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    thresholds = {}
    
    for i, col in enumerate(sig_cols):
        ax = axes[i]
        values = df[col]
        
        # Compute threshold based on method
        if threshold_method == 'percentile':
            thresh = np.percentile(values, threshold_value)
        elif threshold_method == 'mean_sd':
            thresh = values.mean() + threshold_value * values.std()
        elif threshold_method == 'absolute':
            thresh = threshold_value
        else:
            raise ValueError(f"Unknown threshold method: {threshold_method}")
        
        thresholds[col] = thresh
        n_cases = (values >= thresh).sum()
        case_pct = 100 * n_cases / len(values)
        
        # Plot histogram
        ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Add threshold line
        if show_threshold:
            ax.axvline(x=thresh, color='red', linestyle='--', linewidth=2, 
                      label=f'Threshold: {thresh:.2f}')
        
        # Labels
        sig_name = col.replace('_AUC', '')
        ax.set_title(f'{sig_name}\n(Cases: {n_cases:,} = {case_pct:.1f}%)', fontsize=10)
        ax.set_xlabel('AUC Value', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        
        # Add mean line
        ax.axvline(x=values.mean(), color='green', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linestyle='-', linewidth=1.5, label='Mean'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Threshold')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.suptitle(f'Signature AUC Distributions\nThreshold: {threshold_method} = {threshold_value}', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'signature_histograms_{threshold_method}_{threshold_value}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved histograms to {output_path}")
    
    # Also save as PDF
    pdf_path = output_dir / f'signature_histograms_{threshold_method}_{threshold_value}.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved histograms to {pdf_path}")
    
    plt.close()
    
    return thresholds


def create_binary_phenotypes(df, sig_cols, thresholds, output_dir, 
                             plink_format=True, threshold_method='percentile', 
                             threshold_value=75):
    """
    Create binary phenotype files.
    
    Parameters:
    - plink_format: if True, use 1=control, 2=case; if False, use 0=control, 1=case
    """
    # Create combined file with all signatures
    binary_df = df[['FID', 'IID']].copy()
    
    case_counts = {}
    
    for col in sig_cols:
        sig_name = col.replace('_AUC', '')
        thresh = thresholds[col]
        
        # Create binary phenotype
        if plink_format:
            # PLINK format: 1=control, 2=case
            binary_df[f'{sig_name}_BINARY'] = np.where(df[col] >= thresh, 2, 1)
        else:
            # Standard format: 0=control, 1=case
            binary_df[f'{sig_name}_BINARY'] = np.where(df[col] >= thresh, 1, 0)
        
        case_counts[sig_name] = (df[col] >= thresh).sum()
    
    # Save combined file
    combined_path = output_dir / f'signature_binary_phenotypes_{threshold_method}_{threshold_value}.txt'
    binary_df.to_csv(combined_path, sep='\t', index=False)
    print(f"Saved combined binary phenotypes to {combined_path}")
    
    # Save individual signature files (for REGENIE)
    sig_dir = output_dir / f'binary_signatures_{threshold_method}_{threshold_value}'
    sig_dir.mkdir(exist_ok=True)
    
    for col in sig_cols:
        sig_name = col.replace('_AUC', '')
        sig_df = binary_df[['FID', 'IID', f'{sig_name}_BINARY']].copy()
        sig_df.columns = ['FID', 'IID', 'PHENO']
        
        sig_path = sig_dir / f'{sig_name}_binary.txt'
        sig_df.to_csv(sig_path, sep='\t', index=False)
    
    print(f"Saved individual signature files to {sig_dir}/")
    
    return binary_df, case_counts


def create_summary_report(df, sig_cols, thresholds, case_counts, output_dir,
                          threshold_method, threshold_value):
    """Create a summary report of the binarization."""
    report_lines = [
        "=" * 70,
        "BINARY SIGNATURE PHENOTYPE SUMMARY",
        "=" * 70,
        f"\nThreshold Method: {threshold_method}",
        f"Threshold Value: {threshold_value}",
        f"Total Individuals: {len(df):,}",
        "\n" + "-" * 70,
        f"{'Signature':<15} {'Threshold':>12} {'Cases':>10} {'Controls':>10} {'Case %':>10}",
        "-" * 70
    ]
    
    total_n = len(df)
    for col in sig_cols:
        sig_name = col.replace('_AUC', '')
        thresh = thresholds[col]
        n_cases = case_counts[sig_name]
        n_controls = total_n - n_cases
        pct = 100 * n_cases / total_n
        
        report_lines.append(
            f"{sig_name:<15} {thresh:>12.3f} {n_cases:>10,} {n_controls:>10,} {pct:>9.1f}%"
        )
    
    report_lines.append("-" * 70)
    
    # Summary statistics
    avg_cases = np.mean(list(case_counts.values()))
    avg_pct = 100 * avg_cases / total_n
    report_lines.append(f"\nAverage cases per signature: {avg_cases:,.0f} ({avg_pct:.1f}%)")
    
    report = '\n'.join(report_lines)
    print(report)
    
    # Save report
    report_path = output_dir / f'binary_phenotype_summary_{threshold_method}_{threshold_value}.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved summary report to {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Create binary signature phenotypes from continuous AUC values',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Top 25% as cases (percentile 75)
  python create_binary_signature_phenotypes.py --method percentile --value 75
  
  # Top 10% as cases (percentile 90)
  python create_binary_signature_phenotypes.py --method percentile --value 90
  
  # Mean + 1 SD as threshold
  python create_binary_signature_phenotypes.py --method mean_sd --value 1
  
  # Mean + 2 SD as threshold
  python create_binary_signature_phenotypes.py --method mean_sd --value 2
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/for_regenie/nog_bigamp/signature_auc_phenotypes.txt',
                        help='Input signature AUC phenotypes file')
    parser.add_argument('--output', '-o', type=str,
                        default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/binary_signature_phenotypes',
                        help='Output directory')
    parser.add_argument('--method', '-m', type=str, default='percentile',
                        choices=['percentile', 'mean_sd', 'absolute'],
                        help='Threshold method: percentile, mean_sd, or absolute')
    parser.add_argument('--value', '-v', type=float, default=75,
                        help='Threshold value (percentile 0-100, num SDs, or absolute value)')
    parser.add_argument('--plink-format', action='store_true', default=True,
                        help='Use PLINK format (1=control, 2=case)')
    parser.add_argument('--regenie-format', action='store_true',
                        help='Use REGENIE format (0=control, 1=case)')
    
    args = parser.parse_args()
    
    # Determine format
    plink_format = not args.regenie_format
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_signature_data(args.input)
    sig_cols = get_signature_columns(df)
    
    # Compute and display summary statistics
    print("\n" + "=" * 70)
    print("SIGNATURE SUMMARY STATISTICS")
    print("=" * 70)
    stats_df = compute_summary_statistics(df, sig_cols)
    print(stats_df.to_string(index=False))
    
    # Save statistics
    stats_path = output_dir / 'signature_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved statistics to {stats_path}")
    
    # Plot histograms
    print("\n" + "=" * 70)
    print("GENERATING HISTOGRAMS")
    print("=" * 70)
    thresholds = plot_histograms(df, sig_cols, output_dir, 
                                  threshold_method=args.method,
                                  threshold_value=args.value)
    
    # Create binary phenotypes
    print("\n" + "=" * 70)
    print("CREATING BINARY PHENOTYPES")
    print("=" * 70)
    binary_df, case_counts = create_binary_phenotypes(
        df, sig_cols, thresholds, output_dir,
        plink_format=plink_format,
        threshold_method=args.method,
        threshold_value=args.value
    )
    
    # Create summary report
    print("\n")
    create_summary_report(df, sig_cols, thresholds, case_counts, output_dir,
                          args.method, args.value)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
