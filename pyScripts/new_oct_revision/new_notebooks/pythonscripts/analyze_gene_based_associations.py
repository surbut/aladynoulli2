#!/usr/bin/env python3
"""
Analyze Gene-Based Association Test Results from REGENIE

This script processes gene-based association test results (RVAS) from REGENIE
and creates summary tables, visualizations, and identifies significant genes.

Input: Compressed gene-based association files (.gz) with tabix indexes (.tbi)
Output: Summary tables, Manhattan plots, and gene lists
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gzip
import subprocess
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

def read_genebased_file(file_path: Path) -> pd.DataFrame:
    """
    Read a gene-based association test result file from REGENIE.
    
    REGENIE gene-based files are BGZF-compressed tab-delimited files with:
    - Header lines starting with ##
    - Column header starting with #
    - Columns: ENST, SYMBOL, CHROM, GENPOS, ID, ALLELE0, ALLELE1, A1FREQ, N, TEST, BETA, SE, CHISQ, LOG10P, EXTRA
    """
    try:
        # Read BGZF-compressed file
        with gzip.open(file_path, 'rt') as f:
            # Skip header lines starting with ##
            line = f.readline()
            while line.startswith('##'):
                line = f.readline()
            
            # Read column header (starts with #)
            if line.startswith('#'):
                header = line.strip().lstrip('#').split('\t')
            else:
                print(f"Warning: Expected header line starting with #, got: {line[:50]}")
                return pd.DataFrame()
            
            # Read data
            data = []
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(line.strip().split('\t'))
        
        if not data:
            print(f"Warning: No data rows found in {file_path.name}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data, columns=header)
        
        # Convert numeric columns
        numeric_cols = ['GENPOS', 'A1FREQ', 'N', 'BETA', 'SE', 'CHISQ', 'LOG10P']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def extract_significant_genes(df: pd.DataFrame, 
                             pvalue_col: str = 'LOG10P',
                             threshold: float = 2.5e-6) -> pd.DataFrame:
    """
    Extract genes with significant associations.
    
    Default threshold 2.5e-6 is Bonferroni correction for ~20,000 genes.
    """
    if pvalue_col not in df.columns:
        # Try to find p-value column (LOG10P is -log10(p-value))
        pval_cols = [col for col in df.columns if 'LOG10P' in col.upper() or 'PVAL' in col.upper() or ('P' in col.upper() and 'LOG' not in col.upper())]
        if pval_cols:
            pvalue_col = pval_cols[0]
            print(f"Using p-value column: {pvalue_col}")
        else:
            print("Warning: Could not find p-value column")
            return pd.DataFrame()
    
    # LOG10P is -log10(p-value), so convert threshold
    if 'LOG10P' in pvalue_col.upper():
        log10_threshold = -np.log10(threshold)
        significant = df[df[pvalue_col] > log10_threshold].copy()  # Greater than because it's -log10
        significant = significant.sort_values(pvalue_col, ascending=False)  # Higher is more significant
    else:
        significant = df[df[pvalue_col] < threshold].copy()
        significant = significant.sort_values(pvalue_col)
    
    return significant

def analyze_all_signatures(results_dir: Path, 
                           signature_range: range = range(1, 21),
                           pvalue_threshold: float = 2.5e-6,
                           max_signatures: Optional[int] = None) -> Dict:
    """
    Analyze gene-based results for all signatures.
    
    Args:
        results_dir: Directory containing the gene-based result files
        signature_range: Range of signature numbers to analyze
        pvalue_threshold: P-value threshold for significance
        max_signatures: Maximum number of signatures to process (for quick testing)
    
    Returns:
        Dictionary with summary statistics and significant genes
    """
    all_results = {}
    all_significant = []
    
    print("="*80)
    print("ANALYZING GENE-BASED ASSOCIATION RESULTS")
    print("="*80)
    
    sig_list = list(signature_range)
    if max_signatures:
        sig_list = sig_list[:max_signatures]
        print(f"⚠️  Processing only first {max_signatures} signatures for quick test")
    
    for sig_num in sig_list:
        # Look for files matching pattern SIG[number]_AUC...
        pattern = f"SIG{sig_num}_AUC*genebased.gz"
        matching_files = list(results_dir.glob(pattern))
        
        if not matching_files:
            print(f"\n⚠️  No files found for Signature {sig_num}")
            continue
        
        # Use the first matching file
        file_path = matching_files[0]
        print(f"\n📊 Analyzing Signature {sig_num}: {file_path.name}")
        
        # Read the file
        df = read_genebased_file(file_path)
        
        if df.empty:
            print(f"   ⚠️  Empty file or read error")
            continue
        
        print(f"   ✓ Loaded {len(df)} genes")
        
        # Find p-value column (LOG10P for REGENIE gene-based)
        pval_col = None
        for col in df.columns:
            if 'LOG10P' in col.upper():
                pval_col = col
                break
            elif 'SKAT-O' in col or 'SKATO' in col:
                pval_col = col
                break
            elif 'P' in col.upper() and 'VALUE' in col.upper():
                pval_col = col
                break
        
        if pval_col is None:
            print(f"   ⚠️  Could not identify p-value column. Columns: {df.columns.tolist()}")
            all_results[sig_num] = {
                'n_genes': len(df),
                'n_significant': 0,
                'significant_genes': pd.DataFrame()
            }
            continue
        
        # Extract significant genes
        significant = extract_significant_genes(df, pvalue_col=pval_col, threshold=pvalue_threshold)
        
        print(f"   ✓ Found {len(significant)} significant genes (p < {pvalue_threshold:.2e})")
        
        if len(significant) > 0:
            print(f"   Top 5 genes:")
            for idx, row in significant.head(5).iterrows():
                gene = row.get('SYMBOL', row.get('GENE', 'N/A'))
                pval_log10 = row[pval_col]
                if 'LOG10P' in pval_col.upper():
                    pval = 10**(-pval_log10)
                    print(f"      {gene}: LOG10P = {pval_log10:.2f} (p = {pval:.2e})")
                else:
                    print(f"      {gene}: p = {pval_log10:.2e}")
        
        # Store results
        all_results[sig_num] = {
            'n_genes': len(df),
            'n_significant': len(significant),
            'pvalue_column': pval_col,
            'significant_genes': significant,
            'all_genes': df
        }
        
        # Add to combined list
        if len(significant) > 0:
            significant['Signature'] = sig_num
            all_significant.append(significant)
    
    # Combine all significant genes
    if all_significant:
        combined_significant = pd.concat(all_significant, ignore_index=True)
    else:
        combined_significant = pd.DataFrame()
    
    return {
        'by_signature': all_results,
        'all_significant': combined_significant,
        'summary': create_summary_table(all_results)
    }

def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create a summary table across all signatures."""
    summary_data = []
    
    for sig_num, data in results.items():
        summary_data.append({
            'Signature': sig_num,
            'N_Genes_Tested': data.get('n_genes', 0),
            'N_Significant': data.get('n_significant', 0),
            'Perc_Significant': (data.get('n_significant', 0) / max(data.get('n_genes', 1), 1)) * 100
        })
    
    return pd.DataFrame(summary_data)

def create_manhattan_plot(df: pd.DataFrame, 
                          pvalue_col: str,
                          gene_col: str = 'SYMBOL',
                          chrom_col: str = 'CHROM',
                          title: str = 'Gene-Based Association Results',
                          output_path: Optional[Path] = None):
    """Create a Manhattan plot for gene-based associations."""
    if df.empty:
        print("Cannot create Manhattan plot: empty dataframe")
        return None
    
    # Prepare data
    plot_df = df.copy()
    plot_df['CHROM_NUM'] = pd.to_numeric(plot_df[chrom_col], errors='coerce')
    
    # Handle LOG10P (already -log10) vs regular p-values
    if 'LOG10P' in pvalue_col.upper():
        plot_df['-log10P'] = pd.to_numeric(plot_df[pvalue_col], errors='coerce')
    else:
        plot_df['-log10P'] = -np.log10(pd.to_numeric(plot_df[pvalue_col], errors='coerce'))
    
    # Remove invalid values
    plot_df = plot_df.dropna(subset=['CHROM_NUM', '-log10P'])
    
    if plot_df.empty:
        print("Cannot create Manhattan plot: no valid data")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by chromosome
    colors = ['#2c7fb8', '#e74c3c']
    for chrom in sorted(plot_df['CHROM_NUM'].unique()):
        chrom_data = plot_df[plot_df['CHROM_NUM'] == chrom]
        color = colors[chrom % 2]
        ax.scatter(chrom_data['CHROM_NUM'], chrom_data['-log10P'], 
                  c=color, alpha=0.6, s=20)
    
    # Add significance line
    ax.axhline(y=-np.log10(2.5e-6), color='red', linestyle='--', 
              linewidth=1, label='Bonferroni threshold (p=2.5e-6)')
    
    ax.set_xlabel('Chromosome', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log10(P-value)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Manhattan plot to: {output_path}")
    
    return fig

def main(results_dir: str = None,
         output_dir: str = None,
         signature_range: range = range(1, 21),
         pvalue_threshold: float = 2.5e-6,
         max_signatures: Optional[int] = None):
    """
    Main function to analyze gene-based association results.
    
    Args:
        results_dir: Directory containing gene-based result files
        output_dir: Directory to save output files
        signature_range: Range of signatures to analyze
        pvalue_threshold: P-value threshold for significance
    """
    
    # Set default directories
    if results_dir is None:
        # Try to find the directory from the image
        results_dir = Path.home() / "SIG"  # Based on the image showing "SIG" folder
    else:
        results_dir = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_dir / "gene_based_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Analyze all signatures
    results = analyze_all_signatures(results_dir, signature_range, pvalue_threshold, max_signatures)
    
    # Save summary table
    summary_file = output_dir / "gene_based_summary.csv"
    results['summary'].to_csv(summary_file, index=False)
    print(f"\n✓ Saved summary table to: {summary_file}")
    
    # Save all significant genes
    if not results['all_significant'].empty:
        significant_file = output_dir / "significant_genes_all_signatures.csv"
        results['all_significant'].to_csv(significant_file, index=False)
        print(f"✓ Saved significant genes to: {significant_file}")
        
        # Create summary of top genes across all signatures
        gene_col = 'SYMBOL' if 'SYMBOL' in results['all_significant'].columns else 'GENE'
        pval_col = results['by_signature'][list(results['by_signature'].keys())[0]]['pvalue_column']
        
        if gene_col in results['all_significant'].columns:
            top_genes = results['all_significant'].groupby(gene_col).agg({
                'Signature': lambda x: ','.join(map(str, x)),
                pval_col: 'max' if 'LOG10P' in pval_col.upper() else 'min'  # Max for LOG10P (higher is better)
            }).sort_values(pval_col, ascending=('LOG10P' not in pval_col.upper()))
        
        top_genes_file = output_dir / "top_genes_across_signatures.csv"
        top_genes.to_csv(top_genes_file)
        print(f"✓ Saved top genes across signatures to: {top_genes_file}")
    
    # Create Manhattan plots for each signature with significant results
    print("\n" + "="*80)
    print("CREATING MANHATTAN PLOTS")
    print("="*80)
    
    for sig_num, data in results['by_signature'].items():
        if data['n_significant'] > 0:
            pval_col = data['pvalue_column']
            fig = create_manhattan_plot(
                data['all_genes'],
                pvalue_col=pval_col,
                title=f'Signature {sig_num}: Gene-Based Associations',
                output_path=output_dir / f"manhattan_sig{sig_num}.png"
            )
            if fig:
                plt.close(fig)
    
    # Print final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nTotal signatures analyzed: {len(results['by_signature'])}")
    print(f"Total significant genes: {len(results['all_significant'])}")
    print(f"\nSummary by signature:")
    print(results['summary'].to_string(index=False))
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze gene-based association test results')
    parser.add_argument('--results_dir', type=str, help='Directory containing gene-based result files')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--signature_range', type=str, default='1-20', 
                       help='Signature range (e.g., "1-20" or "1,2,3")')
    parser.add_argument('--pvalue_threshold', type=float, default=2.5e-6,
                       help='P-value threshold for significance (default: 2.5e-6)')
    parser.add_argument('--max_signatures', type=int, default=None,
                       help='Maximum number of signatures to process (for quick testing)')
    
    args = parser.parse_args()
    
    # Parse signature range
    if '-' in args.signature_range:
        start, end = map(int, args.signature_range.split('-'))
        sig_range = range(start, end + 1)
    else:
        sig_range = range(1, 21)  # Default
    
    main(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        signature_range=sig_range,
        pvalue_threshold=args.pvalue_threshold,
        max_signatures=args.max_signatures
    )

