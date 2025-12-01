"""
Recreate Signature Correlation Analysis
Analyze signature correlations from loaded theta values
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_signature_correlations(thetas, disease_names=None, save_plots=True):
    """
    Analyze signature correlations and create visualization
    
    Parameters:
    -----------
    thetas : np.ndarray
        Signature loadings (patients x signatures x time) or (patients x signatures)
    disease_names : list, optional
        Disease names for context
    save_plots : bool
        Whether to save plots
        
    Returns:
    --------
    dict : Results dictionary
    """
    
    print(f"\n{'='*80}")
    print(f"SIGNATURE CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Check shape
    if len(thetas.shape) == 3:
        n_patients, n_signatures, n_timepoints = thetas.shape
        print(f"Analyzing {n_patients} patients, {n_signatures} signatures, {n_timepoints} timepoints")
        # Average over time
        thetas_avg = np.mean(thetas, axis=2)  # Shape: (patients, signatures)
    else:
        n_patients, n_signatures = thetas.shape
        print(f"Analyzing {n_patients} patients and {n_signatures} signatures")
        thetas_avg = thetas
    
    # 1. Calculate overall correlation matrix
    print(f"\nCalculating overall signature correlations...")
    overall_corr = np.corrcoef(thetas_avg.T)  # Shape: (signatures, signatures)
    print(f"Correlation matrix shape: {overall_corr.shape}")
    
    # 2. Find top correlated pairs
    corr_pairs = []
    for i in range(n_signatures):
        for j in range(i+1, n_signatures):
            corr_pairs.append({
                'sig1': i,
                'sig2': j,
                'correlation': overall_corr[i, j]
            })
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    print("\nTop 15 most correlated signature pairs:")
    for i, pair in enumerate(corr_pairs[:15]):
        print(f"  {i+1}. Sig {pair['sig1']} vs Sig {pair['sig2']}: {pair['correlation']:.4f}")
    
    # 3. Calculate correlations by age (if 3D data)
    age_correlations = None
    if len(thetas.shape) == 3:
        print(f"\nCalculating correlations by age...")
        ages = []
        avg_corrs = []
        
        # Analyze every 10 years
        for t in range(0, min(51, n_timepoints), 10):  # Up to age 80
            age = t + 30
            sig_at_time = thetas[:, :, t]  # Shape: (patients, signatures)
            corr_matrix = np.corrcoef(sig_at_time.T)  # Shape: (signatures, signatures)
            
            # Average correlation (excluding diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = np.mean(corr_matrix[mask])
            
            ages.append(age)
            avg_corrs.append(avg_corr)
            print(f"  Age {age}: Average correlation = {avg_corr:.4f}")
        
        age_correlations = {'ages': ages, 'avg_correlations': avg_corrs}
    
    # 4. Create visualizations
    print(f"\nCreating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Overall correlation heatmap
    ax = axes[0, 0]
    im = ax.imshow(overall_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title('Overall Signature Correlations', fontsize=14, fontweight='bold')
    ax.set_xlabel('Signature Index', fontsize=12)
    ax.set_ylabel('Signature Index', fontsize=12)
    ax.set_xticks(range(n_signatures))
    ax.set_yticks(range(n_signatures))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11)
    
    # Plot 2: Top 15 correlation pairs
    ax = axes[0, 1]
    top_pairs = corr_pairs[:15]
    pair_labels = [f"Sig {p['sig1']}-{p['sig2']}" for p in top_pairs]
    correlations = [p['correlation'] for p in top_pairs]
    
    bars = ax.barh(range(len(pair_labels)), correlations, 
                   color=['red' if c < 0 else 'blue' for c in correlations])
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Top 15 Signature Correlations', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Correlations by age
    ax = axes[1, 0]
    if age_correlations is not None:
        ages = age_correlations['ages']
        avg_corrs = age_correlations['avg_correlations']
        
        ax.plot(ages, avg_corrs, 'o-', linewidth=2, markersize=8, 
                color='darkblue', markerfacecolor='lightblue', markeredgewidth=2)
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Average Correlation', fontsize=12)
        ax.set_title('Signature Correlations by Age', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(ages[0]-2, ages[-1]+2)
    else:
        ax.text(0.5, 0.5, 'No time-series data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Signature Correlations by Age', fontsize=14, fontweight='bold')
    
    # Plot 4: Empty (reserved for future analysis)
    ax = axes[1, 1]
    ax.axis('off')
    
    plt.suptitle('Signature Correlation Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_plots:
        filename = 'signature_correlation_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as '{filename}'")
    
    plt.show()
    
    return {
        'overall_correlations': overall_corr,
        'correlation_pairs': corr_pairs,
        'age_correlations': age_correlations
    }


def run_analysis():
    """
    Run the signature correlation analysis on loaded data
    """
    import torch
    import pandas as pd
    
    # Load data
    print("Loading data...")
    
    # Load thetas
    thetas = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt').detach().numpy()
    print(f"Loaded thetas shape: {thetas.shape}")
    
    # Load disease names if available
    try:
        disease_names = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/disease_names.csv')['x'].tolist()
        print(f"Loaded {len(disease_names)} disease names")
    except:
        disease_names = None
        print("Could not load disease names")
    
    # Run analysis
    results = analyze_and_plot_signature_correlations(thetas, disease_names=disease_names)
    
    return results

if __name__ == "__main__":
    results = run_analysis()

