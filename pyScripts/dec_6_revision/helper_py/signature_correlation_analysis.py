"""
Signature Correlation Analysis

Load newest lambda values from enrollment model files and analyze signature correlations
overall and for subsets of people at particular times and averaged over time periods.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
import os
from pathlib import Path

def load_lambda_values():
    """
    Load combined theta values from numpy file
    
    Returns:
    --------
    np.ndarray : Combined theta values (patients x signatures x time)
    """
    
    print("Loading combined theta values from PyTorch file...")
    
    filepath = "/Users/sarahurbut/aladynoulli2/pyScripts/all_patient_thetas_alltime.pt"
    
    if os.path.exists(filepath):
        thetas = torch.load(filepath, map_location='cpu').numpy()
        print(f"Loaded theta values shape: {thetas.shape}")
        return thetas
    else:
        print(f"Error: File not found: {filepath}")
        return None

def softmax_normalize_lambdas(lambdas):
    """
    Apply softmax normalization to lambda values
    
    Parameters:
    -----------
    lambdas : np.ndarray
        Raw lambda values (patients x signatures)
    
    Returns:
    --------
    np.ndarray : Softmax normalized lambda values
    """
    
    print("Applying softmax normalization...")
    
    # Apply softmax along signature dimension (axis=1)
    softmax_lambdas = softmax(lambdas, axis=1)
    
    print(f"Softmax normalized shape: {softmax_lambdas.shape}")
    print(f"Softmax values sum to 1: {np.allclose(np.sum(softmax_lambdas, axis=1), 1.0)}")
    
    return softmax_lambdas

def analyze_signature_correlations(lambdas, thetas=None, disease_names=None, 
                                 Y=None, time_points=None):
    """
    Analyze signature correlations overall and for subsets
    
    Parameters:
    -----------
    lambdas : np.ndarray
        Signature loadings (patients x signatures)
    thetas : np.ndarray, optional
        Signature trajectories (patients x signatures x time)
    disease_names : list, optional
        List of disease names
    Y : torch.Tensor, optional
        Binary disease matrix (patients x diseases x time)
    time_points : list, optional
        Specific time points to analyze
    """
    
    print(f"\n{'='*80}")
    print(f"SIGNATURE CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    if len(lambdas.shape) == 3:
        n_patients, n_signatures, n_timepoints = lambdas.shape
        print(f"Analyzing {n_patients} patients, {n_signatures} signatures, {n_timepoints} timepoints")
        print(f"Lambda shape: {lambdas.shape}")
    else:
        n_patients, n_signatures = lambdas.shape
        print(f"Analyzing {n_patients} patients and {n_signatures} signatures")
        print(f"Lambda shape: {lambdas.shape}")
    
    # Initialize variables
    time_correlations = None
    disease_correlations = None
    
    # 1. Overall signature correlations
    print(f"\n1. OVERALL SIGNATURE CORRELATIONS")
    
    if len(lambdas.shape) == 3:
        # For 3D lambdas, average over time first
        lambdas_avg = np.mean(lambdas, axis=2)  # Shape: (patients, signatures)
        overall_corr = np.corrcoef(lambdas_avg.T)
    else:
        overall_corr = np.corrcoef(lambdas.T)
    
    # Find most correlated signature pairs
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
    
    print("Top 10 most correlated signature pairs:")
    for i, pair in enumerate(corr_pairs[:10]):
        print(f"  {i+1}. Sig {pair['sig1']} vs Sig {pair['sig2']}: {pair['correlation']:.4f}")
    
    # 2. Signature correlations by time (use lambdas if 3D, otherwise thetas)
    if len(lambdas.shape) == 3 or thetas is not None:
        print(f"\n2. SIGNATURE CORRELATIONS BY TIME")
        
        # Use lambdas if 3D, otherwise use thetas
        if len(lambdas.shape) == 3:
            time_data = lambdas
            print("Using lambda values for time analysis")
        else:
            time_data = thetas
            print("Using theta values for time analysis")
        
        if time_points is None:
            # Analyze every 5th time point
            time_points = list(range(0, time_data.shape[2], 5))
        
        time_correlations = {}
        
        for t in time_points:
            if t < time_data.shape[2]:
                # Get signature loadings at time t
                sig_at_time = time_data[:, :, t]
                
                # Calculate correlations
                corr_matrix = np.corrcoef(sig_at_time.T)
                
                # Store average correlation
                # Exclude diagonal (self-correlation)
                mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                avg_corr = np.mean(corr_matrix[mask])
                
                time_correlations[t] = {
                    'correlation_matrix': corr_matrix,
                    'average_correlation': avg_corr,
                    'age': t + 30  # Convert to age
                }
                
                print(f"  Age {t+30}: Average correlation = {avg_corr:.4f}")
    
    # 3. Signature correlations by disease subsets (if Y available)
    if Y is not None and disease_names is not None:
        print(f"\n3. SIGNATURE CORRELATIONS BY DISEASE SUBSETS")
        
        # Analyze top diseases by prevalence
        disease_prevalence = []
        for i, disease_name in enumerate(disease_names):
            prevalence = Y[:, i, :].sum().item()
            disease_prevalence.append({
                'disease': disease_name,
                'prevalence': prevalence,
                'index': i
            })
        
        # Sort by prevalence
        disease_prevalence.sort(key=lambda x: x['prevalence'], reverse=True)
        
        # Analyze top 5 diseases
        for i, disease_info in enumerate(disease_prevalence[:5]):
            disease_name = disease_info['disease']
            disease_idx = disease_info['index']
            prevalence = disease_info['prevalence']
            
            print(f"\n  Disease: {disease_name} (prevalence: {prevalence})")
            
            # Find patients with this disease
            disease_patients = []
            for patient_id in range(Y.shape[0]):
                if Y[patient_id, disease_idx, :].sum() > 0:
                    disease_patients.append(patient_id)
            
            if len(disease_patients) > 100:  # Need sufficient sample size
                # Get lambda values for disease patients
                disease_lambdas = lambdas[disease_patients, :]
                
                # Calculate correlations
                disease_corr = np.corrcoef(disease_lambdas.T)
                
                # Average correlation (excluding diagonal)
                mask = ~np.eye(disease_corr.shape[0], dtype=bool)
                avg_corr = np.mean(disease_corr[mask])
                
                print(f"    Average correlation: {avg_corr:.4f}")
                print(f"    Sample size: {len(disease_patients)} patients")
            else:
                print(f"    Insufficient sample size: {len(disease_patients)} patients")
    
    return {
        'overall_correlations': overall_corr,
        'correlation_pairs': corr_pairs,
        'time_correlations': time_correlations,
        'disease_correlations': disease_correlations
    }

def create_correlation_visualizations(results, save_plots=True):
    """
    Create visualizations for signature correlations
    
    Parameters:
    -----------
    results : dict
        Results from analyze_signature_correlations
    save_plots : bool
        Whether to save plots
    """
    
    print(f"\n{'='*80}")
    print(f"CREATING CORRELATION VISUALIZATIONS")
    print(f"{'='*80}")
    
    # 1. Overall correlation heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Overall correlation matrix
    ax = axes[0, 0]
    overall_corr = results['overall_correlations']
    im = ax.imshow(overall_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title('Overall Signature Correlations', fontweight='bold')
    ax.set_xlabel('Signature Index')
    ax.set_ylabel('Signature Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient')
    
    # 2. Top correlation pairs
    ax = axes[0, 1]
    top_pairs = results['correlation_pairs'][:15]
    pair_labels = [f"Sig {p['sig1']}-{p['sig2']}" for p in top_pairs]
    correlations = [p['correlation'] for p in top_pairs]
    
    bars = ax.barh(range(len(pair_labels)), correlations, 
                   color=['red' if c < 0 else 'blue' for c in correlations])
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_title('Top 15 Signature Correlations', fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Time-based correlations (if available)
    if results['time_correlations'] is not None:
        ax = axes[1, 0]
        times = list(results['time_correlations'].keys())
        ages = [results['time_correlations'][t]['age'] for t in times]
        avg_corrs = [results['time_correlations'][t]['average_correlation'] for t in times]
        
        ax.plot(ages, avg_corrs, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Age')
        ax.set_ylabel('Average Correlation')
        ax.set_title('Signature Correlations by Age', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 4. Disease-based correlations (if available)
    if results['disease_correlations'] is not None:
        ax = axes[1, 1]
        diseases = [d['disease'][:20] + '...' if len(d['disease']) > 20 else d['disease'] 
                   for d in results['disease_correlations'][:10]]
        prevalences = [d['prevalence'] for d in results['disease_correlations'][:10]]
        
        bars = ax.bar(range(len(diseases)), prevalences, color='skyblue')
        ax.set_xticks(range(len(diseases)))
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        ax.set_ylabel('Prevalence')
        ax.set_title('Disease Prevalence (Top 10)', fontweight='bold')
    
    plt.tight_layout()
    
    if save_plots:
        filename = 'signature_correlation_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Correlation analysis plot saved as '{filename}'")
    
    plt.show()

def compare_old_vs_new():
    """
    Compare old theta values vs new lambda values
    """
    print("COMPARING OLD vs NEW VALUES")
    print("="*50)
    
    # Load old thetas
    try:
        thetas_old = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/big_stuff/all_patient_thetas_alltime.pt').numpy()
        print(f"Loaded old thetas: {thetas_old.shape}")
    except Exception as e:
        print(f"Could not load old thetas: {e}")
        return
    
    # Load new lambdas
    try:
        lambdas_new = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt')['model_state_dict']['lambda_'].numpy()
        print(f"Loaded new lambdas: {lambdas_new.shape}")
    except Exception as e:
        print(f"Could not load new lambdas: {e}")
        return
    
    # Apply softmax to new lambdas
    lambdas_new_softmax = softmax_normalize_lambdas(lambdas_new)
    
    # Compare shapes
    print(f"\nShape comparison:")
    print(f"Old thetas: {thetas_old.shape}")
    print(f"New lambdas (raw): {lambdas_new.shape}")
    print(f"New lambdas (softmax): {lambdas_new_softmax.shape}")
    
    # Compare first few patients if shapes match
    if thetas_old.shape == lambdas_new_softmax.shape:
        print(f"\nComparing first 5 patients:")
        for i in range(min(5, thetas_old.shape[0])):
            diff = np.abs(thetas_old[i] - lambdas_new_softmax[i]).max()
            mean_diff = np.mean(np.abs(thetas_old[i] - lambdas_new_softmax[i]))
            print(f"Patient {i}: Max diff = {diff:.6f}, Mean diff = {mean_diff:.6f}")
        
        # Overall comparison
        overall_diff = np.abs(thetas_old - lambdas_new_softmax).max()
        overall_mean_diff = np.mean(np.abs(thetas_old - lambdas_new_softmax))
        print(f"\nOverall comparison:")
        print(f"Max difference: {overall_diff:.6f}")
        print(f"Mean difference: {overall_mean_diff:.6f}")
        
        # Check if they're essentially the same
        if overall_diff < 1e-5:
            print("✅ Values are essentially identical!")
        elif overall_diff < 1e-3:
            print("⚠️ Values are very similar (small differences)")
        else:
            print("❌ Values are significantly different")
    else:
        print("❌ Shapes don't match - cannot compare directly")

def main():
    """
    Main function to run signature correlation analysis
    """
    
    print("SIGNATURE CORRELATION ANALYSIS")
    print("="*50)
    
    # Load theta values (these are the signature loadings)
    thetas = load_lambda_values()
    
    if thetas is None:
        print("Failed to load theta values!")
        return
    
    # Apply softmax normalization to lambda values
    print(f"Applying softmax normalization to lambda values...")
    softmax_thetas = softmax_normalize_lambdas(thetas)
    
    # Try to load Y matrix
    try:
        Y = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
        Y = Y[:400000, :, :]  # Match lambda size
        print(f"Loaded Y: {Y.shape}")
    except:
        print("Could not load Y matrix")
        Y = None
    
    # Try to load disease names
    try:
        disease_names = pd.read_csv('/Users/sarahurbut/aladynoulli2/pyScripts/disease_names.csv')['disease_name'].tolist()
        print(f"Loaded {len(disease_names)} disease names")
    except:
        print("Could not load disease names")
        disease_names = None
    
    # Analyze correlations
    results = analyze_signature_correlations(
        softmax_thetas, 
        thetas=softmax_thetas, 
        disease_names=disease_names, 
        Y=Y
    )
    
    # Create visualizations
    create_correlation_visualizations(results)
    
    return results

if __name__ == "__main__":
    results = main()
