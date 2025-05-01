import torch
import numpy as np
import matplotlib.pyplot as plt
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

def plot_cancer_analysis(results):
    """
    Create a visualization of the cancer analysis results
    """
    # Extract cancer types and metrics
    cancer_types = list(results.keys())
    loading_ratios = [results[c]['loading_ratio'] for c in cancer_types]
    risk_ratios = [results[c]['risk_ratio'] for c in cancer_types]
    patient_counts = [results[c]['n_patients'] for c in cancer_types]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.2])
    
    # Plot loading ratios
    bars1 = ax1.bar(cancer_types, loading_ratios, color='skyblue', alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Signature 6 Loading Ratio by Primary Cancer Type', pad=20)
    ax1.set_ylabel('Loading Ratio (Primary/Control)')
    ax1.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax1.set_xticklabels([])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    # Plot risk ratios
    bars2 = ax2.bar(cancer_types, risk_ratios, color='lightgreen', alpha=0.7)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Secondary Cancer Risk Ratio by Primary Cancer Type', pad=20)
    ax2.set_ylabel('Risk Ratio (P(Secondary|Primary) / P(Secondary|No Primary))')
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    ax2.set_xticklabels(cancer_types, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    # Add patient counts as text below each bar
    for i, count in enumerate(patient_counts):
        ax2.text(i, 0, f'n={count}', ha='center', va='top', rotation=90)
    
    plt.tight_layout()
    plt.savefig('cancer_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def analyze_primary_secondary_cancer(model_path, Y, R, W, disease_names, window_size=5):
    """
    Analyze relationship between primary cancers and signature 6 (secondary malignancy)
    """
    # Load model and data
    checkpoint = torch.load(model_path)
    
    # Initialize model with checkpoint parameters
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=10000,
        D=Y.shape[1],
        T=Y.shape[2],
        K=20,
        P=36,
        G=checkpoint['G'][:10000],
        Y=Y[:10000],
        R=R[:10000],
        W=W[:10000],
        prevalence_t=checkpoint['prevalence_t'],
        init_sd_scaler=0.1,
        genetic_scale=1.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get cancer indices
    cancer_indices = {
        'primary': [
            disease_names.index('Cancer of bronchus; lung'),
            disease_names.index('Breast cancer [female]'),
            disease_names.index('Cancer of prostate'),
            disease_names.index('Colon cancer'),
            disease_names.index('Malignant neoplasm of rectum')
        ],
        'secondary': [
            disease_names.index('Secondary malignant neoplasm'),
            disease_names.index('Secondary malignancy of bone'),
            disease_names.index('Secondary malignant neoplasm of digestive systems'),
            disease_names.index('Secondary malignancy of lymph nodes')
        ]
    }
    
    # Get model predictions
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Get signature 6 loadings
    sig6_loading = theta[:, 6, :]
    
    # For each primary cancer type
    results = {}
    for cancer_type in cancer_indices['primary']:
        # Initialize counters for observed outcomes
        n_primary = 0
        n_secondary_after_primary = 0
        n_secondary_without_primary = 0
        n_no_primary = 0
        
        # Get mean signature 6 loading
        has_cancer = Y[:10000, cancer_type, :].any(dim=1)
        if has_cancer.any():
            cancer_sig6 = sig6_loading[has_cancer].mean()
            control_sig6 = sig6_loading[~has_cancer].mean()
            
            cum_pred_prob_with_cancer = []
            cum_pred_prob_without_cancer = []
            
            # For each timepoint
            for t in range(Y.shape[2] - window_size):
                # Get patients who got primary cancer at time t
                new_primary = Y[:10000, cancer_type, t] == 1
                
                if new_primary.any():
                    # Look at their outcomes and predictions in next window_size years
                    future_outcomes = torch.zeros_like(new_primary)
                    future_preds = torch.zeros_like(new_primary, dtype=torch.float)
                    
                    for sec_idx in cancer_indices['secondary']:
                        # Check if they got secondary cancer in window
                        future_outcomes |= Y[:10000, sec_idx, t:t+window_size].any(dim=1)
                        
                        # Get cumulative predicted probability in window
                        future_preds += (1 - (1 - pi[:, sec_idx, t:t+window_size]).prod(dim=1))
                    
                    # Average predictions over secondary cancer types
                    future_preds /= len(cancer_indices['secondary'])
                    
                    # Update observed outcome counters
                    n_primary += new_primary.sum().item()
                    n_secondary_after_primary += (new_primary & future_outcomes).sum().item()
                    
                    # For those without primary at this time
                    no_primary = ~new_primary
                    n_no_primary += no_primary.sum().item()
                    n_secondary_without_primary += (no_primary & future_outcomes).sum().item()
                    
                    # Store prediction results
                    cum_pred_prob_with_cancer.append(future_preds[new_primary].mean().item())
                    cum_pred_prob_without_cancer.append(future_preds[~new_primary].mean().item())
            
            # Calculate final statistics
            p_secondary_given_primary = n_secondary_after_primary / n_primary if n_primary > 0 else 0
            p_secondary_no_primary = n_secondary_without_primary / n_no_primary if n_no_primary > 0 else 0
            risk_ratio = p_secondary_given_primary / p_secondary_no_primary if p_secondary_no_primary > 0 else 0
            
            # Average predictions over all timepoints
            mean_cum_pred_with = np.mean(cum_pred_prob_with_cancer) if cum_pred_prob_with_cancer else 0
            mean_cum_pred_without = np.mean(cum_pred_prob_without_cancer) if cum_pred_prob_without_cancer else 0
            
            results[disease_names[cancer_type]] = {
                'n_patients': has_cancer.sum().item(),
                'n_primary_events': n_primary,
                'mean_sig6_loading': cancer_sig6.item(),
                'control_sig6_loading': control_sig6.item(),
                'loading_ratio': (cancer_sig6 / control_sig6).item(),
                'p_secondary_given_primary': p_secondary_given_primary,
                'p_secondary_no_primary': p_secondary_no_primary,
                'risk_ratio': risk_ratio,
                'cum_pred_prob_with_cancer': mean_cum_pred_with,
                'cum_pred_prob_without_cancer': mean_cum_pred_without,
                'pred_ratio': (mean_cum_pred_with / mean_cum_pred_without if mean_cum_pred_without > 0 else 1.0)
            }
    
    # Print results
    print(f"\nAnalysis of Primary Cancers and Secondary Malignancy Risk ({window_size}-year window):")
    print("-" * 80)
    
    for cancer_type, stats in results.items():
        print(f"\n{cancer_type}:")
        print(f"Number of patients ever: {stats['n_patients']}")
        print(f"Number of primary events: {stats['n_primary_events']}")
        print(f"Mean signature 6 loading: {stats['mean_sig6_loading']:.3f}")
        print(f"Control signature 6 loading: {stats['control_sig6_loading']:.3f}")
        print(f"Loading ratio: {stats['loading_ratio']:.2f}x")
        print(f"Observed outcomes ({window_size}-year window):")
        print(f"  P(Secondary|Primary): {stats['p_secondary_given_primary']:.3f}")
        print(f"  P(Secondary|No Primary): {stats['p_secondary_no_primary']:.3f}")
        print(f"  Risk ratio: {stats['risk_ratio']:.2f}x")
        print(f"Model predictions ({window_size}-year window):")
        print(f"  Cum pred prob (with primary): {stats['cum_pred_prob_with_cancer']:.3f}")
        print(f"  Cum pred prob (without primary): {stats['cum_pred_prob_without_cancer']:.3f}")
        print(f"  Prediction ratio: {stats['pred_ratio']:.2f}x")
    
    return results


def analyze_disease_relationships(model, Y, disease_names, window_size=5):
    """
    Analyze pairwise disease relationships using model predictions
    
    Args:
    model: trained model with forward() returning pi, theta, phi_prob
    Y: observed outcomes tensor (N, D, T)
    disease_names: list of disease names
    window_size: window to look for subsequent conditions
    """
    # Get model predictions
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    N, D, T = Y.shape
    results = {}
    
    # For each disease pair
    for disease_a in range(D):
        results[disease_names[disease_a]] = {}
        
        for disease_b in range(D):
            if disease_a != disease_b:
                # Initialize counters
                n_primary = 0
                n_secondary_after_primary = 0
                n_secondary_without_primary = 0
                n_no_primary = 0
                
                cum_pred_prob_with = []
                cum_pred_prob_without = []
                
                # For each timepoint
                for t in range(T - window_size):
                    # Get patients who got disease A at time t
                    new_primary = Y[:, disease_a, t] == 1
                    
                    if new_primary.any():
                        # Look at disease B outcomes in next window_size years
                        future_outcomes = Y[:, disease_b, t:t+window_size].any(dim=1)
                        
                        # Get cumulative predicted probability in window
                        future_preds = 1 - (1 - pi[:, disease_b, t:t+window_size]).prod(dim=1)
                        
                        # Update counters
                        n_primary += new_primary.sum().item()
                        n_secondary_after_primary += (new_primary & future_outcomes).sum().item()
                        
                        # For those without disease A
                        no_primary = ~new_primary
                        n_no_primary += no_primary.sum().item()
                        n_secondary_without_primary += (no_primary & future_outcomes).sum().item()
                        
                        # Store predictions
                        cum_pred_prob_with.append(future_preds[new_primary].mean().item())
                        cum_pred_prob_without.append(future_preds[~new_primary].mean().item())
                
                # Calculate statistics
                p_b_given_a = n_secondary_after_primary / n_primary if n_primary > 0 else 0
                p_b_no_a = n_secondary_without_primary / n_no_primary if n_no_primary > 0 else 0
                risk_ratio = p_b_given_a / p_b_no_a if p_b_no_a > 0 else 0
                
                # Average predictions
                mean_pred_with = np.mean(cum_pred_prob_with) if cum_pred_prob_with else 0
                mean_pred_without = np.mean(cum_pred_prob_without) if cum_pred_prob_without else 0
                
                results[disease_names[disease_a]][disease_names[disease_b]] = {
                    'p_b_given_a': p_b_given_a,
                    'p_b_no_a': p_b_no_a,
                    'risk_ratio': risk_ratio,
                    'pred_prob_with': mean_pred_with,
                    'pred_prob_without': mean_pred_without,
                    'pred_ratio': (mean_pred_with / mean_pred_without if mean_pred_without > 0 else 1.0)
                }
    
    # Print top relationships
    print(f"\nTop Disease Relationships ({window_size}-year window):")
    print("-" * 80)
    
    # Flatten results for sorting
    flat_results = []
    for disease_a in results:
        for disease_b in results[disease_a]:
            stats = results[disease_a][disease_b]
            flat_results.append({
                'Disease A': disease_a,
                'Disease B': disease_b,
                'Risk Ratio': stats['risk_ratio'],
                'P(B|A)': stats['p_b_given_a'],
                'P(B|no A)': stats['p_b_no_a']
            })
    
    # Sort by risk ratio and print top relationships
    flat_results.sort(key=lambda x: x['Risk Ratio'], reverse=True)
    for rel in flat_results[:20]:  # top 20
        print(f"\n{rel['Disease A']} → {rel['Disease B']}:")
        print(f"Risk Ratio: {rel['Risk Ratio']:.2f}x")
        print(f"P(B|A): {rel['P(B|A)']:.3f}")
        print(f"P(B|no A): {rel['P(B|no A)']:.3f}")
    
    return results


if __name__ == "__main__":
    # Load data
    Y = torch.load("/Users/sarahurbut/Dropbox/data_for_running/Y_tensor.pt")
    Y = Y[:400000]
    
    # Load R and W tensors
    R = torch.load("/Users/sarahurbut/Dropbox/data_for_running/R_tensor.pt")
    W = torch.load("/Users/sarahurbut/Dropbox/data_for_running/W_tensor.pt")
    
    # Load disease names
    disease_names = [
        'Colon cancer',
        'Malignant neoplasm of rectum',
        'Cancer of bronchus; lung',
        'Melanomas of skin',
        'Breast cancer [female]',
        'Cancer of prostate',
        'Malignant neoplasm of kidney',
        'Malignant neoplasm of bladder',
        'Malignant neoplasm, other',
        'Secondary malignant neoplasm',
        'Secondary malignancy of bone',
        'Secondary malignant neoplasm of digestive systems',
        'Secondary malignancy of lymph nodes'
    ]
    
    # Run analysis
    model_path = "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt"
    results = analyze_primary_secondary_cancer(model_path, Y, R, W, disease_names, window_size=5)
    
    # Create visualization
    plot_cancer_analysis(results) 





