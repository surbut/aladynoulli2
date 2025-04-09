import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import softmax

def find_consistent_signatures(mgb_model, aou_model, ukb_model, disease_names):
    """
    Find diseases that consistently appear in cardiovascular and malignancy signatures
    across all three biobanks
    """
    # Get phi values from each model
    mgb_phi = mgb_model.phi.detach().numpy()
    aou_phi = aou_model.phi.detach().numpy()
    ukb_phi = ukb_model.phi.detach().numpy()
    
    # Center phi values by prevalence
    mgb_prev = mgb_model.logit_prev_t.detach().numpy()
    aou_prev = aou_model.logit_prev_t.detach().numpy()
    ukb_prev = ukb_model.logit_prev_t.detach().numpy()
    
    def center_phi(phi, prev):
        phi_centered = np.zeros_like(phi)
        for k in range(phi.shape[0]):
            for d in range(phi.shape[1]):
                phi_centered[k, d, :] = phi[k, d, :] - prev[d, :]
        return phi_centered
    
    mgb_phi_centered = center_phi(mgb_phi, mgb_prev)
    aou_phi_centered = center_phi(aou_phi, aou_prev)
    ukb_phi_centered = center_phi(ukb_phi, ukb_prev)
    
    # Average over time
    mgb_phi_avg = mgb_phi_centered.mean(axis=2)
    aou_phi_avg = aou_phi_centered.mean(axis=2)
    ukb_phi_avg = ukb_phi_centered.mean(axis=2)
    
    # Find top diseases for cardiovascular signature in each biobank
    cv_signature_idx = {'mgb': 5, 'aou': 5, 'ukb': 5}  # Adjust these indices as needed
    malig_signature_idx = {'mgb': 6, 'aou': 6, 'ukb': 6}  # Adjust these indices as needed
    
    def get_top_diseases(phi_avg, sig_idx, n_top=20):
        scores = phi_avg[sig_idx, :]
        return np.argsort(scores)[-n_top:][::-1]
    
    # Get top diseases for each signature in each biobank
    cv_top = {
        'mgb': get_top_diseases(mgb_phi_avg, cv_signature_idx['mgb']),
        'aou': get_top_diseases(aou_phi_avg, cv_signature_idx['aou']),
        'ukb': get_top_diseases(ukb_phi_avg, cv_signature_idx['ukb'])
    }
    
    malig_top = {
        'mgb': get_top_diseases(mgb_phi_avg, malig_signature_idx['mgb']),
        'aou': get_top_diseases(aou_phi_avg, malig_signature_idx['aou']),
        'ukb': get_top_diseases(ukb_phi_avg, malig_signature_idx['ukb'])
    }
    
    # Find diseases that appear in top list across all biobanks
    cv_consistent = set(cv_top['mgb']) & set(cv_top['aou']) & set(cv_top['ukb'])
    malig_consistent = set(malig_top['mgb']) & set(malig_top['aou']) & set(malig_top['ukb'])
    
    return {
        'cardiovascular': {
            'diseases': list(cv_consistent),
            'names': [disease_names[i] for i in cv_consistent],
            'signature_idx': cv_signature_idx
        },
        'malignancy': {
            'diseases': list(malig_consistent),
            'names': [disease_names[i] for i in malig_consistent],
            'signature_idx': malig_signature_idx
        }
    }

def plot_signature_patterns_by_clusters(mgb_checkpoint, aou_checkpoint, ukb_checkpoint, 
                                      mgb_diseases, aou_diseases, ukb_diseases):
    """
    Plot temporal patterns for diseases that are shared across all biobanks in each signature group,
    using consistent colors for the same diseases and showing global averages
    """
    # Define signature mappings for each biobank
    cv_signatures = {
        'mgb': 5,
        'aou': 16,
        'ukb': 5
    }
    
    malig_signatures = {
        'mgb': 11,
        'aou': 11,
        'ukb': 6
    }
    
    # Get clusters from checkpoints
    mgb_clusters = mgb_checkpoint['clusters']
    aou_clusters = aou_checkpoint['clusters']
    ukb_clusters = ukb_checkpoint['clusters']
    
    # Find all diseases in each signature for each biobank
    def get_signature_diseases(diseases, clusters, sig_num):
        return {name: i for i, name in enumerate(diseases) 
                if clusters[i] == sig_num}
    
    # Get cardiovascular diseases for each biobank
    mgb_cv = get_signature_diseases(mgb_diseases, mgb_clusters, 5)
    aou_cv = get_signature_diseases(aou_diseases, aou_clusters, 16)
    ukb_cv = get_signature_diseases(ukb_diseases, ukb_clusters, 5)
    
    # Get malignancy diseases for each biobank
    mgb_malig = get_signature_diseases(mgb_diseases, mgb_clusters, 11)
    aou_malig = get_signature_diseases(aou_diseases, aou_clusters, 11)
    ukb_malig = get_signature_diseases(ukb_diseases, ukb_clusters, 6)
    
    # Find shared diseases across biobanks
    cv_shared = set(mgb_cv.keys()) & set(aou_cv.keys()) & set(ukb_cv.keys())
    malig_shared = set(mgb_malig.keys()) & set(aou_malig.keys()) & set(ukb_malig.keys())
    
    # Create figure with 2 rows and 3 columns
    fig, ((ax1_mgb, ax1_aou, ax1_ukb), 
          (ax2_mgb, ax2_aou, ax2_ukb)) = plt.subplots(2, 3, figsize=(20, 12))
    
    # Create consistent color mappings for shared diseases
    cv_colors = dict(zip(sorted(cv_shared), plt.cm.tab20(np.linspace(0, 1, len(cv_shared)))))
    malig_colors = dict(zip(sorted(malig_shared), plt.cm.tab20(np.linspace(0, 1, len(malig_shared)))))
    
    # Helper function to plot patterns for one biobank
    def plot_biobank_patterns(diseases_dict, shared_diseases, signature_num, checkpoint, 
                            ax, title, colors, other_checkpoints=None, other_sigs=None):
        # First plot average patterns in background
        if other_checkpoints and other_sigs:
            for disease in shared_diseases:
                patterns = []
                for cp, sig in zip(other_checkpoints, other_sigs):
                    idx = diseases_dict[disease]
                    pattern = cp['model_state_dict']['phi'][sig, idx, :51]
                    patterns.append(pattern)
                avg_pattern = np.mean(patterns, axis=0)
                ax.plot(avg_pattern, color='gray', alpha=0.2, linestyle='--')
        
        # Then plot this biobank's patterns
        for disease in shared_diseases:
            idx = diseases_dict[disease]
            pattern = checkpoint['model_state_dict']['phi'][signature_num, idx, :51]
            ax.plot(pattern, color=colors[disease], alpha=0.8,
                   label=f"{disease} ({signature_num})")
        
        ax.set_xlabel('Time (age 30-80)')
        ax.set_ylabel('Phi Value')
        ax.set_title(f"{title} (n={len(shared_diseases)})")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot cardiovascular patterns for each biobank
    other_checkpoints = [mgb_checkpoint, aou_checkpoint, ukb_checkpoint]
    other_cv_sigs = [cv_signatures['mgb'], cv_signatures['aou'], cv_signatures['ukb']]
    
    plot_biobank_patterns(mgb_cv, cv_shared, cv_signatures['mgb'], mgb_checkpoint,
                         ax1_mgb, 'MGB Cardiovascular (Sig 5)', cv_colors,
                         other_checkpoints, other_cv_sigs)
    plot_biobank_patterns(aou_cv, cv_shared, cv_signatures['aou'], aou_checkpoint,
                         ax1_aou, 'AoU Cardiovascular (Sig 16)', cv_colors,
                         other_checkpoints, other_cv_sigs)
    plot_biobank_patterns(ukb_cv, cv_shared, cv_signatures['ukb'], ukb_checkpoint,
                         ax1_ukb, 'UKB Cardiovascular (Sig 5)', cv_colors,
                         other_checkpoints, other_cv_sigs)
    
    # Plot malignancy patterns for each biobank
    other_malig_sigs = [malig_signatures['mgb'], malig_signatures['aou'], malig_signatures['ukb']]
    
    plot_biobank_patterns(mgb_malig, malig_shared, malig_signatures['mgb'], mgb_checkpoint,
                         ax2_mgb, 'MGB Malignancy (Sig 11)', malig_colors,
                         other_checkpoints, other_malig_sigs)
    plot_biobank_patterns(aou_malig, malig_shared, malig_signatures['aou'], aou_checkpoint,
                         ax2_aou, 'AoU Malignancy (Sig 11)', malig_colors,
                         other_checkpoints, other_malig_sigs)
    plot_biobank_patterns(ukb_malig, malig_shared, malig_signatures['ukb'], ukb_checkpoint,
                         ax2_ukb, 'UKB Malignancy (Sig 6)', malig_colors,
                         other_checkpoints, other_malig_sigs)
    
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary of shared diseases across biobanks:")
    print(f"\nCardiovascular signature (shared across MGB:5, AoU:16, UKB:5):")
    print(f"Number of shared diseases: {len(cv_shared)}")
    print("Diseases:")
    for disease in sorted(cv_shared):
        print(f"- {disease}")
    
    print(f"\nMalignancy signature (shared across MGB:11, AoU:11, UKB:6):")
    print(f"Number of shared diseases: {len(malig_shared)}")
    print("Diseases:")
    for disease in sorted(malig_shared):
        print(f"- {disease}")

# Example usage:
# plot_signature_patterns_by_clusters(mgb_checkpoint, aou_checkpoint, ukb_checkpoint,
#                                   mgb_checkpoint['disease_names'], aou_checkpoint['disease_names'], ukb_checkpoint['disease_names']) 