import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_signature_temporal_patterns(model, disease_names, biobank_data):
    """
    Plot temporal patterns for cardiovascular and malignancy signatures across biobanks
    """
    # Get phi and prevalence
    phi = model.phi.detach().numpy()  # Shape: (K, D, T)
    prevalence_logit = model.logit_prev_t.detach().numpy()  # Shape: (D, T)
    
    # Center phi relative to prevalence
    phi_centered = np.zeros_like(phi)
    for k in range(phi.shape[0]):
        for d in range(phi.shape[1]):
            phi_centered[k, d, :] = phi[k, d, :] - prevalence_logit[d, :]
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot cardiovascular signature (blocks 12-17)
    cv_signatures = range(12, 18)
    for biobank in ['MGB', 'AoU', 'UKB']:
        data = biobank_data[biobank]
        temporal_pattern = np.mean([phi_centered[k] for k in cv_signatures], axis=0)
        ax1.plot(temporal_pattern.mean(axis=0), label=biobank, linewidth=2)
        
    ax1.set_title('Cardiovascular Disease Temporal Patterns')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Average Phi Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot malignancy signature (blocks 2-4)
    malig_signatures = range(2, 5)
    for biobank in ['MGB', 'AoU', 'UKB']:
        data = biobank_data[biobank]
        temporal_pattern = np.mean([phi_centered[k] for k in malig_signatures], axis=0)
        ax2.plot(temporal_pattern.mean(axis=0), label=biobank, linewidth=2)
    
    ax2.set_title('Malignancy Temporal Patterns')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Average Phi Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nSummary of temporal patterns:")
    print("\nCardiovascular Disease Blocks (12-17):")
    for k in cv_signatures:
        top_diseases = np.argsort(phi_centered[k].mean(axis=1))[-5:][::-1]
        print(f"\nBlock {k} top diseases:")
        for idx in top_diseases:
            consistency = np.mean([biobank_data[b]['consistency'][k, idx] for b in ['MGB', 'AoU', 'UKB']])
            print(f"- {disease_names[idx]}: {consistency:.2f} consistency across biobanks") 