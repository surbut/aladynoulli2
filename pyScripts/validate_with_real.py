import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

def load_ground_truth(model_path="/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt"):
    """Load the fitted model state"""
    checkpoint = torch.load(model_path)
    return checkpoint

def create_subset_data(full_data, n_individuals=1000, disease_indices=None):
    """Create a subset of the data for validation"""
    if disease_indices is None:
        # Choose diseases with clear patterns
        # This should be modified based on known interesting diseases
        disease_indices = list(range(20))  # placeholder
    
    # Subset the data
    Y_subset = full_data['Y'][:n_individuals, disease_indices, :]
    G_subset = full_data['G'][:n_individuals]
    event_times_subset = full_data['E'][:n_individuals, disease_indices]
    
    return {
        'Y': Y_subset,
        'G': G_subset,
        'event_times': event_times_subset,
        'disease_indices': disease_indices
    }

def plot_signature_comparison(true_sigs, recovered_sigs, save_path=None):
    """Plot true vs recovered signatures"""
    K, T = true_sigs.shape
    fig, axes = plt.subplots(K, 1, figsize=(12, 3*K))
    times = np.arange(T)
    
    for k in range(K):
        ax = axes[k] if K > 1 else axes
        ax.plot(times, true_sigs[k], 'b-', label='True (Full Data)', linewidth=2)
        ax.plot(times, recovered_sigs[k], 'r--', label='Recovered (Subset)', linewidth=2)
        ax.set_title(f'Signature {k+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_psi_comparison(true_psi, recovered_psi, disease_names=None, save_path=None):
    """Plot true vs recovered psi values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot true psi
    sns.heatmap(true_psi, ax=ax1, cmap='RdBu_r', center=0)
    ax1.set_title('True ψ (Full Data)')
    ax1.set_xlabel('Disease')
    ax1.set_ylabel('Signature')
    if disease_names:
        ax1.set_xticklabels(disease_names, rotation=45, ha='right')
    
    # Plot recovered psi
    sns.heatmap(recovered_psi, ax=ax2, cmap='RdBu_r', center=0)
    ax2.set_title('Recovered ψ (Subset)')
    ax2.set_xlabel('Disease')
    ax2.set_ylabel('Signature')
    if disease_names:
        ax2.set_xticklabels(disease_names, rotation=45, ha='right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def find_strong_patterns(model, n_per_signature=5):
    """Find diseases with strongest association to each signature"""
    psi = model.psi.detach().numpy()
    K, D = psi.shape
    
    strong_diseases = []
    for k in range(K):
        # Get indices of top diseases for this signature
        top_indices = np.argsort(-psi[k])[:n_per_signature]
        strong_diseases.extend(top_indices)
    
    # Remove duplicates while preserving order
    strong_diseases = list(dict.fromkeys(strong_diseases))
    
    # Print the selected diseases and their associations
    print("\nSelected diseases with strong signature associations:")
    for d in strong_diseases:
        max_sig = np.argmax(psi[:, d])
        print(f"{model.disease_names[d]}: Strongest with Signature {max_sig+1} (ψ = {psi[max_sig, d]:.3f})")
    
    return strong_diseases

if __name__ == "__main__":
    # Load ground truth
    checkpoint = load_ground_truth()
    
    # Extract key components
    true_model = checkpoint['model']
    
    # Find diseases with strong patterns
    disease_indices = find_strong_patterns(true_model)
    
    # Create subset of data
    subset = create_subset_data(checkpoint['data'], disease_indices=disease_indices)
    
    # Initialize new model on subset
    subset_model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=subset['Y'].shape[0],
        D=subset['Y'].shape[1],
        T=subset['Y'].shape[2],
        K=true_model.K,
        P=true_model.P,
        G=subset['G'],
        Y=subset['Y'],
        prevalence_t=true_model.prevalence_t[subset['disease_indices']],
        disease_names=[true_model.disease_names[i] for i in subset['disease_indices']]
    )
    
    # Fit subset model
    subset_model.fit(subset['event_times'])
    
    # Compare results
    plot_signature_comparison(
        true_model.get_signatures().detach().numpy(),
        subset_model.get_signatures().detach().numpy(),
        save_path='signature_comparison.pdf'
    )
    
    plot_psi_comparison(
        true_model.psi.detach().numpy()[:, disease_indices],
        subset_model.psi.detach().numpy(),
        disease_names=[true_model.disease_names[i] for i in disease_indices],
        save_path='psi_comparison.pdf'
    ) 