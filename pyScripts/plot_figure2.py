import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd

def load_model_data(model_path, sig_refs_path):
    """Load model and signature reference data"""
    model = torch.load(model_path, weights_only=False)
    sig_refs = torch.load(sig_refs_path, weights_only=False)
    
    # Convert tensors to numpy arrays
    sigs = sig_refs['signature_refs'].detach().cpu().numpy()
    phi = model['model_state_dict']['phi'].detach().cpu().numpy()
    lambda_ = model['model_state_dict']['lambda'].detach().cpu().numpy()
    kappa = model['model_state_dict']['kappa'].detach().cpu().numpy()
    
    return sigs, phi, lambda_, kappa

def center_phi(phi):
    """Center phi by temporal mean for each disease"""
    phi_centered = np.zeros_like(phi)
    for k in range(phi.shape[0]):
        for d in range(phi.shape[1]):
            phi_centered[k, d, :] = phi[k, d, :] - np.mean(phi[:, d, :], axis=0)
    return phi_centered

def plot_signature_disease_associations(phi_centered, time_points=[0, 5, 10, 15, 20], n_top_diseases=5):
    """Panel A: Show how signature-disease associations evolve over time"""
    fig, axes = plt.subplots(1, len(time_points), figsize=(15, 5))
    
    # Calculate overall phi average for sorting
    phi_avg = phi_centered.mean(axis=2)
    max_sig = np.argmax(np.abs(phi_avg), axis=0)
    disease_order = np.argsort(max_sig)
    
    for i, t in enumerate(time_points):
        # Get phi at time t
        phi_t = phi_centered[:, :, t]
        phi_t_sorted = phi_t[:, disease_order]
        
        # Create heatmap with centered values
        vmax = np.max(np.abs(phi_centered))
        im = axes[i].imshow(phi_t_sorted, aspect='auto', cmap='RdBu_r', 
                          vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Time {t}')
        
        if i == 0:
            axes[i].set_ylabel('Signature Index')
        if i == len(time_points)//2:
            axes[i].set_xlabel('Diseases (sorted by primary signature)')
    
    plt.colorbar(im, ax=axes, label='Phi (centered)')
    plt.suptitle('A: Signature-Disease Associations Over Time')
    return fig

def plot_disease_signature_relationships(phi_centered, selected_diseases, disease_names=None):
    """Panel B: Show probabilistic relationships between diseases and signatures"""
    if disease_names is None:
        disease_names = [f"Disease {i}" for i in range(phi_centered.shape[1])]
    
    fig, axes = plt.subplots(len(selected_diseases), 1, figsize=(10, 3*len(selected_diseases)))
    
    for i, disease_idx in enumerate(selected_diseases):
        # Plot centered phi values for each signature
        for k in range(phi_centered.shape[0]):
            axes[i].plot(phi_centered[k, disease_idx, :], 
                        label=f'Signature {k}')
        
        axes[i].set_title(f'B: {disease_names[disease_idx][:40]}...')
        axes[i].set_ylabel('Phi (centered)')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    
    axes[-1].set_xlabel('Time')
    return fig

def plot_population_signature_evolution(lambda_):
    """Panel C: Show how signature prevalence changes across age cohorts"""
    # Calculate population-level theta
    theta = np.exp(lambda_ - np.max(lambda_, axis=1, keepdims=True))
    theta = theta / np.sum(theta, axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each signature's prevalence over time
    for k in range(theta.shape[0]):
        ax.plot(theta[k, :], label=f'Signature {k}')
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Population Prevalence')
    ax.set_title('C: Population-Level Signature Evolution')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

def plot_signature_trajectories(phi_centered, selected_signatures, disease_names=None):
    """Panel D: Show complete trajectories for selected signatures"""
    if disease_names is None:
        disease_names = [f"Disease {i}" for i in range(phi_centered.shape[1])]
    
    n_signatures = len(selected_signatures)
    fig, axes = plt.subplots(n_signatures, 2, figsize=(15, 3*n_signatures))
    
    for i, sig_idx in enumerate(selected_signatures):
        # Plot signature's top diseases over time
        top_diseases = np.argsort(-np.abs(phi_centered[sig_idx].mean(axis=1)))[:5]
        for d in top_diseases:
            axes[i, 0].plot(phi_centered[sig_idx, d, :], 
                          label=f'{disease_names[d][:30]}...')
        axes[i, 0].set_title(f'D: Signature {sig_idx} Top Diseases')
        axes[i, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        
        # Plot heatmap of all diseases for this signature
        vmax = np.max(np.abs(phi_centered))
        im = axes[i, 1].imshow(phi_centered[sig_idx], aspect='auto', 
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f'Signature {sig_idx} All Diseases')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Disease Index')
    
    plt.colorbar(im, ax=axes[:, 1], label='Phi (centered)')
    return fig

def main():
    # Load data
    model_path = "/Users/sarahurbut/Dropbox/resultshighamp/results/output_0_10000/model.pt"
    sig_refs_path = "/Users/sarahurbut/Dropbox/data_for_running/reference_trajectories.pt"
    sigs, phi, lambda_, kappa = load_model_data(model_path, sig_refs_path)
    
    # Center phi values
    phi_centered = center_phi(phi)
    
    # Create figure with 4 panels
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    plot_signature_disease_associations(phi_centered)
    
    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    plot_disease_signature_relationships(phi_centered, selected_diseases=[0, 1, 2])
    
    # Panel C
    ax3 = fig.add_subplot(gs[1, 0])
    plot_population_signature_evolution(lambda_)
    
    # Panel D
    ax4 = fig.add_subplot(gs[1, 1])
    plot_signature_trajectories(phi_centered, selected_signatures=[0, 1, 2])
    
    plt.tight_layout()
    plt.savefig('figure2.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main() 