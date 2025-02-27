from clust_huge_amp import *
import numpy as np
import pandas as pd
import torch
import argparse
import os
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering


def plot_signature_temporal_patterns(model, disease_names, plot_dir, n_top=10, selected_signatures=[0,5,6,19]):
    """Show temporal patterns of top diseases for each signature"""
    phi = model.phi.detach().numpy()
    prevalence_logit = model.logit_prev_t.detach().numpy()
    import os
    phi_centered = np.zeros_like(phi)
    for k in range(phi.shape[0]):
        for d in range(phi.shape[1]):
            phi_centered[k, d, :] = phi[k, d, :] - prevalence_logit[d, :]
    
    phi_avg = phi_centered.mean(axis=2)
    
    if selected_signatures is None:
        selected_signatures = range(phi_avg.shape[0])
    
    n_sigs = len(selected_signatures)
    fig, axes = plt.subplots(n_sigs, 1, figsize=(15, 5*n_sigs))
    if n_sigs == 1:
        axes = [axes]
    
    for i, k in enumerate(selected_signatures):
        scores = phi_avg[k, :]
        top_indices = np.argsort(scores)[-n_top:][::-1]
        
        ax = axes[i]
        for idx in top_indices:
            temporal_pattern = phi[k, idx, :]
            disease_name = disease_names[idx]
            ax.plot(temporal_pattern, label=disease_name)
        
        ax.set_title(f'Signature {k} - Top Disease Temporal Patterns')
        ax.set_xlabel('Time')
        ax.set_ylabel('Phi Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'temporal_patterns_withkappa.png'))
    plt.close('all') 

def plot_training_evolution(history, plot_dir):
    """Plot and save training metrics."""
    losses, gradient_history = history
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.yscale('log')
    plt.legend()
    
    # Plot lambda gradients
    plt.subplot(1, 3, 2)
    lambda_norms = [torch.norm(g).item() for g in gradient_history['lambda_grad']]
    plt.plot(lambda_norms, label='Lambda gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Lambda Gradient Evolution')
    plt.legend()
    
    # Plot phi gradients
    plt.subplot(1, 3, 3)
    phi_norms = [torch.norm(g).item() for g in gradient_history['phi_grad']]
    plt.plot(phi_norms, label='Phi gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Phi Gradient Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_evolution.png"))
    plt.close()

def plot_calibration(model, plot_dir):
    """Plot and save calibration metrics."""
    with torch.no_grad():
        predicted = model.forward()
        pi_pred = predicted[0] if isinstance(predicted, tuple) else predicted
        pi_pred = pi_pred.cpu().detach().numpy()
        Y = model.Y.cpu().detach().numpy()

    observed_risk = Y.mean(axis=0).flatten()
    predicted_risk = pi_pred.mean(axis=0).flatten()
    scale_factor = np.mean(observed_risk) / np.mean(predicted_risk)
    calibrated_risk = predicted_risk * scale_factor

    plt.figure(figsize=(12, 5))

    # Original predictions
    plt.subplot(121)
    plt.scatter(observed_risk, predicted_risk, alpha=0.5)
    plt.plot([0, 0.02], [0, 0.02], 'r--')
    plt.title('Original Predictions')
    plt.xlabel('Observed Risk')
    plt.ylabel('Predicted Risk')

    # Calibrated predictions
    plt.subplot(122)
    plt.scatter(observed_risk, calibrated_risk, alpha=0.5)
    plt.plot([0, 0.02], [0, 0.02], 'r--')
    plt.title('Calibrated Predictions')
    plt.xlabel('Observed Risk')
    plt.ylabel('Calibrated Risk')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "calibration_plot.png"))
    plt.close()

    # Save calibration statistics
    stats_file = os.path.join(plot_dir, "calibration_stats.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Mean observed risk: {np.mean(observed_risk):.6f}\n")
        f.write(f"Mean predicted risk (original): {np.mean(predicted_risk):.6f}\n")
        f.write(f"Mean predicted risk (calibrated): {np.mean(calibrated_risk):.6f}\n")
        f.write(f"Calibration scale factor: {scale_factor:.3f}\n")
        
        ss_res = np.sum((observed_risk - calibrated_risk) ** 2)
        ss_tot = np.sum((observed_risk - np.mean(observed_risk)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        f.write(f"R-squared (calibrated): {r2:.3f}\n")

def plot_psi_heatmap(model, disease_names, plot_dir, figsize=(12, 8)):
    """Plot and save psi heatmap."""
    plt.figure(figsize=figsize)
    
    # Get psi values from model and move to CPU if needed
    psi = model.psi.detach().cpu().numpy()
    
    # Create heatmap
    im = plt.imshow(psi, aspect='auto', cmap='RdBu_r', vmin=-5, vmax=5)
    
    plt.colorbar(im, label='ψ value')
    plt.xlabel('Disease')
    plt.ylabel('Signature')
    plt.title('ψ (Signature-Disease Associations)')
    
    if disease_names is not None:
        plt.xticks(range(len(disease_names)), disease_names, 
                  rotation=90, fontsize=8)
    
    plt.yticks(range(psi.shape[0]), [f'Signature {i}' for i in range(psi.shape[0])])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "psi_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()

def plot_theta_differences(model, plot_dir):
    """Plot and save theta distribution differences."""
    diseases = [112, 67, 127, 10, 17, 114]
    signatures = [5, 7, 0, 17, 19, 5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (d, sig) in enumerate(zip(diseases, signatures)):
        ax = axes[i]
        
        # Move tensors to CPU and get diagnosis times
        diagnosis_mask = model.Y[:, d, :].bool().cpu()
        
        # Get thetas
        with torch.no_grad():
            pi, theta, phi_prob = model.forward()
            theta = theta.cpu()
        
        # Plot distributions
        diagnosed_theta = theta[diagnosis_mask, sig].numpy()
        others_theta = theta[~diagnosis_mask, sig].numpy()
        
        ax.hist(diagnosed_theta, alpha=0.5, label='At diagnosis', density=True)
        ax.hist(others_theta, alpha=0.5, label='Others', density=True)
        
        ax.set_title(f'Disease {d} (sig {sig})')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "theta_differences.png"))
    plt.close()


def plot_disease_lambda_alignment(model):
    """
    Plot lambda values aligned with disease occurrences for selected patients
    """
    # Find patients with specific diseases and their diagnosis times
    disease_idx = 112  # MI
    patients_with_disease = []
    diagnosis_times = []
    
    for patient in range(model.Y.shape[0]):
        diag_time = torch.where(model.Y[patient, disease_idx])[0]
        if len(diag_time) > 0:
            patients_with_disease.append(patient)
            diagnosis_times.append(diag_time[0].item())
    
    # Sample a few patients
    n_samples = min(5, len(patients_with_disease))
    sample_indices = np.random.choice(len(patients_with_disease), n_samples, replace=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    time_points = np.arange(model.T)
    
    # Find signature that most strongly associates with this disease
    psi_disease = model.psi[:, disease_idx].detach()
    sig_idx = torch.argmax(psi_disease).item()
    
    # Plot for each sampled patient
    for idx in sample_indices:
        patient = patients_with_disease[idx]
        diag_time = diagnosis_times[idx]
        
        # Plot lambda (detached)
        lambda_values = torch.softmax(model.lambda_[patient].detach(), dim=0)[sig_idx]
        ax.plot(time_points, lambda_values.numpy(),
                alpha=0.5, label=f'Patient {patient}')
        
        # Mark diagnosis time
        ax.axvline(x=diag_time, linestyle=':', alpha=0.3)
    
    ax.set_title(f'Lambda Values for Signature {sig_idx} (Most Associated with MI)\nVertical Lines Show Diagnosis Times')
    ax.set_xlabel('Time')
    ax.set_ylabel('Lambda (proportion)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def load_model_essentials(base_path):
    """Load all essential components."""
    print("Loading components...")
    Y = torch.load(os.path.join(base_path, 'Y_tensor.pt'))
    E = torch.load(os.path.join(base_path, 'E_matrix.pt'))
    G = torch.load(os.path.join(base_path, 'G_matrix.pt'))
    essentials = torch.load(os.path.join(base_path, 'model_essentials.pt'))
    print("Loaded all components successfully!")
    return Y, E, G, essentials

def subset_data(Y, E, G, start_index, end_index):
    """Subset data based on indices."""
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices

def setup_directories(work_dir, start_index, end_index):
    """Create output directories."""
    run_dir = os.path.join(work_dir, f"output_{start_index}_{end_index}")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return run_dir, plot_dir

def save_model_and_results(model, save_dict, run_dir):
    """Save model and results."""
    model_save_path = os.path.join(run_dir, 'model.pt')
    torch.save(save_dict, model_save_path)
    print(f"Model saved to {model_save_path}")

def generate_plots(model, plot_dir, history, essentials):
    """Generate all plots."""
    print("Generating plots...")
    
    # Training evolution plot
    plot_training_evolution(history, plot_dir)
    
    # Calibration plot
    plot_calibration(model, plot_dir)
    
    # Psi heatmap
    plot_psi_heatmap(model, essentials['disease_names'], plot_dir)
    
    # Theta differences
    plot_theta_differences(model, plot_dir)

    plot_signature_temporal_patterns(model, essentials['disease_names'], plot_dir)
    

    
    # Disease lambda alignment for specific diseases
    plot_disease_lambda_alignment()
    
    print("All plots generated successfully!")

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(42)
    else:
        device = torch.device('cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        # Load data
        Y, E, G, essentials = load_model_essentials(args.data_dir)
        
        # Subset data
        Y_subset, E_subset, G_subset, indices = subset_data(
            Y, E, G, 
            start_index=args.start_index,
            end_index=args.end_index
        )
        
        # Setup directories
        run_dir, plot_dir = setup_directories(args.work_dir, args.start_index, args.end_index)
        
        # Log basic information
        with open(os.path.join(run_dir, "run_info.txt"), "w") as f:
            f.write(f"Start index: {args.start_index}\n")
            f.write(f"End index: {args.end_index}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Data shapes:\n")
            f.write(f"Y: {Y_subset.shape}\n")
            f.write(f"E: {E_subset.shape}\n")
            f.write(f"G: {G_subset.shape}\n")

        # Move data to device
        Y_subset = Y_subset.to(device)
        E_subset = E_subset.to(device)
        G_subset = G_subset.to(device)
        
  
        initial_psi = torch.load(os.path.join(args.data_dir, 'initial_psi_400k.pt'))
        initial_clusters = torch.load(os.path.join(args.data_dir, 'initial_clusters_400k.pt'))

        original_cluster_sizes = {}
        unique, counts = np.unique(initial_clusters, return_counts=True)
        for k, count in zip(unique, counts):
            original_cluster_sizes[k] = count
        print("\nOriginal cluster sizes:")
        for k, count in original_cluster_sizes.items():
            print(f"Cluster {k}: {count} diseases")


        print("Initial psi stats:")
        print(f"Shape: {initial_psi.shape}")
        print(f"Range: [{initial_psi.min():.2f}, {initial_psi.max():.2f}]")
        print(f"Number of positive values: {(initial_psi > 0).sum().item()}")




        # Load references (signatures only, no healthy)
        refs = torch.load('reference_trajectories.pt')
        signature_refs = refs['signature_refs']
        # Initialize model
        torch.manual_seed(7)
        np.random.seed(4)
        # Create model without healthy reference
        model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
            N=Y_subset.shape[0], 
            D=Y_subset.shape[1], 
            T=Y_subset.shape[2], 
            K=20,
            P=G_subset.shape[1],
            init_sd_scaler=1e-1,
            G=G_subset, 
            Y=Y_subset,
            genetic_scale=1,
            W=0.0001,
            R=0,
            prevalence_t=essentials['prevalence_t'],
            signature_references=signature_refs,  # Only pass signature refs
            healthy_reference=True,  # Explicitly set to None
            disease_names=essentials['disease_names']
        )

        # Initialize parameters
        torch.manual_seed(0)
        np.random.seed(0)
        model.initialize_params(true_psi=initial_psi)
        model.clusters = initial_clusters

        # Store initial state
        initial_state = {
            'phi': model.phi.detach().clone(),
            'lambda': model.lambda_.detach().clone(),
            'psi': model.psi.detach().clone()
        }

        # Train model
        print("Starting training...")
        history = model.fit(E_subset, num_epochs=200, learning_rate=1e-1, lambda_reg=1e-2)
        print("Training completed!")

        # Get predictions
        with torch.no_grad():
            predicted = model.forward()
            pi_pred = predicted[0] if isinstance(predicted, tuple) else predicted
            pi = pi_pred.cpu().numpy()
            theta = torch.softmax(model.lambda_.detach().cpu(), dim=1).numpy()

        # Save results
        save_dict = {
            'model_state_dict': model.state_dict(),
            'clusters': model.clusters,
            'initial_state': initial_state,
            'pi': pi,
            'theta': theta,
            'Y': Y_subset.cpu(),
            'prevalence_t': essentials['prevalence_t'],
            'logit_prevalence_t': model.logit_prev_t,
            'G': G_subset.cpu(),
            'E': E_subset.cpu(),
            'indices': indices,
            'disease_names': essentials['disease_names'],
            'hyperparameters': {
                'N': Y_subset.shape[0],
                'D': Y_subset.shape[1],
                'T': Y_subset.shape[2],
                'P': G_subset.shape[1],
                'K': model.phi.shape[0]
            },
            'training_history': history
        }
        
        save_model_and_results(model, save_dict, run_dir)
        
        # Generate plots
        generate_plots(model, plot_dir, history, essentials)
        
        print("Job completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        with open(os.path.join(args.work_dir, "error_log.txt"), "a") as f:
            f.write(f"Error in batch {args.start_index}-{args.end_index}: {str(e)}\n")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training on data subset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--work_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--start_index", type=int, required=True, help="Start index for data subset")
    parser.add_argument("--end_index", type=int, required=True, help="End index for data subset")
    args = parser.parse_args()
    
    main(args)
    