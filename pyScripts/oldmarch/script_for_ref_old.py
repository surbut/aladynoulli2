from cluster_g_logit_init_acceptpsi_flatlam_healthtoo import *
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

def load_model_essentials(base_path):
    print("Loading components...")
    Y = torch.load(os.path.join(base_path, 'Y_tensor.pt'))
    print("Y initial storage:", Y.storage().size())
    print("Y initial storage offset:", Y.storage_offset())
    print("Y is contiguous:", Y.is_contiguous())
    E = torch.load(os.path.join(base_path, 'E_matrix.pt'))
    G = torch.load(os.path.join(base_path, 'G_matrix.pt'))
    essentials = torch.load(os.path.join(base_path, 'model_essentials.pt'))
    return Y, E, G, essentials

def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]  # Changed from slice to index list
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices


def main(args):
    # Load and initialize model:
    Y, E, G, essentials = load_model_essentials(args.data_dir)

        # Right after loading the data
    print("Original data shapes:")
    print("Y shape:", Y.shape)
    print("E shape:", E.shape)
    print("G shape:", G.shape)

    Y_subset, E_subset, G_subset, indices = subset_data(
    Y, E, G, 
    start_index=args.start_index,
    end_index=args.end_index
)
    print("Y_subset storage:", Y_subset.storage().size())
    print("Y_subset storage offset:", Y_subset.storage_offset())
    print("Y_subset is contiguous:", Y_subset.is_contiguous())
    print("\nSubset data shapes:")
    print("Y_subset shape:", Y_subset.shape)
    print("E_subset shape:", E_subset.shape)
    print("G_subset shape:", G_subset.shape)
    print("Number of indices:", len(indices))

    print(f"Subsetting from {args.start_index} to {args.end_index}")
    print("Y_subset shape:", Y_subset.shape)
    print("First index:", indices[0])
    print("Last index:", indices[-1])

    run_dir = os.path.join(args.work_dir, f"output_{args.start_index}_{args.end_index}")
    os.makedirs(run_dir, exist_ok=True)
    # Generate unique output directory for each run
    with open(os.path.join(run_dir, "subset_indices.txt"), "w") as f:
        f.write(f"Start index: {args.start_index}\n")
        f.write(f"End index: {args.end_index}\n")

        # Create a subdirectory for plots
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Initialize model with subsetted data

    # When initializing the model:
    original_G = G_subset # Store the original G - proper tensor copy

    # Now in your batch run, load and verify:
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
    # When initializing the model:


    torch.manual_seed(7)
    np.random.seed(4)
    # Create model without healthy reference
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_subset.shape[0],
        D=Y_subset.shape[1],
        T=Y_subset.shape[2],
        K=20,
        P=G_subset.shape[1],
        init_var_scaler=1e-1,
        G=G_subset,
        Y=Y_subset,
        genetic_scale=3,
        prevalence_t=essentials['prevalence_t'],
        signature_references=torch.load(os.path.join(args.data_dir, 'reference_trajectories.pt'))['signature_refs'],
        healthy_reference=True,
        disease_names=essentials['disease_names'],
    )

    torch.manual_seed(0)
    np.random.seed(0)
    # Initialize with psi and clusters
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters
    # Verify clusters match
    clusters_match = np.array_equal(initial_clusters, model.clusters)
    print(f"\nClusters match exactly: {clusters_match}")


    print(model.K_total)
    print(model.K)


    
    # Sample patients and set parameters
    
    sample_patients = [4376, 6640]  # Use specific patients
    n_top_states = 5  # Show only top 5 states per patient

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    time_points = np.arange(model.T)

    # Get dominant states for these patients
    lambda_patients = model.lambda_[sample_patients]  # [2, K, T]
    patient_props = torch.softmax(lambda_patients, dim=1)  # [2, K, T]
    mean_props = patient_props.mean(dim=-1)  # Average over time, now [2, K]
    top_states = mean_props.topk(n_top_states, dim=-1).indices  # Get top states for each patient

    # Color map for consistent colors across patients
    colors = plt.cm.tab20(np.linspace(0, 1, model.K))

    # Plot for each patient
    for i, patient in enumerate(sample_patients):
        for j, k in enumerate(top_states[i]):
            k = k.item()  # Convert tensor to int
            color = colors[k]
            
            # Plot lambda values
            ax1.plot(time_points, signature_refs[k], '--', color=color, alpha=0.3, 
                    label=f'Ref {k}')
            ax1.plot(time_points, model.lambda_[patient, k].detach(), '-', color=color,
                    label=f'Patient {patient} - State {k}')
            
            # Plot proportions
            ref_props = torch.softmax(signature_refs, dim=0)[k]
            patient_props = torch.softmax(model.lambda_[patient].detach(), dim=0)[k]
            ax2.plot(time_points, ref_props, '--', color=color, alpha=0.3, 
                    label=f'Ref {k}')
            ax2.plot(time_points, patient_props, '-', color=color,
                    label=f'Patient {patient} - State {k}')

    # Customize plots
    ax1.set_title('Initial Lambda Values\n(dashed=reference, solid=patient-specific)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lambda (logit scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_title('Initial Proportions\n(dashed=reference, solid=patient-specific)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Proportion')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "patient_lambda_and_proportions.png"))  # Save plot
    plt.close() 

    # Print genetic effects for these patients
    print("\nGenetic Effect Statistics for Sample Patients:")
    for i, patient in enumerate(sample_patients):
        genetic_effects = model.genetic_scale * (model.G[patient] @ model.gamma).detach()
        print(f"\nPatient {patient}:")
        print(f"Mean shift: {genetic_effects.mean():.3f}")
        print(f"Std of shifts: {genetic_effects.std():.3f}")
        print(f"Range of shifts: [{genetic_effects.min():.3f}, {genetic_effects.max():.3f}]")
        
        # Print top states and their proportions
        patient_mean_props = mean_props[i]  # Already computed above
        top_props, top_indices = patient_mean_props.topk(n_top_states)
        print("\nTop states and average proportions:")
        for state, prop in zip(top_indices, top_props):
            print(f"State {state.item()}: {prop:.3f}")


    plt.ioff()  # Turn off interactive mode

    # Then your plotting code:
    plt.figure(figsize=(12, 8))
    model.visualize_initialization()
    plt.tight_layout()  # Adjust layout to prevent cutoff
    plt.savefig(os.path.join(plot_dir, "initialization.png"), bbox_inches='tight', dpi=300)
    plt.close()

    initial_gamma = model.gamma.detach().clone()
    initial_phi = model.phi.detach().clone()
    initial_lambda = model.lambda_.detach().clone()
    initial_psi = model.psi.detach().clone()
    clusters_match = np.array_equal(initial_clusters, model.clusters)
    print(f"\nClusters match exactly: {clusters_match}")
    #history_new = model.fit(E_100k, num_epochs=100, learning_rate=1e-4, lambda_reg=1e-2)
    history_new = model.fit(E_subset, num_epochs=1, learning_rate=1e-4, lambda_reg=1e-2)
    final_lambda = model.lambda_.detach().clone()
    diff = torch.abs(final_lambda - initial_lambda)

    print(f"Lambda changes with lr=1e-4:")
    print(f"Mean absolute change: {torch.mean(diff):.3e}")
    print(f"Max absolute change: {torch.max(diff):.3e}")
    print(f"Std of changes: {torch.std(diff):.3e}")


    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot loss
    ax1.plot(history_new['loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)

    # Plot gradients
    ax2.plot(history_new['max_grad_lambda'], label='Lambda')
    ax2.plot(history_new['max_grad_phi'], label='Phi')
    ax2.plot(history_new['max_grad_psi'], label='Psi')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Max Gradient Magnitude')
    ax2.set_title('Parameter Gradients')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_loss_and_gradients.png"))  # Save plot
    plt.close() 

    os.makedirs(args.work_dir, exist_ok=True)
    print(Y_subset.shape)
    model_save_path = os.path.join(run_dir, 'model.pt')

    print("\nDebugging model state:")
    state_dict = model.state_dict()
    total_size = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            size_gb = value.element_size() * value.nelement() / (1024**3)
            print(f"{key}: shape {value.shape}, size {size_gb:.3f} GB")
            total_size += size_gb
    print(f"\nTotal state dict size: {total_size:.3f} GB")

    # Also check other large objects
    print("\nOther large objects:")
    print(f"initial_phi shape: {initial_phi.shape}")
    print(f"initial_lambda shape: {initial_lambda.shape}")
    print(f"model.lambda_ shape: {model.lambda_.shape}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'clusters': model.clusters,
        'initial_phi': initial_phi, 
        'initial_lambda': initial_lambda, 
        'psi': model.psi,
        'Y': Y_subset,
        'prevalence_t': essentials['prevalence_t'],
        'logit_prevalence_t': model.logit_prev_t,
        'G': G_subset,
        'E': E_subset,
        'indices': indices,
        'disease_names': essentials['disease_names'],
        'hyperparameters': {
            'N': Y_subset.shape[0],
            'D': Y_subset.shape[1],
            'T': Y_subset.shape[2],
            'P': G_subset.shape[1],
            'K': model.phi.shape[0]
        }
    }, model_save_path)
    print(f"Model saved to {model_save_path}")
    


    # 1. Get predictions and actual values
    predicted = model.forward()
    pi_pred = predicted[0] if isinstance(predicted, tuple) else predicted
    pi_pred = pi_pred.cpu().detach().numpy()
    Y = model.Y.cpu().detach().numpy()

    # 2. Calculate marginal risks directly
    # Assuming dimensions are: [N, D, T] for both Y and pi_pred
    observed_risk = Y.mean(axis=0).flatten()  # average across individuals
    predicted_risk = pi_pred.mean(axis=0).flatten()

    # 3. Apply calibration
    scale_factor = np.mean(observed_risk) / np.mean(predicted_risk)
    calibrated_risk = predicted_risk * scale_factor

    # 4. Plot final calibration
    plt.figure(figsize=(12, 5))

    # Original predictions
    plt.subplot(121)
    plt.scatter(observed_risk, predicted_risk, alpha=0.5)
    plt.plot([0, 0.02], [0, 0.02], 'r--')  # y=x line
    plt.title('Original Predictions (After Training)')
    plt.xlabel('Observed Risk')
    plt.ylabel('Predicted Risk')

    # Calibrated predictions
    plt.subplot(122)
    plt.scatter(observed_risk, calibrated_risk, alpha=0.5)
    plt.plot([0, 0.02], [0, 0.02], 'r--')  # y=x line
    plt.title('Calibrated Predictions (After Training)')
    plt.xlabel('Observed Risk')
    plt.ylabel('Calibrated Risk')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "calibration_plot_final.png"))  # Changed filename
    plt.close()

    # Create a log file for final calibration statistics
    stats_file = os.path.join(run_dir, "calibration_stats_final.txt")  # Changed filename
    with open(stats_file, 'w') as f:
        # Write statistics
        f.write(f"Mean observed risk: {np.mean(observed_risk):.6f}\n")
        f.write(f"Mean predicted risk (original): {np.mean(predicted_risk):.6f}\n")
        f.write(f"Mean predicted risk (calibrated): {np.mean(calibrated_risk):.6f}\n")
        f.write(f"Calibration scale factor: {scale_factor:.3f}\n\n")

        ss_res = np.sum((observed_risk - calibrated_risk) ** 2)
        ss_tot = np.sum((observed_risk - np.mean(observed_risk)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        f.write(f"Rsquared_calibrated: {r2:.3f}\n\n")



    # Sample patients and set parameters
    n_samples = 2
    sample_patients = [4376, 6640]  # Use specific patients
    n_top_states = 5  # Show only top 5 states per patient

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    time_points = np.arange(model.T)

    # Get dominant states for these patients
    lambda_patients = model.lambda_[sample_patients]  # [2, K, T]
    patient_props = torch.softmax(lambda_patients, dim=1)  # [2, K, T]
    mean_props = patient_props.mean(dim=-1)  # Average over time, now [2, K]
    top_states = mean_props.topk(n_top_states, dim=-1).indices  # Get top states for each patient

    # Color map for consistent colors across patients
    colors = plt.cm.tab20(np.linspace(0, 1, model.K))

    # Plot for each patient
    for i, patient in enumerate(sample_patients):
        for j, k in enumerate(top_states[i]):
            k = k.item()  # Convert tensor to int
            color = colors[k]
            
            # Plot lambda values
            ax1.plot(time_points, signature_refs[k], '--', color=color, alpha=0.3, 
                    label=f'Ref {k}')
            ax1.plot(time_points, model.lambda_[patient, k].detach(), '-', color=color,
                    label=f'Patient {patient} - State {k}')
            
            # Plot proportions
            ref_props = torch.softmax(signature_refs, dim=0)[k]
            patient_props = torch.softmax(model.lambda_[patient].detach(), dim=0)[k]
            ax2.plot(time_points, ref_props, '--', color=color, alpha=0.3, 
                    label=f'Ref {k}')
            ax2.plot(time_points, patient_props, '-', color=color,
                    label=f'Patient {patient} - State {k}')

    # Customize plots
    ax1.set_title('Final Lambda Values\n(dashed=reference, solid=patient-specific)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lambda (logit scale)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    ax2.set_title('Final Proportions\n(dashed=reference, solid=patient-specific)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Proportion')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "patient_lambda_and_proportions.png"))  # Save plot
    plt.close()  # Close the figure to free memory

    # Print genetic effects for these patients
    print("\nGenetic Effect Statistics for Sample Patients:")
    for i, patient in enumerate(sample_patients):
        genetic_effects = model.genetic_scale * (model.G[patient] @ model.gamma).detach()
        print(f"\nPatient {patient}:")
        print(f"Mean shift: {genetic_effects.mean():.3f}")
        print(f"Std of shifts: {genetic_effects.std():.3f}")
        print(f"Range of shifts: [{genetic_effects.min():.3f}, {genetic_effects.max():.3f}]")
        
        # Print top states and their proportions
        patient_mean_props = mean_props[i]  # Already computed above
        top_props, top_indices = patient_mean_props.topk(n_top_states)
        print("\nTop states and average proportions:")
        for state, prop in zip(top_indices, top_props):
            print(f"State {state.item()}: {prop:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training script with specific indices.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--work_dir", type=str, default="./", help="Directory to save outputs.")
    parser.add_argument("--start_index", type=int, required=True, help="Start index for subsetting data.")
    parser.add_argument("--end_index", type=int, required=True, help="End index for subsetting data.")
    args = parser.parse_args()

    main(args)