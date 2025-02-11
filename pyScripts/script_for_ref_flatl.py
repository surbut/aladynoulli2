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
    """
    Load all essential components.
    """
    print("Loading components...")
    Y = torch.load(os.path.join(base_path, 'Y_tensor.pt'))
    E = torch.load(os.path.join(base_path, 'E_matrix.pt'))
    G = torch.load(os.path.join(base_path, 'G_matrix.pt'))
    essentials = torch.load(os.path.join(base_path, 'model_essentials.pt'))
    print("Loaded all components successfully!")
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

    Y_subset, E_subset, G_subset, indices = subset_data(
    Y, E, G, 
    start_index=args.start_index,
    end_index=args.end_index
)
    
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
        genetic_scale=1,
        flat_lambda=True,
        prevalence_t=essentials['prevalence_t'],
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
    history_new = model.fit(E_subset, num_epochs=100, learning_rate=1e-4, lambda_reg=1e-2)
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
    model_save_path = os.path.join(run_dir, 'model.pt')
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


    def plot_signature_top_diseases_centered(model, disease_names, plot_dir, n_top=10):
        """
        Show top diseases for each signature, centered relative to prevalence.
        """
        phi = model.phi.detach().numpy()  # Shape: (K, D, T)
        prevalence_logit = model.logit_prev_t.detach().numpy()  # Shape: (D, T)

        # Create a text file to save the results
        output_file = os.path.join(plot_dir, "signature_top_diseases.txt")
        with open(output_file, 'w') as f:
            # Center phi relative to prevalence
            phi_centered = np.zeros_like(phi)
            for k in range(phi.shape[0]):
                for d in range(phi.shape[1]):
                    phi_centered[k, d, :] = phi[k, d, :] - prevalence_logit[d, :]

            # Average over time
            phi_avg = phi_centered.mean(axis=2)  # Shape: (K, D)

            for k in range(phi_avg.shape[0]):
                scores = phi_avg[k, :]
                top_indices = np.argsort(scores)[-n_top:][::-1]

                output = f"\nTop {n_top} diseases in Signature {k} (relative to baseline):\n"
                f.write(output)
                print(output)
                
                for idx in top_indices:
                    avg_effect = scores[idx]
                    temporal_std = np.std(phi_centered[k, idx, :])
                    odds_ratio = np.exp(avg_effect)

                    # Handle disease name indexing
                    try:
                        if isinstance(disease_names, pd.DataFrame):
                            disease_name = disease_names.iloc[idx, 0]
                        elif isinstance(disease_names, pd.Series):
                            disease_name = disease_names.iloc[idx]
                        else:
                            disease_name = disease_names[idx]
                    except (IndexError, KeyError):
                        disease_name = f"Disease_{idx}"

                    output = f"{disease_name}: effect={avg_effect:.3f} (OR={odds_ratio:.2f}), std={temporal_std:.3f}\n"
                    f.write(output)
                    print(output)

        # Create visualization
        plt.figure(figsize=(15, 10))
        for k in range(min(4, phi_avg.shape[0])):  # Plot first 4 signatures
            plt.subplot(2, 2, k+1)
            scores = phi_avg[k, :]
            top_indices = np.argsort(scores)[-n_top:][::-1]
            
            y_pos = np.arange(len(top_indices))
            values = scores[top_indices]
            
            # Get disease names for top indices
            disease_labels = []
            for idx in top_indices:
                try:
                    if isinstance(disease_names, pd.DataFrame):
                        disease_name = disease_names.iloc[idx, 0]
                    elif isinstance(disease_names, pd.Series):
                        disease_name = disease_names.iloc[idx]
                    else:
                        disease_name = disease_names[idx]
                except (IndexError, KeyError):
                    disease_name = f"Disease_{idx}"
                disease_labels.append(disease_name)
            
            plt.barh(y_pos, values)
            plt.yticks(y_pos, disease_labels)
            plt.title(f'Signature {k} Top Diseases')
            plt.xlabel('Effect Size (centered)')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "signature_top_diseases.png"))
        plt.close()

    # Run visualization
    plot_signature_top_diseases_centered(model, essentials['disease_names'], plot_dir)


    def compare_disease_rankings(model, disease_names, output_file, n_top=10):
        """
        Compare initial vs final disease rankings for each signature and save to file.
        """
        # Open file for writing
        with open(output_file, 'w') as f:
            psi = model.psi.detach().numpy()  # Shape: (K, D)
            phi = model.phi.detach().numpy()  # Shape: (K, D, T)
            prevalence_logit = model.logit_prev_t.detach().numpy()  # Shape: (D, T)

            # Center phi relative to prevalence
            phi_centered = np.zeros_like(phi)
            for k in range(phi.shape[0]):
                for d in range(phi.shape[1]):
                    phi_centered[k, d, :] = phi[k, d, :] - prevalence_logit[d, :]

            # Average over time
            phi_avg = phi_centered.mean(axis=2)  # Shape: (K, D)

            for k in range(phi_avg.shape[0]):
                output = f"\nSignature {k}:\n"
                f.write(output)
                print(output, end='')

                # Get initial and final top diseases
                initial_scores = psi[k, :]
                initial_top = np.argsort(initial_scores)[-n_top:][::-1]
                final_scores = phi_avg[k, :]
                final_top = np.argsort(final_scores)[-n_top:][::-1]

                output = "\nInitial top diseases:\n"
                f.write(output)
                print(output, end='')
                
                for i, idx in enumerate(initial_top):
                    try:
                        if isinstance(disease_names, pd.DataFrame):
                            disease_name = disease_names.iloc[idx, 0]
                        elif isinstance(disease_names, pd.Series):
                            disease_name = disease_names.iloc[idx]
                        else:
                            disease_name = disease_names[idx]
                    except (IndexError, KeyError):
                        disease_name = f"Disease_{idx}"

                    output = f"{i+1}. {disease_name}: {initial_scores[idx]:.3f}\n"
                    f.write(output)
                    print(output, end='')

                output = "\nFinal top diseases:\n"
                f.write(output)
                print(output, end='')
                
                for i, idx in enumerate(final_top):
                    try:
                        if isinstance(disease_names, pd.DataFrame):
                            disease_name = disease_names.iloc[idx, 0]
                        elif isinstance(disease_names, pd.Series):
                            disease_name = disease_names.iloc[idx]
                        else:
                            disease_name = disease_names[idx]
                    except (IndexError, KeyError):
                        disease_name = f"Disease_{idx}"

                    output = f"{i+1}. {disease_name}: {final_scores[idx]:.3f}\n"
                    f.write(output)
                    print(output, end='')

                output = "\nSignificant rank changes:\n"
                f.write(output)
                print(output, end='')
                
                # Calculate significant rank changes
                initial_ranks = {disease: rank for rank, disease in enumerate(initial_top)}
                final_ranks = {disease: rank for rank, disease in enumerate(final_top)}

                for disease in set(initial_top) | set(final_top):
                    initial_rank = initial_ranks.get(disease, n_top + 1)
                    final_rank = final_ranks.get(disease, n_top + 1)
                    if abs(final_rank - initial_rank) > 2:
                        try:
                            if isinstance(disease_names, pd.DataFrame):
                                disease_name = disease_names.iloc[disease, 0]
                            elif isinstance(disease_names, pd.Series):
                                disease_name = disease_names.iloc[disease]
                            else:
                                disease_name = disease_names[disease]
                        except (IndexError, KeyError):
                            disease_name = f"Disease_{disease}"
                        output = f"{disease_name}: Rank changed from {initial_rank + 1} to {final_rank + 1}\n"
                        f.write(output)
                        print(output, end='')

    # Run comparison and save to file
    rankings_file = os.path.join(run_dir, "disease_rankings_comparison.txt")
    compare_disease_rankings(model, essentials['disease_names'], rankings_file)


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