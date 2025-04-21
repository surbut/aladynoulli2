from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any 

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_training_evolution(history_tuple):
    losses, gradient_history = history_tuple
    
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
    plt.show()


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



def plot_theta_differences(model, diseases=None, signatures=None):
    """
    Plot theta distributions for diagnosed vs non-diagnosed patients
    
    Parameters:
    model: The trained model (can be enrollment-constrained or full-data)
    diseases: List of disease indices to plot, default [112, 67, 127, 10, 17, 114]
    signatures: List of signature indices for each disease, default [5, 7, 0, 17, 19, 5]
    """
    if diseases is None:
        diseases = [112, 67, 127, 10, 17, 114]
    if signatures is None:
        signatures = [5, 7, 0, 17, 19, 5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (d, sig) in enumerate(zip(diseases, signatures)):
        ax = axes[i]
        
        # Get diagnosis times
        diagnosis_mask = model.Y[:, d, :].bool()
        diagnosed = torch.where(diagnosis_mask)[0]
        
        # Get thetas
        pi, theta, phi_prob = model.forward()
        
        # Plot distributions
        diagnosed_theta = theta[diagnosis_mask, sig].detach().numpy()
        others_theta = theta[~diagnosis_mask, sig].detach().numpy()
        
        ax.hist(diagnosed_theta, alpha=0.5, label='At diagnosis', density=True)
        ax.hist(others_theta, alpha=0.5, label='Others', density=True)
        
        ax.set_title(f'Disease {d} (sig {sig})')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.show()




def plot_roc_curve(y_true, y_pred, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=label)


def compare_with_pce(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff_fuull'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_with_pce(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff_fuull'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))

def plot_signature_temporal_patterns_assigned(
    phi_log_odds,          # Raw log-odds phi from model
    disease_names,
    disease_primary_signature, # Array where index=disease_idx, value=primary_signature_idx
    selected_signatures=None   # List of signature indices to plot
    ):
    """
    Show temporal patterns of sigmoid(phi) for diseases assigned to each signature.
    """
    # Ensure input is numpy array on CPU
    if isinstance(phi_log_odds, torch.Tensor):
        phi_log_odds = phi_log_odds.detach().cpu().numpy()
    if isinstance(disease_primary_signature, torch.Tensor):
        disease_primary_signature = disease_primary_signature.detach().cpu().numpy()

    # 1. Calculate phi_prob using sigmoid
    phi_prob = sigmoid(phi_log_odds)
    # phi_prob shape: (n_signatures, n_diseases, n_timepoints)

    # 2. Setup plotting
    n_total_signatures = phi_prob.shape[0]
    if selected_signatures is None:
        # Ensure selected_signatures are valid indices
        selected_signatures = sorted(list(set(disease_primary_signature))) # Plot signatures that have diseases assigned
        selected_signatures = [s for s in selected_signatures if 0 <= s < n_total_signatures]
    else:
        # Filter selected_signatures to be valid indices
         selected_signatures = [s for s in selected_signatures if 0 <= s < n_total_signatures]


    n_sigs_to_plot = len(selected_signatures)
    if n_sigs_to_plot == 0:
        print("No valid signatures selected or found for plotting.")
        return

    # Create subplots, ensuring axes is always 2D
    fig, axes = plt.subplots(n_sigs_to_plot, 1,
                             figsize=(15, 5 * n_sigs_to_plot),
                             squeeze=False, sharex=True) # Share x-axis

    # 3. Plotting loop
    plotted_signatures_count = 0
    for i, k in enumerate(selected_signatures):
        ax = axes[i, 0] # Access subplot correctly

        # Find diseases assigned to this signature k based on input
        assigned_disease_indices = np.where(disease_primary_signature == k)[0]

        if len(assigned_disease_indices) == 0:
            ax.text(0.5, 0.5, f'No diseases assigned to Signature {k}',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        else:
            plotted_signatures_count += 1
            # Plot temporal pattern (using phi_prob) for assigned diseases
            for disease_idx in assigned_disease_indices:
                # Check if disease_idx is valid
                if disease_idx < phi_prob.shape[1]:
                    temporal_pattern = phi_prob[k, disease_idx, :]
                    # Ensure disease_names has this index
                    disease_name = disease_names[disease_idx] if disease_idx < len(disease_names) else f"Disease_{disease_idx}"
                    ax.plot(temporal_pattern, label=disease_name)
                else:
                     print(f"Warning: Disease index {disease_idx} out of bounds for phi_prob.")

            # Add legend to this specific subplot
            if len(assigned_disease_indices) > 15: # Threshold for moving legend outside
                 ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title="Diseases")
            elif len(assigned_disease_indices) > 0:
                 ax.legend(loc='best', fontsize='small', title="Diseases")

        # Common subplot settings
        ax.set_title(f'Signature {k} - Temporal Patterns')
        ax.set_ylabel("Prob(Disease | Sig k, Time)") # Or similar interpretation
        ax.grid(True, alpha=0.3)
        if i == n_sigs_to_plot - 1: # Only add x-label to the bottom plot
             ax.set_xlabel("Time (e.g., Age)")
        else:
             ax.tick_params(labelbottom=False) # Hide x-labels for upper plots


    # Final adjustments
    if plotted_signatures_count > 0:
        # Add a main title (optional)
        # fig.suptitle("Disease-Signature Association Probabilities over Time", fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 0.9, 0.98]) # Adjust rect to make space for legends/titles
        plt.show()
        # Optional: Save the figure
        # plt.savefig(f'phi_prob_assigned_patterns.pdf', bbox_inches='tight')
    else:
        plt.close(fig)
        print("Finished plotting: No signatures had assigned diseases among the selected ones.")


import torch
import numpy as np
import pandas as pd # Ensure pandas is imported
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union # Import Union for type hint

# utils.py

import torch
import numpy as np
import pandas as pd # Ensure pandas is imported
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Union # Import Union for type hint


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec # Explicit import
from typing import List, Dict, Optional, Union

import torch
import numpy as np

def softmax_by_k(x):
    """
    Apply softmax along K dimension (dimension 1 in Python/PyTorch)
    Args:
        x: tensor of shape [N, K, T] or [K, T]
    Returns:
        softmaxed tensor of same shape
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    
    # Handle both 2D and 3D cases
    if x.dim() == 2:  # Shape [K, T]
        return torch.softmax(x, dim=0)
    elif x.dim() == 3:  # Shape [N, K, T]
        return torch.softmax(x, dim=1)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")

def calculate_pi_pred(lambda_params, phi, kappa):
    """
    Calculate pi predictions using softmax and tensor multiplication
    
    Args:
        lambda_params: tensor of shape [N, K, T]
        phi: tensor of shape [K, D, T]
        kappa: scalar value
    
    Returns:
        pi_pred: tensor of shape [N, D, T]
    """
    # Convert inputs to torch tensors if they aren't already
    if isinstance(lambda_params, np.ndarray):
        lambda_params = torch.from_numpy(lambda_params)
    if isinstance(phi, np.ndarray):
        phi = torch.from_numpy(phi)
    if isinstance(kappa, np.ndarray):
        kappa = torch.from_numpy(kappa).item()

    # Get dimensions
    N, K, T = lambda_params.shape
    _, D, _ = phi.shape

    # 1. Calculate all_thetas using softmax
    all_thetas = softmax_by_k(lambda_params)  # Shape: [N, K, T]

    phi_prob = sigmoid(phi) 

    # 2. Calculate pi_pred using einsum
    # This is equivalent to the R loop but much faster
    pi_pred = torch.einsum('nkt,kdt->ndt', all_thetas, phi_prob) * kappa

    return pi_pred


def plot_all_disease_probabilities_heatmap(
    pi_pred: Union[np.ndarray, torch.Tensor],
    psi: Union[np.ndarray, torch.Tensor],
    disease_names: List[str],
    selected_indices: List[int],
    age_offset: int = 30,
    figsize: tuple = (20, 12), # Reset default figsize
    output_path: Optional[str] = None,
    plot_title: str = "Disease Probabilities Over Time", # Base title
    cmap: str = "RdBu_r", # Default to RdBu_r as per psi example
    scale_type: str = 'log', # Default to 'log', can be 'linear'
    epsilon: float = 1e-10, # For log scale stability
    quantile_clip: Optional[float] = 0.99, # For linear scale vmax (if robust=False)
    robust_scaling: bool = True # Use robust=True in heatmap for auto-scaling
):
    """
    Creates a heatmap of disease probabilities over time, clustered by signature,
    with selected diseases indicated by pointers and listed on the right.
    Allows plotting on either log or linear probability scale.
    FIXED: Corrected annotation coordinate issues causing excessive image size.

    Args:
        pi_pred: Predicted probabilities array/tensor [N, D, T] or [D, T].
        psi: Static associations array/tensor [K, D] used for clustering.
        disease_names: List of all disease names [D].
        selected_indices: List of original indices [0 to D-1] of diseases to select.
        age_offset: Value to add to time index for Age axis.
        figsize: Figure size.
        output_path: Path to save the plot. If None, displays plot.
        plot_title: Base title for the plot (scale type will be appended).
        cmap: Colormap for probabilities. Consider 'Blues' or 'viridis' for linear scale.
        scale_type: How to scale probabilities for color ('linear' or 'log').
        epsilon: Small value for log scale stability log(p + epsilon).
        quantile_clip: Max color value quantile for linear scale if robust_scaling=False.
        robust_scaling: Whether to use seaborn's robust=True for color scale limits.
    """
    # --- Data Conversions and Checks (Identical to previous version) ---
    if isinstance(pi_pred, torch.Tensor): pi_pred_np = pi_pred.detach().cpu().numpy()
    elif isinstance(pi_pred, np.ndarray): pi_pred_np = pi_pred
    else: raise TypeError(f"pi_pred type {type(pi_pred)} not supported.")

    if isinstance(psi, torch.Tensor): psi_np = psi.detach().cpu().numpy()
    elif isinstance(psi, np.ndarray): psi_np = psi
    else: raise TypeError(f"psi type {type(psi)} not supported.")

    if not isinstance(disease_names, list):
        try: disease_names = list(disease_names)
        except TypeError: raise TypeError("disease_names must be a list or convertible to one.")

    if pi_pred_np.ndim == 3: pi_pred_avg = np.mean(pi_pred_np, axis=0)
    elif pi_pred_np.ndim == 2: pi_pred_avg = pi_pred_np
    else: raise ValueError(f"pi_pred shape {pi_pred_np.shape} not supported.")

    n_diseases, n_time_points = pi_pred_avg.shape
    if psi_np.shape[1] != n_diseases: raise ValueError("psi D dim != pi_pred D dim")
    if len(disease_names) != n_diseases: raise ValueError("disease_names length != pi_pred D dim")
    # --- End Conversions and Checks ---

    # --- Sort diseases based on psi (Identical) ---
    primary_sigs = psi_np.argmax(axis=0)
    max_psi_values = psi_np.max(axis=0)
    sorted_indices = np.lexsort((-max_psi_values, primary_sigs))
    pi_pred_avg_sorted = pi_pred_avg[sorted_indices, :]
    primary_sigs_sorted = primary_sigs[sorted_indices]
    # --- End Sorting ---

    # --- Mapping: Original index -> Sorted row index (Identical) ---
    original_to_sorted_row = {original_idx: row_idx for row_idx, original_idx in enumerate(sorted_indices)}
    selected_rows = []
    selected_names_in_order = []
    for original_idx in selected_indices:
        if original_idx in original_to_sorted_row:
            row_idx = original_to_sorted_row[original_idx]
            selected_rows.append(row_idx)
            selected_names_in_order.append(disease_names[original_idx])
        else:
            print(f"Warning: Selected index {original_idx} not found in disease list.")

    if not selected_rows:
        print("Warning: No selected diseases found to point to.")
    # --- End Mapping ---

    # --- Prepare data and labels based on scale_type (Identical) ---
    vmin, vmax = None, None
    if scale_type.lower() == 'log':
        plot_data = np.log10(pi_pred_avg_sorted + epsilon)
        cbar_label = 'log10(Average Probability)'
        final_plot_title = f"{plot_title} (Log Scale)"
        if not robust_scaling:
             vmin, vmax = np.percentile(plot_data[np.isfinite(plot_data)], [1, 99])
    elif scale_type.lower() == 'linear':
        plot_data = pi_pred_avg_sorted
        cbar_label = 'Average Probability'
        final_plot_title = f"{plot_title} (Linear Scale)"
        vmin = 0
        if not robust_scaling and quantile_clip is not None:
            vmax = np.quantile(plot_data[np.isfinite(plot_data)], quantile_clip)
    else:
        raise ValueError(f"Unknown scale_type: '{scale_type}'. Choose 'log' or 'linear'.")
    # --- End Data Prep ---

    # --- Plotting ---
    fig = plt.figure(figsize=figsize)
    # Use a wider ratio for text/colorbar side
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1.5], wspace=0.1)
    ax_heat = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')

    # Define colorbar axes BEFORE heatmap to prevent heatmap stealing space
    cbar_ax = fig.add_axes([0.72, 0.25, 0.02, 0.5]) # Fine-tune [left, bottom, width, height]

    sns.heatmap(
        plot_data,
        cmap=cmap,
        cbar_ax=cbar_ax, # Use the predefined axes for the colorbar
        cbar_kws={'label': cbar_label},
        xticklabels=5,
        yticklabels=False,
        vmin=vmin,
        vmax=vmax,
        robust=robust_scaling,
        linewidths=0.2,
        linecolor='lightgrey',
        ax=ax_heat
    )

    ax_heat.set_title(final_plot_title, pad=20, fontsize=14)
    ax_heat.set_xlabel(f"Age (t + {age_offset})", fontsize=12)
    ax_heat.set_ylabel(f"Diseases ({n_diseases}, Clustered by Signature)", fontsize=12)
    ax_heat.tick_params(axis='y', length=0)

    tick_positions = np.arange(0, n_time_points, 5) + 0.5
    tick_labels = np.arange(0, n_time_points, 5) + age_offset
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels(tick_labels, rotation=0)

    # --- Add white lines between signatures (Identical) ---
    prev_sig = -1
    line_positions = []
    for i, sig in enumerate(primary_sigs_sorted):
        if sig != prev_sig and i > 0:
            line_positions.append(i)
        prev_sig = sig
    ax_heat.hlines(line_positions, *ax_heat.get_xlim(), color='white', linewidth=1.5)
    # --- End Lines ---

    # --- Add pointers and text labels (REVISED) ---
    # Sort selected rows and names together for consistent vertical order in the label list
    sorted_selection = sorted(zip(selected_rows, selected_names_in_order))
    num_selected = len(sorted_selection)
    # Distribute text labels vertically in the text axis (coordinates 0-1)
    text_y_positions = np.linspace(0.95, 0.05, num_selected) if num_selected > 0 else []

    arrow_props = dict(
        arrowstyle="->",
        color='black',
        lw=0.8, # Thinner arrow
        connectionstyle="arc3,rad=-0.1" # Slight curve
    )

    for i, (row_idx, name) in enumerate(sorted_selection):
        text_y = text_y_positions[i]
        # Add the text label in the text axis
        ax_text.text(
            0.05, # x position (slightly indented) in the text axis (0-1 range)
            text_y, # y position (calculated above) in the text axis (0-1 range)
            name,
            transform=ax_text.transAxes, # Use axes coordinates for placement
            verticalalignment='center',
            fontsize=9
        )

        # Add annotation arrow pointing from text to heatmap row
        ax_text.annotate(
            "", # No text for the annotation itself
            xy=(n_time_points - 0.5, row_idx + 0.5), # Target point: Mid-right edge of the row in heatmap DATA coordinates
            xycoords=ax_heat.transData, # Specify that xy is in ax_heat's data coordinates
            xytext=(0.0, text_y), # Starting point: Left edge (x=0) of text label in text AXES coordinates
            textcoords=ax_text.transAxes, # Specify that xytext is in ax_text's axes coordinates
            arrowprops=arrow_props,
            annotation_clip=False # Allow arrow to be drawn outside axes if needed
        )
    # --- End Pointers (REVISED) ---

    # Adjust layout slightly if needed, but avoid aggressive tight_layout
    plt.subplots_adjust(left=0.1, right=0.7, bottom=0.1, top=0.9) # Adjust margins

    # Save or show plot
    if output_path:
        try:
            # Ensure directory exists if specified in path
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created directory: {output_dir}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Plot saved to {output_path}")
        except Exception as e: print(f"Error saving plot to {output_path}: {e}")
    else:
        plt.show()

    plt.close(fig)


def plot_disease_probabilities_heatmap(
    # Assuming pi_pred is shape (num_samples, num_diseases, num_time_points)
    pi_pred: Union[np.ndarray, torch.Tensor],
    selected_disease_indices: List[int],
    selected_disease_names: List[str], # Names corresponding to indices above
    age_offset: int = 30, # To convert time index to age
    num_time_points: Optional[int] = None, # Optional: specify if pi_pred shape is different
    figsize: tuple = (10, 8),
    cmap: str = "Blues", # Colormap similar to 'navy' gradient
    output_path: Optional[str] = None
):
    """
    Creates a heatmap of average disease probabilities over time for selected diseases.

    Args:
        pi_pred (Union[np.ndarray, torch.Tensor]): Predicted probabilities, typically
            with shape (num_samples, num_diseases, num_time_points).
        selected_disease_indices (List[int]): List of indices for the diseases to plot.
        selected_disease_names (List[str]): List of names corresponding exactly
            to the selected_disease_indices.
        age_offset (int): Value to add to the time index for the x-axis labels (Age).
            Default is 30.
        num_time_points (Optional[int]): Specify the number of time points if not
            determinable from pi_pred.shape[-1]. Default None.
        figsize (tuple): Figure size.
        cmap (str): Matplotlib colormap name for the heatmap gradient.
                    Examples: "Blues", "Purples", "viridis". Default "Blues".
        output_path (Optional[str]): Path to save the plot. If None, displays the plot.
    """
    # --- Data Preparation ---
    if isinstance(pi_pred, torch.Tensor):
        pi_pred_np = pi_pred.detach().cpu().numpy()
    elif isinstance(pi_pred, np.ndarray):
        pi_pred_np = pi_pred
    else:
         try: pi_pred_np = np.array(pi_pred)
         except Exception as e: raise TypeError(f"pi_pred must be Tensor, ndarray, or convertible. Error {e}")

    # Calculate average probability across samples (axis 0)
    # Expected shape after mean: (num_diseases, num_time_points)
    if pi_pred_np.ndim == 3:
        pi_pred_avg = np.mean(pi_pred_np, axis=0)
    elif pi_pred_np.ndim == 2:
         # Assume shape is already (num_diseases, num_time_points)
         print("Warning: pi_pred has 2 dimensions, assuming shape is (num_diseases, num_time_points).")
         pi_pred_avg = pi_pred_np
    else:
         raise ValueError(f"pi_pred has unexpected shape {pi_pred_np.shape}. Expected 3D or 2D.")

    if num_time_points is None:
        num_time_points = pi_pred_avg.shape[1]

    # Select the data for the chosen diseases
    selected_probs = pi_pred_avg[selected_disease_indices, :num_time_points]

    # Check consistency
    if len(selected_disease_names) != len(selected_disease_indices):
        raise ValueError("Mismatch between number of selected indices and selected names.")
    if selected_probs.shape[0] != len(selected_disease_indices):
         raise ValueError("Mismatch fetching probabilities for selected indices.")
    # --- End Data Preparation ---

    # --- Create DataFrame for Heatmap ---
    # Ensure correct ordering matches R code (reverse levels for y-axis)
    df_plot = pd.DataFrame(
        selected_probs,
        index=pd.Categorical(selected_disease_names, categories=selected_disease_names[::-1], ordered=True), # Reverse order for plotting
        columns=np.arange(num_time_points) + age_offset # Columns are Ages
    )
    # --- End DataFrame Creation ---

    # --- Plotting ---
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        df_plot,
        cmap=cmap,
        linewidths=0.5, # Optional: add lines between cells
        linecolor='lightgrey', # Optional: line color
        cbar_kws={'label': 'Average Probability'} # Color bar label
        # vmin=0 # Ensure scale starts at 0 (usually default for Blues etc.)
    )

    # Adjust labels and title
    plt.title("Disease Probabilities Over Time", fontsize=14, pad=15)
    plt.xlabel(f"Age (t + {age_offset})", fontsize=12)
    plt.ylabel("") # Remove default y-axis label
    plt.yticks(fontsize=9) # Adjust disease name font size
    plt.xticks(fontsize=9) # Adjust age font size

    plt.tight_layout()
    # --- End Plotting ---

    # Save or show
    if output_path:
        try:
            # Ensure directory exists if specified in path
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created directory: {output_dir}")

            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Probability heatmap saved to {output_path}")
        except Exception as e:
            print(f"Error saving probability heatmap to {output_path}: {e}")
    else:
        plt.show()

    plt.close() # Close the figure to free memory


def plot_signature_psi_heatmap(
    psi: Union[torch.Tensor, np.ndarray],
    disease_names: Union[List[str], pd.Series, pd.DataFrame],
    n_top_diseases: int = 5,
    figsize: tuple = (15, 20),
    output_path: Optional[str] = 'signature_psi_heatmap.pdf',
    plot_title: str = "Signature-Disease Specificity (psi_kd)",
    y_label_style: str = 'disease_index'  # 'disease_index' or 'signature_number'
):
    """
    Create a heatmap of psi values showing signature-disease associations.
    
    Args:
        psi: Tensor/array of shape (n_signatures, n_diseases) containing raw psi values
        disease_names: List/Series/DataFrame of disease names for legend
        n_top_diseases: Number of top diseases to show per signature in legend
        figsize: Figure size (width, height)
        output_path: Path to save the plot (PDF recommended)
        plot_title: Title for the plot
        y_label_style: Either 'disease_index' or 'signature_number' for y-axis labels
    """
    # Convert disease_names to list
    if isinstance(disease_names, pd.DataFrame):
        if disease_names.shape[1] > 1:
            print("Warning: disease_names is a DataFrame, taking first column.")
        disease_names_list = disease_names.iloc[:, 0].tolist()
    elif isinstance(disease_names, pd.Series):
        disease_names_list = disease_names.tolist()
    elif isinstance(disease_names, list):
        disease_names_list = disease_names
    else:
        try:
            disease_names_list = list(disease_names)
            print(f"Warning: Converted disease_names from {type(disease_names)} to list.")
        except TypeError:
            raise TypeError(f"disease_names must be list, Series, DataFrame, or convertible, not {type(disease_names)}")

    # Convert psi to numpy
    if isinstance(psi, torch.Tensor):
        psi_np = psi.detach().cpu().numpy()
    elif isinstance(psi, np.ndarray):
        psi_np = psi
    else:
        try:
            psi_np = np.array(psi)
            print(f"Warning: Converted psi from {type(psi)} to numpy array.")
        except Exception as e:
            raise TypeError(f"psi must be Tensor, ndarray, or convertible. Error: {e}")

    # Get dimensions
    n_signatures, n_diseases = psi_np.shape

    # Check dimensions
    if n_diseases != len(disease_names_list):
        raise ValueError(f"Dimension mismatch: psi has {n_diseases} diseases, but found {len(disease_names_list)} names.")
    if n_diseases == 0:
        print("Warning: No diseases."); return
    if n_signatures == 0:
        print("Warning: No signatures."); return

    # Sort diseases by primary signature and max psi value
    primary_sigs = psi_np.argmax(axis=0)
    max_values = psi_np.max(axis=0)
    sorted_indices = np.lexsort((-max_values, primary_sigs))

    # Create sorted data
    psi_sorted = psi_np[:, sorted_indices]
    primary_sigs_sorted = primary_sigs[sorted_indices]

    # Create figure
    plt.figure(figsize=figsize)

    # Determine y-axis labels based on style
    if y_label_style == 'disease_index':
        yticklabels = sorted_indices  # Original disease numbers
        ylabel = "Disease Index"
    else:  # 'signature_number'
        yticklabels = False
        ylabel = ""

    # Create heatmap
    ax = sns.heatmap(
        psi_sorted.T,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Log Odds Ratio (psi)'},
        xticklabels=[f"{i}" for i in range(n_signatures)],
        yticklabels=yticklabels,
        linewidths=0.5,
        linecolor='lightgrey'
    )

    # Add white lines and collect signature boundaries
    sig_boundaries = [0]  # Start with 0
    prev_sig = primary_sigs_sorted[0]
    for i, sig in enumerate(primary_sigs_sorted[1:], 1):
        if sig != prev_sig:
            plt.axhline(y=i, color='white', linewidth=2.5)
            sig_boundaries.append(i)
        prev_sig = sig
    sig_boundaries.append(n_diseases)  # Add end boundary

    if y_label_style == 'signature_number':
        # Add large signature numbers centered in each cluster
        for i in range(len(sig_boundaries)-1):
            mid_point = (sig_boundaries[i] + sig_boundaries[i+1]) / 2
            plt.text(-0.1, mid_point, str(i), 
                    fontsize=14, fontweight='bold',
                    ha='center', va='center')

    # Adjust labels and title
    plt.title(plot_title, pad=20, fontsize=14)
    plt.xlabel("Signatures", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Adjust tick labels
    if y_label_style == 'disease_index':
        plt.yticks(fontsize=6)  # Smaller font for disease indices
    plt.xticks(rotation=0, fontsize=12)  # Make x-axis numbers more prominent

    # Create legend text with top diseases per signature
    legend_entries = []
    for sig_idx in range(n_signatures):
        sig_disease_indices = np.where(primary_sigs == sig_idx)[0]
        if len(sig_disease_indices) > 0:
            # Sort by psi value
            sig_psi_values = psi_np[sig_idx, sig_disease_indices]
            top_disease_indices_local = np.argsort(sig_psi_values)[::-1]
            top_n_global_indices = sig_disease_indices[top_disease_indices_local[:n_top_diseases]]
            # Create text
            top_disease_names = [f"• {disease_names_list[d_idx]}" for d_idx in top_n_global_indices]
            legend_entries.append(f"Signature {sig_idx}:\n" + "\n".join(top_disease_names))
        else:
            legend_entries.append(f"Signature {sig_idx}:\n(No primary diseases)")

    # Add text box with top diseases
    plt.figtext(1.01, 0.95, "\n\n".join(legend_entries),
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
                fontsize=7, ha='left', va='top')

    # Adjust layout
    plt.tight_layout(rect=[0.05 if y_label_style == 'signature_number' else 0, 
                          0, 0.85, 1])

    # Save or show plot
    if output_path:
        try:
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
    else:
        plt.show()

    plt.close()



import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

# Assume sigmoid function is defined elsewhere or here
def sigmoid(x):
    """Numerically stable sigmoid function."""
    # Clip values to avoid overflow/underflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def plot_signature_temporal_patterns_assigned(
    phi_log_odds: torch.Tensor,
    disease_names: List[str],
    disease_to_signature_map: Dict[int, int],
    selected_signatures: List[int],
    num_time_points: int = 51,
    age_offset: int = 30,
    top_n_diseases: Optional[int] = None,
    plot_style: str = 'seaborn-v0_8-whitegrid',
    # --- Adjusted figsize default: Wider, slightly shorter per plot ---
    figsize: tuple = (15, 5), # Changed default: width=15, height_per_plot=5
    output_path: Optional[str] = None,
    plot_probability: bool = True, # Add flag to switch between log-odds and probability
    legend_fontsize: Optional[int] = None # Optional: Control legend font size
):
    """
    Plots temporal patterns for diseases assigned to selected signatures,
    showing either probability (sigmoid(phi)) or raw log-odds (phi).
    Stretched horizontally for better time visualization.

    Args:
        phi_log_odds (torch.Tensor): Tensor of shape (num_signatures, num_diseases, num_time_points)
                                     containing the raw phi values (log-odds scale).
        disease_names (List[str]): List of disease names corresponding to the second dimension of phi.
        disease_to_signature_map (Dict[int, int]): Dictionary mapping disease index to its primary signature index.
        selected_signatures (List[int]): List of signature indices to plot.
        num_time_points (int): Number of time points (length of the last dimension of phi). Default is 51.
        age_offset (int): Value to add to the time index to get the actual age. Default is 30.
        top_n_diseases (Optional[int]): If set, plot only the top N diseases per signature based on
                                        max probability or max absolute log-odds. Default is None (plot all).
        plot_style (str): Matplotlib/Seaborn style for the plot. Default is 'seaborn-v0_8-whitegrid'.
        figsize (tuple): Figure size (width, height_per_plot) for the plot.
                         Default is (15, 5). Total height will be height_per_plot * num_selected_signatures.
        output_path (Optional[str]): Path to save the plot image. If None, plot is displayed. Default is None.
        plot_probability (bool): If True, plot sigmoid(phi). If False, plot raw phi (log-odds). Default True.
        legend_fontsize (Optional[int]): Font size for the legend text. Default is None (matplotlib default).
    """
    plt.style.use(plot_style)

    # Ensure phi is on CPU and converted to numpy
    if isinstance(phi_log_odds, torch.Tensor):
        phi_log_odds_np = phi_log_odds.detach().cpu().numpy()
    else:
        phi_log_odds_np = np.array(phi_log_odds) # Assuming it's already numpy or list-like

    # Calculate plot values based on the flag
    if plot_probability:
        phi_plot_values = sigmoid(phi_log_odds_np)
        y_label = f"Prob(Disease | Sig k, Age)"
        plot_title_suffix = "Temporal Patterns (Probability)"
    else:
        phi_plot_values = phi_log_odds_np
        y_label = f"Phi Value (Log-Odds)"
        plot_title_suffix = "Temporal Patterns (Log-Odds)"


    num_selected_signatures = len(selected_signatures)
    if num_selected_signatures == 0:
        print("No signatures selected for plotting.")
        return

    # --- Adjust figsize calculation for aspect ratio ---
    plot_width = figsize[0]
    height_per_plot = figsize[1]
    total_height = height_per_plot * num_selected_signatures

    fig, axes = plt.subplots(num_selected_signatures, 1,
                             figsize=(plot_width, total_height),
                             sharex=True) # Share x-axis

    # Handle case where only one signature is selected (axes is not an array)
    if num_selected_signatures == 1:
        axes = [axes]

    # Generate time points representing actual ages
    time_points = np.arange(num_time_points)
    age_points = time_points + age_offset

    for i, sig_idx in enumerate(selected_signatures):
        ax = axes[i]

        # Find diseases assigned to this signature
        assigned_disease_indices = [
            d_idx for d_idx, prim_sig in disease_to_signature_map.items()
            if prim_sig == sig_idx and 0 <= d_idx < len(disease_names) # Ensure disease index is valid
        ]

        # Validate signature index access
        if sig_idx < 0 or sig_idx >= phi_plot_values.shape[0]:
             print(f"Warning: Signature index {sig_idx} is out of bounds for phi_plot_values shape {phi_plot_values.shape}. Skipping.")
             ax.set_title(f"Signature {sig_idx} - Error: Index out of bounds")
             continue

        if not assigned_disease_indices:
            ax.set_title(f"Signature {sig_idx} - {plot_title_suffix}\n(No diseases primarily assigned)")
            ax.text(0.5, 0.5, "No diseases assigned", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7) # Add grid even if empty
            continue # Skip to the next signature if no diseases are assigned

        # Filter out invalid disease indices before accessing phi_plot_values
        valid_assigned_disease_indices = [idx for idx in assigned_disease_indices if 0 <= idx < phi_plot_values.shape[1]]
        if len(valid_assigned_disease_indices) != len(assigned_disease_indices):
            print(f"Warning: Some disease indices for signature {sig_idx} were out of bounds for phi_plot_values shape {phi_plot_values.shape}.")
            assigned_disease_indices = valid_assigned_disease_indices
            if not assigned_disease_indices:
                 ax.set_title(f"Signature {sig_idx} - {plot_title_suffix}\n(No valid diseases found)")
                 ax.text(0.5, 0.5, "No valid diseases", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
                 continue


        # Select the relevant phi values for these diseases and the current signature
        # Shape: (num_assigned_diseases, num_time_points)
        signature_phi_values = phi_plot_values[sig_idx, assigned_disease_indices, :]
        assigned_disease_names = [disease_names[d_idx] for d_idx in assigned_disease_indices]


        # --- Optional: Select Top N diseases ---
        if top_n_diseases is not None and len(assigned_disease_indices) > top_n_diseases:
            if signature_phi_values.size == 0: # Check if array is empty before ranking
                 print(f"Warning: No data to rank for top N diseases in signature {sig_idx}. Plotting available ones.")
                 plot_title = f"Signature {sig_idx} - {plot_title_suffix}"
            else:
                # Calculate metric for ranking (max value across time)
                if plot_probability:
                     # For probability, rank by max probability
                    ranking_metric = np.max(signature_phi_values, axis=1)
                else:
                     # For log-odds, rank by max absolute value to capture strong positive/negative effects
                    ranking_metric = np.max(np.abs(signature_phi_values), axis=1)

                # Get indices of top N diseases based on the metric
                # Ensure we don't request more indices than available
                num_to_select = min(top_n_diseases, len(ranking_metric))
                # Argsort gives indices; select the top ones
                top_indices_local = np.argsort(ranking_metric)[-num_to_select:] # Indices within the assigned group

                # Select the data and names for the top N
                signature_phi_values = signature_phi_values[top_indices_local, :]
                assigned_disease_names = [assigned_disease_names[j] for j in top_indices_local]
                plot_title = f"Signature {sig_idx} - Top {len(top_indices_local)} Disease {plot_title_suffix}"
        else:
             plot_title = f"Signature {sig_idx} - {plot_title_suffix}"
             if top_n_diseases is not None: # Adjust title if N > available
                 plot_title = f"Signature {sig_idx} - Top {len(assigned_disease_indices)} Disease {plot_title_suffix}"


        # --- Plotting Section ---
        if signature_phi_values.size > 0: # Ensure there's data to plot
            # Create DataFrame for easier plotting with Seaborn/Pandas
            plot_data = pd.DataFrame(signature_phi_values.T, columns=assigned_disease_names)
            plot_data['Age'] = age_points # Use age_points for the x-axis

            # Plot each disease trajectory
            for disease_name in assigned_disease_names:
                ax.plot(plot_data['Age'], plot_data[disease_name], label=disease_name)

            # Place legend outside the plot
            # Use smaller fontsize if specified
            legend = ax.legend(title='Diseases', bbox_to_anchor=(1.02, 1), loc='upper left',
                               fontsize=legend_fontsize)
            if legend_fontsize:
                 plt.setp(legend.get_title(), fontsize=legend_fontsize) # Adjust title size too if needed


        ax.set_title(plot_title)
        ax.set_ylabel(y_label)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)


    # Set common x-axis label only for the bottom plot
    axes[-1].set_xlabel(f"Age (t + {age_offset})") # Simplified label

    # --- Adjust layout to reduce white space around legend ---
    # rect=[left, bottom, right, top]
    # Increase 'right' closer to 1 to give less space to legend
    plt.tight_layout(rect=[0, 0, 0.92, 1])



def plot_signature_temporal_patterns(model, disease_names, n_top=10, selected_signatures=None):
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
        ax.set_ylabel('log OR Disease | Signature k')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
   



def compare_with_pce_using_enrollment_time(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using the model's predicted trajectory
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using the full trajectory
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Calculate cumulative risk from enrollment to 10 years later
        max_t = min(enroll_time + 10, model.T - 1)
        p_not_disease = 1.0
        for t in range(enroll_time, max_t+1):
            for d_idx in ascvd_indices:
                p_not_disease *= (1 - pi_calibrated[patient_idx, d_idx, t])
        
        risk = 1 - p_not_disease
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff_fuull'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_with_pce_one_year(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 1-year predictions using single timepoint prediction
    """
    our_1yr_risks = []
    actual_1yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Apply calibration as before (optional)
    pi_calibrated = pi  # Or apply calibration if desired
    
    # Calculate 1-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 1 >= model.T:
            continue
            
        # Only use predictions at enrollment time for ASCVD indices
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk (combine across ASCVD diseases)
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        our_1yr_risks.append(yearly_risk)
        
        # Look at actual events over 1 year
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+1]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_1yr.append(actual.item())
   
    # Convert to arrays
    our_1yr_risks = np.array(our_1yr_risks)
    actual_1yr = np.array(actual_1yr)
    
    # Calculate ROC AUC
    our_auc = roc_auc_score(actual_1yr, our_1yr_risks)
    
    print(f"\nROC AUC for 1-year prediction from enrollment:")
    print(f"Our model: {our_auc:.3f}")
    
    # No PCE comparison for 1-year risk (PCE is designed for 10-year risk)
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_1yr, our_1yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 1-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return our_auc, our_1yr_risks, actual_1yr

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Optional

def plot_disease_blocks(
    mgb_checkpoint: Dict,
    aou_checkpoint: Dict,
    ukb_checkpoint: Dict,
    output_path: Optional[str] = None,
    figsize: tuple = (15, 6)
):
    """
    Creates heatmaps showing cluster correspondence between biobanks.
    
    Args:
        mgb_checkpoint: MGB model checkpoint dictionary
        aou_checkpoint: AoU model checkpoint dictionary
        ukb_checkpoint: UKB model checkpoint dictionary
        output_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
    """
    # Extract disease names and clusters from checkpoints
    mgb_diseases = mgb_checkpoint['disease_names']
    aou_diseases = aou_checkpoint['disease_names']
    ukb_diseases = ukb_checkpoint['disease_names']
    
    # Create DataFrames for each biobank
    mgb_df = pd.DataFrame({
        'Disease': mgb_diseases,
        'MGB_cluster': mgb_checkpoint['clusters']
    })
    
    aou_df = pd.DataFrame({
        'Disease': aou_diseases,
        'AoU_cluster': aou_checkpoint['clusters']
    })
    
    ukb_df = pd.DataFrame({
        'Disease': ukb_diseases,
        'UKB_cluster': ukb_checkpoint['clusters']
    })
    
    # Find common diseases
    common_diseases = list(set(mgb_df['Disease']) & 
                         set(aou_df['Disease']) & 
                         set(ukb_df['Disease']))
    
    # Create merged dataframe for common diseases
    df_common = pd.DataFrame({'Disease': common_diseases})
    df_common = (df_common
                 .merge(mgb_df, on='Disease', how='left')
                 .merge(aou_df, on='Disease', how='left')
                 .merge(ukb_df, on='Disease', how='left'))
    
    # Create cross tabs and normalize
    cross_tab_mgb = pd.crosstab(
        df_common['UKB_cluster'], 
        df_common['MGB_cluster'], 
        normalize='index'
    )
    
    cross_tab_aou = pd.crosstab(
        df_common['UKB_cluster'], 
        df_common['AoU_cluster'], 
        normalize='index'
    )
    
    # Find best matches for ordering
    best_matches_mgb = pd.DataFrame({
        'UKB': cross_tab_mgb.index,
        'MGB': cross_tab_mgb.idxmax(axis=1)
    }).sort_values('MGB')
    
    best_matches_aou = pd.DataFrame({
        'UKB': cross_tab_aou.index,
        'AOU': cross_tab_aou.idxmax(axis=1)
    }).sort_values('AOU')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot UKB vs MGB
    sns.heatmap(
        cross_tab_mgb.loc[best_matches_mgb['UKB']],
        cmap='Reds',
        vmin=0, vmax=1,
        ax=ax1,
        cbar_kws={'label': 'Proportion'}
    )
    ax1.set_title('Cluster Correspondence: UKB vs MGB\n(Common Diseases Only)')
    ax1.set_xlabel('MGB_cluster')
    ax1.set_ylabel('UKB_cluster')
    
    # Plot UKB vs AoU
    sns.heatmap(
        cross_tab_aou.loc[best_matches_aou['UKB']],
        cmap='Reds',
        vmin=0, vmax=1,
        ax=ax2,
        cbar_kws={'label': 'Proportion'}
    )
    ax2.set_title('Cluster Correspondence: UKB vs AoU\n(Common Diseases Only)')
    ax2.set_xlabel('AoU_cluster')
    ax2.set_ylabel('UKB_cluster')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
        
    plt.close()
    
    # Return the cross tabs and best matches for further analysis if needed
    return {
        'cross_tab_mgb': cross_tab_mgb,
        'cross_tab_aou': cross_tab_aou,
        'best_matches_mgb': best_matches_mgb,
        'best_matches_aou': best_matches_aou
    }

# Example usage:
# Load checkpoints
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set

# Renaming back to the original function name
def plot_signature_patterns_by_clusters(
    mgb_checkpoint: Dict,
    aou_checkpoint: Dict,
    ukb_checkpoint: Dict,
    mgb_diseases: List[str],
    aou_diseases: List[str],
    ukb_diseases: List[str],
    output_path: Optional[str] = None, # For saving PDF
    figsize: tuple = (20, 12) # Default size
):
    """
    Plot temporal patterns (lines) for diseases shared across biobanks
    in specified signature groups, using consistent colors, global averages,
    correct time slicing, and clear legends next to each plot.

    Args:
        mgb_checkpoint: MGB model checkpoint dictionary.
        aou_checkpoint: AoU model checkpoint dictionary.
        ukb_checkpoint: UKB model checkpoint dictionary.
        mgb_diseases: List of disease names for MGB.
        aou_diseases: List of disease names for AoU.
        ukb_diseases: List of disease names for UKB.
        output_path: Optional path to save the plot as PDF. If None, shows the plot.
        figsize: Figure size tuple.
    """
    # Define signature mappings
    cv_signatures = {'mgb': 5, 'aou': 16, 'ukb': 5}
    malig_signatures = {'mgb': 11, 'aou': 11, 'ukb': 6}

    # Get clusters
    mgb_clusters = mgb_checkpoint['clusters']
    aou_clusters = aou_checkpoint['clusters']
    ukb_clusters = ukb_checkpoint['clusters']

    # --- Determine Minimum Time Points ---
    def get_phi_time_dim(checkpoint: Dict) -> int:
        phi = checkpoint.get('model_state_dict', {}).get('phi', checkpoint.get('phi'))
        if phi is None or not hasattr(phi, 'shape') or len(phi.shape) != 3: return 0
        return phi.shape[2]

    time_dims = [get_phi_time_dim(cp) for cp in [mgb_checkpoint, aou_checkpoint, ukb_checkpoint]]
    valid_time_dims = [td for td in time_dims if td > 0]
    if not valid_time_dims: raise ValueError("Could not determine valid time dimension from 'phi'.")
    min_time_points = min(valid_time_dims)
    print(f"Using minimum time points across biobanks: {min_time_points}")
    # --- End Determine Minimum Time Points ---

    # --- Find Shared Diseases ---
    def get_signature_diseases(diseases: List[str], clusters: np.ndarray, sig_num: int) -> Dict[str, int]:
        if isinstance(clusters, torch.Tensor): clusters_np = clusters.cpu().numpy()
        else: clusters_np = np.array(clusters)
        if len(diseases) != len(clusters_np): raise ValueError(f"Length mismatch: {len(diseases)} vs {len(clusters_np)}.")
        indices_in_sig = np.where(clusters_np == sig_num)[0]
        return {diseases[i]: i for i in indices_in_sig}

    mgb_cv = get_signature_diseases(mgb_diseases, mgb_clusters, cv_signatures['mgb'])
    aou_cv = get_signature_diseases(aou_diseases, aou_clusters, cv_signatures['aou'])
    ukb_cv = get_signature_diseases(ukb_diseases, ukb_clusters, cv_signatures['ukb'])
    mgb_malig = get_signature_diseases(mgb_diseases, mgb_clusters, malig_signatures['mgb'])
    aou_malig = get_signature_diseases(aou_diseases, aou_clusters, malig_signatures['aou'])
    ukb_malig = get_signature_diseases(ukb_diseases, ukb_clusters, malig_signatures['ukb'])

    cv_shared: Set[str] = set(mgb_cv.keys()) & set(aou_cv.keys()) & set(ukb_cv.keys())
    malig_shared: Set[str] = set(mgb_malig.keys()) & set(aou_malig.keys()) & set(ukb_malig.keys())
    # --- End Find Shared Diseases ---

    # Create figure and axes
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharey='row') # Share Y axis within rows
    ax1_mgb, ax1_aou, ax1_ukb = axes[0]
    ax2_mgb, ax2_aou, ax2_ukb = axes[1]

    # Create consistent color mappings
    cmap_cv = plt.cm.tab10 if len(cv_shared) <= 10 else plt.cm.tab20 # Use tab10/20 for distinct colors
    cmap_malig = plt.cm.tab10 if len(malig_shared) <= 10 else plt.cm.tab20
    cv_colors = dict(zip(sorted(cv_shared), cmap_cv(np.linspace(0, 1, len(cv_shared))))) if cv_shared else {}
    malig_colors = dict(zip(sorted(malig_shared), cmap_malig(np.linspace(0, 1, len(malig_shared))))) if malig_shared else {}

    # Helper function to plot patterns for one biobank (back to line plots)
    def plot_biobank_patterns(
        biobank_all_diseases: List[str], # Full list needed for index lookup
        biobank_diseases_dict: Dict[str, int], # {name: index} for diseases in target sig
        shared_diseases_set: Set[str],
        target_signature_num: int,
        target_checkpoint: Dict,
        ax: plt.Axes,
        title: str,
        colors: Dict[str, tuple],
        num_time_points_to_plot: int, # Use consistent time points
        all_checkpoints: Optional[List[Dict]] = None,
        all_corresponding_sigs: Optional[List[int]] = None
    ):
        """Plots line patterns using a consistent number of time points."""
        phi_key = 'phi'
        phi = target_checkpoint.get('model_state_dict', {}).get(phi_key, target_checkpoint.get(phi_key))
        if phi is None:
            print(f"Warning: Cannot find '{phi_key}' for {title}. Skipping.")
            ax.set_title(f"{title}\n(Error: Phi not found)")
            ax.text(0.5, 0.5, "Phi not found", ha='center', va='center')
            return

        if isinstance(phi, torch.Tensor): phi_np = phi.detach().cpu().numpy()
        else: phi_np = np.array(phi)

        if len(phi_np.shape) != 3 or phi_np.shape[2] < num_time_points_to_plot:
             print(f"Warning: Phi for {title} has invalid shape/time. Shape: {phi_np.shape}, Required time: {num_time_points_to_plot}. Skipping.")
             ax.set_title(f"{title}\n(Error: Invalid Phi Shape/Time)")
             ax.text(0.5, 0.5, f"Invalid Phi Shape/Time\nShape: {phi_np.shape}", ha='center', va='center')
             return

        x_axis = np.arange(num_time_points_to_plot)

        # Plot average patterns using num_time_points_to_plot
        if all_checkpoints and all_corresponding_sigs and len(all_checkpoints) == len(all_corresponding_sigs):
            # avg_patterns_dict = {} # Not strictly needed if just plotting
            for disease in shared_diseases_set:
                patterns = []
                for i, cp in enumerate(all_checkpoints):
                    sig = all_corresponding_sigs[i]
                    current_diseases = cp.get('disease_names', [])
                    if not isinstance(current_diseases, list): current_diseases = list(current_diseases)
                    try:
                        original_idx = current_diseases.index(disease)
                        # No need to check cluster here, assuming shared_diseases_set is correct
                        current_phi = cp.get('model_state_dict', {}).get('phi', cp.get('phi'))
                        if current_phi is not None:
                            current_phi_np = current_phi.detach().cpu().numpy() if isinstance(current_phi, torch.Tensor) else np.array(current_phi)
                            if (len(current_phi_np.shape) == 3 and
                                sig < current_phi_np.shape[0] and
                                original_idx < current_phi_np.shape[1] and
                                current_phi_np.shape[2] >= num_time_points_to_plot):
                                 patterns.append(current_phi_np[sig, original_idx, :num_time_points_to_plot])
                            # else: print(f"Warning: Skipping pattern for avg calc due to shape/index issue: {disease}...") # Optionally add verbose warnings
                    except (ValueError, IndexError): pass # Ignore if disease/index not found in a specific checkpoint

                if patterns:
                    try:
                        avg_pattern = np.mean(np.stack(patterns), axis=0)
                        # avg_patterns_dict[disease] = avg_pattern
                        ax.plot(x_axis, avg_pattern, color='gray', alpha=0.3, linestyle='--') # Made average lines fainter
                    except ValueError as e:
                         print(f"Error stacking patterns for avg {disease}: {e}")

        # Plot this specific biobank's patterns using num_time_points_to_plot
        # Keep track of handles/labels for the legend later
        handles, labels = [], []
        for disease in shared_diseases_set:
            if disease in biobank_diseases_dict: # Check if disease actually belongs to this sig in this biobank
                 idx = biobank_diseases_dict[disease]
                 if target_signature_num < phi_np.shape[0] and idx < phi_np.shape[1]:
                    pattern = phi_np[target_signature_num, idx, :num_time_points_to_plot]
                    line, = ax.plot(x_axis, pattern, color=colors.get(disease, 'black'), alpha=0.8, label=disease) # Use disease name as label
                    # Store handle and label IF it's not already stored (relevant if multiple lines had same label somehow)
                    if disease not in labels:
                        handles.append(line)
                        labels.append(disease)
                 # else: print(f"Warning: Index OOB for {disease} in {title}.") # Optionally add warning

        # Customize plot
        ax.set_xlabel(f'Time Steps (0-{num_time_points_to_plot-1})')
        ax.set_ylabel('Phi Value (Log Odds Ratio)')
        ax.set_title(f"{title}\n(n={len(shared_diseases_set)} shared diseases)")
        ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey', alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

        # Return handles and labels for external legend
        return handles, labels

    # --- Plotting Calls ---
    all_checkpoints_list = [mgb_checkpoint, aou_checkpoint, ukb_checkpoint]
    cv_handles_labels = {} # Store handles/labels per axis {axis_index: (handles, labels)}
    malig_handles_labels = {}

    # Cardiovascular
    other_cv_sigs = [cv_signatures['mgb'], cv_signatures['aou'], cv_signatures['ukb']]
    cv_handles_labels[0] = plot_biobank_patterns(mgb_diseases, mgb_cv, cv_shared, cv_signatures['mgb'], mgb_checkpoint, ax1_mgb, 'MGB Cardiovascular (Sig 5)', cv_colors, min_time_points, all_checkpoints_list, other_cv_sigs)
    cv_handles_labels[1] = plot_biobank_patterns(aou_diseases, aou_cv, cv_shared, cv_signatures['aou'], aou_checkpoint, ax1_aou, 'AoU Cardiovascular (Sig 16)', cv_colors, min_time_points, all_checkpoints_list, other_cv_sigs)
    cv_handles_labels[2] = plot_biobank_patterns(ukb_diseases, ukb_cv, cv_shared, cv_signatures['ukb'], ukb_checkpoint, ax1_ukb, 'UKB Cardiovascular (Sig 5)', cv_colors, min_time_points, all_checkpoints_list, other_cv_sigs)

    # Malignancy
    other_malig_sigs = [malig_signatures['mgb'], malig_signatures['aou'], malig_signatures['ukb']]
    malig_handles_labels[0] = plot_biobank_patterns(mgb_diseases, mgb_malig, malig_shared, malig_signatures['mgb'], mgb_checkpoint, ax2_mgb, 'MGB Malignancy (Sig 11)', malig_colors, min_time_points, all_checkpoints_list, other_malig_sigs)
    malig_handles_labels[1] = plot_biobank_patterns(aou_diseases, aou_malig, malig_shared, malig_signatures['aou'], aou_checkpoint, ax2_aou, 'AoU Malignancy (Sig 11)', malig_colors, min_time_points, all_checkpoints_list, other_malig_sigs)
    malig_handles_labels[2] = plot_biobank_patterns(ukb_diseases, ukb_malig, malig_shared, malig_signatures['ukb'], ukb_checkpoint, ax2_ukb, 'UKB Malignancy (Sig 6)', malig_colors, min_time_points, all_checkpoints_list, other_malig_sigs)
    # --- End Plotting Calls ---

    # --- Add Legends Externally ---
    for i, ax in enumerate(axes[0]): # CV row
        handles, labels = cv_handles_labels.get(i, ([], []))
        if handles: # Only add legend if there are items
            ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, borderaxespad=0.)

    for i, ax in enumerate(axes[1]): # Malignancy row
        handles, labels = malig_handles_labels.get(i, ([], []))
        if handles:
            ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, borderaxespad=0.)
    # --- End Legends ---

    # Adjust layout carefully to prevent legend overlap
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Leave space on right: rect=[left, bottom, right, top]

    # --- Save or Show ---
    if output_path:
        try:
            import os
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created directory: {output_dir}")
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            print(f"Plot saved to {output_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Error saving plot to {output_path}: {e}")
            plt.close(fig)
    else:
        plt.show()
    # --- End Save or Show ---

    # --- Print Summary ---
    # (Summary printing code remains the same)
    print("\nSummary of shared diseases across biobanks:")
    print(f"\nCardiovascular signature (shared across MGB:{cv_signatures['mgb']}, AoU:{cv_signatures['aou']}, UKB:{cv_signatures['ukb']}):")
    print(f"Number of shared diseases: {len(cv_shared)}")
    if cv_shared: print("Diseases:"); [print(f"- {d}") for d in sorted(cv_shared)]
    else: print("No shared CV diseases found.")

    print(f"\nMalignancy signature (shared across MGB:{malig_signatures['mgb']}, AoU:{malig_signatures['aou']}, UKB:{malig_signatures['ukb']}):")
    print(f"Number of shared diseases: {len(malig_shared)}")
    if malig_shared: print("Diseases:"); [print(f"- {d}") for d in sorted(malig_shared)]
    else: print("No shared Malignancy diseases found.")
    # --- End Summary ---

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

import torch
from sklearn.metrics import roc_curve, auc

def evaluate_major_diseases(model, Y_100k, E_100k, disease_names, pce_df, Y_full,enrollment_df):  # Add Y_full parameter
    """
    Evaluate model performance on major diseases, using full Y for event rates
    """
    """
    Evaluate model performance on major diseases
    
    Parameters:
    - model: trained model
    - Y_100k: disease status matrix (PyTorch tensor)
    - E_100k: event times matrix (PyTorch tensor)
    - disease_names: list of disease names
    - pce_df: DataFrame with patient characteristics
    """
    # Define major diseases to evaluate
    major_diseases = {
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
              'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
    'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'],
    'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
    # Add to major_diseases dictionary:
    'All_Cancers': ['Colon cancer', 
                'Malignant neoplasm of rectum, rectosigmoid junction, and anus',
                'Cancer of bronchus; lung',
                'Breast cancer [female]',
                'Malignant neoplasm of female breast',
                'Cancer of prostate',
                'Malignant neoplasm of bladder',
                'Secondary malignant neoplasm',
                'Secondary malignancy of lymph nodes',
                'Secondary malignancy of respiratory organs',
                'Secondary malignant neoplasm of digestive systems',
                'Secondary malignant neoplasm of liver',
                'Secondary malignancy of bone'],
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
    'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
    'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
    'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
    'Hip_Fracture': ['Hip fracture'],  # if this exact term exists in disease_names
    'Osteoporosis': ['Osteoporosis NOS'],
    'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
    'Alzheimer': ['Alzheimer disease and other dementias'],
    'Esophageal_Cancer': ['Cancer of esophagus'],  # adjust if different in disease_names
    'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
    'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
    'Prostate_Cancer': ['Cancer of prostate'],
    'Lung_Cancer': ['Cancer of bronchus; lung'],
    'Bladder_Cancer': ['Malignant neoplasm of bladder'],
    'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 
                        'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
    'Depression': ['Major depressive disorder'],
    'Anxiety': ['Anxiety disorder'],
    'Bipolar_Disorder': ['Bipolar'],
    'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
    'Psoriasis': ['Psoriasis vulgaris'],
    'Ulcerative_Colitis': ['Ulcerative colitis'],
    'Crohns_Disease': ['Regional enteritis'],
    'Asthma': ['Asthma'],
    #'Allergic_Rhinitis': ['Allergic rhinitis'],
    # Additional common conditions
    'Parkinsons': ["Parkinson's disease"],
    'Multiple_Sclerosis': ['Multiple sclerosis'],
    #'Sleep_Apnea': ['Sleep apnea'],
    #'Glaucoma': ['Glaucoma', 'Primary open angle glaucoma'],
    #'Cataract': ['Cataract', 'Senile cataract'],
    'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
}


    
    # Get model predictions
    with torch.no_grad():
        pi, _, _ = model.forward()
    
    results = {}
    
    # For each disease group
    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group}...")
        
        # Find disease indices
        disease_indices = []
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            disease_indices.extend(indices)
        
        if not disease_indices:
            print(f"No matching diseases found for {disease_group}")
            continue
            
        # Get predictions at enrollment time
        N = len(pce_df)
        risks = torch.zeros(N, device=pi.device)
        outcomes = torch.zeros(N, device=pi.device)
        
        # Calculate risks and outcomes
        for i in range(N):
            age = pce_df.iloc[i]['age']
            t_enroll = int(age - 30)  # Convert age to time index
            
            if t_enroll >= pi.shape[2]:
                continue
                
            # Get prediction at enrollment
            pi_diseases = pi[i, disease_indices, t_enroll]
            yearly_risk = 1 - torch.prod(1 - pi_diseases)
            risks[i] = 1 - (1 - yearly_risk)**10  # 10-year risk
            
            # Check for actual events in next 10 years
            end_time = min(t_enroll + 10, Y_100k.shape[2])
            for d_idx in disease_indices:
                if torch.any(Y_100k[i, d_idx, t_enroll:end_time] > 0):
                    outcomes[i] = 1
                    break
        
        # Convert to numpy for sklearn metrics
        risks_np = risks.cpu().numpy()
        outcomes_np = outcomes.cpu().numpy()
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
        auc_score = auc(fpr, tpr)
        
        # NEW: Calculate event rate using full Y tensor
        full_outcomes = torch.zeros(Y_full.shape[0], device=Y_full.device)
        for i in range(Y_full.shape[0]):
            age_at_enrollment = enrollment_df.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)  # Convert age to time index
            end_time = min(t_enroll + 10, Y_full.shape[2])
            for d_idx in disease_indices:
                if torch.any(Y_full[i, d_idx, t_enroll:end_time] > 0):
                    full_outcomes[i] = 1
                    break
        
        full_event_rate = (full_outcomes.mean() * 100).item()
        full_event_count = int(full_outcomes.sum().item())
        
        results[disease_group] = {
            'auc': auc_score,
            'n_events': full_event_count,  # Use full data count
            'event_rate': full_event_rate  # Use full data rate
        }
        

        
        print(f"AUC: {auc_score:.3f}")
        print(f"Events: {int(outcomes.sum().item())} ({outcomes.mean()*100:.1f}%)")
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 50)
    print(f"{'Disease Group':<15} {'AUC':<8} {'Events':<8} {'Rate':<8}")
    print("-" * 50)
    for group, res in results.items():
        print(f"{group:<15} {res['auc']:.3f}   {res['n_events']:<8d} {res['event_rate']:.1f}%")
    
    return results



def evaluate_major_diseases_wsex(model, Y_100k, E_100k, disease_names, pce_df, Y_full, enrollment_df):
    """
    Evaluate model performance on major diseases, using full Y for event rates 
    and handling sex-specific diseases correctly. 
    FIX 2: Uses integer positional indices consistently after filtering.
    
    Parameters are the same as before. Assumes alignment between pce_df rows (0..N-1) and Y_100k/pi rows (0..N-1),
    and alignment between enrollment_df rows (0..M-1) and Y_full rows (0..M-1).
    """
    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus', 'Cancer of bronchus; lung', 'Breast cancer [female]', 'Malignant neoplasm of female breast', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver', 'Secondary malignancy of bone'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'], # Sex-specific
        'Prostate_Cancer': ['Cancer of prostate'], # Sex-specific
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    # --- Input Validation ---
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'sex' not in enrollment_df.columns: raise ValueError("'Sex' column not found in enrollment_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")
    if 'age' not in enrollment_df.columns: raise ValueError("'age' column not found in enrollment_df")

    with torch.no_grad():
        pi, _, _ = model.forward()
        
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    N_yfull = Y_full.shape[0]
    N_enroll = len(enrollment_df)

    # Ensure alignment for AUC calculation cohort
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for AUC cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N_auc = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N_auc]
        pce_df = pce_df.iloc[:min_N_auc]
        Y_100k = Y_100k[:min_N_auc]
        N_auc_cohort = min_N_auc
    else:
        N_auc_cohort = N_pce

    # Ensure alignment for Rate calculation cohort
    if not (N_yfull == N_enroll):
        print(f"Warning: Size mismatch for Rate cohort. Y_full: {N_yfull}, enrollment_df: {N_enroll}. Using minimum size.")
        min_N_rate = min(N_yfull, N_enroll)
        Y_full = Y_full[:min_N_rate]
        enrollment_df = enrollment_df.iloc[:min_N_rate]
        N_rate_cohort = min_N_rate
    else:
        N_rate_cohort = N_enroll
        
    # Reset index after potential slicing to ensure 0-based sequential index for iloc
    pce_df = pce_df.reset_index(drop=True)
    enrollment_df = enrollment_df.reset_index(drop=True)

    results = {}
    
    # --- Main Loop ---
    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group}...")
        
        # --- Get Disease Indices ---
        disease_indices = []
        # ... (same logic as before to find indices, check bounds against pi.shape[1]) ...
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                 if idx not in unique_indices:
                      disease_indices.append(idx)
                      unique_indices.add(idx)
        
        max_model_disease_idx = pi.shape[1] - 1
        original_indices_count = len(disease_indices)
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
             print(f"No valid matching disease indices found for {disease_group} within model output bounds.")
             results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0}
             continue

        # --- Sex Filtering ---
        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'

        # Get boolean masks based on the potentially trimmed and re-indexed DataFrames
        mask_pce = pd.Series(True, index=pce_df.index)
        mask_enroll = pd.Series(True, index=enrollment_df.index)
        
        if target_sex:
            mask_pce = (pce_df['Sex'] == target_sex)
            mask_enroll = (enrollment_df['sex'] == target_sex)
            # Find integer positions (iloc indices) where mask is True
            int_indices_pce = np.where(mask_pce)[0]
            int_indices_enroll = np.where(mask_enroll)[0]
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} in AUC cohort, {len(int_indices_enroll)} in Rate cohort")
            if len(int_indices_pce) == 0 or len(int_indices_enroll) == 0:
                 print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                 results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0}
                 continue
        else:
            # Use all integer indices if not sex-specific
            int_indices_pce = np.arange(N_auc_cohort)
            int_indices_enroll = np.arange(N_rate_cohort)

        # --- Calculate AUC (using integer positions) ---
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events_auc = 0; n_processed_auc = 0; outcomes_np = np.array([]) # Handle empty case
        else:
            # Slice tensors and DataFrame using the integer positions
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce] # Use iloc with integer positions
            current_N_auc = len(int_indices_pce)

            risks_auc = torch.zeros(current_N_auc, device=pi.device)
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device)
            processed_count_auc = 0

            # Iterate based on the length of the filtered integer indices
            for i in range(current_N_auc):
                # Access DataFrame row using iloc with relative index i
                age = current_pce_df_auc.iloc[i]['age'] 
                t_enroll = int(age - 30)

                if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue

                # Access tensors using relative index i
                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                # --- MODIFICATION: Evaluate 1-year risk ---
                risks_auc[i] = yearly_risk # Use 1-year risk directly

                # --- MODIFICATION: Check event in next 1 year ---
                end_time = min(t_enroll + 10, current_Y_100k_auc.shape[2]) # Look only 1 year ahead
                if end_time <= t_enroll: continue
                
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): # Check t_enroll to end_time
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                processed_count_auc += 1 # Increment count if this iteration was valid

            # Calculate AUC based on processed data
            if processed_count_auc == 0:
                 auc_score = np.nan; outcomes_np = np.array([])
            else:
                 # Only use results from processed indices - NOTE: this slicing might be tricky if indices are sparse
                 # It's simpler to create new lists and convert at the end
                 valid_risks_list = []
                 valid_outcomes_list = []
                 temp_risks_cpu = risks_auc.cpu().numpy()
                 temp_outcomes_cpu = outcomes_auc.cpu().numpy()

                 # Re-iterate to gather valid pairs (safer than complex slicing)
                 processed_indices_auc_final = [] # Store indices relative to the loop (0 to current_N_auc-1)
                 for i in range(current_N_auc):
                     age = current_pce_df_auc.iloc[i]['age'] 
                     t_enroll = int(age - 30)
                     if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue
                     end_time = min(t_enroll + 1, current_Y_100k_auc.shape[2])
                     if end_time <= t_enroll: continue
                     processed_indices_auc_final.append(i)

                 if not processed_indices_auc_final:
                      auc_score = np.nan; outcomes_np = np.array([])
                 else:
                      risks_np = temp_risks_cpu[processed_indices_auc_final]
                      outcomes_np = temp_outcomes_cpu[processed_indices_auc_final]

                      if len(np.unique(outcomes_np)) > 1:
                           fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                           auc_score = auc(fpr, tpr)
                      else:
                           auc_score = np.nan
                           print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
            n_processed_auc = len(outcomes_np) # Number used for final AUC calc

        # --- Calculate Event Rate/Count (using integer positions) ---
        if len(int_indices_enroll) == 0:
            full_event_rate = 0.0; full_event_count = 0; num_processed_for_rate = 0 # Handle empty case
        else:
            # Slice tensors and DataFrame using the integer positions
            current_Y_full_rate = Y_full[int_indices_enroll]
            current_enrollment_df_rate = enrollment_df.iloc[int_indices_enroll] # Use iloc
            current_N_rate = len(int_indices_enroll)

            full_outcomes_rate = torch.zeros(current_N_rate, device=Y_full.device)
            processed_count_rate = 0

            # Iterate based on the length of the filtered integer indices
            for i in range(current_N_rate):
                # Access DataFrame row using iloc with relative index i
                age_at_enrollment = current_enrollment_df_rate.iloc[i]['age'] 
                t_enroll = int(age_at_enrollment - 30)

                if t_enroll < 0 or t_enroll >= current_Y_full_rate.shape[2]: continue
                
                # --- MODIFICATION: Check event in next 1 year ---
                end_time = min(t_enroll + 10, current_Y_full_rate.shape[2]) # Look only 1 year ahead
                if end_time <= t_enroll: continue

                event_found_rate = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_full_rate.shape[1]: continue
                    if torch.any(current_Y_full_rate[i, d_idx, t_enroll:end_time] > 0): # Check t_enroll to end_time
                        full_outcomes_rate[i] = 1
                        event_found_rate = True
                        break
                processed_count_rate += 1 # Increment count if this iteration was valid
            
            # Calculate rate/count based on processed data
            if processed_count_rate == 0:
                 full_event_rate = 0.0; full_event_count = 0
            else:
                 # Similar to AUC, safer to collect valid outcomes
                 valid_outcomes_rate_list = []
                 temp_outcomes_rate_cpu = full_outcomes_rate.cpu().numpy()
                 processed_indices_rate_final = []
                 for i in range(current_N_rate):
                     age_at_enrollment = current_enrollment_df_rate.iloc[i]['age'] 
                     t_enroll = int(age_at_enrollment - 30)
                     if t_enroll < 0 or t_enroll >= current_Y_full_rate.shape[2]: continue
                     end_time = min(t_enroll + 1, current_Y_full_rate.shape[2])
                     if end_time <= t_enroll: continue
                     processed_indices_rate_final.append(i)
                 
                 if not processed_indices_rate_final:
                      full_event_rate = 0.0; full_event_count = 0
                 else:
                      full_outcomes_valid = temp_outcomes_rate_cpu[processed_indices_rate_final]
                      full_event_count = int(np.sum(full_outcomes_valid))
                      # Rate is based on the number actually processed
                      full_event_rate = (full_event_count / processed_count_rate * 100) if processed_count_rate > 0 else 0.0 
            num_processed_for_rate = processed_count_rate

        # Store results
        results[disease_group] = {
            'auc': auc_score,
            'n_events': full_event_count,
            'event_rate': full_event_rate
        }
        
        print(f"AUC (1-Year): {auc_score if not np.isnan(auc_score) else 'N/A'} (calculated on {n_processed_auc} individuals)")
        print(f"Events (1-Year, Full Cohort, Filtered): {full_event_count} ({full_event_rate:.1f}%) (calculated on {num_processed_for_rate} individuals)")

    # Print summary table
    print("\nSummary of Results (Prospective 1-Year, Sex-Adjusted):")
    # ... (rest of printing code is the same) ...
    print("-" * 60)
    print(f"{'Disease Group':<20} {'AUC':<8} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 60)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f}" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<8} {res['n_events']:<10d} {rate_str}")
    print("-" * 60)

    return results
# Usage:
#results = evaluate_major_diseases(model, Y_100k, E_100k, disease_names, pce_df)

def compare_with_pce_filtered(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction, handling missing PCE values
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff'].values[:len(our_10yr_risks)]

    # Get indices of non-missing PCE values
    non_missing_idx = ~np.isnan(pce_risks)

    # Filter all arrays to only include non-missing cases
    our_10yr_risks_filtered = our_10yr_risks[non_missing_idx]
    actual_10yr_filtered = actual_10yr[non_missing_idx]
    pce_risks_filtered = pce_risks[non_missing_idx]

    # Calculate ROC AUCs on filtered data
    our_auc = roc_auc_score(actual_10yr_filtered, our_10yr_risks_filtered)
    pce_auc = roc_auc_score(actual_10yr_filtered, pce_risks_filtered)

    # Print results with sample size info
    n_total = len(our_10yr_risks)
    n_complete = len(our_10yr_risks_filtered)
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Sample size: {n_complete}/{n_total} ({n_complete/n_total*100:.1f}% complete cases)")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr_filtered, our_10yr_risks_filtered, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr_filtered, pce_risks_filtered, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

    return our_10yr_risks_filtered, actual_10yr_filtered, pce_risks_filtered 



def compare_clusters_across_biobanks(mgb_checkpoint, aou_checkpoint, ukb_checkpoint, disease_names_all):
    """
    Compare cluster assignments across biobanks, handling different disease sets
    """
    import pandas as pd
    import numpy as np
    
    # Create initial dataframes for each biobank with their diseases
    mgb_df = pd.DataFrame({
        'Disease': disease_names_all[:len(mgb_checkpoint['clusters'])],
        'MGB_cluster': mgb_checkpoint['clusters']
    })
    
    aou_df = pd.DataFrame({
        'Disease': disease_names_all[:len(aou_checkpoint['clusters'])],
        'AoU_cluster': aou_checkpoint['clusters']
    })
    
    ukb_df = pd.DataFrame({
        'Disease': disease_names_all[:len(ukb_checkpoint['clusters'])],
        'UKB_cluster': ukb_checkpoint['clusters']
    })
    
    # Merge dataframes on Disease column
    df = mgb_df.merge(aou_df, on='Disease', how='outer')\
               .merge(ukb_df, on='Disease', how='outer')
    
    print("Number of diseases in each biobank:")
    print(f"MGB: {len(mgb_df)}")
    print(f"AoU: {len(aou_df)}")
    print(f"UKB: {len(ukb_df)}")
    print(f"Total unique diseases: {len(df)}")
    
    # Print cluster sizes for each biobank
    print("\nCluster sizes in each biobank:")
    for col in ['MGB_cluster', 'AoU_cluster', 'UKB_cluster']:
        if col in df.columns:
            print(f"\n{col.split('_')[0]}:")
            print(df[col].value_counts().sort_index())
    
    # Find common diseases across all biobanks
    common_diseases = df.dropna(subset=['MGB_cluster', 'AoU_cluster', 'UKB_cluster'])
    print(f"\nNumber of diseases common to all biobanks: {len(common_diseases)}")
    
    # Create heatmap for common diseases
    if len(common_diseases) > 0:
        plt.figure(figsize=(15, 10))
        
        n_diseases = len(common_diseases)
        disease_list = common_diseases['Disease'].tolist()
        
        # Create binary matrices for co-clustering
        mgb_cocluster = np.zeros((n_diseases, n_diseases))
        aou_cocluster = np.zeros((n_diseases, n_diseases))
        ukb_cocluster = np.zeros((n_diseases, n_diseases))
        
        for i in range(n_diseases):
            for j in range(n_diseases):
                mgb_cocluster[i,j] = common_diseases['MGB_cluster'].iloc[i] == common_diseases['MGB_cluster'].iloc[j]
                aou_cocluster[i,j] = common_diseases['AoU_cluster'].iloc[i] == common_diseases['AoU_cluster'].iloc[j]
                ukb_cocluster[i,j] = common_diseases['UKB_cluster'].iloc[i] == common_diseases['UKB_cluster'].iloc[j]
        
        # Average co-clustering across biobanks
        avg_cocluster = (mgb_cocluster + aou_cocluster + ukb_cocluster) / 3
        
        # Plot heatmap
        sns.heatmap(avg_cocluster, 
                    xticklabels=disease_list,
                    yticklabels=disease_list,
                    cmap='YlOrRd')
        plt.title('Disease Co-clustering Consistency Across Biobanks\n(Common Diseases Only)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Find most consistent disease pairs
        print("\nMost consistently co-clustered diseases (among common diseases):")
        consistent_pairs = []
        for i in range(n_diseases):
            for j in range(i+1, n_diseases):
                consistency = avg_cocluster[i,j]
                if consistency > 0.66:  # Co-clustered in at least 2 biobanks
                    consistent_pairs.append((disease_list[i], disease_list[j], consistency))
        
        consistent_pairs.sort(key=lambda x: x[2], reverse=True)
        for d1, d2, score in consistent_pairs[:10]:
            print(f"{d1} - {d2}: {score:.2f}")
    
    return df

# Function to look at specific disease clusters
def examine_disease_clusters(df, disease_of_interest):
    """
    Examine clusters containing a specific disease across biobanks
    """
    print(f"\nClusters containing {disease_of_interest}:")
    
    for biobank in ['MGB', 'AoU', 'UKB']:
        col = f'{biobank}_cluster'
        if col in df.columns:
            # Get the cluster number for the disease of interest
            disease_cluster = df[df['Disease'] == disease_of_interest][col].iloc[0]
            if pd.notna(disease_cluster):  # Check if disease exists in this biobank
                cohort_diseases = df[df[col] == disease_cluster]['Disease'].tolist()
                print(f"\n{biobank} cluster {disease_cluster}:")
                print(cohort_diseases)
            else:
                print(f"\n{biobank}: Disease not present")

# Use the functions:
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_psi_heatmap_with_selective_labels(psi, disease_names, threshold=0.9):
    """
    Plot heatmap with selective labeling based on membership strength
    """
    # Get normalized values
    if psi.requires_grad:
        psi = psi.detach()
    norm_psi = torch.nn.functional.softmax(psi, dim=0).numpy().T
    
    # Sort by max cluster and value
    max_values = norm_psi.max(axis=1)
    max_clusters = norm_psi.argmax(axis=1)
    sorted_indices = np.lexsort((max_values, max_clusters))
    
    # Determine which labels to show
    show_labels = []
    for idx in sorted_indices:
        # Show label if:
        # 1. Has strong membership in any cluster (> threshold)
        # 2. Is first or last in its cluster group
        cluster = max_clusters[idx]
        cluster_group = np.where(max_clusters[sorted_indices] == cluster)[0]
        is_boundary = idx == sorted_indices[cluster_group[0]] or idx == sorted_indices[cluster_group[-1]]
        has_strong_membership = max_values[idx] > threshold
        
        show_labels.append(disease_names[idx] if (is_boundary or has_strong_membership) else '')
    
    # Create figure
    plt.figure(figsize=(20, 30))
    
    # Plot heatmap
    sns.heatmap(norm_psi[sorted_indices], 
                xticklabels=[f"Cluster {i}" for i in range(norm_psi.shape[1])],
                yticklabels=show_labels,
                cmap='YlOrRd',
                vmin=0, vmax=1,
                cbar_kws={'label': 'Normalized Membership Strength'})
    
    # Add white lines between clusters
    current_cluster = max_clusters[sorted_indices[0]]
    for i, idx in enumerate(sorted_indices[1:], 1):
        if max_clusters[idx] != current_cluster:
            plt.axhline(y=i, color='white', linewidth=2)
            current_cluster = max_clusters[idx]
    
    plt.title("Disease-Cluster Mixed Membership", fontsize=20, pad=20)
    plt.xlabel("Clusters", fontsize=16)
    plt.ylabel("Diseases", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

def plot_psi_heatmap_with_zoom(psi, disease_names, n_clusters_to_show=5):
    """
    Create multiple plots focusing on different clusters
    """
    if psi.requires_grad:
        psi = psi.detach()
    norm_psi = torch.nn.functional.softmax(psi, dim=0).numpy().T
    
    # Sort by max cluster and value
    max_values = norm_psi.max(axis=1)
    max_clusters = norm_psi.argmax(axis=1)
    
    # Create subplots for each cluster group
    n_groups = (norm_psi.shape[1] + n_clusters_to_show - 1) // n_clusters_to_show
    fig, axes = plt.subplots(n_groups, 1, figsize=(20, 15*n_groups))
    if n_groups == 1:
        axes = [axes]
    
    for group in range(n_groups):
        start_cluster = group * n_clusters_to_show
        end_cluster = min(start_cluster + n_clusters_to_show, norm_psi.shape[1])
        
        # Get diseases primarily belonging to these clusters
        cluster_mask = (max_clusters >= start_cluster) & (max_clusters < end_cluster)
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # Sort within this group
            sorted_indices = cluster_indices[np.lexsort((max_values[cluster_indices], 
                                                       max_clusters[cluster_indices]))]
            
            # Plot heatmap for this group
            sns.heatmap(norm_psi[sorted_indices, start_cluster:end_cluster],
                       xticklabels=[f"Cluster {i}" for i in range(start_cluster, end_cluster)],
                       yticklabels=[disease_names[i] for i in sorted_indices],
                       cmap='YlOrRd',
                       vmin=0, vmax=1,
                       ax=axes[group],
                       cbar_kws={'label': 'Membership Strength'})
            
            axes[group].set_title(f"Clusters {start_cluster}-{end_cluster-1}", 
                                fontsize=16, pad=20)
    
    plt.tight_layout()
    return fig

# Usage:
# fig1 = plot_psi_heatmap_with_selective_labels(psi, disease_names)
# fig2 = plot_psi_heatmap_with_zoom(psi, disease_names)
# 
# fig1.savefig('selective_labels.pdf', bbox_inches='tight', dpi=300)
# fig2.savefig('cluster_groups.pdf', bbox_inches='tight', dpi=300)