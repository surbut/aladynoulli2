"""
Simple comparison script - update paths as needed
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def compare_lambda_trajectories():
    """
    Compare lambda trajectories for sample individuals between two models
    """
    
    print("COMPARING LAMBDA TRAJECTORIES: WITH PCs vs WITHOUT PCs")
    print("="*60)
    
    # Update these paths to the correct ones
    #old_model_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enorlleobecjt/enrollment_model_W0.0001_fulldata_sexspecific.pt"
    old_model_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox/ret_full_nopc_withsex/enrollment_model_W0.0001_batch_0_10000.pt"

    new_model_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt"
    
    print(f"Trying to load OLD model from: {old_model_path}")
    print(f"Trying to load NEW model from: {new_model_path}")
    
    # Load model without PCs (OLD)
    try:
        model_without_pcs = torch.load(old_model_path, map_location='cpu')
        lambdas_without_pcs = model_without_pcs['model_state_dict']['lambda_'].numpy()
        print(f"✅ Loaded model WITHOUT PCs (OLD): {lambdas_without_pcs.shape}")
    except Exception as e:
        print(f"❌ Error loading model without PCs: {e}")
        print("Please check the file path and update it in the script")
        return
    
    # Load model with PCs (NEW)
    try:
        model_with_pcs = torch.load(new_model_path, map_location='cpu')
        lambdas_with_pcs = model_with_pcs['model_state_dict']['lambda_'].numpy()
        print(f"✅ Loaded model WITH PCs (NEW): {lambdas_with_pcs.shape}")
    except Exception as e:
        print(f"❌ Error loading model with PCs: {e}")
        print("Please check the file path and update it in the script")
        return
    
    # Apply softmax normalization to both
    lambdas_with_pcs_softmax = softmax(lambdas_with_pcs, axis=1)
    lambdas_without_pcs_softmax = softmax(lambdas_without_pcs, axis=1)
    
    print(f"After softmax normalization:")
    print(f"  With PCs: {lambdas_with_pcs_softmax.shape}")
    print(f"  Without PCs: {lambdas_without_pcs_softmax.shape}")
    
    # Sample individuals to compare
    sample_patients = [0, 100, 500, 1000, 5000]
    
    # Create figure
    fig, axes = plt.subplots(len(sample_patients), 1, figsize=(12, 3*len(sample_patients)))
    if len(sample_patients) == 1:
        axes = [axes]
    
    # Colors for signatures
    colors = plt.cm.tab20(np.linspace(0, 1, lambdas_with_pcs_softmax.shape[1]))
    
    # Determine age range based on actual data dimensions
    print(f"Data shapes:")
    print(f"  With PCs: {lambdas_with_pcs_softmax.shape}")
    print(f"  Without PCs: {lambdas_without_pcs_softmax.shape}")
    
    # Handle different time dimensions
    if len(lambdas_with_pcs_softmax.shape) == 3:
        time_dim_with_pcs = lambdas_with_pcs_softmax.shape[2]
        time_dim_without_pcs = lambdas_without_pcs_softmax.shape[2]
        print(f"Time dimensions: With PCs={time_dim_with_pcs}, Without PCs={time_dim_without_pcs}")
        
        # Use the smaller time dimension to match both
        min_time_dim = min(time_dim_with_pcs, time_dim_without_pcs)
        ages = np.arange(30, 30 + min_time_dim)
    else:
        # If 2D, create a single time point
        ages = np.array([30])
    
    for i, patient_idx in enumerate(sample_patients):
        ax = axes[i]
        
        if patient_idx >= lambdas_with_pcs_softmax.shape[0] or patient_idx >= lambdas_without_pcs_softmax.shape[0]:
            print(f"Patient {patient_idx} not available in both models")
            continue
        
        # Get trajectories for this patient over time
        # Assuming shape is (patients, signatures, time)
        if len(lambdas_with_pcs_softmax.shape) == 3:
            # Truncate to the smaller time dimension to match
            traj_with_pcs = lambdas_with_pcs_softmax[patient_idx, :, :min_time_dim]  # Shape: (signatures, time)
            traj_without_pcs = lambdas_without_pcs_softmax[patient_idx, :, :min_time_dim]  # Shape: (signatures, time)
        else:
            # If 2D, assume it's already averaged over time
            traj_with_pcs = lambdas_with_pcs_softmax[patient_idx, :]  # Shape: (signatures,)
            traj_without_pcs = lambdas_without_pcs_softmax[patient_idx, :]  # Shape: (signatures,)
            # Create constant trajectories over time
            traj_with_pcs = np.tile(traj_with_pcs[:, np.newaxis], (1, len(ages)))
            traj_without_pcs = np.tile(traj_without_pcs[:, np.newaxis], (1, len(ages)))
        
        # Plot signatures over time
        for sig_idx in range(traj_with_pcs.shape[0]):
            # Plot NEW model (with PCs) - solid line
            ax.plot(ages, traj_with_pcs[sig_idx, :], '-', 
                   color=colors[sig_idx], alpha=0.7, linewidth=2,
                   label=f'Sig {sig_idx}' if sig_idx < 10 else '')
            
            # Plot OLD model (without PCs) - dashed line
            ax.plot(ages, traj_without_pcs[sig_idx, :], '--', 
                   color=colors[sig_idx], alpha=0.7, linewidth=2)
        
        ax.set_title(f'Patient {patient_idx}: Signature Loading (theta) over Age (years)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Signature Loading (theta)')
        ax.grid(True, alpha=0.3)
        
        # Add legend for first subplot only
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_xlim(30, 80)
    
    plt.tight_layout()
    plt.show()
    
    # Print numerical differences
    print(f"\nNUMERICAL COMPARISON:")
    print(f"{'Patient':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Sig with Max Diff':<15}")
    print("-" * 50)
    
    for patient_idx in sample_patients:
        if patient_idx >= lambdas_with_pcs_softmax.shape[0] or patient_idx >= lambdas_without_pcs_softmax.shape[0]:
            continue
        
        traj_with_pcs = lambdas_with_pcs_softmax[patient_idx, :]
        traj_without_pcs = lambdas_without_pcs_softmax[patient_idx, :]
        
        diff = np.abs(traj_with_pcs - traj_without_pcs)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        sig_with_max_diff = np.argmax(diff)
        
        print(f"{patient_idx:<8} {max_diff:<10.6f} {mean_diff:<10.6f} {sig_with_max_diff:<15}")
    
    # Overall comparison
    print(f"\nOVERALL COMPARISON:")
    overall_diff = np.abs(lambdas_with_pcs_softmax - lambdas_without_pcs_softmax)
    overall_max_diff = np.max(overall_diff)
    overall_mean_diff = np.mean(overall_diff)
    
    print(f"Max difference across all patients and signatures: {overall_max_diff:.6f}")
    print(f"Mean difference across all patients and signatures: {overall_mean_diff:.6f}")
    
    if overall_max_diff < 1e-5:
        print("✅ Models are essentially identical!")
    elif overall_max_diff < 1e-3:
        print("⚠️ Models are very similar (small differences)")
    else:
        print("❌ Models are significantly different")

if __name__ == "__main__":
    compare_lambda_trajectories()
