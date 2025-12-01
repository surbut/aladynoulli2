"""
Compare lambda trajectories for sample individuals between models with PCs vs without PCs
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
    
    # Load model without PCs (OLD)
    try:
        model_without_pcs = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal (9-23-25 4:48 PM)/resultshighamp/results/output_0_10000/enrollment_model_W0.0001_batch_0_10000.pt', map_location='cpu')
        lambdas_without_pcs = model_without_pcs['model_state_dict']['lambda_'].numpy()
        print(f"Loaded model WITHOUT PCs (OLD): {lambdas_without_pcs.shape}")
    except Exception as e:
        print(f"Error loading model without PCs: {e}")
        return
    
    # Load model with PCs (NEW)
    try:
        model_with_pcs = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt', map_location='cpu')
        lambdas_with_pcs = model_with_pcs['model_state_dict']['lambda_'].numpy()
        print(f"Loaded model WITH PCs (NEW): {lambdas_with_pcs.shape}")
    except Exception as e:
        print(f"Error loading model with PCs: {e}")
        return
    
    # Apply softmax normalization to both
    lambdas_with_pcs_softmax = softmax(lambdas_with_pcs, axis=1)
    lambdas_without_pcs_softmax = softmax(lambdas_without_pcs, axis=1)
    
    print(f"After softmax normalization:")
    print(f"  With PCs: {lambdas_with_pcs_softmax.shape}")
    print(f"  Without PCs: {lambdas_without_pcs_softmax.shape}")
    
    # Sample individuals to compare
    sample_patients = [0, 100, 500, 1000]
    
    # Create figure
    fig, axes = plt.subplots(len(sample_patients), 1, figsize=(12, 3*len(sample_patients)))
    if len(sample_patients) == 1:
        axes = [axes]
    
    # Colors for signatures
    colors = plt.cm.tab20(np.linspace(0, 1, lambdas_with_pcs_softmax.shape[1]))
    
    for i, patient_idx in enumerate(sample_patients):
        ax = axes[i]
        
        if patient_idx >= lambdas_with_pcs_softmax.shape[0] or patient_idx >= lambdas_without_pcs_softmax.shape[0]:
            print(f"Patient {patient_idx} not available in both models")
            continue
        
        # Get trajectories for this patient
        traj_with_pcs = lambdas_with_pcs_softmax[patient_idx, :]  # Shape: (signatures,)
        traj_without_pcs = lambdas_without_pcs_softmax[patient_idx, :]  # Shape: (signatures,)
        
        # Plot signatures
        for sig_idx in range(lambdas_with_pcs_softmax.shape[1]):
            # Plot NEW model (with PCs) - solid line
            ax.plot([sig_idx], [traj_with_pcs[sig_idx]], 'o-', 
                   color=colors[sig_idx], alpha=0.7, linewidth=2, markersize=4,
                   label=f'Sig {sig_idx}' if sig_idx < 10 else '')
            
            # Plot OLD model (without PCs) - dashed line
            ax.plot([sig_idx], [traj_without_pcs[sig_idx]], 's--', 
                   color=colors[sig_idx], alpha=0.7, linewidth=2, markersize=4)
        
        ax.set_title(f'Patient {patient_idx}: Lambda Trajectories (With PCs vs Without PCs)', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Signature Index')
        ax.set_ylabel('Signature Loading (softmax normalized)')
        ax.grid(True, alpha=0.3)
        
        # Add legend for first subplot only
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
    
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
