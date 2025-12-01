"""
Comprehensive comparison of all three theta/lambda datasets
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def compare_all_datasets():
    """
    Compare all three datasets:
    1. all_patient_thetas_alltime.pt
    2. thetas.npy  
    3. Softmax from PC runs
    """
    
    print("COMPREHENSIVE DATASET COMPARISON")
    print("="*60)
    
    # Load dataset 1: all_patient_thetas_alltime.pt
    try:
        dataset1 = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/big_stuff/all_patient_thetas_alltime.pt', map_location='cpu').numpy()
        print(f"✅ Dataset 1 (all_patient_thetas_alltime.pt): {dataset1.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset 1: {e}")
        return
    
    # Load dataset 2: thetas.npy
    try:
        dataset2 = np.load('/Users/sarahurbut/aladynoulli2/pyScripts/thetas.npy')
        print(f"✅ Dataset 2 (thetas.npy): {dataset2.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset 2: {e}")
        return
    
    # Load dataset 3: Softmax from PC runs (NEW model)
    try:
        pc_model = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt', map_location='cpu')
        dataset3_raw = pc_model['model_state_dict']['lambda_'].numpy()
        dataset3 = softmax(dataset3_raw, axis=1)
        print(f"✅ Dataset 3 (NEW model with PCs - softmax): {dataset3.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset 3: {e}")
        return
    
    # Print shape analysis
    print(f"\nSHAPE ANALYSIS:")
    print(f"Dataset 1: {dataset1.shape}")
    print(f"Dataset 2: {dataset2.shape}")
    print(f"Dataset 3: {dataset3.shape}")
    
    # Determine common dimensions for comparison
    min_patients = min(dataset1.shape[0], dataset2.shape[0], dataset3.shape[0])
    min_signatures = min(dataset1.shape[1], dataset2.shape[1], dataset3.shape[1])
    
    print(f"\nCommon dimensions for comparison:")
    print(f"Patients: {min_patients}")
    print(f"Signatures: {min_signatures}")
    
    # Sample patients for comparison
    sample_patients = [0, 100, 500, 1000]
    
    # Create comparison plots
    fig, axes = plt.subplots(len(sample_patients), 1, figsize=(15, 4*len(sample_patients)))
    if len(sample_patients) == 1:
        axes = [axes]
    
    # Colors for datasets
    colors = ['blue', 'red', 'green']
    labels = ['Dataset 1 (all_patient_thetas_alltime.pt)', 'Dataset 2 (thetas.npy - should be same as D1)', 'Dataset 3 (NEW model with PCs)']
    
    for i, patient_idx in enumerate(sample_patients):
        ax = axes[i]
        
        if patient_idx >= min_patients:
            print(f"Patient {patient_idx} not available in all datasets")
            continue
        
        # Get trajectories for this patient
        traj1 = dataset1[patient_idx, :min_signatures]
        traj2 = dataset2[patient_idx, :min_signatures]
        traj3 = dataset3[patient_idx, :min_signatures]
        
        # Plot signatures
        x_pos = np.arange(min_signatures)
        
        ax.plot(x_pos, traj1, 'o-', color=colors[0], alpha=0.7, linewidth=2, markersize=4, label=labels[0])
        ax.plot(x_pos, traj2, 's-', color=colors[1], alpha=0.7, linewidth=2, markersize=4, label=labels[1])
        ax.plot(x_pos, traj3, '^-', color=colors[2], alpha=0.7, linewidth=2, markersize=4, label=labels[2])
        
        ax.set_title(f'Patient {patient_idx}: Signature Loadings Comparison', fontweight='bold', fontsize=12)
        ax.set_xlabel('Signature Index')
        ax.set_ylabel('Signature Loading')
        ax.grid(True, alpha=0.3)
        
        # Add legend for first subplot only
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical comparison
    print(f"\nNUMERICAL COMPARISON:")
    print(f"{'Patient':<8} {'D1 vs D2':<12} {'D1 vs D3':<12} {'D2 vs D3':<12}")
    print("-" * 50)
    
    for patient_idx in sample_patients:
        if patient_idx >= min_patients:
            continue
        
        traj1 = dataset1[patient_idx, :min_signatures]
        traj2 = dataset2[patient_idx, :min_signatures]
        traj3 = dataset3[patient_idx, :min_signatures]
        
        diff_12 = np.mean(np.abs(traj1 - traj2))
        diff_13 = np.mean(np.abs(traj1 - traj3))
        diff_23 = np.mean(np.abs(traj2 - traj3))
        
        print(f"{patient_idx:<8} {diff_12:<12.6f} {diff_13:<12.6f} {diff_23:<12.6f}")
    
    # Overall comparison
    print(f"\nOVERALL COMPARISON:")
    
    # Truncate all datasets to common dimensions
    d1_trunc = dataset1[:min_patients, :min_signatures]
    d2_trunc = dataset2[:min_patients, :min_signatures]
    d3_trunc = dataset3[:min_patients, :min_signatures]
    
    diff_12 = np.mean(np.abs(d1_trunc - d2_trunc))
    diff_13 = np.mean(np.abs(d1_trunc - d3_trunc))
    diff_23 = np.mean(np.abs(d2_trunc - d3_trunc))
    
    print(f"Dataset 1 vs Dataset 2: {diff_12:.6f}")
    print(f"Dataset 1 vs Dataset 3: {diff_13:.6f}")
    print(f"Dataset 2 vs Dataset 3: {diff_23:.6f}")
    
    # Correlation analysis
    print(f"\nCORRELATION ANALYSIS:")
    
    # Flatten for correlation
    d1_flat = d1_trunc.flatten()
    d2_flat = d2_trunc.flatten()
    d3_flat = d3_trunc.flatten()
    
    corr_12 = np.corrcoef(d1_flat, d2_flat)[0, 1]
    corr_13 = np.corrcoef(d1_flat, d3_flat)[0, 1]
    corr_23 = np.corrcoef(d2_flat, d3_flat)[0, 1]
    
    print(f"Dataset 1 vs Dataset 2 correlation: {corr_12:.6f}")
    print(f"Dataset 1 vs Dataset 3 correlation: {corr_13:.6f}")
    print(f"Dataset 2 vs Dataset 3 correlation: {corr_23:.6f}")
    
    # Summary
    print(f"\nSUMMARY:")
    if diff_12 < 1e-5 and diff_13 < 1e-5 and diff_23 < 1e-5:
        print("✅ All datasets are essentially identical!")
    elif diff_12 < 1e-3 and diff_13 < 1e-3 and diff_23 < 1e-3:
        print("⚠️ All datasets are very similar (small differences)")
    else:
        print("❌ Datasets have significant differences")
        
        # Identify which datasets are most similar
        min_diff = min(diff_12, diff_13, diff_23)
        if min_diff == diff_12:
            print("Dataset 1 and Dataset 2 are most similar")
        elif min_diff == diff_13:
            print("Dataset 1 and Dataset 3 are most similar")
        else:
            print("Dataset 2 and Dataset 3 are most similar")

if __name__ == "__main__":
    compare_all_datasets()
