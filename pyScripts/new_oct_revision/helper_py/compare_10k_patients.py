"""
Compare first 10K patients from NEW model with PCs vs OLD analysis
Create time-varying plots like we saw before
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def compare_10k_patients():
    """
    Compare first 10K patients from OLD analysis vs OLD model (no sex)
    """
    
    print("COMPARING FIRST 10K PATIENTS: OLD ANALYSIS vs OLD MODEL (NO SEX)")
    print("="*60)
    
    # Load OLD analysis (Dataset 1)
    try:
        old_data = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/big_stuff/all_patient_thetas_alltime.pt', map_location='cpu').numpy()
        print(f"✅ OLD analysis (all_patient_thetas_alltime.pt): {old_data.shape}")
    except Exception as e:
        print(f"❌ Error loading OLD analysis: {e}")
        return
    
    # Load OLD model without sex (first 10K patients)
    try:
        old_model_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enorlleobecjt/enrollment_model_W0.0001_fulldata_sexspecific.pt"
        print(f"Trying to load OLD model from: {old_model_path}")
        old_model = torch.load(old_model_path, map_location='cpu')
        old_data_raw = old_model['model_state_dict']['lambda_'].numpy()
        old_data_model = softmax(old_data_raw, axis=1)
        print(f"✅ OLD model without sex (first 10K): {old_data_model.shape}")
    except Exception as e:
        print(f"❌ Error loading OLD model: {e}")
        return
    
    # Take first 10K patients from OLD analysis for fair comparison
    old_data_10k = old_data[:10000, :, :] if len(old_data.shape) == 3 else old_data[:10000, :]
    old_model_10k = old_data_model[:10000, :, :] if len(old_data_model.shape) == 3 else old_data_model[:10000, :]
    
    print(f"\nComparison dimensions:")
    print(f"OLD analysis (10K): {old_data_10k.shape}")
    print(f"OLD model without sex (10K): {old_model_10k.shape}")
    
    # Sample patients for time-varying plots
    sample_patients = [0, 100, 500, 1000, 5000]
    
    # Create time-varying plots
    fig, axes = plt.subplots(len(sample_patients), 1, figsize=(12, 3*len(sample_patients)))
    if len(sample_patients) == 1:
        axes = [axes]
    
    # Colors for signatures
    colors = plt.cm.tab20(np.linspace(0, 1, old_data_10k.shape[1]))
    
    # Handle time dimensions
    if len(old_data_10k.shape) == 3 and len(old_model_10k.shape) == 3:
        time_dim_old = old_data_10k.shape[2]
        time_dim_model = old_model_10k.shape[2]
        min_time_dim = min(time_dim_old, time_dim_model)
        ages = np.arange(30, 30 + min_time_dim)
        print(f"Time dimensions: OLD analysis={time_dim_old}, OLD model={time_dim_model}, Using={min_time_dim}")
    else:
        # If 2D, create constant trajectories
        ages = np.arange(30, 81)
        min_time_dim = len(ages)
    
    for i, patient_idx in enumerate(sample_patients):
        ax = axes[i]
        
        if patient_idx >= 10000:
            print(f"Patient {patient_idx} not available in 10K subset")
            continue
        
        # Get trajectories for this patient
        if len(old_data_10k.shape) == 3:
            old_traj = old_data_10k[patient_idx, :, :min_time_dim]  # Shape: (signatures, time)
        else:
            old_traj = np.tile(old_data_10k[patient_idx, :, np.newaxis], (1, min_time_dim))
        
        if len(old_model_10k.shape) == 3:
            model_traj = old_model_10k[patient_idx, :, :min_time_dim]  # Shape: (signatures, time)
        else:
            model_traj = np.tile(old_model_10k[patient_idx, :, np.newaxis], (1, min_time_dim))
        
        # Plot signatures over time
        for sig_idx in range(min(old_traj.shape[0], model_traj.shape[0])):
            # Plot OLD analysis (solid line)
            ax.plot(ages, old_traj[sig_idx, :], '-', 
                   color=colors[sig_idx], alpha=0.7, linewidth=2,
                   label=f'Sig {sig_idx}' if sig_idx < 10 else '')
            
            # Plot OLD model without sex (dashed line)
            ax.plot(ages, model_traj[sig_idx, :], '--', 
                   color=colors[sig_idx], alpha=0.7, linewidth=2)
        
        ax.set_title(f'Patient {patient_idx}: Signature Loading (theta) over Age (years)\nOLD Analysis vs OLD Model (No Sex)', 
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
    
    # Numerical comparison
    print(f"\nNUMERICAL COMPARISON (First 10K patients):")
    print(f"{'Patient':<8} {'Max Diff':<10} {'Mean Diff':<10} {'Sig with Max Diff':<15}")
    print("-" * 50)
    
    for patient_idx in sample_patients:
        if patient_idx >= 10000:
            continue
        
        if len(old_data_10k.shape) == 3:
            old_traj = old_data_10k[patient_idx, :, :min_time_dim]
        else:
            old_traj = np.tile(old_data_10k[patient_idx, :, np.newaxis], (1, min_time_dim))
        
        if len(old_model_10k.shape) == 3:
            model_traj = old_model_10k[patient_idx, :, :min_time_dim]
        else:
            model_traj = np.tile(old_model_10k[patient_idx, :, np.newaxis], (1, min_time_dim))
        
        # Calculate differences
        diff = np.abs(old_traj - model_traj)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Find signature with max difference
        max_diff_indices = np.unravel_index(np.argmax(diff), diff.shape)
        sig_with_max_diff = max_diff_indices[0]
        
        print(f"{patient_idx:<8} {max_diff:<10.6f} {mean_diff:<10.6f} {sig_with_max_diff:<15}")
    
    # Overall comparison
    print(f"\nOVERALL COMPARISON (First 10K patients):")
    
    # Flatten for overall comparison
    if len(old_data_10k.shape) == 3:
        old_flat = old_data_10k[:, :, :min_time_dim].flatten()
    else:
        old_flat = old_data_10k.flatten()
    
    if len(old_model_10k.shape) == 3:
        model_flat = old_model_10k[:, :, :min_time_dim].flatten()
    else:
        model_flat = old_model_10k.flatten()
    
    overall_diff = np.mean(np.abs(old_flat - model_flat))
    overall_max_diff = np.max(np.abs(old_flat - model_flat))
    correlation = np.corrcoef(old_flat, model_flat)[0, 1]
    
    print(f"Mean difference: {overall_diff:.6f}")
    print(f"Max difference: {overall_max_diff:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    # Summary
    print(f"\nSUMMARY:")
    if overall_max_diff < 1e-5:
        print("✅ OLD analysis and OLD model (no sex) are essentially identical!")
    elif overall_max_diff < 1e-3:
        print("⚠️ OLD analysis and OLD model (no sex) are very similar (small differences)")
    else:
        print("❌ OLD analysis and OLD model (no sex) have significant differences")
        print("This suggests the OLD analysis might have used a different model or preprocessing!")

if __name__ == "__main__":
    compare_10k_patients()
