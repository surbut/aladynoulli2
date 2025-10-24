"""
Assemble NEW model with PCs from all batches and save as new_thetas_with_pcs_retrospective.pt
Also analyze correlation structure of OLD vs NEW
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import os

def assemble_new_model_with_pcs():
    """
    Assemble NEW model with PCs from all batches (0-400K)
    """
    
    print("ASSEMBLING NEW MODEL WITH PCS FROM ALL BATCHES")
    print("="*60)
    
    base_path = "/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full"
    total_patients = 400000
    batch_size = 10000
    
    all_lambdas = []
    
    for start_idx in range(0, total_patients, batch_size):
        end_idx = min(start_idx + batch_size, total_patients)
        filename = f"enrollment_model_W0.0001_batch_{start_idx}_{end_idx}.pt"
        filepath = os.path.join(base_path, filename)
        
        print(f"Loading batch {start_idx}-{end_idx}...")
        
        try:
            model = torch.load(filepath, map_location='cpu')
            lambda_batch = model['model_state_dict']['lambda_'].numpy()
            all_lambdas.append(lambda_batch)
            print(f"✅ Loaded: {lambda_batch.shape}")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            continue
    
    if not all_lambdas:
        print("❌ No batches loaded successfully!")
        return None
    
    # Concatenate all batches
    print(f"\nConcatenating {len(all_lambdas)} batches...")
    all_lambdas_combined = np.concatenate(all_lambdas, axis=0)
    print(f"✅ Combined shape: {all_lambdas_combined.shape}")
    
    # Apply softmax normalization
    print("Applying softmax normalization...")
    all_thetas_combined = softmax(all_lambdas_combined, axis=1)
    print(f"✅ Softmax applied: {all_thetas_combined.shape}")
    
    # Save the combined thetas
    output_path = "/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt"
    print(f"Saving to: {output_path}")
    
    torch.save(torch.from_numpy(all_thetas_combined), output_path)
    print(f"✅ Saved successfully!")
    
    return all_thetas_combined

def analyze_correlation_structure():
    """
    Analyze correlation structure of OLD vs NEW models
    """
    
    print("\nANALYZING CORRELATION STRUCTURE")
    print("="*60)
    
    # Load OLD analysis
    try:
        old_data = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/big_stuff/all_patient_thetas_alltime.pt', map_location='cpu').numpy()
        print(f"✅ OLD analysis: {old_data.shape}")
    except Exception as e:
        print(f"❌ Error loading OLD analysis: {e}")
        return
    
    # Load NEW model with PCs
    try:
        new_data = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt', map_location='cpu').numpy()
        print(f"✅ NEW model with PCs: {new_data.shape}")
    except Exception as e:
        print(f"❌ Error loading NEW model: {e}")
        return
    
    # Ensure same dimensions for comparison
    min_patients = min(old_data.shape[0], new_data.shape[0])
    old_data_subset = old_data[:min_patients]
    new_data_subset = new_data[:min_patients]
    
    print(f"Comparing {min_patients} patients")
    
    # Calculate overall correlations
    if len(old_data_subset.shape) == 3 and len(new_data_subset.shape) == 3:
        # 3D case: average over time
        old_avg = np.mean(old_data_subset, axis=2)  # Shape: (patients, signatures)
        new_avg = np.mean(new_data_subset, axis=2)  # Shape: (patients, signatures)
        
        # Calculate signature correlations
        signature_correlations = []
        for sig_idx in range(min(old_avg.shape[1], new_avg.shape[1])):
            corr = np.corrcoef(old_avg[:, sig_idx], new_avg[:, sig_idx])[0, 1]
            signature_correlations.append(corr)
        
        signature_correlations = np.array(signature_correlations)
        
        print(f"\nSignature correlations (averaged over time):")
        print(f"Mean correlation: {np.mean(signature_correlations):.6f}")
        print(f"Std correlation: {np.std(signature_correlations):.6f}")
        print(f"Min correlation: {np.min(signature_correlations):.6f}")
        print(f"Max correlation: {np.max(signature_correlations):.6f}")
        
        # Plot signature correlations
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(signature_correlations, 'o-', markersize=4)
        plt.title('Signature Correlations (OLD vs NEW)')
        plt.xlabel('Signature Index')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(signature_correlations, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Signature Correlations')
        plt.xlabel('Correlation')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Overall correlation matrix
        plt.subplot(2, 2, 3)
        correlation_matrix = np.corrcoef(old_avg.T, new_avg.T)
        n_sigs = old_avg.shape[1]
        # Extract cross-correlation block
        cross_corr = correlation_matrix[:n_sigs, n_sigs:]
        sns.heatmap(cross_corr, annot=False, cmap='RdBu_r', center=0, 
                   xticklabels=False, yticklabels=False)
        plt.title('Cross-Correlation Matrix\n(OLD signatures vs NEW signatures)')
        
        # Time-varying correlations
        plt.subplot(2, 2, 4)
        time_correlations = []
        min_time = min(old_data_subset.shape[2], new_data_subset.shape[2])
        for t in range(min_time):
            old_t = old_data_subset[:, :, t]
            new_t = new_data_subset[:, :, t]
            # Average correlation across signatures at this timepoint
            time_corr = np.mean([np.corrcoef(old_t[:, sig], new_t[:, sig])[0, 1] 
                               for sig in range(min(old_t.shape[1], new_t.shape[1]))])
            time_correlations.append(time_corr)
        
        ages = np.arange(30, 30 + min_time)
        plt.plot(ages, time_correlations, 'o-', markersize=3)
        plt.title('Time-Varying Correlations')
        plt.xlabel('Age (years)')
        plt.ylabel('Average Correlation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # 2D case
        old_flat = old_data_subset.flatten()
        new_flat = new_data_subset.flatten()
        
        overall_corr = np.corrcoef(old_flat, new_flat)[0, 1]
        print(f"\nOverall correlation: {overall_corr:.6f}")
        
        # Plot scatter
        plt.figure(figsize=(8, 6))
        plt.scatter(old_flat[::1000], new_flat[::1000], alpha=0.5, s=1)
        plt.xlabel('OLD Analysis')
        plt.ylabel('NEW Model with PCs')
        plt.title(f'Overall Correlation: {overall_corr:.6f}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Summary
    print(f"\nSUMMARY:")
    if len(signature_correlations) > 0:
        mean_corr = np.mean(signature_correlations)
        if mean_corr > 0.95:
            print("✅ OLD and NEW models are highly correlated!")
        elif mean_corr > 0.8:
            print("⚠️ OLD and NEW models are moderately correlated")
        else:
            print("❌ OLD and NEW models have low correlation")
            print("This suggests significant differences between the models!")
    else:
        print("❌ Could not calculate correlations")

def main():
    """
    Main function to assemble NEW model and analyze correlations
    """
    
    # Assemble NEW model with PCs
    new_thetas = assemble_new_model_with_pcs()
    
    if new_thetas is not None:
        # Analyze correlation structure
        analyze_correlation_structure()
    else:
        print("❌ Failed to assemble NEW model, skipping correlation analysis")

if __name__ == "__main__":
    main()
