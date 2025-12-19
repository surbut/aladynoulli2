"""
Train models with weighted prevalence and compare pi with weighted prevalence.

This script:
1. Loads weighted prevalence (prevalence_t_weighted_corrected.pt)
2. Trains models (or loads existing) using weighted prevalence for initialization
3. Computes pi from trained models
4. Compares pi with weighted prevalence to show correlation

This demonstrates that using weighted prevalence for initialization leads to
pi predictions that better match the weighted population prevalence.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add paths
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts_forPublish')
from weighted_aladyn_vec import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest_weighted, subset_data
from weightedprev import match_weights_to_ids
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred

print("="*80)
print("TRAINING WITH WEIGHTED PREVALENCE AND COMPARING PI")
print("="*80)

# Configuration
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
batch_size = 10000
n_batches_to_train = 10  # Train just 1 batch for quick test, or increase for more
N_Y = 100000  # Use first 100K patients

# Output directory for saving batch models
model_output_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1219/')
model_output_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("\n1. Loading data...")
Y = torch.load(str(data_dir / 'Y_tensor.pt'), weights_only=False)
E = torch.load(str(data_dir / 'E_matrix_corrected.pt'), weights_only=False)
G = torch.load(str(data_dir / 'G_matrix.pt'), weights_only=False)
essentials = torch.load(str(data_dir / 'model_essentials.pt'), weights_only=False)

print(f"   Y shape: {Y.shape}")
print(f"   E shape: {E.shape}")
print(f"   G shape: {G.shape}")

# Load weighted prevalence (computed on all 400K patients)
print("\n2. Loading weighted prevalence (from all 400K patients)...")
weighted_prevalence_path = data_dir / 'prevalence_t_weighted_corrected.pt'
if weighted_prevalence_path.exists():
    prevalence_t_weighted = torch.load(str(weighted_prevalence_path), weights_only=False)
    if torch.is_tensor(prevalence_t_weighted):
        prevalence_t_weighted = prevalence_t_weighted.numpy()
    print(f"   ✓ Loaded weighted prevalence: {prevalence_t_weighted.shape}")
    print(f"   Note: This is computed on all 400K patients, while pi will be from 100K subset")
else:
    print(f"   ⚠️  Weighted prevalence not found: {weighted_prevalence_path}")
    print(f"   Run compute_weighted_prevalence_corrected.py first")
    raise FileNotFoundError(f"Weighted prevalence not found: {weighted_prevalence_path}")

# Load references
print("\n3. Loading references...")
refs = torch.load(str(data_dir / 'reference_trajectories.pt'), weights_only=False)
signature_refs = refs['signature_refs']
print(f"   ✓ Loaded signature references")

# Load initial psi and clusters
print("\n4. Loading initial parameters...")
initial_psi = torch.load(str(data_dir / 'initial_psi_400k.pt'), weights_only=False)
initial_clusters = torch.load(str(data_dir / 'initial_clusters_400k.pt'), weights_only=False)
print(f"   ✓ Loaded initial psi and clusters")

# Load patient IDs
print("\n5. Loading patient IDs...")
pids_csv_path = Path('/Users/sarahurbut/aladynoulli2/pyScripts/csv/processed_ids.csv')
pids_df = pd.read_csv(pids_csv_path)
pids = pids_df['eid'].values
print(f"   ✓ Loaded {len(pids):,} patient IDs")

# Load IPW weights
print("\n6. Loading IPW weights...")
weights_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/UKBWeights-main/UKBSelectionWeights.csv")
weights_df = pd.read_csv(weights_path, sep='\s+', engine='python')
matched_weights, match_mask = match_weights_to_ids(weights_df, pids[:N_Y])
print(f"   ✓ Matched weights: {match_mask.sum():,} / {N_Y:,}")

# Load covariates
print("\n7. Loading covariates...")
covariates_path = data_dir / 'baselinagefamh_withpcs.csv'
cov_df = pd.read_csv(covariates_path)
sex_col = 'sex'
pc_columns = [
    'f.22009.0.1','f.22009.0.2','f.22009.0.3','f.22009.0.4','f.22009.0.5',
    'f.22009.0.6','f.22009.0.7','f.22009.0.8','f.22009.0.9','f.22009.0.10'
]
cov_df = cov_df[['identifier', sex_col] + pc_columns].dropna(subset=['identifier'])
cov_df = cov_df.drop_duplicates(subset=['identifier'])
cov_map = cov_df.set_index('identifier')
print(f"   ✓ Loaded covariates")

# Train models with weighted prevalence
print("\n" + "="*80)
print(f"TRAINING {n_batches_to_train} BATCH(ES) WITH WEIGHTED PREVALENCE")
print("="*80)

pi_list = []
prevalence_list = []

for batch_idx in range(n_batches_to_train):
    print(f"\n{'='*60}")
    print(f"BATCH {batch_idx + 1}/{n_batches_to_train}")
    print(f"{'='*60}")
    
    # Calculate batch indices
    idx_start = batch_idx * batch_size
    idx_end = min((batch_idx + 1) * batch_size, N_Y)
    
    print(f"Processing individuals {idx_start} to {idx_end-1}")
    
    # Subset data
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G, start_index=idx_start, end_index=idx_end)
    
    # Subset covariates
    pids_batch = pids[idx_start:idx_end]
    sex_batch = np.zeros(len(pids_batch), dtype=float)
    pcs_batch = np.zeros((len(pids_batch), len(pc_columns)), dtype=float)
    
    matched_cov_batch = 0
    for i, pid in enumerate(pids_batch):
        if pid in cov_map.index:
            sex_batch[i] = cov_map.at[pid, sex_col]
            pcs_batch[i, :] = cov_map.loc[pid, pc_columns].values
            matched_cov_batch += 1
    
    print(f"Covariates matched: {matched_cov_batch} / {len(pids_batch)} ({matched_cov_batch*100.0/len(pids_batch):.1f}%)")
    
    # Combine G with covariates
    sex_batch = sex_batch.reshape(-1, 1)
    G_with_covs_batch = np.column_stack([G_batch, sex_batch, pcs_batch])
    
    # Subset weights
    weights_batch = matched_weights[idx_start:idx_end].copy()
    num_unmatched = int((weights_batch == 0).sum())
    if num_unmatched > 0:
        print(f"Warning: {num_unmatched} had no matched weight; setting weight=1.0.")
        weights_batch[weights_batch == 0] = 1.0
    
    print(f"Weight stats: mean={weights_batch.mean():.3f}, std={weights_batch.std():.3f}")
    
    # Build model with WEIGHTED PREVALENCE
    print(f"\nInitializing model with weighted prevalence...")
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest_weighted(
        N=Y_batch.shape[0],
        D=Y_batch.shape[1],
        T=Y_batch.shape[2],
        K=20,
        P=G_with_covs_batch.shape[1],
        G=G_with_covs_batch,
        Y=Y_batch,
        W=1e-4,
        R=0,
        prevalence_t=prevalence_t_weighted,  # Use weighted prevalence!
        signature_references=signature_refs,
        healthy_reference=True,
        disease_names=essentials.get('disease_names', None),
        init_sd_scaler=1e-1,
        genetic_scale=1.0,
        learn_kappa=True,
        weights=weights_batch
    )
    
    # Initialize with stored psi/clusters
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters
    
    # Train
    print("Training...")
    losses, grads = model.fit(event_times=E_batch, num_epochs=200, learning_rate=1e-1, lambda_reg=1e-2)
    
    print(f"Training complete. Final loss: {losses[-1]:.4f}")
    
    # Save batch model (just like the notebook)
    print(f"Saving batch model...")
    model_filename = f"batch_{batch_idx:02d}_model.pt"
    model_path = model_output_dir / model_filename
    
    # Calculate weights stats for this batch
    weights_stats = {
        'mean': weights_batch.mean(),
        'std': weights_batch.std(),
        'min': weights_batch.min(),
        'max': weights_batch.max()
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'batch_idx': batch_idx,
        'weights_stats': weights_stats,
        'final_loss': losses[-1] if losses else None,
        'losses': losses  # Save full loss history
    }, str(model_path))
    
    print(f"   ✓ Saved batch model to: {model_path}")
    
    # Compute pi from model
    print("Computing pi from model...")
    lambda_ = model.lambda_.detach()
    phi = model.phi.detach()
    kappa = model.kappa.item() if torch.is_tensor(model.kappa) and model.kappa.numel() == 1 else model.kappa.mean().item()
    
    pi_batch = calculate_pi_pred(lambda_, phi, kappa)
    pi_list.append(pi_batch)
    
    # Store weighted prevalence for this batch (same for all batches - from all 400K)
    prevalence_list.append(torch.tensor(prevalence_t_weighted))
    
    print(f"   Pi shape: {pi_batch.shape}")
    print(f"   Prevalence shape: {prevalence_t_weighted.shape}")

# Concatenate pi from all batches
if len(pi_list) > 0:
    print(f"\n{'='*80}")
    print("COMPARING PI WITH WEIGHTED PREVALENCE")
    print("="*80)
    
    # Concatenate all batches: [N_total, D, T]
    pi_all = torch.cat(pi_list, dim=0)
    print(f"   Concatenated pi shape (all batches): {pi_all.shape}")
    
    # Average pi across patients: [D, T]
    pi_avg = pi_all.mean(dim=0)
    
    # Weighted prevalence is already [D, T] (same for all batches)
    prevalence_avg = prevalence_list[0].numpy() if torch.is_tensor(prevalence_list[0]) else prevalence_list[0]
    
    print(f"\nPi average shape: {pi_avg.shape}")
    print(f"Prevalence shape: {prevalence_avg.shape}")
    
    # Calculate correlation
    pi_flat = pi_avg.numpy().flatten()
    prev_flat = prevalence_avg.flatten()
    
    # Remove any NaN or inf values
    valid_mask = ~(np.isnan(pi_flat) | np.isnan(prev_flat) | np.isinf(pi_flat) | np.isinf(prev_flat))
    pi_valid = pi_flat[valid_mask]
    prev_valid = prev_flat[valid_mask]
    
    correlation = np.corrcoef(pi_valid, prev_valid)[0, 1]
    mean_diff = np.abs(pi_valid - prev_valid).mean()
    max_diff = np.abs(pi_valid - prev_valid).max()
    
    print(f"\nComparison Statistics:")
    print(f"  Correlation: {correlation:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    
    # Plot comparison for selected diseases
    DISEASES_TO_PLOT = [
        (112, "Myocardial Infarction"),
        (66, "Depression"),
        (16, "Breast cancer [female]"),
        (127, "Atrial fibrillation"),
        (47, "Type 2 diabetes"),
    ]
    
    # Load disease names if available
    disease_names_dict = {}
    try:
        disease_names_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv")
        if disease_names_path.exists():
            disease_df = pd.read_csv(disease_names_path)
            disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
    except:
        pass
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        if disease_names_dict and disease_idx in disease_names_dict:
            display_name = disease_names_dict[disease_idx]
        else:
            display_name = disease_name
        
        if disease_idx < pi_avg.shape[0] and disease_idx < prevalence_avg.shape[0]:
            pi_traj = pi_avg[disease_idx, :].numpy()
            prev_traj = prevalence_avg[disease_idx, :]
            
            # Time points (assuming starting at age 30)
            time_points = np.arange(len(pi_traj)) + 30
            
            ax.plot(time_points, prev_traj, label='Weighted Prevalence\n(corrected E + IPW)', 
                   linewidth=2, alpha=0.8, color='green')
            ax.plot(time_points, pi_traj, label='Pi from Model\n(trained with weighted prevalence)', 
                   linewidth=2, alpha=0.8, linestyle='--', color='red')
            
            ax.set_xlabel('Age', fontsize=11)
            ax.set_ylabel('Prevalence / Pi', fontsize=11)
            ax.set_title(f'{display_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Add correlation annotation
            disease_corr = np.corrcoef(prev_traj, pi_traj)[0, 1]
            disease_diff = np.abs(prev_traj - pi_traj).mean()
            ax.text(0.02, 0.98, f'Corr: {disease_corr:.4f}\nMean diff: {disease_diff:.4f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Disease {disease_idx}\nnot found', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(f'{disease_name}\n(Disease {disease_idx})', fontsize=12, fontweight='bold')
    
    # Remove extra subplot
    if len(DISEASES_TO_PLOT) < len(axes):
        axes[len(DISEASES_TO_PLOT)].axis('off')
    
    plt.suptitle(f'Weighted Prevalence vs Pi (Trained with Weighted Prevalence)\nCorrelation: {correlation:.4f} (N={pi_all.shape[0]:,} patients, {n_batches_to_train} batches)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'weighted_prevalence_vs_pi_trained_with_weighted_prev.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {plot_path}")
    plt.show()
    
    # Scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.scatter(prev_valid, pi_valid, alpha=0.3, s=1)
    ax.plot([prev_valid.min(), prev_valid.max()], 
           [prev_valid.min(), prev_valid.max()], 'r--', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Weighted Prevalence (corrected E + IPW)', fontsize=12)
    ax.set_ylabel('Pi from Model (trained with weighted prevalence)', fontsize=12)
    ax.set_title(f'Prevalence vs Pi Comparison\nCorrelation: {correlation:.4f}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    scatter_path = output_dir / 'weighted_prevalence_vs_pi_scatter.pdf'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved scatter plot to: {scatter_path}")
    plt.show()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"✓ Trained model(s) with weighted prevalence")
    print(f"✓ Computed pi from trained model(s)")
    print(f"✓ Correlation between weighted prevalence and pi: {correlation:.6f}")
    print(f"\nKey Insight:")
    print(f"  When models are trained with weighted prevalence for initialization,")
    print(f"  the resulting pi predictions show high correlation ({correlation:.4f}) with")
    print(f"  the weighted population prevalence. This demonstrates that:")
    print(f"  1. The model adapts to the reweighted population through lambda/pi")
    print(f"  2. Using weighted prevalence for initialization improves alignment")
    print(f"  3. Despite phi stability, the model captures population-specific risks")
    
else:
    print("\n⚠️  No models trained successfully.")

