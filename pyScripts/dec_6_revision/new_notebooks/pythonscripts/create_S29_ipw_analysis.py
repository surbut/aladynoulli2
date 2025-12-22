"""
Create Supplementary Figure S29: IPW Analysis

This figure includes:
1. IPW weights distribution
2. Phi/Pi/Prevalence comparison (3 columns: phi stability, pi changes, prevalence changes)
3. Lambda comparison (6 panels showing individual lambda differences)

This demonstrates the full IPW story: weights, phi stability, lambda/pi adaptability, and prevalence changes.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add path for utils
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from utils import calculate_pi_pred

print("="*80)
print("CREATING SUPPLEMENTARY FIGURE S29: IPW ANALYSIS")
print("="*80)

# Data directories
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
model_1218_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/batch_models_weighted_vec_censoredE_1218/')
unweighted_model_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/')
weights_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/UKBWeights-main/UKBSelectionWeights.csv")

# ============================================================================
# PART 1: IPW Weights Distribution
# ============================================================================
print("\n1. Loading and plotting IPW weights distribution...")

weights_df = pd.read_csv(weights_path, sep='\s+', engine='python')
weights = weights_df['LassoWeight'].values

print(f"   ✓ Loaded {len(weights):,} weights")
print(f"   Mean: {weights.mean():.3f}, Std: {weights.std():.3f}")
print(f"   Min: {weights.min():.3f}, Max: {weights.max():.3f}")

# ============================================================================
# PART 2: Load data for Phi/Pi/Prevalence comparison
# ============================================================================
print("\n2. Loading prevalences...")
weighted_prevalence_path = data_dir / 'prevalence_t_weighted_corrected.pt'
unweighted_prevalence_path = data_dir / 'prevalence_t_corrected.pt'

prevalence_t_weighted = torch.load(str(weighted_prevalence_path), weights_only=False)
if torch.is_tensor(prevalence_t_weighted):
    prevalence_t_weighted = prevalence_t_weighted.numpy()

prevalence_t_unweighted = torch.load(str(unweighted_prevalence_path), weights_only=False)
if torch.is_tensor(prevalence_t_unweighted):
    prevalence_t_unweighted = prevalence_t_unweighted.numpy()

print(f"   ✓ Loaded prevalences")

# Load models
print("\n3. Loading models (weighted and unweighted)...")
n_batches = 10

phi_1218_list = []
phi_unweighted_list = []
pi_1218_list = []
pi_unweighted_list = []
lambda_1218_list = []
lambda_unweighted_list = []

for batch_idx in range(n_batches):
    path_1218 = model_1218_dir / f"batch_{batch_idx:02d}_model.pt"
    if not path_1218.exists():
        path_1218 = model_1218_dir / f"batch_{batch_idx}_model.pt"
    
    path_unweighted = unweighted_model_dir / f"enrollment_model_W0.0001_batch_{batch_idx*10000}_{(batch_idx+1)*10000}.pt"
    
    if path_1218.exists() and path_unweighted.exists():
        # Load weighted model
        ckpt_1218 = torch.load(path_1218, weights_only=False, map_location='cpu')
        if 'model_state_dict' in ckpt_1218:
            phi_1218 = ckpt_1218['model_state_dict']['phi'].detach()
            lambda_1218 = ckpt_1218['model_state_dict']['lambda_'].detach()
            kappa_1218 = ckpt_1218['model_state_dict'].get('kappa', torch.tensor(1.0))
        else:
            phi_1218 = ckpt_1218['phi'].detach()
            lambda_1218 = ckpt_1218['lambda_'].detach()
            kappa_1218 = ckpt_1218.get('kappa', torch.tensor(1.0))
        
        if torch.is_tensor(kappa_1218):
            kappa_1218 = kappa_1218.item() if kappa_1218.numel() == 1 else kappa_1218.mean().item()
        
        # Load unweighted
        ckpt_unweighted = torch.load(path_unweighted, weights_only=False, map_location='cpu')
        if 'model_state_dict' in ckpt_unweighted:
            phi_unweighted = ckpt_unweighted['model_state_dict']['phi'].detach()
            lambda_unweighted = ckpt_unweighted['model_state_dict']['lambda_'].detach()
            kappa_unweighted = ckpt_unweighted['model_state_dict'].get('kappa', torch.tensor(1.0))
        else:
            phi_unweighted = ckpt_unweighted['phi'].detach()
            lambda_unweighted = ckpt_unweighted['lambda_'].detach()
            kappa_unweighted = ckpt_unweighted.get('kappa', torch.tensor(1.0))
        
        if torch.is_tensor(kappa_unweighted):
            kappa_unweighted = kappa_unweighted.item() if kappa_unweighted.numel() == 1 else kappa_unweighted.mean().item()
        
        # Store phi and lambda
        phi_1218_list.append(phi_1218)
        phi_unweighted_list.append(phi_unweighted)
        lambda_1218_list.append(lambda_1218)
        lambda_unweighted_list.append(lambda_unweighted)
        
        # Compute pi and average
        pi_1218_batch = calculate_pi_pred(lambda_1218, phi_1218, kappa_1218)
        pi_unweighted_batch = calculate_pi_pred(lambda_unweighted, phi_unweighted, kappa_unweighted)
        
        pi_1218_list.append(pi_1218_batch.mean(dim=0))
        pi_unweighted_list.append(pi_unweighted_batch.mean(dim=0))

# Average across batches
phi_1218_avg = torch.stack(phi_1218_list).mean(dim=0)  # [K, D, T]
phi_unweighted_avg = torch.stack(phi_unweighted_list).mean(dim=0)
phi_1218_avg_over_sigs = phi_1218_avg.mean(dim=0)  # [D, T]
phi_unweighted_avg_over_sigs = phi_unweighted_avg.mean(dim=0)

pi_1218_avg = torch.stack(pi_1218_list).mean(dim=0)  # [D, T]
pi_unweighted_avg = torch.stack(pi_unweighted_list).mean(dim=0)

# Concatenate lambda across batches for individual comparison
lambda_1218_all = torch.cat(lambda_1218_list, dim=0)  # [N_total, K, T]
lambda_unweighted_all = torch.cat(lambda_unweighted_list, dim=0)

print(f"   ✓ Loaded and processed {n_batches} batches")

# ============================================================================
# PART 3: Create combined figure
# ============================================================================
print("\n4. Creating combined S29 figure...")

# Create figure with multiple sections
# Layout: 
# - Top: Weights distribution (1 panel)
# - Middle: Phi/Pi/Prevalence (4 diseases × 3 columns = 12 subplots)
# - Bottom: Lambda comparison (2 rows × 3 columns = 6 panels)

fig = plt.figure(figsize=(22, 30))

# Create gridspec for flexible layout
# Total rows: 1 (weights) + 1 (column titles) + 4 (diseases) + 2 (lambda) = 8 rows
from matplotlib.gridspec import GridSpec
gs = GridSpec(8, 3, figure=fig, hspace=0.4, wspace=0.35, 
              left=0.06, right=0.96, top=0.97, bottom=0.03,
              height_ratios=[0.7, 0.25, 1.1, 1.1, 1.1, 1.1, 1.3, 1.3])

# ===== SECTION 1: IPW Weights Distribution =====
ax_weights = fig.add_subplot(gs[0, :])
ax_weights.hist(weights, bins=100, alpha=0.75, edgecolor='black', linewidth=0.5, color='#4A90E2')
ax_weights.axvline(weights.mean(), color='#E74C3C', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {weights.mean():.3f}')
ax_weights.axvline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Unweighted (1.0)')
ax_weights.set_xlabel('IPW Weight', fontsize=13, fontweight='bold')
ax_weights.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax_weights.set_title('IPW Weights Distribution', fontsize=15, fontweight='bold', pad=10)
ax_weights.legend(fontsize=11, framealpha=0.9, loc='upper right')
ax_weights.grid(True, alpha=0.3, linestyle='--')
ax_weights.tick_params(labelsize=11)
ax_weights.text(0.98, 0.98, 
               f'N = {len(weights):,}\nMean = {weights.mean():.3f}\nStd = {weights.std():.3f}\n'
               f'Min = {weights.min():.3f}\nMax = {weights.max():.3f}',
               transform=ax_weights.transAxes, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='darkorange', linewidth=2), 
               fontsize=11, fontweight='bold')

# ===== SECTION 2: Phi/Pi/Prevalence Comparison =====
DISEASES_TO_PLOT = [
    (112, "Myocardial Infarction"),  # MI
    (47, "Type 2 diabetes"),  # DM
    (16, "Breast cancer [female]"),  # BC
    (127, "Atrial fibrillation"),  # AFib
]

disease_names_dict = {}
try:
    disease_names_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/disease_names.csv")
    if disease_names_path.exists():
        disease_df = pd.read_csv(disease_names_path)
        disease_names_dict = dict(zip(disease_df['index'], disease_df['name']))
except:
    pass

time_points = np.arange(phi_1218_avg_over_sigs.shape[1]) + 30

# Calculate overall correlations
phi_correlation = np.corrcoef(phi_1218_avg_over_sigs.numpy().flatten(), 
                             phi_unweighted_avg_over_sigs.numpy().flatten())[0, 1]
pi_correlation = np.corrcoef(pi_1218_avg.numpy().flatten(), 
                             pi_unweighted_avg.numpy().flatten())[0, 1]
prev_correlation = np.corrcoef(prevalence_t_weighted.flatten(), 
                               prevalence_t_unweighted.flatten())[0, 1]

# Add column titles (row 1, after weights)
ax_phi_title = fig.add_subplot(gs[1, 0])
ax_phi_title.axis('off')
ax_phi_title.text(0.5, 0.5, 'Phi: Weighted vs Unweighted\n(Same Init, Averaged over All Signatures)', 
                 transform=ax_phi_title.transAxes, ha='center', va='center',
                 fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='darkgreen', linewidth=2))

ax_pi_title = fig.add_subplot(gs[1, 1])
ax_pi_title.axis('off')
ax_pi_title.text(0.5, 0.5, 'Pi: Weighted vs Unweighted\n(Same Init, Lambda Adapts)', 
                 transform=ax_pi_title.transAxes, ha='center', va='center',
                 fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='darkorange', linewidth=2))

ax_prev_title = fig.add_subplot(gs[1, 2])
ax_prev_title.axis('off')
ax_prev_title.text(0.5, 0.5, 'Prevalence: Weighted vs Unweighted\n(All 400K, Population Demographics)', 
                 transform=ax_prev_title.transAxes, ha='center', va='center',
                 fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='darkblue', linewidth=2))

for idx, (disease_idx, disease_name) in enumerate(DISEASES_TO_PLOT):
    if disease_idx >= phi_1218_avg_over_sigs.shape[0]:
        continue
    
    display_name = disease_names_dict.get(disease_idx, disease_name)
    row_idx = idx + 2  # Row 0 is weights, row 1 is column titles, rows 2-5 are diseases
    
    # Column 1: Phi
    ax1 = fig.add_subplot(gs[row_idx, 0])
    
    phi_1218_traj = phi_1218_avg_over_sigs[disease_idx, :].numpy()
    phi_unweighted_traj = phi_unweighted_avg_over_sigs[disease_idx, :].numpy()
    
    # Calculate correlation for this disease
    phi_corr_disease = np.corrcoef(phi_1218_traj, phi_unweighted_traj)[0, 1]
    phi_mean_diff_disease = np.abs(phi_1218_traj - phi_unweighted_traj).mean()
    
    ax1.plot(time_points, phi_unweighted_traj, linewidth=2.5, alpha=0.85, color='#2E86AB', label='Unweighted' if idx == 0 else '')
    ax1.plot(time_points, phi_1218_traj, linewidth=2.5, alpha=0.85, linestyle='--', color='#A23B72', label='Weighted (IPW)' if idx == 0 else '')
    
    if idx == 0:
        ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    if idx == len(DISEASES_TO_PLOT) - 1:
        ax1.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Phi', fontsize=11, fontweight='bold')
    ax1.set_title(display_name, fontsize=12, fontweight='bold', pad=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=10)
    
    # Add correlation text box
    ax1.text(0.02, 0.98, f'Corr: {phi_corr_disease:.4f}\nMean diff: {phi_mean_diff_disease:.4f}', 
             transform=ax1.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1.5))
    
    # Column 2: Pi
    ax2 = fig.add_subplot(gs[row_idx, 1])
    
    pi_1218_traj = pi_1218_avg[disease_idx, :].numpy()
    pi_unweighted_traj = pi_unweighted_avg[disease_idx, :].numpy()
    
    # Calculate correlation for this disease
    pi_corr_disease = np.corrcoef(pi_1218_traj, pi_unweighted_traj)[0, 1]
    pi_mean_diff_disease = np.abs(pi_1218_traj - pi_unweighted_traj).mean()
    
    ax2.plot(time_points, pi_unweighted_traj, linewidth=2.5, alpha=0.85, color='#2E86AB')
    ax2.plot(time_points, pi_1218_traj, linewidth=2.5, alpha=0.85, linestyle='--', color='#A23B72')
    
    if idx == len(DISEASES_TO_PLOT) - 1:
        ax2.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Pi', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_title(display_name, fontsize=12, fontweight='bold', pad=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=10)
    
    # Add correlation text box
    ax2.text(0.02, 0.98, f'Corr: {pi_corr_disease:.4f}\nMean diff: {pi_mean_diff_disease:.4f}', 
             transform=ax2.transAxes, verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='darkorange', linewidth=1.5))
    
    # Column 3: Prevalence
    ax3 = fig.add_subplot(gs[row_idx, 2])
    
    if disease_idx < prevalence_t_weighted.shape[0]:
        weighted_prev_traj = prevalence_t_weighted[disease_idx, :]
        unweighted_prev_traj = prevalence_t_unweighted[disease_idx, :]
        min_T = min(len(weighted_prev_traj), len(time_points))
        
        # Calculate correlation for this disease
        prev_corr_disease = np.corrcoef(weighted_prev_traj[:min_T], unweighted_prev_traj[:min_T])[0, 1]
        prev_mean_diff_disease = np.abs(weighted_prev_traj[:min_T] - unweighted_prev_traj[:min_T]).mean()
        
        ax3.plot(time_points[:min_T], unweighted_prev_traj[:min_T], linewidth=2.5, alpha=0.85, color='#2E86AB')
        ax3.plot(time_points[:min_T], weighted_prev_traj[:min_T], linewidth=2.5, alpha=0.85, linestyle='--', color='#A23B72')
        
        # Add correlation text box
        ax3.text(0.02, 0.98, f'Corr: {prev_corr_disease:.4f}\nMean diff: {prev_mean_diff_disease:.4f}', 
                 transform=ax3.transAxes, verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5))
    
    if idx == len(DISEASES_TO_PLOT) - 1:
        ax3.set_xlabel('Age', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Prevalence', fontsize=11, fontweight='bold')
    ax3.set_yscale('log')
    ax3.set_title(display_name, fontsize=12, fontweight='bold', pad=8)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=10)

# Add overall correlation text (positioned after weights section)
fig.text(0.5, 0.965, 
        f'Overall Correlations:  Phi = {phi_correlation:.4f} (STABLE)  |  '
        f'Pi = {pi_correlation:.4f} (CAN CHANGE)  |  '
        f'Prevalence = {prev_correlation:.4f} (CAN CHANGE)',
        ha='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5))

# ===== SECTION 3: Lambda Comparison =====
# Compute statistics
weighted_lambda_flat = lambda_1218_all.numpy().flatten()
unweighted_lambda_flat = lambda_unweighted_all.numpy().flatten()
lambda_correlation = np.corrcoef(weighted_lambda_flat, unweighted_lambda_flat)[0, 1]
lambda_mean_diff = np.abs(weighted_lambda_flat - unweighted_lambda_flat).mean()

# Average lambda for heatmaps
weighted_lambda_avg = lambda_1218_all.mean(dim=0)  # [K, T]
unweighted_lambda_avg = lambda_unweighted_all.mean(dim=0)
lambda_diff_avg = weighted_lambda_avg - unweighted_lambda_avg

# Variance difference
weighted_lambda_var = lambda_1218_all.var(dim=0)
unweighted_lambda_var = lambda_unweighted_all.var(dim=0)
lambda_var_diff = weighted_lambda_var - unweighted_lambda_var

# Panel 1: Scatter plot
ax_l1 = fig.add_subplot(gs[6, 0])
n_sample = min(50000, len(weighted_lambda_flat))
sample_idx = np.random.choice(len(weighted_lambda_flat), n_sample, replace=False)
ax_l1.scatter(unweighted_lambda_flat[sample_idx], weighted_lambda_flat[sample_idx], 
             alpha=0.15, s=1, color='#4A90E2', edgecolors='none')
ax_l1.plot([unweighted_lambda_flat.min(), unweighted_lambda_flat.max()], 
          [unweighted_lambda_flat.min(), unweighted_lambda_flat.max()], 
          'r--', alpha=0.8, linewidth=2.5, label='y=x')
ax_l1.set_xlabel('Unweighted Lambda', fontsize=11, fontweight='bold')
ax_l1.set_ylabel('Weighted (IPW) Lambda', fontsize=11, fontweight='bold')
ax_l1.set_title(f'Individual Lambda: All N×K×T\nCorrelation: {lambda_correlation:.4f}', 
               fontsize=12, fontweight='bold', pad=8)
ax_l1.grid(True, alpha=0.3, linestyle='--')
ax_l1.tick_params(labelsize=10)
ax_l1.legend(fontsize=9, framealpha=0.9)

# Panel 2: Distribution of differences
ax_l2 = fig.add_subplot(gs[6, 1])
diff_flat = weighted_lambda_flat - unweighted_lambda_flat
ax_l2.hist(diff_flat, bins=100, alpha=0.75, edgecolor='black', linewidth=0.5, color='#4A90E2')
ax_l2.axvline(0, color='#E74C3C', linestyle='--', linewidth=2.5, label='No difference')
ax_l2.axvline(lambda_mean_diff, color='orange', linestyle=':', linewidth=2, label=f'Mean: {lambda_mean_diff:.4f}')
ax_l2.set_xlabel('Lambda Difference (Weighted - Unweighted)', fontsize=11, fontweight='bold')
ax_l2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax_l2.set_title(f'Distribution of Individual Differences\nMean: {lambda_mean_diff:.4f}', 
               fontsize=12, fontweight='bold', pad=8)
ax_l2.legend(fontsize=10, framealpha=0.9)
ax_l2.grid(True, alpha=0.3, linestyle='--')
ax_l2.tick_params(labelsize=10)

# Panel 3: Mean difference heatmap
ax_l3 = fig.add_subplot(gs[6, 2])
im3 = ax_l3.imshow(lambda_diff_avg.numpy(), aspect='auto', cmap='RdBu_r', 
                  vmin=-lambda_diff_avg.abs().max().item(), 
                  vmax=lambda_diff_avg.abs().max().item())
ax_l3.set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
ax_l3.set_ylabel('Signature', fontsize=11, fontweight='bold')
ax_l3.set_title('Mean Lambda Difference\n(Averaged across patients)', 
               fontsize=12, fontweight='bold', pad=8)
cbar3 = plt.colorbar(im3, ax=ax_l3, label='Difference')
cbar3.ax.tick_params(labelsize=9)
ax_l3.tick_params(labelsize=10)

# Panel 4: Variance difference heatmap
ax_l4 = fig.add_subplot(gs[7, 0])
im4 = ax_l4.imshow(lambda_var_diff.numpy(), aspect='auto', cmap='RdBu_r',
                  vmin=-lambda_var_diff.abs().max().item(),
                  vmax=lambda_var_diff.abs().max().item())
ax_l4.set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
ax_l4.set_ylabel('Signature', fontsize=11, fontweight='bold')
ax_l4.set_title('Variance Difference\n(Shows where differences vary most)', 
               fontsize=12, fontweight='bold', pad=8)
cbar4 = plt.colorbar(im4, ax=ax_l4, label='Variance Difference')
cbar4.ax.tick_params(labelsize=9)
ax_l4.tick_params(labelsize=10)

# Panel 5: Sample signature trajectories
ax_l5 = fig.add_subplot(gs[7, 1])
sample_sigs = [0, 5, 10, 15]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
for i, sig_idx in enumerate(sample_sigs):
    if sig_idx < weighted_lambda_avg.shape[0]:
        unweighted_traj = unweighted_lambda_avg[sig_idx, :].numpy()
        weighted_traj = weighted_lambda_avg[sig_idx, :].numpy()
        unweighted_std = lambda_unweighted_all[:, sig_idx, :].std(dim=0).numpy()
        weighted_std = lambda_1218_all[:, sig_idx, :].std(dim=0).numpy()
        
        color = colors[i % len(colors)]
        ax_l5.plot(unweighted_traj, label=f'Sig {sig_idx} (Unweighted)', 
                  alpha=0.8, linewidth=2, color=color)
        ax_l5.fill_between(range(len(unweighted_traj)), 
                          unweighted_traj - unweighted_std, 
                          unweighted_traj + unweighted_std, alpha=0.2, color=color)
        ax_l5.plot(weighted_traj, label=f'Sig {sig_idx} (Weighted)', 
                  linestyle='--', alpha=0.8, linewidth=2, color=color)

ax_l5.set_xlabel('Time (Age)', fontsize=11, fontweight='bold')
ax_l5.set_ylabel('Lambda Value (Mean ± SD)', fontsize=11, fontweight='bold')
ax_l5.set_title('Lambda Trajectories: Sample Signatures', 
               fontsize=12, fontweight='bold', pad=8)
ax_l5.legend(fontsize=9, ncol=2, framealpha=0.9, loc='best')
ax_l5.grid(True, alpha=0.3, linestyle='--')
ax_l5.tick_params(labelsize=10)

# Panel 6: Correlation by signature
ax_l6 = fig.add_subplot(gs[7, 2])
sig_correlations = []
for sig_idx in range(lambda_1218_all.shape[1]):
    sig_weighted = lambda_1218_all[:, sig_idx, :].numpy().flatten()
    sig_unweighted = lambda_unweighted_all[:, sig_idx, :].numpy().flatten()
    sig_corr = np.corrcoef(sig_weighted, sig_unweighted)[0, 1]
    sig_correlations.append(sig_corr)

bars = ax_l6.bar(range(len(sig_correlations)), sig_correlations, alpha=0.75, 
                edgecolor='black', linewidth=1, color='#4A90E2')
ax_l6.axhline(lambda_correlation, color='#E74C3C', linestyle='--', linewidth=2.5, 
            label=f'Overall: {lambda_correlation:.4f}')
ax_l6.set_xlabel('Signature', fontsize=11, fontweight='bold')
ax_l6.set_ylabel('Correlation', fontsize=11, fontweight='bold')
ax_l6.set_title('Lambda Correlation by Signature', 
               fontsize=12, fontweight='bold', pad=8)
ax_l6.legend(fontsize=10, framealpha=0.9, loc='best')
ax_l6.grid(True, alpha=0.3, axis='y', linestyle='--')
ax_l6.tick_params(labelsize=10)
ax_l6.set_ylim([0, 1.05])

# Overall title
fig.suptitle('S29: IPW Analysis - Weights, Phi Stability, Lambda/Pi Adaptability, and Prevalence Changes', 
            fontsize=17, fontweight='bold', y=0.998)

# Save
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s29')
output_dir.mkdir(parents=True, exist_ok=True)
plot_path = output_dir / 'S29.pdf'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved S29 figure to: {plot_path}")
plt.show()

print(f"\n{'='*80}")
print("S29 COMPLETE")
print("="*80)
print(f"✓ IPW weights distribution")
print(f"✓ Phi/Pi/Prevalence comparison (phi stable, pi/prevalence can change)")
print(f"✓ Lambda comparison (6 panels showing individual differences)")
print(f"\nThis figure demonstrates the full IPW story:")
print(f"  - Weights distribution shows the reweighting scheme")
print(f"  - Phi remains stable (signature structure preserved)")
print(f"  - Lambda/Pi adapts (model adjusts to reweighted population)")
print(f"  - Prevalence changes (population demographics shift)")

