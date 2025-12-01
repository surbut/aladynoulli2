#!/usr/bin/env python3
"""
Signature Loadings by Ancestry - Continuous Analysis

Creates visualizations showing signature loadings vs. ancestry probability (pred)
for the 5 most variable signatures between populations.

For each ancestry (AFR, EAS, EUR, SAS), creates a figure showing:
- X-axis: pred (percent ancestry/probability, 0-1)
- Y-axis: Average deviation on signature X over baseline
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import f_oneway
from scipy.special import softmax

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Paths
ANCESTRY_PATH = Path('/Users/sarahurbut/aladynoulli2/ukb.kgp_projected.tsv')
THETA_PATH = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt')
OUTPUT_DIR = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/ancestry_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SIGNATURE LOADINGS BY ANCESTRY - CONTINUOUS ANALYSIS")
print("="*80)

# 1. Load ancestry data
print("\n1. Loading ancestry data...")
ancestry_df = pd.read_csv(ANCESTRY_PATH, sep='\t')
print(f"   ✓ Loaded {len(ancestry_df):,} rows")
print(f"   Columns: {list(ancestry_df.columns)}")

# Filter to patients with ancestry predictions
ancestry_df = ancestry_df[ancestry_df['rf'].notna() & ancestry_df['pred'].notna()]
print(f"   ✓ {len(ancestry_df):,} patients with ancestry predictions")

# 2. Load signature loadings (theta) - using thetas with PCs
print("\n2. Loading signature loadings (with PCs)...")
thetas = torch.load(THETA_PATH, map_location='cpu')
if hasattr(thetas, 'numpy'):
    thetas = thetas.numpy()
elif isinstance(thetas, torch.Tensor):
    thetas = thetas.numpy()
print(f"   ✓ Loaded theta shape: {thetas.shape} (patients × signatures × time)")

# Calculate average signature loadings per patient (across time)
print("\n3. Calculating average signature loadings per patient...")
avg_signature_loadings = thetas.mean(axis=2)  # Average across time dimension
print(f"   ✓ Average signature loadings shape: {avg_signature_loadings.shape} (patients × signatures)")

# 4. Match ancestry data to signature loadings
print("\n4. Matching ancestry data to signature loadings...")
# Assuming eid corresponds to patient index, or we need to load processed_ids
# For now, assume they're aligned or we need to load processed_ids
PROCESSED_IDS_PATH = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/processed_ids.npy')
try:
    processed_ids = np.load(PROCESSED_IDS_PATH)
    print(f"   ✓ Loaded processed_ids: {len(processed_ids):,}")
    
    # Create mapping from eid to index
    ancestry_df['eid_numeric'] = pd.to_numeric(ancestry_df['eid'], errors='coerce')
    eid_to_idx = {int(eid): idx for idx, eid in enumerate(processed_ids[:len(avg_signature_loadings)])}
    
    # Match ancestry to signature loadings
    matched_indices = []
    matched_ancestry = []
    matched_pred = []
    
    for idx in range(len(avg_signature_loadings)):
        eid = int(processed_ids[idx])
        ancestry_row = ancestry_df[ancestry_df['eid_numeric'] == eid]
        if len(ancestry_row) > 0:
            matched_indices.append(idx)
            matched_ancestry.append(ancestry_row.iloc[0]['rf'])
            matched_pred.append(ancestry_row.iloc[0]['pred'])
    
    matched_indices = np.array(matched_indices)
    matched_ancestry = np.array(matched_ancestry)
    matched_pred = np.array(matched_pred)
    
    print(f"   ✓ Matched {len(matched_indices):,} patients ({len(matched_indices)/len(avg_signature_loadings)*100:.1f}%)")
    
except FileNotFoundError:
    print(f"   ⚠️  Processed IDs file not found. Assuming direct alignment...")
    # Assume direct alignment if processed_ids not available
    n_patients = min(len(ancestry_df), len(avg_signature_loadings))
    matched_indices = np.arange(n_patients)
    matched_ancestry = ancestry_df['rf'].values[:n_patients]
    matched_pred = ancestry_df['pred'].values[:n_patients]

# 5. Calculate baseline (population average) for each signature
print("\n5. Calculating baseline (population average) for each signature...")
baseline_loadings = avg_signature_loadings.mean(axis=0)  # Average across all patients
print(f"   ✓ Baseline shape: {baseline_loadings.shape} (signatures)")

# 6. Calculate deviations from baseline
print("\n6. Calculating deviations from baseline...")
deviations = avg_signature_loadings[matched_indices] - baseline_loadings  # Shape: (matched_patients, signatures)
print(f"   ✓ Deviations shape: {deviations.shape}")

# 7. Identify 5 most variable signatures between populations
print("\n7. Identifying 5 most variable signatures between populations...")
ancestry_groups = ['AFR', 'EAS', 'EUR', 'SAS']
signature_variances = []

for sig_idx in range(deviations.shape[1]):
    # Get deviations for this signature across all ancestry groups
    sig_deviations_by_ancestry = []
    for ancestry in ancestry_groups:
        mask = matched_ancestry == ancestry
        if mask.sum() > 0:
            sig_deviations_by_ancestry.append(deviations[mask, sig_idx])
    
    # Calculate F-statistic (variance between groups)
    if len(sig_deviations_by_ancestry) >= 2:
        try:
            f_stat, p_val = f_oneway(*sig_deviations_by_ancestry)
            signature_variances.append({
                'signature': sig_idx,
                'f_statistic': f_stat,
                'p_value': p_val,
                'variance': np.var([np.mean(group) for group in sig_deviations_by_ancestry])
            })
        except:
            signature_variances.append({
                'signature': sig_idx,
                'f_statistic': 0,
                'p_value': 1.0,
                'variance': 0
            })

variance_df = pd.DataFrame(signature_variances)
top_5_signatures = variance_df.nlargest(5, 'f_statistic')['signature'].values
print(f"   ✓ Top 5 most variable signatures: {top_5_signatures}")

# 8. Create visualizations for each ancestry
print("\n8. Creating visualizations...")

ancestries_to_plot = ['AFR', 'EAS', 'EUR', 'SAS']

for ancestry in ancestries_to_plot:
    print(f"\n   Creating figure for {ancestry}...")
    
    # Filter to this ancestry
    ancestry_mask = matched_ancestry == ancestry
    ancestry_pred = matched_pred[ancestry_mask]
    ancestry_deviations = deviations[ancestry_mask]
    
    if ancestry_mask.sum() < 10:
        print(f"   ⚠️  Skipping {ancestry}: only {ancestry_mask.sum()} patients")
        continue
    
    # Create figure with 5 subplots (one per signature)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for plot_idx, sig_idx in enumerate(top_5_signatures):
        ax = axes[plot_idx]
        
        # Get deviations for this signature
        sig_deviations = ancestry_deviations[:, sig_idx]
        
        # Create scatter plot with smooth trend line
        ax.scatter(ancestry_pred, sig_deviations, alpha=0.3, s=10, color='steelblue')
        
        # Add LOESS/smooth trend line
        from scipy.interpolate import UnivariateSpline
        # Sort by pred for smooth line
        sort_idx = np.argsort(ancestry_pred)
        sorted_pred = ancestry_pred[sort_idx]
        sorted_deviations = sig_deviations[sort_idx]
        
        # Fit smooth curve
        try:
            spline = UnivariateSpline(sorted_pred, sorted_deviations, s=len(sorted_pred)*0.1)
            pred_smooth = np.linspace(sorted_pred.min(), sorted_pred.max(), 100)
            deviations_smooth = spline(pred_smooth)
            ax.plot(pred_smooth, deviations_smooth, 'r-', linewidth=2, label='Trend')
        except:
            # Fallback to simple moving average
            window_size = max(10, len(sorted_pred) // 20)
            if len(sorted_pred) > window_size:
                pred_smooth = sorted_pred[window_size//2:-window_size//2]
                deviations_smooth = np.convolve(sorted_deviations, np.ones(window_size)/window_size, mode='valid')
                ax.plot(pred_smooth, deviations_smooth, 'r-', linewidth=2, label='Trend')
        
        # Add horizontal line at y=0 (baseline)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Ancestry Probability (pred)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Deviation from Baseline', fontsize=10, fontweight='bold')
        ax.set_title(f'Signature {sig_idx}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle(f'Signature Loadings vs. Ancestry Probability: {ancestry} (n={ancestry_mask.sum():,})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = OUTPUT_DIR / f'signature_loadings_by_ancestry_{ancestry}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {fig_path}")
    plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\n✓ Created 4 figures (one per ancestry)")
print(f"✓ Analyzed {len(matched_indices):,} patients")
print(f"✓ Top 5 most variable signatures: {top_5_signatures}")
print(f"\nOutput directory: {OUTPUT_DIR}")

