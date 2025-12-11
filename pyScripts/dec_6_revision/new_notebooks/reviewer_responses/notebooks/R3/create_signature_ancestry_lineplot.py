#!/usr/bin/env python3
"""
Create signature ancestry line plot (No-PCs vs With-PCs) and save as PDF
This creates the 8-panel figure showing signature deviations over age for different ancestries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ----------------------------
# Inputs
# ----------------------------
ancestry_path = '/Users/sarahurbut/aladynoulli2/ukb.kgp_projected.tsv'   # cols: eid, rf80
processed_ids_path = '/Users/sarahurbut/aladynoulli2/pyScripts/notebook2/processed_patient_ids.npy'
thetas_nopcs = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/pt/new_thetas_with_sex_nopcs_retrospective.pt', map_location='cpu')
if torch.is_tensor(thetas_nopcs):
    thetas_nopcs = thetas_nopcs.detach().numpy()
thetas_withpcs = torch.load('/Users/sarahurbut/aladynoulli2/pyScripts/pt/new_thetas_with_pcs_retrospective.pt', map_location='cpu')
if torch.is_tensor(thetas_withpcs):
    thetas_withpcs = thetas_withpcs.detach().numpy()
reference_theta_csv = '/Users/sarahurbut/dtwin_noulli/reference_thetas.csv'  # K x T (or >= T)

# X-axis as ages
use_age_axis = True
age_start = 30  # baseline age for t=0

# Output paths
# Local output directory
output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/figures')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'signature_ancestry_lineplot.pdf'

# Also save to Dropbox Overleaf figures folder if it exists
dropbox_figures_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/figures')
dropbox_output_path = None
if dropbox_figures_dir.exists():
    dropbox_output_path = dropbox_figures_dir / 'signature_ancestry_lineplot.pdf'

print("="*80)
print("CREATING SIGNATURE ANCESTRY LINE PLOT")
print("="*80)

# Load metadata
print("\n1. Loading ancestry data...")
ancestry = pd.read_csv(ancestry_path, sep='\t', usecols=['eid', 'rf80']).drop_duplicates('eid')
processed_ids = np.load(processed_ids_path).astype(int)
df = pd.DataFrame({'eid': processed_ids}).merge(ancestry, on='eid', how='left')
print(f"   ✓ Loaded {len(df):,} patients with ancestry data")

# Shapes
N, K, T = thetas_nopcs.shape
assert thetas_withpcs.shape == (N, K, T)
assert K == 21, f"Expected 21 signatures, got {K}"
print(f"   ✓ Theta shapes: {thetas_nopcs.shape} (patients × signatures × time)")

# Reference (K x T). If CSV missing/short, fallback to zeros to show raw means.
print("\n2. Loading reference theta...")
try:
    ref_theta = pd.read_csv(reference_theta_csv, header=0).values  # shape (K, T_ref)
    if ref_theta.shape[1] >= T:
        ref_slice = ref_theta[:, -T:]  # use last T columns
    else:
        pad = np.zeros((K, T - ref_theta.shape[1]))
        ref_slice = np.concatenate([ref_theta, pad], axis=1)
    print(f"   ✓ Loaded reference theta: {ref_slice.shape}")
except Exception as e:
    print(f"   ⚠️  Could not load reference theta: {e}")
    print("   Using zeros as reference (will show raw means)")
    ref_slice = np.zeros((K, T))

# Choose ancestries (columns). Order them explicitly if you like.
print("\n3. Identifying major ancestries...")
min_samples = 1000
anc_counts = df['rf80'].value_counts(dropna=True)
major_ancestries = [anc for anc, n in anc_counts.items() if n >= min_samples]
# Example order preference if present:
preferred_order = ['AFR', 'EAS', 'SAS', 'EUR', 'AMR', 'MID', 'OTH']
ancestries_to_show = [a for a in preferred_order if a in major_ancestries]
if not ancestries_to_show:
    ancestries_to_show = sorted(major_ancestries)
n_anc = len(ancestries_to_show)
assert n_anc > 0, "No ancestries with sufficient samples."
print(f"   ✓ Found {n_anc} ancestries with ≥{min_samples} samples: {ancestries_to_show}")

# Consistent colors per signature
tab20 = plt.get_cmap('tab20').colors
extra_color = plt.get_cmap('tab10').colors[0]
palette_21 = list(tab20[:20]) + [extra_color]
sig_color = {k: palette_21[k % len(palette_21)] for k in range(K)}

# X-axis
x = np.arange(T)
if use_age_axis:
    x = age_start + x
x_label = 'Age (years)' if use_age_axis else 'Time index'

# Figure: 2 rows (No-PCs, With-PCs) × n_anc columns
print("\n4. Creating figure...")
fig, axes = plt.subplots(2, n_anc, figsize=(4.6 * n_anc, 7.6), sharex=True, sharey=True)

for j, anc in enumerate(ancestries_to_show):
    ax_top = axes[0, j]
    ax_bot = axes[1, j]

    idx = df.index[df['rf80'] == anc].values
    idx = idx[idx < N]
    if len(idx) == 0:
        ax_top.text(0.5, 0.5, f'No data for {anc}', ha='center', va='center', transform=ax_top.transAxes)
        ax_top.axis('off'); ax_bot.axis('off')
        continue

    # Mean trajectories per ancestry
    mean_nopcs = np.nanmean(thetas_nopcs[idx, :, :], axis=0)      # [K, T]
    mean_withpcs = np.nanmean(thetas_withpcs[idx, :, :], axis=0)  # [K, T]

    # Deviations from reference
    dev_nopcs = mean_nopcs - ref_slice
    dev_withpcs = mean_withpcs - ref_slice

    # Plot: one line per signature, same color, solid vs dashed not needed (rows separate sets)
    for k in range(K):
        c = sig_color[k]
        ax_top.plot(x, dev_nopcs[k], color=c, linewidth=1.6, alpha=0.95)
        ax_bot.plot(x, dev_withpcs[k], color=c, linewidth=1.6, alpha=0.95)

    ax_top.axhline(0, color='#888', lw=1, alpha=0.4)
    ax_bot.axhline(0, color='#888', lw=1, alpha=0.4)
    ax_top.set_title(f'{anc} — No PCs (Δθ)', fontsize=12, fontweight='bold')
    ax_bot.set_title(f'{anc} — With PCs (Δθ)', fontsize=12, fontweight='bold')
    ax_top.grid(True, alpha=0.25); ax_bot.grid(True, alpha=0.25)

# Labels
for j in range(n_anc):
    axes[1, j].set_xlabel(x_label, fontsize=11)
axes[0, 0].set_ylabel('Deviation from Reference (Δθ)', fontsize=11)
axes[1, 0].set_ylabel('Deviation from Reference (Δθ)', fontsize=11)

# Signature legend (colors only) – place below
legend_handles = [plt.Line2D([0], [0], color=sig_color[k], lw=2, label=f'Sig {k}') for k in range(K)]
by_label = {h.get_label(): h for h in legend_handles}
sig_legend = list(by_label.values())
fig.legend(handles=sig_legend, loc='lower center', ncol=min(7, K), bbox_to_anchor=(0.5, -0.02),
           frameon=True, title='Signatures', fontsize=9)

plt.tight_layout(rect=[0.03, 0.03, 1, 0.98])

# Save as PDF
print(f"\n5. Saving figure to PDF...")
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"   ✓ Saved to: {output_path}")

# Also save to Dropbox if directory exists
if dropbox_output_path is not None:
    plt.savefig(dropbox_output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"   ✓ Also saved to Dropbox: {dropbox_output_path}")

# Also save as PNG for reference
png_path = output_dir / 'signature_ancestry_lineplot.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
print(f"   ✓ Also saved PNG to: {png_path}")

print("\n" + "="*80)
print("✅ FIGURE CREATION COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  - PDF (local): {output_path}")
if dropbox_output_path is not None:
    print(f"  - PDF (Dropbox): {dropbox_output_path}")
print(f"  - PNG (local): {png_path}")

plt.close()

