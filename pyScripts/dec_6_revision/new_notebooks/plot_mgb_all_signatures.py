#!/usr/bin/env python3
"""
Plot all MGB signatures with diseases organized in boxes.
For each signature, shows probability trajectories for all diseases in that signature.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import expit as sigmoid

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Load MGB model checkpoint
mgb_path = '/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt'
print(f"Loading MGB model from: {mgb_path}")
mgb_ckpt = torch.load(mgb_path, map_location='cpu', weights_only=False)

# Load clusters
clusters = mgb_ckpt['clusters']
if torch.is_tensor(clusters):
    clusters = clusters.numpy()

# Load disease names
disease_names = mgb_ckpt.get('disease_names', None)
if disease_names is None:
    # Try to load from old checkpoint as fallback
    try:
        mgb_old = torch.load('/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt', 
                            map_location='cpu', weights_only=False)
        disease_names = mgb_old.get('disease_names', None)
    except:
        pass

if disease_names is None:
    # Create default names if not found
    D = len(clusters) if hasattr(clusters, '__len__') else clusters.shape[0] if hasattr(clusters, 'shape') else 0
    disease_names = [f'Disease {i}' for i in range(D)]
    print(f"Warning: disease_names not found, using default names")

if isinstance(disease_names, (list, tuple)):
    disease_names = list(disease_names)
elif hasattr(disease_names, 'values'):
    disease_names = disease_names.values.tolist()

# Load phi from model_state_dict
if 'model_state_dict' in mgb_ckpt and 'phi' in mgb_ckpt['model_state_dict']:
    phi = mgb_ckpt['model_state_dict']['phi']
elif 'phi' in mgb_ckpt:
    phi = mgb_ckpt['phi']
else:
    raise ValueError("No phi found in checkpoint!")

if torch.is_tensor(phi):
    phi = phi.detach().cpu().numpy()

K, D, T = phi.shape
ages = np.arange(30, 30 + T)

print(f"\nMGB Model Info:")
print(f"  Number of signatures: {K}")
print(f"  Number of diseases: {D}")
print(f"  Number of timepoints: {T}")
print(f"  Phi shape: {phi.shape}")

# Group diseases by signature
sig_to_diseases = {}
for d in range(D):
    sig = int(clusters[d])
    if sig not in sig_to_diseases:
        sig_to_diseases[sig] = []
    sig_to_diseases[sig].append(d)

# Create figure with subplots for each signature
# Arrange in a grid: 5 columns, 5 rows (adjust as needed)
n_cols = 5
n_rows = int(np.ceil(K / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)
axes = axes.flatten()

for sig in range(K):
    ax = axes[sig]
    diseases_in_sig = sig_to_diseases.get(sig, [])
    
    if len(diseases_in_sig) == 0:
        ax.text(0.5, 0.5, f'Signature {sig}\n(No diseases)', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(f'Signature {sig}', fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    
    # Find top 5 diseases by maximum probability
    disease_max_probs = []
    for d in diseases_in_sig:
        prob_mean = sigmoid(phi[sig, d, :])
        max_prob = prob_mean.max()
        max_prob_age_idx = np.argmax(prob_mean)
        disease_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
        disease_max_probs.append((d, max_prob, max_prob_age_idx, disease_name, prob_mean))
    
    # Sort by max probability and get top 5
    disease_max_probs.sort(key=lambda x: x[1], reverse=True)
    top_5_diseases = disease_max_probs[:5]
    other_diseases = disease_max_probs[5:]
    
    # Colors for top diseases
    top_colors = sns.color_palette("tab10", 5)
    
    # Plot top 5 diseases with labels
    for idx, (d, max_prob, max_age_idx, disease_name, prob_mean) in enumerate(top_5_diseases):
        color = top_colors[idx]
        ax.plot(ages, prob_mean, linewidth=2, alpha=0.8, color=color, 
               label=f'{disease_name[:35]}')
        
        # Add text label at peak
        max_age = ages[max_age_idx]
        ax.text(max_age, max_prob, f'{idx+1}', 
               fontsize=8, fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor=color, alpha=0.8),
               ha='center', va='bottom')
    
    # Plot other diseases in gray
    for d, max_prob, max_age_idx, disease_name, prob_mean in other_diseases:
        ax.plot(ages, prob_mean, linewidth=0.8, alpha=0.3, color='lightgray')
    
    ax.set_title(f'Signature {sig} (n={len(diseases_in_sig)} diseases)', 
                fontsize=11, fontweight='bold')
    ax.set_xlabel('Age (yr)', fontsize=9)
    ax.set_ylabel('Prob (disease | sig k, age)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([30, 81])
    ax.set_ylim([0, None])
    
    # Add legend for top 5 diseases
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=1)

# Remove extra subplots
for sig in range(K, len(axes)):
    axes[sig].axis('off')

plt.suptitle('MGB: Disease Probabilities by Signature', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_path = '/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/mgb_all_signatures_probabilities.pdf'
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_path}")
plt.show()

# Print summary
print(f"\n{'='*60}")
print("SIGNATURE SUMMARY")
print(f"{'='*60}")
for sig in sorted(sig_to_diseases.keys()):
    diseases = sig_to_diseases[sig]
    print(f"Signature {sig}: {len(diseases)} diseases")
    if len(diseases) > 0:
        # Show top 5 diseases by max probability
        max_probs = []
        for d in diseases:
            prob_mean = sigmoid(phi[sig, d, :])
            max_probs.append((d, prob_mean.max()))
        max_probs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Top diseases:")
        for d, max_prob in max_probs[:5]:
            name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
            print(f"    {name[:50]}: max prob = {max_prob:.6f}")