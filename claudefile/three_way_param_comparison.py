#!/usr/bin/env python3
"""
Three-way parameter comparison: nolr vs reparam v1 vs nokappa.
Covers: pooled param correlations, PRS-signature associations, psi stability.
"""

import csv
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def corr(a, b):
    a, b = a.flatten(), b.flatten()
    return np.corrcoef(a, b)[0, 1]

# ============================================================================
# Load data
# ============================================================================
data_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
reparam_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM')
nolr_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_nolr')
nokappa_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa')

nolr_pooled = torch.load(data_dir / 'pooled_phi_kappa_gamma_nolr.pt', weights_only=False)
reparam_pooled = torch.load(data_dir / 'pooled_phi_kappa_gamma_reparam.pt', weights_only=False)
nokappa_pooled = torch.load(data_dir / 'pooled_phi_kappa_gamma_nokappa.pt', weights_only=False)
initial_psi = to_np(torch.load(data_dir / 'initial_psi_400k.pt', weights_only=False))

phi_n, phi_r, phi_k = to_np(nolr_pooled['phi']), to_np(reparam_pooled['phi']), to_np(nokappa_pooled['phi'])
psi_n, psi_r, psi_k = to_np(nolr_pooled['psi']), to_np(reparam_pooled['psi']), to_np(nokappa_pooled['psi'])
gamma_n, gamma_r, gamma_k = to_np(nolr_pooled['gamma']), to_np(reparam_pooled['gamma']), to_np(nokappa_pooled['gamma'])
kappa_n = float(nolr_pooled['kappa'])
kappa_r = float(reparam_pooled['kappa'])
kappa_k = float(nokappa_pooled['kappa'])

disease_csv = Path('/Users/sarahurbut/aladynoulli2/claudefile/aladyn_project/pyScripts_forPublish/disease_names.csv')
disease_names = []
with open(disease_csv) as f:
    for row in csv.DictReader(f):
        disease_names.append(row.get('x', ''))

prs_names = pd.read_csv('/Users/sarahurbut/aladynoulli2/prs_names.csv', header=None).iloc[:, 0].tolist()
feat_names = prs_names + ['Sex'] + [f'PC{i}' for i in range(1, 11)]

SIG_NAMES = {
    0: 'Cardiac Arrhythmias', 1: 'Musculoskeletal', 2: 'Upper GI/Esophageal',
    3: 'Mixed/General', 4: 'Upper Respiratory', 5: 'Ischemic CV',
    6: 'Metastatic Cancer', 7: 'Pain/Inflammation', 8: 'Gynecologic',
    9: 'Spinal', 10: 'Ophthalmologic', 11: 'Cerebrovascular',
    12: 'Renal/Urologic', 13: 'Male Urogenital', 14: 'Pulmonary/Smoking',
    15: 'Metabolic/Diabetes', 16: 'Infectious/Crit Care', 17: 'Lower GI/Colon',
    18: 'Hepatobiliary', 19: 'Dermatologic/Onc', 20: 'Healthy'
}

print(f'phi: nolr={phi_n.shape}, reparam={phi_r.shape}, nokappa={phi_k.shape}')
print(f'psi: nolr={psi_n.shape}, reparam={psi_r.shape}, nokappa={psi_k.shape}')
print(f'gamma: nolr={gamma_n.shape}, reparam={gamma_r.shape}, nokappa={gamma_k.shape}')

# ============================================================================
# 1. Pooled parameter comparison
# ============================================================================
print(f"\n{'='*100}")
print("1. POOLED PARAMETER COMPARISON")
print(f"{'='*100}")

print(f"\n{'Parameter':<10} {'nolr-v1':>8} {'nolr-nk':>8} {'v1-nk':>8} "
      f"{'|nolr|':>8} {'|v1|':>8} {'|nk|':>8}")
print('-' * 65)
for name, a, b, c in [('phi', phi_n, phi_r, phi_k),
                        ('psi', psi_n, psi_r, psi_k),
                        ('gamma', gamma_n, gamma_r, gamma_k)]:
    r_nv = corr(a, b)
    r_nk = corr(a, c)
    r_vk = corr(b, c)
    print(f'{name:<10} {r_nv:>8.3f} {r_nk:>8.3f} {r_vk:>8.3f} '
          f'{np.abs(a).mean():>8.4f} {np.abs(b).mean():>8.4f} {np.abs(c).mean():>8.4f}')
print(f"{'kappa':<10} {'':>8} {'':>8} {'':>8} "
      f'{kappa_n:>8.3f} {kappa_r:>8.3f} {kappa_k:>8.3f}')

# Save comparison CSV
comp_rows = []
for name, a, b, c in [('phi', phi_n, phi_r, phi_k),
                        ('psi', psi_n, psi_r, psi_k),
                        ('gamma', gamma_n, gamma_r, gamma_k)]:
    comp_rows.append({
        'param': name,
        'corr_nolr_v1': round(corr(a, b), 4),
        'corr_nolr_nokappa': round(corr(a, c), 4),
        'corr_v1_nokappa': round(corr(b, c), 4),
        'mean_abs_nolr': round(np.abs(a).mean(), 4),
        'mean_abs_v1': round(np.abs(b).mean(), 4),
        'mean_abs_nokappa': round(np.abs(c).mean(), 4),
    })
comp_rows.append({
    'param': 'kappa',
    'corr_nolr_v1': None, 'corr_nolr_nokappa': None, 'corr_v1_nokappa': None,
    'mean_abs_nolr': round(kappa_n, 4),
    'mean_abs_v1': round(kappa_r, 4),
    'mean_abs_nokappa': round(kappa_k, 4),
})
pd.DataFrame(comp_rows).to_csv(
    Path(__file__).parent / 'three_way_param_comparison.csv', index=False)

# Plot: 3x3 scatter matrix
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
params = [('phi', phi_n, phi_r, phi_k),
          ('psi', psi_n, psi_r, psi_k),
          ('gamma', gamma_n, gamma_r, gamma_k)]
pairs = [('nolr', 'v1', 0, 1), ('nolr', 'nokappa', 0, 2), ('v1', 'nokappa', 1, 2)]

for row, (pname, a, b, c) in enumerate(params):
    vals = [a, b, c]
    labels = ['nolr', 'v1 reparam', 'nokappa']
    colors_p = ['steelblue', 'coral', 'seagreen']
    for col, (l1, l2, i, j) in enumerate(pairs):
        ax = axes[row][col]
        x, y = vals[i].flatten(), vals[j].flatten()
        ax.scatter(x, y, alpha=0.1, s=2, c=colors_p[col])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        r = corr(vals[i], vals[j])
        ax.set_title(f'{pname}: {labels[i]} vs {labels[j]}\nr={r:.3f}', fontsize=10)
        if row == 2:
            ax.set_xlabel(labels[i])
        if col == 0:
            ax.set_ylabel(labels[j])

plt.suptitle('3-Way Pooled Parameter Comparison', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'three_way_param_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: three_way_param_scatter.png")

# ============================================================================
# 2. PRS-Signature associations (batch 0)
# ============================================================================
print(f"\n{'='*100}")
print("2. PRS-SIGNATURE ASSOCIATIONS (batch 0)")
print(f"{'='*100}")

nolr_b0 = torch.load(nolr_dir / 'enrollment_model_VECTORIZED_W0.0001_nolr_batch_0_10000.pt', weights_only=False)
reparam_b0 = torch.load(reparam_dir / 'enrollment_model_REPARAM_W0.0001_batch_0_10000.pt', weights_only=False)
nokappa_files = sorted(glob.glob(str(nokappa_dir / 'enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_0_10000.pt')))
if nokappa_files:
    nokappa_b0 = torch.load(nokappa_files[0], weights_only=False)
else:
    nokappa_files = sorted(glob.glob(str(nokappa_dir / '*batch_0_10000*')))
    nokappa_b0 = torch.load(nokappa_files[0], weights_only=False) if nokappa_files else None

g_n_b0 = nolr_b0['model_state_dict']['gamma'].detach().numpy()
g_r_b0 = reparam_b0['model_state_dict']['gamma'].detach().numpy()
if nokappa_b0 is not None:
    g_k_b0 = (nokappa_b0.get('model_state_dict', nokappa_b0).get('gamma', nokappa_b0.get('gamma')))
    g_k_b0 = g_k_b0.detach().numpy() if torch.is_tensor(g_k_b0) else np.array(g_k_b0)
else:
    g_k_b0 = gamma_k  # fallback to pooled

print(f"Batch 0 |gamma|: nolr={np.abs(g_n_b0).mean():.4f}, "
      f"v1={np.abs(g_r_b0).mean():.4f}, nokappa={np.abs(g_k_b0).mean():.4f}")

# Table: top 3 features per signature for all 3 models
print(f"\n{'Sig':<4} {'Name':<22} {'Centered (nolr)':<35} {'Reparam v1':<35} {'Nokappa':<35}")
print('-' * 135)
for k in range(min(g_n_b0.shape[1], 20)):
    gn, gr, gk = g_n_b0[:, k], g_r_b0[:, k], g_k_b0[:, k]
    top_n = np.argsort(np.abs(gn))[-3:][::-1]
    top_r = np.argsort(np.abs(gr))[-3:][::-1]
    top_k = np.argsort(np.abs(gk))[-3:][::-1]

    nolr_str = ', '.join(f'{feat_names[i]}={gn[i]:.3f}' for i in top_n)
    reparam_str = ', '.join(f'{feat_names[i]}={gr[i]:.3f}' for i in top_r)
    nokappa_str = ', '.join(f'{feat_names[i]}={gk[i]:.3f}' for i in top_k)

    print(f'{k:<4} {SIG_NAMES.get(k, ""):<22} {nolr_str:<35} {reparam_str:<35} {nokappa_str:<35}')

# Plot: Sig 5 (Ischemic CV) and Sig 15 (Metabolic) — three-way bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, k, title in [(axes[0], 5, 'Sig 5: Ischemic CV'),
                      (axes[1], 15, 'Sig 15: Metabolic/Diabetes')]:
    gn, gr, gk = g_n_b0[:len(feat_names), k], g_r_b0[:len(feat_names), k], g_k_b0[:len(feat_names), k]
    importance = np.maximum(np.abs(gn), np.maximum(np.abs(gr), np.abs(gk)))
    top_idx = np.argsort(importance)[-10:][::-1]
    names_top = [feat_names[i] for i in top_idx]
    x = np.arange(len(top_idx))
    w = 0.25

    ax.barh(x - w, gn[top_idx], w, label='Centered (nolr)', color='steelblue', alpha=0.8)
    ax.barh(x, gr[top_idx], w, label='Reparam v1', color='coral', alpha=0.8)
    ax.barh(x + w, gk[top_idx], w, label='Nokappa', color='seagreen', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names_top)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('gamma coefficient')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

plt.suptitle('PRS-Signature Associations: 3-Way Comparison (batch 0)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'three_way_prs_signatures.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: three_way_prs_signatures.png")

# ============================================================================
# 3. Psi stability (cluster assignments)
# ============================================================================
print(f"\n{'='*100}")
print("3. PSI STABILITY (disease -> signature assignments)")
print(f"{'='*100}")

K_dis = 20
init_psi = initial_psi[:K_dis]
argmax_init = np.argmax(init_psi, axis=0)

psi_n_dis = psi_n[:K_dis]
psi_r_dis = psi_r[:K_dis]
psi_k_dis = psi_k[:K_dis]

argmax_nolr = np.argmax(psi_n_dis, axis=0)
argmax_reparam = np.argmax(psi_r_dis, axis=0)
argmax_nokappa = np.argmax(psi_k_dis, axis=0)

flipped_nolr = np.where(argmax_init != argmax_nolr)[0]
flipped_reparam = np.where(argmax_init != argmax_reparam)[0]
flipped_nokappa = np.where(argmax_init != argmax_nokappa)[0]

diff_n = psi_n_dis - init_psi
diff_r = psi_r_dis - init_psi
diff_k = psi_k_dis - init_psi

print(f"\n{'Metric':<40} {'Centered':>12} {'Reparam v1':>12} {'Nokappa':>12}")
print('-' * 80)
print(f"{'Mean |delta psi|':<40} {np.abs(diff_n).mean():>12.4f} {np.abs(diff_r).mean():>12.4f} {np.abs(diff_k).mean():>12.4f}")
print(f"{'Corr(initial, final)':<40} {corr(init_psi, psi_n_dis):>12.4f} {corr(init_psi, psi_r_dis):>12.4f} {corr(init_psi, psi_k_dis):>12.4f}")
print(f"{'Same primary signature':<40} {f'{348-len(flipped_nolr)}/348':>12} {f'{348-len(flipped_reparam)}/348':>12} {f'{348-len(flipped_nokappa)}/348':>12}")
print(f"{'Diseases that flip':<40} {len(flipped_nolr):>12} {len(flipped_reparam):>12} {len(flipped_nokappa):>12}")

# Detail on nokappa flips
if len(flipped_nokappa) > 0:
    print(f"\nNokappa flipped diseases ({len(flipped_nokappa)}):")
    for d in flipped_nokappa:
        sig_init = argmax_init[d]
        sig_final = argmax_nokappa[d]
        d_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
        print(f"  {d_name}: {SIG_NAMES.get(sig_init, sig_init)} -> {SIG_NAMES.get(sig_final, sig_final)}")

# Check overlap: which diseases flip in both reparam and nokappa?
shared_flips = set(flipped_reparam) & set(flipped_nokappa)
if shared_flips:
    print(f"\nShared flips (both reparam & nokappa): {len(shared_flips)}")
    for d in sorted(shared_flips):
        d_name = disease_names[d] if d < len(disease_names) else f'Disease {d}'
        print(f"  {d_name}: init={SIG_NAMES.get(argmax_init[d], argmax_init[d])}, "
              f"v1={SIG_NAMES.get(argmax_reparam[d], argmax_reparam[d])}, "
              f"nk={SIG_NAMES.get(argmax_nokappa[d], argmax_nokappa[d])}")

# Plot: psi change distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(np.abs(diff_n).flatten(), bins=50, alpha=0.5, label=f'Centered ({np.abs(diff_n).mean():.3f})',
        color='steelblue', density=True)
ax.hist(np.abs(diff_r).flatten(), bins=50, alpha=0.5, label=f'Reparam v1 ({np.abs(diff_r).mean():.3f})',
        color='coral', density=True)
ax.hist(np.abs(diff_k).flatten(), bins=50, alpha=0.5, label=f'Nokappa ({np.abs(diff_k).mean():.3f})',
        color='seagreen', density=True)
ax.set_xlabel('|delta psi| (initial -> final)')
ax.set_ylabel('Density')
ax.set_title('Distribution of psi changes from initialization', fontweight='bold')
ax.set_xlim(0, 4)
ax.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'three_way_psi_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: three_way_psi_stability.png")

# Save psi stability CSV
psi_rows = [
    {'metric': 'mean_abs_delta_psi', 'nolr': round(np.abs(diff_n).mean(), 4),
     'reparam_v1': round(np.abs(diff_r).mean(), 4), 'nokappa': round(np.abs(diff_k).mean(), 4)},
    {'metric': 'corr_init_final', 'nolr': round(corr(init_psi, psi_n_dis), 4),
     'reparam_v1': round(corr(init_psi, psi_r_dis), 4), 'nokappa': round(corr(init_psi, psi_k_dis), 4)},
    {'metric': 'same_primary_sig', 'nolr': 348-len(flipped_nolr),
     'reparam_v1': 348-len(flipped_reparam), 'nokappa': 348-len(flipped_nokappa)},
    {'metric': 'n_flipped', 'nolr': len(flipped_nolr),
     'reparam_v1': len(flipped_reparam), 'nokappa': len(flipped_nokappa)},
]
pd.DataFrame(psi_rows).to_csv(
    Path(__file__).parent / 'three_way_psi_stability.csv', index=False)

print(f"\n{'='*100}")
print("DONE")
print(f"{'='*100}")
print("\nOutput files:")
print("  three_way_param_comparison.csv")
print("  three_way_param_scatter.png")
print("  three_way_prs_signatures.png")
print("  three_way_psi_stability.png")
print("  three_way_psi_stability.csv")
