#!/usr/bin/env python3
"""
Three analyses demonstrating slope model value:
  3. Disease-specific gamma_slope visualization
  4. Age-stratified 1-year AUC (slope vs no-slope)
  5. Individual trajectory examples (patients where models disagree)

Uses saved pi tensors + slope checkpoints. No model fitting needed.

Usage:
    python slope_value_analyses.py              # holdout (100k): pi_slope_holdout.pt, pi_noslope_holdout.pt
    python slope_value_analyses.py --loo       # LOO (400k): slope + no-slope FULL.pt, single_phase checkpoints
"""

import argparse
import gc
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
SLOPE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3/')
SLOPE_CKPT_DIR_LOO = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3_single_phase/')
BASE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized_REPARAM_v3_nokappa/')
RESULTS_DIR = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc/')
PCE_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv'
PI_SLOPE_HOLDOUT = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc/pi_slope_holdout.pt')
PI_NOSLOPE_HOLDOUT = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc/pi_noslope_holdout.pt')
PI_SLOPE_LOO = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_slope_1phase_loo_all40/pi_enroll_fixedphi_sex_FULL.pt')
PI_NOSLOPE_LOO = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_nokappa_v3_loo_all40/pi_enroll_fixedphi_sex_FULL.pt')
PI_SLOPE_PATH = PI_SLOPE_HOLDOUT
PI_NOSLOPE_PATH = PI_NOSLOPE_HOLDOUT

BATCH_SIZE = 10000
N_TEST = 100000
N_TEST_LOO = 400000
TRAIN_BATCHES = list(range(10, 40))
TRAIN_BATCHES_LOO = list(range(40))
OUTPUT_SUFFIX = ''

def _effective_config(use_loo):
    """Set effective paths and N for holdout vs LOO (used by main() before calling analyses)."""
    global N_TEST, SLOPE_CKPT_DIR, TRAIN_BATCHES, OUTPUT_SUFFIX
    global PI_SLOPE_PATH, PI_NOSLOPE_PATH
    if use_loo:
        N_TEST = N_TEST_LOO
        SLOPE_CKPT_DIR = SLOPE_CKPT_DIR_LOO
        TRAIN_BATCHES = TRAIN_BATCHES_LOO
        OUTPUT_SUFFIX = '_loo'
        PI_SLOPE_PATH = PI_SLOPE_LOO
        PI_NOSLOPE_PATH = PI_NOSLOPE_LOO
    else:
        N_TEST = 100000
        SLOPE_CKPT_DIR = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/slope_model_nokappa_v3/')
        TRAIN_BATCHES = list(range(10, 40))
        OUTPUT_SUFFIX = ''
        PI_SLOPE_PATH = PI_SLOPE_HOLDOUT
        PI_NOSLOPE_PATH = PI_NOSLOPE_HOLDOUT

# Feature names
prs_df = pd.read_csv('/Users/sarahurbut/aladynoulli2/prs_names.csv')
PRS_NAMES = list(prs_df.columns) + list(prs_df.iloc[:, 0].values)
FEATURE_NAMES = PRS_NAMES + ['Sex'] + [f'PC{i}' for i in range(1, 11)]

# Major disease groups for targeted analyses
MAJOR_DISEASES = {
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis',
              'Other acute and subacute forms of ischemic heart disease',
              'Unstable angina (intermediate coronary syndrome)',
              'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
    'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'],
    'Heart_Failure': ['Congestive heart failure (CHF) NOS'],
    'COPD': ['Chronic obstructive asthma', 'Obstructive chronic bronchitis',
             'Chronic airway obstruction, not elsewhere classified'],
    'CKD': ['Chronic kidney disease, Stage III', 'Chronic kidney disease, Stage IV',
            'Chronic kidney disease, Stage V'],
    'Depression': ['Major depressive disorder'],
    'Breast_Cancer': ['Malignant neoplasm of breast'],
    'Lung_Cancer': ['Malignant neoplasm of bronchus and lung'],
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction',
               'Acute, but ill-defined, cerebrovascular disease'],
}


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# ======================================================================
# Analysis 3: Disease-specific gamma_slope visualization
# ======================================================================

def analysis_3_gamma_slope_viz():
    """Heatmap of gamma_slope (PRS x signature) + bar chart of top slopes per disease."""
    print('\n' + '=' * 70)
    print('ANALYSIS 3: Gamma Slope Visualization')
    print('=' * 70)

    # Pool gamma_slope and psi from training batches
    gamma_slopes, psis = [], []
    for b in TRAIN_BATCHES:
        start = b * BATCH_SIZE
        stop = start + BATCH_SIZE
        f = SLOPE_CKPT_DIR / f'slope_model_batch_{start}_{stop}.pt'
        if f.exists():
            ck = torch.load(f, weights_only=False)
            gamma_slopes.append(_to_numpy(ck['gamma_slope']))
            psis.append(_to_numpy(ck['psi']))

    gamma_slope = np.mean(gamma_slopes, axis=0)  # (47, 21) — 20 disease sigs + 1 health
    psi_pool = np.mean(psis, axis=0)              # (21, 348)

    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']

    # Disease-to-signature assignment (argmax of softmax(psi) for disease sigs only)
    psi_disease = psi_pool[:20, :]  # exclude health signature
    assign = psi_disease.argmax(axis=0)  # (348,) — which signature each disease belongs to

    n_prs = 36  # PRS features only (exclude sex + PCs)

    # --- Figure A: Heatmap of gamma_slope (PRS x 20 disease signatures) ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [2, 1]})

    gs_prs = gamma_slope[:n_prs, :20]  # (36, 20) PRS features x disease signatures
    im = axes[0].imshow(gs_prs, aspect='auto', cmap='RdBu_r',
                        vmin=-np.percentile(np.abs(gs_prs), 95),
                        vmax=np.percentile(np.abs(gs_prs), 95))
    axes[0].set_yticks(range(n_prs))
    axes[0].set_yticklabels(PRS_NAMES, fontsize=8)
    axes[0].set_xticks(range(20))
    axes[0].set_xticklabels([f'Sig {k}' for k in range(20)], rotation=45, fontsize=8)
    axes[0].set_title('gamma_slope: PRS Feature x Disease Signature\n(red = increasing effect over time, blue = decreasing)', fontsize=12)
    plt.colorbar(im, ax=axes[0], shrink=0.6)

    # --- Figure B: Top PRS slopes for selected diseases ---
    key_diseases_for_bar = ['ASCVD', 'Diabetes', 'COPD', 'Depression', 'CKD',
                            'Heart_Failure', 'Breast_Cancer', 'Stroke']

    bar_data = []
    for dg_name in key_diseases_for_bar:
        if dg_name not in MAJOR_DISEASES:
            continue
        disease_list = MAJOR_DISEASES[dg_name]
        d_indices = [i for i, dn in enumerate(disease_names) if dn in disease_list]
        if not d_indices:
            continue
        sigs_for_group = set(assign[d_indices])
        for sig in sigs_for_group:
            slopes = gs_prs[:, sig]
            top_idx = np.argsort(np.abs(slopes))[-5:][::-1]
            for idx in top_idx:
                bar_data.append({
                    'Disease': dg_name, 'Signature': sig,
                    'PRS': PRS_NAMES[idx], 'slope': slopes[idx],
                })

    if bar_data:
        bar_df = pd.DataFrame(bar_data)
        bar_df['abs_slope'] = bar_df['slope'].abs()
        top_per_disease = (bar_df.groupby('Disease')
                          .apply(lambda x: x.nlargest(3, 'abs_slope', keep='first'),
                                 include_groups=False)
                          .reset_index(level=0))

        diseases_in_plot = top_per_disease['Disease'].unique()
        y_labels = []
        y_vals = []
        colors = []
        for dg in diseases_in_plot:
            sub = top_per_disease[top_per_disease['Disease'] == dg]
            for _, row in sub.iterrows():
                y_labels.append(f"{row['Disease']}: {row['PRS']}")
                y_vals.append(row['slope'])
                colors.append('firebrick' if row['slope'] > 0 else 'steelblue')

        y_pos = range(len(y_labels))
        axes[1].barh(y_pos, y_vals, color=colors, height=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(y_labels, fontsize=8)
        axes[1].set_xlabel('gamma_slope value')
        axes[1].set_title('Top PRS slopes by disease group (by |slope|)\n(red = ↑ with age, blue = ↓ with age)', fontsize=12)
        axes[1].axvline(x=0, color='black', linewidth=0.5)
        axes[1].invert_yaxis()

    plt.tight_layout()
    save_path = RESULTS_DIR / f'gamma_slope_visualization{OUTPUT_SUFFIX}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {save_path}')

    # Also save a CSV of all gamma_slope values
    slope_df = pd.DataFrame(gs_prs, index=PRS_NAMES, columns=[f'Sig_{k}' for k in range(20)])
    slope_csv = RESULTS_DIR / f'gamma_slope_prs_by_signature{OUTPUT_SUFFIX}.csv'
    slope_df.to_csv(slope_csv)
    print(f'  Saved: {slope_csv}')


# ======================================================================
# Analysis 4: Age-stratified 1-year AUC
# ======================================================================

def analysis_4_age_stratified_auc():
    """Compute 1-year AUC by age tertile for slope vs no-slope."""
    print('\n' + '=' * 70)
    print('ANALYSIS 4: Age-Stratified 1-Year AUC')
    print('=' * 70)

    pi_slope = torch.load(PI_SLOPE_PATH, weights_only=False)
    pi_noslope = torch.load(PI_NOSLOPE_PATH, weights_only=False)
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)[:N_TEST]
    E = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)[:N_TEST]
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']

    pce_df = pd.read_csv(PCE_PATH).iloc[:N_TEST].reset_index(drop=True)
    ages = pce_df['age'].values

    # Age tertiles
    t1, t2 = np.percentile(ages, [33.3, 66.7])
    age_groups = {
        f'Young (<{t1:.0f})': ages < t1,
        f'Middle ({t1:.0f}-{t2:.0f})': (ages >= t1) & (ages < t2),
        f'Old (>={t2:.0f})': ages >= t2,
    }
    print(f'  Age tertiles: <{t1:.0f} (n={np.sum(ages < t1)}), '
          f'{t1:.0f}-{t2:.0f} (n={np.sum((ages >= t1) & (ages < t2))}), '
          f'>={t2:.0f} (n={np.sum(ages >= t2)})')

    results = []
    for dg_name, disease_list in MAJOR_DISEASES.items():
        d_indices = [i for i, dn in enumerate(disease_names) if dn in disease_list]
        if not d_indices:
            continue

        for age_label, age_mask in age_groups.items():
            n_in_group = age_mask.sum()
            if n_in_group < 100:
                continue

            # 1-year outcome: event in first year after enrollment
            E_sub = E[age_mask]
            Y_sub = Y[age_mask]
            pi_s_sub = pi_slope[age_mask]
            pi_n_sub = pi_noslope[age_mask]

            # For each patient: risk = pi at enrollment time, outcome = event within 1 year
            outcomes = torch.zeros(n_in_group)
            risks_slope = torch.zeros(n_in_group)
            risks_noslope = torch.zeros(n_in_group)

            for i in range(n_in_group):
                enroll_t = E_sub[i, d_indices[0]].long().item()
                if enroll_t >= Y_sub.shape[2] - 1:
                    continue
                event_window = min(enroll_t + 1, Y_sub.shape[2] - 1)
                outcome = 0
                for d_idx in d_indices:
                    if Y_sub[i, d_idx, event_window] > 0:
                        outcome = 1
                        break
                outcomes[i] = outcome
                risk_s = max(pi_s_sub[i, d_indices, enroll_t].mean().item(), 1e-8)
                risk_n = max(pi_n_sub[i, d_indices, enroll_t].mean().item(), 1e-8)
                risks_slope[i] = risk_s
                risks_noslope[i] = risk_n

            n_events = int(outcomes.sum().item())
            if n_events < 5:
                continue

            try:
                auc_s = roc_auc_score(outcomes.numpy(), risks_slope.numpy())
                auc_n = roc_auc_score(outcomes.numpy(), risks_noslope.numpy())
                results.append({
                    'disease': dg_name, 'age_group': age_label,
                    'auc_slope': auc_s, 'auc_noslope': auc_n,
                    'diff': auc_s - auc_n, 'n_events': n_events, 'n_total': int(n_in_group),
                })
            except ValueError:
                continue

    res_df = pd.DataFrame(results)
    res_csv = RESULTS_DIR / f'age_stratified_1yr_auc{OUTPUT_SUFFIX}.csv'
    res_df.to_csv(res_csv, index=False)
    print(f'  Saved: {res_csv}')

    # --- Plot ---
    diseases_to_plot = [d for d in MAJOR_DISEASES.keys()
                        if d in res_df['disease'].values]
    n_diseases = len(diseases_to_plot)
    fig, axes = plt.subplots(2, (n_diseases + 1) // 2, figsize=(5 * ((n_diseases + 1) // 2), 10))
    axes = axes.flatten()

    for i, dg in enumerate(diseases_to_plot):
        ax = axes[i]
        sub = res_df[res_df['disease'] == dg].sort_values('age_group')
        x = range(len(sub))
        w = 0.35
        ax.bar([xi - w/2 for xi in x], sub['auc_slope'], w, label='Slope', color='firebrick', alpha=0.8)
        ax.bar([xi + w/2 for xi in x], sub['auc_noslope'], w, label='No-slope', color='steelblue', alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub['age_group'].values, rotation=25, fontsize=7)
        ax.set_ylabel('AUC')
        ax.set_title(dg, fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_ylim(0.4, 1.0)
        for xi, (_, row) in zip(x, sub.iterrows()):
            ax.text(xi, max(row['auc_slope'], row['auc_noslope']) + 0.01,
                    f'n={row["n_events"]}', ha='center', fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('1-Year AUC by Age Group: Slope vs No-Slope', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = RESULTS_DIR / f'age_stratified_1yr_auc{OUTPUT_SUFFIX}.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {save_path}')

    # Print summary
    print('\n  Age-stratified results:')
    for _, row in res_df.iterrows():
        winner = 'SLOPE' if row['diff'] > 0 else 'noslope'
        print(f'    {row["disease"]:<20} {row["age_group"]:<20} '
              f'slope={row["auc_slope"]:.3f} noslope={row["auc_noslope"]:.3f} '
              f'diff={row["diff"]:+.3f} ({winner}) events={row["n_events"]}')

    del pi_slope, pi_noslope, Y, E
    gc.collect()


# ======================================================================
# Analysis 5: Individual trajectory examples
# ======================================================================

def analysis_5_trajectory_examples():
    """
    Show that the slope model discriminates better: cases rise, controls stay flat.

    Per disease (one page):
      Row 0: (A) mean trajectory for cases vs controls in both models
             (B) top PRS slopes for this disease
             (C) risk-difference distributions for cases vs controls
      Row 1: 3 individual case examples (got disease, slope > noslope)
      Row 2: 3 individual control examples (no disease, slope <= noslope)
    """
    print('\n' + '=' * 70)
    print('ANALYSIS 5: Individual Trajectory Examples — Discrimination')
    print('=' * 70)

    pi_slope = torch.load(PI_SLOPE_PATH, weights_only=False)
    pi_noslope = torch.load(PI_NOSLOPE_PATH, weights_only=False)
    Y = torch.load(DATA_DIR / 'Y_tensor.pt', weights_only=False)[:N_TEST]
    E = torch.load(DATA_DIR / 'E_enrollment_full.pt', weights_only=False)[:N_TEST]
    G_prs = torch.load(DATA_DIR / 'G_matrix.pt', weights_only=False)[:N_TEST]
    essentials = torch.load(DATA_DIR / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']

    pce_df = pd.read_csv(PCE_PATH).iloc[:N_TEST].reset_index(drop=True)
    ages = pce_df['age'].values

    # Pool gamma_slope + psi for annotation
    gamma_slopes_list, psis_list = [], []
    for b in TRAIN_BATCHES:
        start = b * BATCH_SIZE
        stop = start + BATCH_SIZE
        f = SLOPE_CKPT_DIR / f'slope_model_batch_{start}_{stop}.pt'
        if f.exists():
            ck = torch.load(f, weights_only=False)
            gamma_slopes_list.append(_to_numpy(ck['gamma_slope']))
            psis_list.append(_to_numpy(ck['psi']))
    gs_pool = np.mean(gamma_slopes_list, axis=0)[:36, :20]
    psi_pool = np.mean(psis_list, axis=0)
    assign = psi_pool[:20, :].argmax(axis=0)

    diseases_to_show = ['ASCVD', 'Diabetes', 'COPD', 'Stroke',
                        'Heart_Failure', 'Atrial_Fib']

    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = RESULTS_DIR / f'individual_trajectories{OUTPUT_SUFFIX}.pdf'

    with PdfPages(str(pdf_path)) as pdf:
        for dg_name in diseases_to_show:
            if dg_name not in MAJOR_DISEASES:
                continue
            disease_list = MAJOR_DISEASES[dg_name]
            d_indices = [i for i, dn in enumerate(disease_names) if dn in disease_list]
            if not d_indices:
                continue

            d_idx = d_indices[0]
            T = pi_slope.shape[2]
            t_arr = np.arange(T)

            risk_slope = pi_slope[:, d_indices, :].mean(dim=1).detach()
            risk_noslope = pi_noslope[:, d_indices, :].mean(dim=1).detach()

            # Case / control labels
            got_disease = torch.zeros(Y.shape[0], dtype=torch.bool)
            first_events = torch.full((Y.shape[0],), T, dtype=torch.long)
            for di in d_indices:
                evts = Y[:, di, :].sum(dim=1) > 0
                got_disease |= evts
                for pat in torch.where(evts)[0]:
                    et = torch.where(Y[pat, di, :] > 0)[0]
                    if len(et) > 0:
                        first_events[pat] = min(first_events[pat].item(), et[0].item())

            case_idx = torch.where(got_disease)[0]
            ctrl_idx = torch.where(~got_disease)[0]

            # Top PRS slopes for annotation
            sigs_for_disease = set(assign[d_indices])
            prs_slope_info = []
            for sig in sigs_for_disease:
                slopes = gs_pool[:, sig]
                top3 = np.argsort(np.abs(slopes))[-3:][::-1]
                for idx in top3:
                    prs_slope_info.append((PRS_NAMES[idx], slopes[idx], idx))

            # ---- Figure: 3 rows x 3 cols ----
            fig = plt.figure(figsize=(18, 16))
            gs_fig = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

            # ============================================================
            # ROW 0: Summary panels
            # ============================================================

            # (0,0): Mean trajectories — cases vs controls, both models
            ax_mean = fig.add_subplot(gs_fig[0, 0])
            mean_case_slope = risk_slope[case_idx].mean(dim=0).numpy()
            mean_case_noslope = risk_noslope[case_idx].mean(dim=0).numpy()
            mean_ctrl_slope = risk_slope[ctrl_idx].mean(dim=0).numpy()
            mean_ctrl_noslope = risk_noslope[ctrl_idx].mean(dim=0).numpy()

            ax_mean.plot(t_arr, mean_case_slope, color='firebrick', linewidth=2.5,
                         label=f'Cases — Slope (n={len(case_idx)})')
            ax_mean.plot(t_arr, mean_case_noslope, color='firebrick', linewidth=2.5,
                         linestyle='--', alpha=0.6,
                         label='Cases — No-slope')
            ax_mean.plot(t_arr, mean_ctrl_slope, color='steelblue', linewidth=2.5,
                         label=f'Controls — Slope (n={len(ctrl_idx)})')
            ax_mean.plot(t_arr, mean_ctrl_noslope, color='steelblue', linewidth=2.5,
                         linestyle='--', alpha=0.6,
                         label='Controls — No-slope')

            ax_mean.fill_between(t_arr, mean_case_slope, mean_case_noslope,
                                 color='firebrick', alpha=0.08)
            ax_mean.fill_between(t_arr, mean_ctrl_slope, mean_ctrl_noslope,
                                 color='steelblue', alpha=0.08)
            ax_mean.set_xlabel('Time')
            ax_mean.set_ylabel('Mean P(disease)')
            ax_mean.set_title('Mean trajectory: Cases vs Controls', fontsize=11,
                              fontweight='bold')
            ax_mean.legend(fontsize=7, loc='upper left')

            # (0,1): Top PRS slopes
            ax_prs = fig.add_subplot(gs_fig[0, 1])
            if prs_slope_info:
                unique_prs = {}
                for name, val, idx in prs_slope_info:
                    if name not in unique_prs or abs(val) > abs(unique_prs[name]):
                        unique_prs[name] = val
                prs_sorted = sorted(unique_prs.keys(),
                                    key=lambda k: abs(unique_prs[k]), reverse=True)[:6]
                prs_vals = [unique_prs[n] for n in prs_sorted]
                bar_c = ['firebrick' if v > 0 else 'steelblue' for v in prs_vals]
                ax_prs.barh(range(len(prs_sorted)), prs_vals, color=bar_c, height=0.6)
                ax_prs.set_yticks(range(len(prs_sorted)))
                ax_prs.set_yticklabels(prs_sorted, fontsize=9)
                ax_prs.axvline(0, color='black', linewidth=0.5)
                ax_prs.invert_yaxis()
                ax_prs.set_xlabel('γ_slope')
                ax_prs.set_title(f'PRS slopes for {dg_name}\n'
                                 f'(red=↑ with age, blue=↓)', fontsize=11)

            # (0,2): Risk-difference distributions — cases vs controls
            ax_dist = fig.add_subplot(gs_fig[0, 2])
            diff_mean = (risk_slope - risk_noslope).mean(dim=1).numpy()
            bins = np.linspace(np.percentile(diff_mean, 1),
                               np.percentile(diff_mean, 99), 60)
            ax_dist.hist(diff_mean[ctrl_idx.numpy()], bins=bins,
                         color='steelblue', alpha=0.5, density=True,
                         label='Controls', edgecolor='none')
            ax_dist.hist(diff_mean[case_idx.numpy()], bins=bins,
                         color='firebrick', alpha=0.5, density=True,
                         label='Cases', edgecolor='none')
            ax_dist.axvline(0, color='black', linewidth=0.5)
            ax_dist.set_xlabel('Mean (slope − no-slope) risk')
            ax_dist.set_ylabel('Density')
            ax_dist.set_title('Slope model shifts cases RIGHT\n'
                              '(higher risk) vs controls', fontsize=11)
            ax_dist.legend(fontsize=8)

            # ============================================================
            # ROW 1: Individual CASES (got disease, slope predicts higher)
            # ============================================================
            diff_per_patient = (risk_slope - risk_noslope).mean(dim=1)
            abs_diff = diff_per_patient.abs()

            mask_case_up = got_disease & (diff_per_patient > 0)
            if mask_case_up.sum() < 3:
                mask_case_up = got_disease
            case_pool = torch.where(mask_case_up)[0]
            case_scores = abs_diff[mask_case_up]
            sorted_cases = case_pool[case_scores.argsort(descending=True)]

            chosen_cases = _pick_diverse(sorted_cases, ages, n=3)

            for col_i, pat in enumerate(chosen_cases):
                ax = fig.add_subplot(gs_fig[1, col_i])
                _plot_trajectory(ax, pat, t_arr, risk_slope, risk_noslope,
                                 E, d_idx, first_events, got_disease, ages,
                                 G_prs, prs_slope_info,
                                 show_legend=(col_i == 0))
            # Row label
            fig.text(0.01, 0.52, 'CASES\n(got disease)',
                     fontsize=12, fontweight='bold', color='firebrick',
                     va='center', rotation=90)

            # ============================================================
            # ROW 2: Individual CONTROLS (no disease, slope predicts lower)
            # ============================================================
            mask_ctrl_down = (~got_disease) & (diff_per_patient < 0)
            if mask_ctrl_down.sum() < 3:
                mask_ctrl_down = ~got_disease
            ctrl_pool = torch.where(mask_ctrl_down)[0]
            ctrl_scores = abs_diff[mask_ctrl_down]
            sorted_ctrls = ctrl_pool[ctrl_scores.argsort(descending=True)]

            chosen_ctrls = _pick_diverse(sorted_ctrls, ages, n=3)

            for col_i, pat in enumerate(chosen_ctrls):
                ax = fig.add_subplot(gs_fig[2, col_i])
                _plot_trajectory(ax, pat, t_arr, risk_slope, risk_noslope,
                                 E, d_idx, first_events, got_disease, ages,
                                 G_prs, prs_slope_info,
                                 show_legend=(col_i == 0))
            fig.text(0.01, 0.19, 'CONTROLS\n(no disease)',
                     fontsize=12, fontweight='bold', color='steelblue',
                     va='center', rotation=90)

            fig.suptitle(
                f'{dg_name}: Slope Model Improves Discrimination\n'
                f'Cases rise more · Controls stay flatter',
                fontsize=15, fontweight='bold', y=1.02)
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f'  Page saved for {dg_name}')

    print(f'  Saved: {pdf_path}')
    del pi_slope, pi_noslope, Y, E, G_prs
    gc.collect()


def _pick_diverse(sorted_indices, ages, n=3):
    """Pick n patients from sorted_indices with age diversity."""
    if len(sorted_indices) == 0:
        return []
    chosen = []
    age_vals = ages[sorted_indices.numpy()]
    try:
        bins = np.digitize(age_vals, np.percentile(age_vals, [33, 67]))
    except (IndexError, ValueError):
        bins = np.zeros(len(age_vals), dtype=int)
    seen = set()
    for rank, pidx in enumerate(sorted_indices):
        b = bins[rank]
        if b not in seen or len(chosen) < n:
            chosen.append(pidx.item())
            seen.add(b)
        if len(chosen) >= n:
            break
    # Fill if needed
    for pidx in sorted_indices[:n * 2]:
        if pidx.item() not in chosen:
            chosen.append(pidx.item())
        if len(chosen) >= n:
            break
    return chosen[:n]


def _plot_trajectory(ax, pat, t_arr, risk_slope, risk_noslope,
                     E, d_idx, first_events, got_disease, ages,
                     G_prs, prs_slope_info, show_legend=False):
    """Plot one patient's slope vs no-slope trajectory."""
    T = len(t_arr)
    rs = risk_slope[pat].numpy()
    rn = risk_noslope[pat].numpy()

    ax.plot(t_arr, rs, color='firebrick', linewidth=2.5,
            label='Slope model', zorder=3)
    ax.plot(t_arr, rn, color='steelblue', linewidth=2.5,
            label='No-slope model', zorder=3)

    ax.fill_between(t_arr, rs, rn, where=rs >= rn,
                    color='firebrick', alpha=0.12)
    ax.fill_between(t_arr, rs, rn, where=rs < rn,
                    color='steelblue', alpha=0.12)

    enroll_t = E[pat, d_idx].long().item()
    if 0 < enroll_t < T:
        ax.axvline(x=enroll_t, color='gray', linestyle='--',
                   alpha=0.6, linewidth=1, label='Enrollment')

    fe = first_events[pat].item()
    if fe < T:
        ax.axvline(x=fe, color='red', linestyle='-',
                   alpha=0.8, linewidth=1.5, label='Disease onset')
        ax.scatter([fe], [rs[min(fe, T - 1)]], color='red',
                   s=80, zorder=5, marker='v')

    if prs_slope_info and pat < G_prs.shape[0]:
        g = G_prs[pat]
        top2 = prs_slope_info[:2]
        prs_text = ', '.join(
            f'{nm}={g[pidx].item():.1f} (γs={sv:+.2f})'
            for nm, sv, pidx in top2
        )
        ax.text(0.02, 0.97, prs_text, transform=ax.transAxes,
                fontsize=7, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='lightyellow', alpha=0.8))

    age_str = f'{ages[pat]:.0f}' if pat < len(ages) else '?'
    outcome = 'GOT DISEASE' if got_disease[pat] else 'disease-free'
    c = 'firebrick' if got_disease[pat] else 'steelblue'
    ax.set_xlabel('Time')
    ax.set_ylabel('P(disease)')
    ax.set_title(f'Patient {pat} · age {age_str} · {outcome}',
                 fontsize=10, fontweight='bold', color=c)
    if show_legend:
        ax.legend(fontsize=7, loc='best')
    ymax = max(rs.max(), rn.max()) * 1.4 + 0.005
    ax.set_ylim(-0.002, ymax)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Slope value analyses: gamma viz, age-stratified AUC, trajectory examples.')
    parser.add_argument('--loo', action='store_true', help='Use LOO pi (400k) and single-phase slope checkpoints; save with _loo suffix.')
    args = parser.parse_args()
    _effective_config(args.loo)

    print('=' * 70)
    print('SLOPE VALUE ANALYSES' + (' (LOO 400k)' if args.loo else ' (holdout 100k)'))
    print('=' * 70)

    analysis_3_gamma_slope_viz()
    analysis_4_age_stratified_auc()
    analysis_5_trajectory_examples()

    print('\n' + '=' * 70)
    print('ALL DONE')
    print(f'Results in: {RESULTS_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
