#!/usr/bin/env python3
"""
Generate presentation slides summarizing the progression:
  Original (nolr) → Reparameterized (nokappa) → Slope model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_holdout_auc')
FEB18   = Path('/Users/sarahurbut/aladynoulli2/claudefile/results_feb18')
OUT     = Path('/Users/sarahurbut/aladynoulli2/claudefile/progression_slides.pdf')

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 13,
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.3,
})

ACCENT  = '#2563eb'
ACCENT2 = '#16a34a'
ACCENT3 = '#dc2626'
GRAY    = '#6b7280'


def title_slide(pdf):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.fill_between([0, 1], 0.55, 1.0, color=ACCENT, alpha=0.08)
    ax.text(0.5, 0.78, 'Aladynoulli Model Improvements', fontsize=30,
            fontweight='bold', ha='center', va='center', color=ACCENT)
    ax.text(0.5, 0.68, 'From gradient repair to time-varying genetic effects',
            fontsize=18, ha='center', va='center', color=GRAY, style='italic')

    items = [
        '1.  Original model: incomplete gradient flow to genetic effects (γ)',
        '2.  Reparameterization fix: direct NLL gradients → no κ shrinkage needed',
        '3.  Slope extension: time-varying genetic effects (γ_slope)',
        '4.  Holdout evaluation: AUC gains across 28 diseases',
    ]
    for i, item in enumerate(items):
        ax.text(0.12, 0.48 - i * 0.09, item, fontsize=15, va='center', color='#1f2937')

    ax.axhline(y=0.10, xmin=0.1, xmax=0.9, color=GRAY, lw=0.5)
    ax.text(0.5, 0.06, 'S. Urbut — February 2026', fontsize=12,
            ha='center', color=GRAY)

    pdf.savefig(fig); plt.close(fig)


def slide_problem(pdf):
    """Slide: The gradient problem in the original model."""
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.fill_between([0, 1], 0.88, 1.0, color=ACCENT, alpha=0.08)
    ax.text(0.5, 0.94, 'The Problem: Incomplete Gradient Flow',
            fontsize=24, fontweight='bold', ha='center', color=ACCENT)

    # Original model box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.50), 0.42, 0.32, boxstyle='round,pad=0.02',
        facecolor='#fef2f2', edgecolor=ACCENT3, lw=2))
    ax.text(0.26, 0.78, 'Original Model ("nolr")', fontsize=16,
            fontweight='bold', ha='center', color=ACCENT3)
    lines_orig = [
        'λ, φ  are direct nn.Parameters',
        'γ, ψ  appear ONLY in GP prior',
        '',
        '∂NLL/∂γ = 0  (no direct gradient!)',
        'γ learns only from weak GP signal',
        '',
        'Required κ shrinkage to compensate',
    ]
    for i, line in enumerate(lines_orig):
        color = ACCENT3 if '∂NLL' in line or 'κ' in line else '#1f2937'
        ax.text(0.08, 0.72 - i * 0.04, line, fontsize=12,
                fontfamily='monospace', color=color)

    # Reparam model box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.53, 0.50), 0.42, 0.32, boxstyle='round,pad=0.02',
        facecolor='#f0fdf4', edgecolor=ACCENT2, lw=2))
    ax.text(0.74, 0.78, 'Reparameterized Model', fontsize=16,
            fontweight='bold', ha='center', color=ACCENT2)
    lines_reparam = [
        'λ = mean_λ(γ) + δ',
        'φ = mean_φ(ψ) + ε',
        '',
        '∂NLL/∂γ ≠ 0  (direct gradient!)',
        'γ, ψ get strong NLL signal',
        '',
        'κ = 1 (no shrinkage needed!)',
    ]
    for i, line in enumerate(lines_reparam):
        color = ACCENT2 if '∂NLL/∂γ ≠' in line or 'κ = 1' in line else '#1f2937'
        ax.text(0.56, 0.72 - i * 0.04, line, fontsize=12,
                fontfamily='monospace', color=color)

    # Arrow
    ax.annotate('', xy=(0.53, 0.66), xytext=(0.47, 0.66),
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=3))
    ax.text(0.50, 0.69, 'fix', fontsize=12, ha='center',
            fontweight='bold', color=ACCENT)

    # Key insight
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.10, 0.10), 0.80, 0.28, boxstyle='round,pad=0.02',
        facecolor='#eff6ff', edgecolor=ACCENT, lw=2))
    ax.text(0.50, 0.33, 'Key Insight', fontsize=16, fontweight='bold',
            ha='center', color=ACCENT)
    ax.text(0.50, 0.27,
            'The need for κ shrinkage was a symptom, not the disease.',
            fontsize=14, ha='center', color='#1f2937')
    ax.text(0.50, 0.21,
            'The real problem was that γ had no direct NLL gradient in the original architecture.',
            fontsize=13, ha='center', color='#1f2937')
    ax.text(0.50, 0.15,
            'With reparameterization, γ receives direct gradients → κ = 1 works best.',
            fontsize=13, ha='center', color=ACCENT2, fontweight='bold')

    pdf.savefig(fig); plt.close(fig)


def slide_reparam_results(pdf):
    """Slide: Reparam (nokappa v3) vs Delphi baselines — 1yr AUC at enrollment."""
    comp = pd.read_csv(FEB18 / 'three_way_comparison_t0.csv')
    comp = comp.sort_values('Nokappa_v3_t0', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    y = np.arange(len(comp))
    diseases = comp['Disease'].values
    v3 = comp['Nokappa_v3_t0'].values
    delphi_med = comp['Delphi_median'].values
    centered = comp['Centered_t0'].values

    ax.barh(y - 0.22, v3, height=0.22, color=ACCENT, alpha=0.85, label='Reparam (nokappa v3)')
    ax.barh(y, delphi_med, height=0.22, color=ACCENT3, alpha=0.5, label='Delphi (median ICD)')
    ax.barh(y + 0.22, centered, height=0.22, color=GRAY, alpha=0.5, label='Original (centered)')

    ax.set_yticks(y)
    ax.set_yticklabels(diseases, fontsize=9)
    ax.set_xlabel('1-Year AUC at Enrollment', fontsize=14)
    ax.set_title('Reparameterized Model vs Baselines\n1-Year Static AUC at Enrollment (400k patients)',
                 fontsize=18, fontweight='bold', color=ACCENT, pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim(0.4, 1.0)
    ax.axvline(0.5, color='gray', ls=':', alpha=0.5)

    n_wins = (v3 > delphi_med).sum()
    ax.text(0.98, 0.02, f'Reparam wins {n_wins}/{len(comp)} vs Delphi median',
            transform=ax.transAxes, fontsize=12, ha='right', va='bottom',
            fontweight='bold', color=ACCENT,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)


def slide_slope_architecture(pdf):
    """Slide: The slope model extension."""
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.fill_between([0, 1], 0.88, 1.0, color=ACCENT2, alpha=0.08)
    ax.text(0.5, 0.94, 'Extension: Time-Varying Genetic Effects',
            fontsize=24, fontweight='bold', ha='center', color=ACCENT2)

    # Reparam box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.55), 0.42, 0.28, boxstyle='round,pad=0.02',
        facecolor='#eff6ff', edgecolor=ACCENT, lw=2))
    ax.text(0.26, 0.79, 'Reparam (Level Only)', fontsize=15,
            fontweight='bold', ha='center', color=ACCENT)
    level_lines = [
        'λ = σ_ref + scale · (G @ γ) + δ',
        '',
        'γ captures static genetic',
        'effects — same at all ages',
        '',
        'Cannot model PRS effects',
        'that wane or strengthen',
    ]
    for i, line in enumerate(level_lines):
        ax.text(0.08, 0.73 - i * 0.035, line, fontsize=11,
                fontfamily='monospace', color='#1f2937')

    # Slope box
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.53, 0.55), 0.42, 0.28, boxstyle='round,pad=0.02',
        facecolor='#f0fdf4', edgecolor=ACCENT2, lw=2))
    ax.text(0.74, 0.79, 'Slope Model', fontsize=15,
            fontweight='bold', ha='center', color=ACCENT2)
    slope_lines = [
        'λ = σ_ref + scale·(G @ γ_level)',
        '    + t · scale·(G @ γ_slope) + δ',
        '',
        'γ_level: baseline genetic effect',
        'γ_slope: time-varying change',
        '',
        'Captures waning/strengthening',
    ]
    for i, line in enumerate(slope_lines):
        fw = 'bold' if 'γ_slope' in line and 'time' in line else 'normal'
        color = ACCENT2 if 'waning' in line.lower() else '#1f2937'
        ax.text(0.56, 0.73 - i * 0.035, line, fontsize=11,
                fontfamily='monospace', color=color, fontweight=fw)

    ax.annotate('', xy=(0.53, 0.69), xytext=(0.47, 0.69),
                arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=3))

    # Two-phase training
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.05, 0.08), 0.90, 0.38, boxstyle='round,pad=0.02',
        facecolor='#fefce8', edgecolor='#ca8a04', lw=2))
    ax.text(0.50, 0.42, 'Two-Phase Training Strategy', fontsize=16,
            fontweight='bold', ha='center', color='#ca8a04')

    # Phase 1
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, 0.12), 0.38, 0.22, boxstyle='round,pad=0.01',
        facecolor='white', edgecolor='#ca8a04', lw=1.5))
    ax.text(0.27, 0.30, 'Phase 1 (200 epochs)', fontsize=13,
            fontweight='bold', ha='center', color='#ca8a04')
    p1_lines = [
        'δ FROZEN (patient residual)',
        'γ_slope forced to learn',
        'population-level time effects',
        'from NLL directly',
    ]
    for i, line in enumerate(p1_lines):
        fw = 'bold' if 'FROZEN' in line else 'normal'
        ax.text(0.10, 0.25 - i * 0.035, line, fontsize=10.5, color='#1f2937',
                fontweight=fw)

    # Phase 2
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.54, 0.12), 0.38, 0.22, boxstyle='round,pad=0.01',
        facecolor='white', edgecolor='#ca8a04', lw=1.5))
    ax.text(0.73, 0.30, 'Phase 2 (100 epochs)', fontsize=13,
            fontweight='bold', ha='center', color='#ca8a04')
    p2_lines = [
        'ALL parameters unfrozen',
        'Fine-tune together',
        'δ absorbs patient-specific',
        'residual after slope is set',
    ]
    for i, line in enumerate(p2_lines):
        ax.text(0.56, 0.25 - i * 0.035, line, fontsize=10.5, color='#1f2937')

    ax.annotate('', xy=(0.54, 0.23), xytext=(0.46, 0.23),
                arrowprops=dict(arrowstyle='->', color='#ca8a04', lw=2.5))
    ax.text(0.50, 0.20, 'then', fontsize=10, ha='center', color='#ca8a04')

    pdf.savefig(fig); plt.close(fig)


def slide_holdout_nll(pdf):
    """Slide: Holdout NLL — slope consistently lower."""
    nll = pd.read_csv(RESULTS / 'holdout_nll_slope_vs_noslope.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor('white')
    fig.suptitle('Holdout NLL: Slope vs No-Slope (100k patients)',
                 fontsize=20, fontweight='bold', color=ACCENT, y=0.98)

    ax = axes[0]
    x = nll['batch'].values
    ax.plot(x, nll['nll_slope'], 'o-', color=ACCENT2, ms=8, lw=2, label='Slope')
    ax.plot(x, nll['nll_noslope'], 's-', color=ACCENT3, ms=8, lw=2, label='No-slope')
    ax.set_xlabel('Holdout Batch')
    ax.set_ylabel('NLL (lower = better)')
    ax.set_title('Per-Batch Holdout NLL')
    ax.legend(fontsize=12)
    for i in range(len(nll)):
        ax.annotate(f'{nll["delta_nll"].iloc[i]:+.2f}',
                    (x[i], (nll['nll_slope'].iloc[i] + nll['nll_noslope'].iloc[i]) / 2),
                    fontsize=9, ha='center', color=GRAY)

    ax2 = axes[1]
    mean_slope = nll['nll_slope'].mean()
    mean_noslope = nll['nll_noslope'].mean()
    bars = ax2.bar(['Slope', 'No-slope'], [mean_slope, mean_noslope],
                   color=[ACCENT2, ACCENT3], alpha=0.7, width=0.5)
    ax2.set_ylabel('Mean Holdout NLL')
    ax2.set_title('Average (10 batches)')
    improvement = (mean_noslope - mean_slope) / mean_noslope * 100
    ax2.text(0.5, 0.85, f'Δ = {mean_noslope - mean_slope:.2f}\n({improvement:.1f}% lower)',
             transform=ax2.transAxes, ha='center', fontsize=14,
             fontweight='bold', color=ACCENT2,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def slide_auc_10yr(pdf):
    """Slide: 10-year static AUC — slope vs no-slope."""
    auc = pd.read_csv(RESULTS / 'holdout_auc_slope_vs_noslope.csv')
    static = auc[auc['horizon'] == 'static_10yr']

    slope_df = static[static['model'] == 'slope'].set_index('disease')
    noslope_df = static[static['model'] == 'noslope'].set_index('disease')
    common = slope_df.index.intersection(noslope_df.index)

    diseases = []
    s_aucs, n_aucs = [], []
    for d in common:
        diseases.append(d)
        s_aucs.append(slope_df.loc[d, 'auc'])
        n_aucs.append(noslope_df.loc[d, 'auc'])
    s_aucs = np.array(s_aucs); n_aucs = np.array(n_aucs)
    diffs = s_aucs - n_aucs
    order = np.argsort(diffs)
    diseases = [diseases[i] for i in order]
    s_aucs = s_aucs[order]; n_aucs = n_aucs[order]; diffs = diffs[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor('white')
    fig.suptitle('Holdout AUC: Static 10-Year (Slope vs No-Slope)',
                 fontsize=20, fontweight='bold', color=ACCENT, y=0.98)

    y = np.arange(len(diseases))
    ax1.scatter(n_aucs, y, color=ACCENT3, s=50, zorder=3, label='No-slope')
    ax1.scatter(s_aucs, y, color=ACCENT2, s=50, zorder=3, label='Slope')
    for i in range(len(diseases)):
        color = ACCENT2 if diffs[i] > 0 else ACCENT3
        ax1.plot([n_aucs[i], s_aucs[i]], [y[i], y[i]], color=color, lw=1.5, alpha=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(diseases, fontsize=8)
    ax1.set_xlabel('AUC')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_title('Paired Dot Plot')

    colors = [ACCENT2 if d > 0 else ACCENT3 for d in diffs]
    ax2.barh(y, diffs, color=colors, alpha=0.7, height=0.7)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('ΔAUC (slope − no-slope)')
    ax2.set_title('Waterfall')

    n_wins = (diffs > 0).sum()
    ax2.text(0.95, 0.02, f'Slope wins: {n_wins}/{len(diseases)}',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=12, fontweight='bold', color=ACCENT2,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def slide_auc_1yr(pdf):
    """Slide: 1-year AUC — where slope model really shines."""
    auc = pd.read_csv(RESULTS / 'holdout_auc_1yr_slope_vs_noslope.csv')
    static = auc[auc['horizon'] == 'static_1yr']

    slope_df = static[static['model'] == 'slope'].set_index('disease')
    noslope_df = static[static['model'] == 'noslope'].set_index('disease')
    common = slope_df.index.intersection(noslope_df.index)

    diseases, s_aucs, n_aucs = [], [], []
    for d in common:
        diseases.append(d)
        s_aucs.append(slope_df.loc[d, 'auc'])
        n_aucs.append(noslope_df.loc[d, 'auc'])
    s_aucs = np.array(s_aucs); n_aucs = np.array(n_aucs)
    diffs = s_aucs - n_aucs
    order = np.argsort(diffs)
    diseases = [diseases[i] for i in order]
    s_aucs = s_aucs[order]; n_aucs = n_aucs[order]; diffs = diffs[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor('white')
    fig.suptitle('Holdout AUC: Static 1-Year at Enrollment\n(Slope vs No-Slope)',
                 fontsize=20, fontweight='bold', color=ACCENT2, y=0.99)

    y = np.arange(len(diseases))
    ax1.scatter(n_aucs, y, color=ACCENT3, s=50, zorder=3, label='No-slope')
    ax1.scatter(s_aucs, y, color=ACCENT2, s=50, zorder=3, label='Slope')
    for i in range(len(diseases)):
        color = ACCENT2 if diffs[i] > 0 else ACCENT3
        ax1.plot([n_aucs[i], s_aucs[i]], [y[i], y[i]], color=color, lw=1.5, alpha=0.6)
    ax1.set_yticks(y)
    ax1.set_yticklabels(diseases, fontsize=8)
    ax1.set_xlabel('AUC')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_title('Paired Dot Plot')
    ax1.set_xlim(0.55, 1.02)

    colors = [ACCENT2 if d > 0 else ACCENT3 for d in diffs]
    ax2.barh(y, diffs, color=colors, alpha=0.7, height=0.7)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel('ΔAUC (slope − no-slope)')
    ax2.set_title('Waterfall')

    n_wins = (diffs > 0).sum()
    mean_gain = diffs[diffs > 0].mean()
    ax2.text(0.95, 0.02,
             f'Slope wins: {n_wins}/{len(diseases)}\nMean gain: +{mean_gain:.3f}',
             transform=ax2.transAxes, ha='right', va='bottom',
             fontsize=12, fontweight='bold', color=ACCENT2,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def slide_1yr_highlights(pdf):
    """Slide: Highlight table of biggest 1yr AUC gains."""
    wide = pd.read_csv(RESULTS / 'wide_comparison.csv')
    wide['gain_1yr'] = wide['static_1yr_slope'] - wide['static_1yr_noslope']
    wide = wide.sort_values('gain_1yr', ascending=False)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.fill_between([0, 1], 0.88, 1.0, color=ACCENT2, alpha=0.08)
    ax.text(0.5, 0.94, 'Biggest 1-Year AUC Improvements with Slope Model',
            fontsize=22, fontweight='bold', ha='center', color=ACCENT2)

    headers = ['Disease', 'Slope AUC', 'No-Slope AUC', 'Gain', '% Improvement']
    col_x = [0.10, 0.38, 0.54, 0.70, 0.82]

    for j, h in enumerate(headers):
        ax.text(col_x[j], 0.84, h, fontsize=13, fontweight='bold',
                color=ACCENT, va='center')
    ax.axhline(0.82, xmin=0.08, xmax=0.95, color=ACCENT, lw=1)

    top = wide.head(15)
    for i, (_, row) in enumerate(top.iterrows()):
        y_pos = 0.78 - i * 0.042
        pct = row['gain_1yr'] / row['static_1yr_noslope'] * 100

        bg_alpha = 0.05 if i % 2 == 0 else 0.0
        if bg_alpha > 0:
            ax.fill_between([0.08, 0.95], y_pos - 0.018, y_pos + 0.018,
                           color=ACCENT2, alpha=bg_alpha)

        ax.text(col_x[0], y_pos, row['Disease'].replace('_', ' '), fontsize=11, va='center')
        ax.text(col_x[1], y_pos, f"{row['static_1yr_slope']:.3f}", fontsize=11,
                va='center', fontweight='bold', color=ACCENT2)
        ax.text(col_x[2], y_pos, f"{row['static_1yr_noslope']:.3f}", fontsize=11,
                va='center', color=ACCENT3)
        ax.text(col_x[3], y_pos, f"+{row['gain_1yr']:.3f}", fontsize=11,
                va='center', fontweight='bold', color=ACCENT2)
        ax.text(col_x[4], y_pos, f"+{pct:.1f}%", fontsize=11,
                va='center', color=ACCENT2)

    # Summary box
    mean_gain = wide['gain_1yr'].mean()
    mean_pct = (wide['gain_1yr'] / wide['static_1yr_noslope']).mean() * 100
    n_wins = (wide['gain_1yr'] > 0).sum()
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, 0.04), 0.70, 0.10, boxstyle='round,pad=0.01',
        facecolor='#f0fdf4', edgecolor=ACCENT2, lw=2))
    ax.text(0.50, 0.09,
            f'Across all 28 diseases: slope wins {n_wins}/28  |  '
            f'Mean AUC gain = +{mean_gain:.3f}  |  Mean improvement = +{mean_pct:.1f}%',
            fontsize=13, ha='center', va='center', fontweight='bold', color=ACCENT2)

    pdf.savefig(fig); plt.close(fig)


def slide_all_horizons(pdf):
    """Slide: Summary across all 4 evaluation horizons."""
    wide = pd.read_csv(RESULTS / 'wide_comparison.csv')

    horizons = [
        ('static_1yr', 'Static 1yr\n(enrollment)'),
        ('dynamic_1yr', 'Dynamic 1yr'),
        ('static_10yr', 'Static 10yr'),
        ('dynamic_10yr', 'Dynamic 10yr'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=False)
    fig.patch.set_facecolor('white')
    fig.suptitle('Slope Model Advantage Across All Evaluation Horizons',
                 fontsize=20, fontweight='bold', color=ACCENT, y=0.99)

    for idx, (prefix, label) in enumerate(horizons):
        ax = axes[idx]
        s_col = f'{prefix}_slope'
        n_col = f'{prefix}_noslope'
        d_col = f'{prefix}_diff'

        diffs = wide[d_col].dropna().values
        n_pos = (diffs > 0).sum()
        n_neg = (diffs < 0).sum()
        mean_d = diffs.mean()

        ax.hist(diffs, bins=15, color=ACCENT2 if mean_d > 0 else ACCENT3,
                alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', lw=1, ls='-')
        ax.axvline(mean_d, color=ACCENT, lw=2, ls='--')
        ax.set_xlabel('ΔAUC')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.text(0.5, 0.92, f'{n_pos}/{n_pos+n_neg} wins',
                transform=ax.transAxes, ha='center', fontsize=11,
                fontweight='bold', color=ACCENT2)
        ax.text(0.5, 0.82, f'mean: {mean_d:+.4f}',
                transform=ax.transAxes, ha='center', fontsize=10, color=ACCENT)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig); plt.close(fig)


def slide_summary(pdf):
    """Final summary slide."""
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.fill_between([0, 1], 0.85, 1.0, color=ACCENT, alpha=0.08)
    ax.text(0.5, 0.92, 'Summary & Next Steps', fontsize=28,
            fontweight='bold', ha='center', color=ACCENT)

    # Findings
    ax.text(0.08, 0.78, 'Key Findings', fontsize=18, fontweight='bold', color=ACCENT2)
    findings = [
        '1.  Reparameterization fixed the gradient flow problem — κ shrinkage was a band-aid',
        '2.  Reparam (nokappa v3) beats Delphi ICD-level baselines on most diseases',
        '3.  Time-varying γ_slope captures waning/strengthening PRS effects with age',
        '4.  Holdout NLL: slope model ~16% lower (better fit on unseen patients)',
        '5.  1-year AUC: slope wins 27/28 diseases, mean improvement +10%',
        '6.  10-year AUC: slope wins 18/28 diseases with smaller but consistent gains',
    ]
    for i, f in enumerate(findings):
        ax.text(0.10, 0.71 - i * 0.055, f, fontsize=13, color='#1f2937')

    # NLL stat
    nll = pd.read_csv(RESULTS / 'holdout_nll_slope_vs_noslope.csv')
    mean_delta = nll['delta_nll'].mean()
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, 0.26), 0.36, 0.14, boxstyle='round,pad=0.01',
        facecolor='#f0fdf4', edgecolor=ACCENT2, lw=2))
    ax.text(0.26, 0.36, 'Holdout NLL', fontsize=14, fontweight='bold',
            ha='center', color=ACCENT2)
    ax.text(0.26, 0.30, f'Mean Δ = {mean_delta:.2f} (slope lower)',
            fontsize=13, ha='center', color='#1f2937')

    # 1yr AUC stat
    wide = pd.read_csv(RESULTS / 'wide_comparison.csv')
    mean_1yr = wide['static_1yr_diff'].mean()
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.56, 0.26), 0.36, 0.14, boxstyle='round,pad=0.01',
        facecolor='#eff6ff', edgecolor=ACCENT, lw=2))
    ax.text(0.74, 0.36, '1-Year AUC', fontsize=14, fontweight='bold',
            ha='center', color=ACCENT)
    ax.text(0.74, 0.30, f'Mean gain = +{mean_1yr:.3f} (27/28 wins)',
            fontsize=13, ha='center', color='#1f2937')

    # Next steps
    ax.text(0.08, 0.18, 'In Progress', fontsize=18, fontweight='bold', color='#ca8a04')
    nexts = [
        '•  Ablation study: is two-phase training necessary? (single-phase comparison running)',
        '•  Disease-specific γ_slope interpretation and biological validation',
    ]
    for i, n in enumerate(nexts):
        ax.text(0.10, 0.12 - i * 0.05, n, fontsize=13, color='#1f2937')

    pdf.savefig(fig); plt.close(fig)


def main():
    print('Generating progression slides...')
    with PdfPages(str(OUT)) as pdf:
        title_slide(pdf)
        slide_problem(pdf)
        slide_reparam_results(pdf)
        slide_slope_architecture(pdf)
        slide_holdout_nll(pdf)
        slide_auc_10yr(pdf)
        slide_auc_1yr(pdf)
        slide_1yr_highlights(pdf)
        slide_all_horizons(pdf)
        slide_summary(pdf)
    print(f'Done! {OUT}')


if __name__ == '__main__':
    main()
