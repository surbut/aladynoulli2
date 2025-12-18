"""
Visualize lead variants (all_loci_annotated.tsv) and mask3 canonical rare variants
in a comprehensive multipanel figure.

NEW LAYOUT:
- Panel A: Back-to-back plot (vertical): Sig 0 at top, Sig 20 at bottom
           Common variants pointing LEFT, Rare variants pointing RIGHT
- Panel B: Bar plot with number of lead variants, common variants, and rare variants per signature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans']

# ============================================================================
# LOAD DATA
# ============================================================================

# Load lead variants
loci_file = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/all_loci_annotated.tsv")
print(f"Loading lead variants from {loci_file}...")
loci_df = pd.read_csv(loci_file, sep='\t')

# Extract signature number from SIG column (e.g., "SIG0" -> 0)
loci_df['SIG_NUM'] = loci_df['SIG'].str.replace('SIG', '').astype(int)

print(f"Loaded {len(loci_df)} lead variants across {loci_df['SIG_NUM'].nunique()} signatures")

# ============================================================================
# DEFINE THE 10 UNIQUE SIGNATURE 5 LOCI (to be highlighted in green)
# ============================================================================
THE_10_UNIQUE_SIG5_LOCI = {
    'rs6687726': 'IL6R',
    'rs2509121': 'HYOU1',
    'rs4760278': 'R3HDM2',
    'rs1532085': 'LIPC',  # Hepatic lipase (not ALDH1A2)
    'rs7168222': 'NR2F2-AS1',
    'rs35039495': 'PLCG2',
    'rs8121509': 'OPRL1',
    'rs1499813': 'FNDC3B',
    '4:96088139': 'UNC5C',
    'rs4732365': 'C7orf55'
}

# Mark these in the dataframe
loci_df['is_top10_sig5'] = False
if 'rsid' in loci_df.columns:
    for rsid, gene in THE_10_UNIQUE_SIG5_LOCI.items():
        mask = loci_df['rsid'].str.contains(rsid, case=False, na=False)
        loci_df.loc[mask, 'is_top10_sig5'] = True
        if not mask.any() and ':' in rsid:
            chr_pos = rsid.split(':')
            if len(chr_pos) == 2:
                mask_pos = (loci_df['#CHR'] == int(chr_pos[0])) & (loci_df['POS'] == int(chr_pos[1]))
                loci_df.loc[mask_pos, 'is_top10_sig5'] = True

# Replace ALDH1A2 with LIPC if found
if 'nearestgene' in loci_df.columns:
    loci_df['nearestgene'] = loci_df['nearestgene'].replace('ALDH1A2', 'LIPC')
    
    for rsid, gene in THE_10_UNIQUE_SIG5_LOCI.items():
        gene_mask = (loci_df['SIG_NUM'] == 5) & (loci_df['nearestgene'].str.contains(gene, case=False, na=False))
        loci_df.loc[gene_mask, 'is_top10_sig5'] = True

print(f"Marked {loci_df['is_top10_sig5'].sum()} variants as 'Top 10 Unique Sig 5' loci")

# ============================================================================
# IDENTIFY NOVEL VS KNOWN LOCI (optional - for highlighting)
# ============================================================================

if 'KNOWN' in loci_df.columns or 'is_novel' in loci_df.columns:
    if 'is_novel' not in loci_df.columns:
        loci_df['is_novel'] = (loci_df['KNOWN'] == 0) if 'KNOWN' in loci_df.columns else False
    print(f"Novelty information available: {loci_df['is_novel'].sum()} novel loci")
else:
    loci_df['is_novel'] = False

# ============================================================================
# SIGNATURE LABELS (for legend)
# ============================================================================

SIGNATURE_LABELS = {
    0: 'Cardiac Arrhythmias',
    1: 'Musculoskeletal',
    2: 'Upper GI/Esophageal',
    3: 'Mixed/General Medical',
    4: 'Upper Respiratory',
    5: 'Ischemic cardiovascular',
    6: 'Metastatic Cancer',
    7: 'Pain/Inflammation',
    8: 'Gynecologic',
    9: 'Spinal Disorders',
    10: 'Ophthalmologic',
    11: 'Cerebrovascular',
    12: 'Renal/Urologic',
    13: 'Male Urogenital',
    14: 'Pulmonary/Smoking',
    15: 'Metabolic/Diabetes',
    16: 'Infectious/Critical Care',
    17: 'Lower GI/Colon',
    18: 'Hepatobiliary',
    19: 'Dermatologic/Oncologic',
    20: 'Health'
}

# ============================================================================
# LOAD RARE VARIANT DATA (mask3)
# ============================================================================

results_dir = Path("/Users/sarahurbut/Desktop/SIG/gene_based_analysis")
canonical_dir = results_dir / "canonical"

print(f"\nLoading mask3 (LoF variants only) results from {canonical_dir}...")
mask3_files = sorted(canonical_dir.glob("Mask3_*_significant_canonical.tsv"))

if len(mask3_files) == 0:
    print(f"⚠️  No mask3 files found in {canonical_dir}")
    mask3_df = None
    best_results = None
else:
    mask3_results = []
    
    for file in mask3_files:
        parts = file.stem.replace("_significant_canonical", "").split("_")
        maf = parts[1]
        
        df = pd.read_csv(file, sep='\t')
        df['MAF'] = maf
        mask3_results.append(df)
    
    mask3_df = pd.concat(mask3_results, ignore_index=True)
    print(f"Loaded {len(mask3_df)} significant associations from mask3")
    
    # Get best result per gene-signature (across MAF thresholds)
    best_results = mask3_df.loc[mask3_df.groupby(['SIG', 'SYMBOL'])['LOG10P'].idxmax()].copy()
    best_results = best_results.sort_values(['SIG', 'LOG10P'], ascending=[True, False])
    
    print(f"Best results per gene-signature: {len(best_results)} associations")
    print(f"Unique genes: {best_results['SYMBOL'].nunique()}")

# ============================================================================
# CREATE MULTIPANEL FIGURE
# ============================================================================

# Layout: 2 rows (back-to-back plot full width, counts bar plot full width), 1 column for plots + 1 for legend
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, 
                      width_ratios=[3, 0.8], height_ratios=[2, 1])

# Color palette for signatures
n_sigs = 21
sig_colors = plt.cm.tab20(np.linspace(0, 1, 20))
sig_colors = np.vstack([sig_colors, plt.cm.tab20b(0.5)])
sig_color_dict = {i: sig_colors[i] for i in range(n_sigs)}

# ============================================================================
# PANEL A: Back-to-back plot (vertical: Sig 0 top, Sig 20 bottom) - FULL WIDTH
# ============================================================================

ax1 = fig.add_subplot(gs[0, 0])

# Prepare data: y-position = signature number (reversed so Sig 0 is at top)
y_sig_dict = {i: n_sigs - 1 - i for i in range(n_sigs)}  # Sig 0 -> y=20, Sig 20 -> y=0

# Truncate LOG10P values at 50 for better visualization
MAX_LOG10P = 50
loci_df['LOG10P_clipped'] = loci_df['LOG10P'].clip(upper=MAX_LOG10P)
if best_results is not None and len(best_results) > 0:
    best_results['LOG10P_clipped'] = best_results['LOG10P'].clip(upper=MAX_LOG10P)

# Plot GWAS (pointing LEFT - negative x-axis)
for sig_num in sorted(loci_df['SIG_NUM'].unique()):
    sig_data = loci_df[loci_df['SIG_NUM'] == sig_num]
    y_pos = y_sig_dict[sig_num]
    
    # Add small jitter in y-direction
    y_jitter = y_pos + np.random.normal(0, 0.1, len(sig_data))
    
    # Plot all variants (regular markers, no stars)
    # Top 10 Unique Sig 5 in green, others in signature color
    if sig_num == 5:
        # Separate top10 and others for Sig 5
        top10_data = sig_data[sig_data['is_top10_sig5'] == True]
        other_data = sig_data[sig_data['is_top10_sig5'] == False]
        
        if len(other_data) > 0:
            ax1.scatter(-other_data['LOG10P_clipped'], 
                       y_pos + np.random.normal(0, 0.1, len(other_data)),
                       alpha=0.6, s=60, color=sig_color_dict[sig_num],
                       edgecolors='black', linewidths=0.5, marker='o', zorder=2)
        
        if len(top10_data) > 0:
            # Sort by significance for consistent annotation
            top10_data = top10_data.sort_values('LOG10P', ascending=False)
            y_top10 = y_pos + np.random.normal(0, 0.1, len(top10_data))
            ax1.scatter(-top10_data['LOG10P_clipped'], y_top10,
                       alpha=0.9, s=100, color='#2ecc71',  # Green
                       edgecolors='darkgreen', linewidths=2, marker='o', zorder=3)
            
            # Annotate Top 5 genes (most significant ones)
            for idx in range(min(5, len(top10_data))):
                row = top10_data.iloc[idx]
                gene = row.get('nearestgene', 'N/A')
                if pd.notna(gene) and gene != 'N/A':
                    ax1.annotate(gene, 
                               xy=(-row['LOG10P_clipped'], y_top10[idx]),
                               xytext=(-row['LOG10P_clipped'] - 2, y_top10[idx]),
                               fontsize=9, alpha=0.9, fontweight='bold',
                               ha='right', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5),
                               arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.7, lw=1.5))
    else:
        ax1.scatter(-sig_data['LOG10P_clipped'], y_jitter,
                   alpha=0.6, s=60, color=sig_color_dict[sig_num],
                   edgecolors='black', linewidths=0.5, marker='o', zorder=2)
        
        # Annotate top lead variant per signature (if very significant)
        if len(sig_data) > 0:
            top_idx = sig_data['LOG10P'].idxmax()
            top_variant = sig_data.loc[top_idx]
            if top_variant['LOG10P'] > 6:  # Only annotate if very significant (p < 1e-6)
                position_in_array = list(sig_data.index).index(top_idx)
                if position_in_array < len(y_jitter):
                    gene = top_variant.get('nearestgene', 'N/A')
                    rsid = top_variant.get('rsid', 'N/A')
                    label = gene if pd.notna(gene) and gene != 'N/A' else (str(rsid) if pd.notna(rsid) else f'Sig{sig_num}')
                    ax1.annotate(label, 
                               xy=(-top_variant['LOG10P_clipped'], y_jitter[position_in_array]),
                               xytext=(-top_variant['LOG10P_clipped'] - 2, y_jitter[position_in_array]),
                               fontsize=9, alpha=0.9, fontweight='bold',
                               ha='right', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5),
                               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, lw=1.5))

# Plot RVAS (pointing RIGHT - positive x-axis)
if best_results is not None and len(best_results) > 0:
    for sig_num in sorted(best_results['SIG'].unique()):
        sig_data = best_results[best_results['SIG'] == sig_num]
        y_pos = y_sig_dict[sig_num]
        y_jitter = y_pos + np.random.normal(0, 0.1, len(sig_data))
        
        ax1.scatter(sig_data['LOG10P_clipped'], y_jitter,
                   alpha=0.7, s=80, color=sig_color_dict[sig_num],
                   edgecolors='black', linewidths=1, marker='s',  # Square markers for RVAS
                   zorder=2)
        
        # Annotate top gene per signature (if significant)
        if len(sig_data) > 0:
            top_idx = sig_data['LOG10P'].idxmax()
            top_gene = sig_data.loc[top_idx]
            if top_gene['LOG10P'] > 6:  # Only annotate if very significant (p < 1e-6)
                # Find the y position for this top gene
                position_in_array = list(sig_data.index).index(top_idx)
                if position_in_array < len(y_jitter):
                    ax1.annotate(top_gene['SYMBOL'], 
                               xy=(top_gene['LOG10P_clipped'], y_jitter[position_in_array]),
                               xytext=(top_gene['LOG10P_clipped'] + 2, y_jitter[position_in_array]),
                               fontsize=9, alpha=0.9, fontweight='bold',
                               ha='left', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5),
                               arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, lw=1.5))

# Add vertical line at x=0
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1.5, zorder=1)

# Add significance thresholds (vertical lines)
ax1.axvline(x=-7.3, color='red', linestyle='--', alpha=0.7, linewidth=1.5, zorder=1)  # GWAS threshold at -log10(5e-8) ≈ 7.3
ax1.text(-7.3, n_sigs - 1, 'GWAS threshold', ha='center', va='top', fontsize=9, color='red', rotation=90)
if best_results is not None and len(best_results) > 0:
    ax1.axvline(x=5.6, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, zorder=1)  # RVAS threshold at -log10(2.5e-6) ≈ 5.6
    ax1.text(5.6, n_sigs - 1, 'RVAS threshold', ha='center', va='top', fontsize=9, color='orange', rotation=90)

# Formatting
ax1.set_xlabel('-log₁₀(P-value)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Signature', fontsize=13, fontweight='bold')
ax1.set_title('A. Genetic Associations: GWAS (left) and RVAS (right)', 
              fontsize=14, fontweight='bold', pad=15)

# Set y-axis: signatures from 0 (top) to 20 (bottom)
ax1.set_yticks(range(n_sigs))
ax1.set_yticklabels([f'Sig {n_sigs-1-i}' for i in range(n_sigs)], fontsize=10)
ax1.set_ylim([-0.5, n_sigs - 0.5])
ax1.invert_yaxis()  # Sig 0 at top

# Set x-axis: symmetric around 0, truncated at 50
ax1.set_xlim([-55, 55])
# Create symmetric ticks with same numbers on both sides
x_ticks = list(range(-50, 0, 5)) + [0] + list(range(5, 55, 5))
ax1.set_xticks(x_ticks)
# Use absolute values for labels (since both sides represent -log10(p))
ax1.set_xticklabels([abs(x) if x != 0 else '0' for x in x_ticks])

ax1.grid(True, alpha=0.3, axis='x')

# Add legend for markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markersize=8, label='GWAS (Common Variants)', markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
           markersize=8, label='RVAS (Rare Variants)', markeredgecolor='black'),
]
if loci_df['is_top10_sig5'].sum() > 0:
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', 
               markersize=10, label='Top 10 Unique Sig 5', 
               markeredgecolor='darkgreen', markeredgewidth=2)
    )
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

# ============================================================================
# PANEL B: Bar plot with counts per signature - FULL WIDTH BELOW PANEL A
# ============================================================================

ax2 = fig.add_subplot(gs[1, 0])

# Count lead variants (all), common variants (GWAS), and rare variants (RVAS) per signature
counts_data = []
for sig_num in range(n_sigs):
    # All lead variants (this includes all GWAS hits)
    all_lead = len(loci_df[loci_df['SIG_NUM'] == sig_num])
    
    # Rare variants (from mask3)
    if best_results is not None and len(best_results) > 0:
        rare_count = len(best_results[best_results['SIG'] == sig_num])
    else:
        rare_count = 0
    
    counts_data.append({
        'Signature': sig_num,
        'Lead Variants (GWAS)': all_lead,
        'Rare Variants (RVAS)': rare_count
    })

counts_df = pd.DataFrame(counts_data)

# Create grouped bar chart
x = np.arange(n_sigs)
width = 0.35

bars1 = ax2.bar(x - width/2, counts_df['Lead Variants (GWAS)'], width,
                label='Lead Variants (GWAS)', color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
bars2 = ax2.bar(x + width/2, counts_df['Rare Variants (RVAS)'], width,
                label='Rare Variants (RVAS)', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_xlabel('Signature', fontsize=13, fontweight='bold')
ax2.set_ylabel('Number of Associations', fontsize=13, fontweight='bold')
ax2.set_title('B. Number of Associations per Signature', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(range(n_sigs))
ax2.set_xticklabels([f'Sig {i}' for i in range(n_sigs)], rotation=45, ha='right', fontsize=9)
y_max = max(counts_df['Lead Variants (GWAS)'].max(), 
           counts_df['Rare Variants (RVAS)'].max()) * 1.15 if len(counts_df) > 0 else 1
ax2.set_ylim([0, max(y_max, 1)])
ax2.set_xlim([-0.5, n_sigs - 0.5])
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3, axis='y')

# ============================================================================
# RIGHT PANEL: Signature Labels Legend + Top 10 Unique Sig 5 List
# ============================================================================

ax_legend = fig.add_subplot(gs[:, 1])  # Spans both rows
ax_legend.axis('off')

# Create signature labels table
legend_text = "Signature Categories:\n\n"
for i in range(n_sigs):
    sig_label = SIGNATURE_LABELS.get(i, f'Sig {i}')
    legend_text += f"Sig {i:2d}: {sig_label}\n"

ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
              fontsize=9.5, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# Add Top 10 Unique Sig 5 list
if 'is_top10_sig5' in loci_df.columns:
    top10_sig5 = loci_df[(loci_df['SIG_NUM'] == 5) & (loci_df['is_top10_sig5'] == True)].copy()
    
    if len(top10_sig5) > 0:
        top10_sig5 = top10_sig5.sort_values('LOG10P', ascending=False)
        
        top10_list = []
        for idx, row in top10_sig5.iterrows():
            gene = row.get('nearestgene', 'N/A')
            rsid = row.get('rsid', 'N/A')
            if pd.isna(gene) or gene == '' or gene == 'N/A':
                for r, g in THE_10_UNIQUE_SIG5_LOCI.items():
                    if (pd.notna(rsid) and str(rsid) == str(r)) or \
                       (':' in r and pd.notna(row.get('#CHR')) and pd.notna(row.get('POS')) and 
                        str(row.get('#CHR', '')) + ':' + str(row.get('POS', '')) == r):
                        gene = g
                        break
            
            gene_name = gene if pd.notna(gene) else 'N/A'
            rsid_str = str(rsid) if pd.notna(rsid) else 'N/A'
            top10_list.append((gene_name, rsid_str))
        
        top10_text = "\n\n" + "─"*35 + "\n"
        top10_text += "Top 10 Unique Sig 5 Loci\n"
        top10_text += "─"*35 + "\n\n"
        
        for i, (gene, rsid) in enumerate(top10_list, 1):
            top10_text += f"{i:2d}. {gene}\n"
            if len(str(rsid)) > 20:
                rsid = str(rsid)[:17] + "..."
            top10_text += f"    {rsid}\n\n"
        
        ax_legend.text(0.05, 0.40, top10_text, transform=ax_legend.transAxes,
                      fontsize=8.5, verticalalignment='top', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# ============================================================================
# SAVE FIGURE
# ============================================================================

output_dir = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig5")
output_dir.mkdir(parents=True, exist_ok=True)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_file = output_dir / "genetic_validation_multipanel.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n✓ Saved figure to: {output_file}")

plt.show()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("GENETIC ASSOCIATION SUMMARY")
print("="*80)
print(f"Lead Variants (GWAS):")
print(f"  Total: {len(loci_df)}")
print(f"  Signatures with associations: {loci_df['SIG_NUM'].nunique()}")

if best_results is not None and len(best_results) > 0:
    print(f"\nRare Variants (RVAS, mask3):")
    print(f"  Total significant genes: {len(best_results)}")
    print(f"  Unique genes: {best_results['SYMBOL'].nunique()}")
    print(f"  Signatures with associations: {best_results['SIG'].nunique()}")

if 'is_top10_sig5' in loci_df.columns:
    top10_count = loci_df['is_top10_sig5'].sum()
    print(f"\nTop 10 Unique Sig 5 Loci: {top10_count}")

print("="*80)
