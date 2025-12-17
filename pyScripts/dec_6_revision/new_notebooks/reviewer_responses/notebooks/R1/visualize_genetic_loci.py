"""
Visualize lead variants (all_loci_annotated.tsv) and mask3 canonical rare variants
in a comprehensive multipanel figure.
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
plt.rcParams['figure.figsize'] = (22, 16)

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
# These are the 10 loci that are genome-wide significant in Signature 5
# but NOT present in any constituent trait GWAS (novel discoveries)
THE_10_UNIQUE_SIG5_LOCI = {
    'rs6687726': 'IL6R',
    'rs2509121': 'HYOU1',
    'rs4760278': 'R3HDM2',
    'rs1532085': 'LIPC',
    'rs7168222': 'NR2F2-AS1',
    'rs35039495': 'PLCG2',
    'rs8121509': 'OPRL1',
    'rs1499813': 'FNDC3B',
    '4:96088139': 'UNC5C',  # Position-based ID
    'rs4732365': 'C7orf55'
}

# Biological roles for annotation
BIOLOGICAL_ROLES = {
    'IL6R': 'Interleukin-6 receptor, inflammatory pathway',
    'HYOU1': 'Hypoxia up-regulated protein, ER stress response',
    'R3HDM2': 'RNA binding protein',
    'LIPC': 'Hepatic lipase, HDL metabolism',
    'NR2F2-AS1': 'Nuclear receptor antisense RNA',
    'PLCG2': 'Phospholipase C gamma 2, platelet activation',
    'OPRL1': 'Opioid receptor, pain/stress signaling',
    'FNDC3B': 'Fibronectin domain, insulin signaling',
    'UNC5C': 'Netrin receptor, axon guidance',
    'C7orf55': 'Unknown function'
}

# Mark these in the dataframe
loci_df['is_top10_sig5'] = False
if 'rsid' in loci_df.columns:
    for rsid, gene in THE_10_UNIQUE_SIG5_LOCI.items():
        # Try to match by rsid column
        mask = loci_df['rsid'].str.contains(rsid, case=False, na=False)
        loci_df.loc[mask, 'is_top10_sig5'] = True
        # Also try matching by position if rsid doesn't work
        if not mask.any() and ':' in rsid:
            # Handle position-based IDs like '4:96088139'
            chr_pos = rsid.split(':')
            if len(chr_pos) == 2:
                mask_pos = (loci_df['#CHR'] == int(chr_pos[0])) & (loci_df['POS'] == int(chr_pos[1]))
                loci_df.loc[mask_pos, 'is_top10_sig5'] = True

# Also check by gene name if available
if 'nearestgene' in loci_df.columns:
    for rsid, gene in THE_10_UNIQUE_SIG5_LOCI.items():
        gene_mask = (loci_df['SIG_NUM'] == 5) & (loci_df['nearestgene'].str.contains(gene, case=False, na=False))
        loci_df.loc[gene_mask, 'is_top10_sig5'] = True

print(f"\nMarked {loci_df['is_top10_sig5'].sum()} variants as 'Top 10 Unique Sig 5' loci")

# ============================================================================
# IDENTIFY NOVEL VS KNOWN LOCI
# ============================================================================
# Note: Component trait GWAS files are only available for Signature 5 (cardiovascular)
# So we only check for novelty for Sig 5 loci

# Load component trait GWAS files to identify novel loci (for Sig 5 only)
component_trait_dir = Path("/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/tetgwas/result/10_loci")
component_trait_files = [
    "Angina_pectoris_ukb_eur_regenie_af1.sig.lead.sumstats.txt",
    "Coronary_atherosclerosis_ukb_eur_regenie_af1.sig.lead.sumstats.txt",
    "Hypercholesterolemia_ukb_eur_regenie_af1.sig.lead.sumstats.txt",
    "Myocardial_infarction_ukb_eur_regenie_af1.sig.lead.sumstats.txt",
    "Other_acute_and_subacute_forms_of_ischemic_heart_disease_ukb_eur_regenie_af1.sig.lead.sumstats.txt",
    "Other_chronic_ischemic_heart_disease,_unspecified_ukb_eur_regenie_af1.sig.lead.sumstats.txt"
]

print(f"\nChecking for novel loci (not in component trait GWAS) - Signature 5 only...")
component_trait_loci = []

for trait_file in component_trait_files:
    trait_path = component_trait_dir / trait_file
    if trait_path.exists():
        try:
            # Read file - header starts with #CHR, so we need to handle it specially
            # First line is the header (starts with #)
            with open(trait_path, 'r') as f:
                header_line = f.readline().strip()
                # Remove leading # from header
                if header_line.startswith('#'):
                    header_line = header_line[1:]
                # Read the rest
                trait_df = pd.read_csv(trait_path, sep='\t', skiprows=1)
                # Rename columns if needed (remove # from #CHR)
                if '#CHR' in trait_df.columns:
                    trait_df = trait_df.rename(columns={'#CHR': 'CHR'})
                elif 'CHR' not in trait_df.columns:
                    # Try to set column names from header
                    trait_df.columns = header_line.split('\t')
                    if '#CHR' in trait_df.columns:
                        trait_df = trait_df.rename(columns={'#CHR': 'CHR'})
                
                if len(trait_df) > 0 and 'CHR' in trait_df.columns and 'POS' in trait_df.columns:
                    component_trait_loci.append(trait_df)
                    print(f"  Loaded {len(trait_df)} loci from {trait_file}")
        except Exception as e:
            print(f"  ⚠️  Could not load {trait_file}: {e}")

# Initialize novelty column - default to None (unknown) for all signatures
loci_df['is_novel'] = None
loci_df['component_trait_overlap'] = False

if len(component_trait_loci) > 0:
    # Combine all component trait loci
    all_component_loci = pd.concat(component_trait_loci, ignore_index=True)
    print(f"  Total component trait loci: {len(all_component_loci)}")
    
    # Standardize column names
    chr_col = 'CHR' if 'CHR' in all_component_loci.columns else '#CHR'
    if chr_col not in all_component_loci.columns:
        print(f"  ⚠️  Could not find chromosome column in component trait files")
        chr_col = None
    
    if chr_col and 'POS' in all_component_loci.columns:
        # Create a set of (chr, pos) tuples for component trait loci
        component_loci_set = set()
        for _, row in all_component_loci.iterrows():
            chr_num = row[chr_col]
            pos = row['POS']
            component_loci_set.add((chr_num, pos))
        
        # Create a more efficient lookup: group component trait loci by chromosome
        component_loci_by_chr = {}
        for comp_chr, comp_pos in component_loci_set:
            if comp_chr not in component_loci_by_chr:
                component_loci_by_chr[comp_chr] = []
            component_loci_by_chr[comp_chr].append(comp_pos)
        
        # Only check for Sig 5 loci (component traits are cardiovascular)
        sig5_loci = loci_df[loci_df['SIG_NUM'] == 5].copy()
        
        for idx in sig5_loci.index:
            row = loci_df.loc[idx]
            sig_chr = row['#CHR']
            sig_pos = row['POS']
            
            # Check if this locus overlaps with any component trait locus (within 1MB window)
            sig_locus_from = row.get('LOCUS_FROM', sig_pos - 500000)
            sig_locus_to = row.get('LOCUS_TO', sig_pos + 500000)
            
            # Check for overlap with component trait loci on same chromosome
            if sig_chr in component_loci_by_chr:
                for comp_pos in component_loci_by_chr[sig_chr]:
                    # Check if component trait locus is within signature locus window
                    if sig_locus_from <= comp_pos <= sig_locus_to:
                        loci_df.at[idx, 'is_novel'] = False
                        loci_df.at[idx, 'component_trait_overlap'] = True
                        break
                    # Also check if signature locus is within 1MB of component trait locus
                    elif abs(comp_pos - sig_pos) <= 1000000:
                        loci_df.at[idx, 'is_novel'] = False
                        loci_df.at[idx, 'component_trait_overlap'] = True
                        break
            
            # If we didn't find overlap, mark as novel
            if loci_df.at[idx, 'is_novel'] is None:
                loci_df.at[idx, 'is_novel'] = True
        
        # For Sig 5, show summary
        sig5_novel = loci_df[(loci_df['SIG_NUM'] == 5) & (loci_df['is_novel'] == True)]
        sig5_known = loci_df[(loci_df['SIG_NUM'] == 5) & (loci_df['is_novel'] == False)]
        print(f"  Signature 5: {len(sig5_novel)} novel, {len(sig5_known)} known loci")
        
        # Show examples of novel Sig 5 loci
        if len(sig5_novel) > 0:
            print(f"\n  Examples of novel Sig 5 loci:")
            novel_examples = sig5_novel.nlargest(5, 'LOG10P')
            for _, row in novel_examples.iterrows():
                gene = row.get('nearestgene', 'N/A')
                rsid = row.get('rsid', 'N/A')
                print(f"    {gene:15} {rsid:15} LOG10P={row['LOG10P']:6.2f}")
    else:
        print("  ⚠️  Could not process component trait loci - column names not found")
else:
    print("  ⚠️  No component trait GWAS files found - cannot identify novel loci")

# ============================================================================
# SIGNATURE LABELS (based on top disease associations)
# ============================================================================
SIGNATURE_LABELS = {
    0: 'Cardiac Arrhythmia/Heart Failure',
    1: 'Musculoskeletal',
    2: 'GI/Upper Digestive',
    3: 'Vascular/Misc',
    4: 'Upper Respiratory',
    5: 'Ischemia/Cardiovascular',
    6: 'Neoplastic',
    7: 'Asthma/Migraine/Metabolic',
    8: 'Female Reproductive',
    9: 'Back/Spine',
    10: 'Ophthalmologic',
    11: 'Cerebrovascular',
    12: 'Renal',
    13: 'Male Urogenital',
    14: 'Pulmonary',
    15: 'Diabetes/Metabolic',
    16: 'Infections/Septic',
    17: 'GI/Colorectal',
    18: 'Gallbladder/Biliary',
    19: 'Skin/Kidney',
    20: 'Healthy/Protective'
}

# Load mask3 canonical rare variants
results_dir = Path("/Users/sarahurbut/Desktop/SIG/gene_based_analysis")
canonical_dir = results_dir / "canonical"

print(f"\nLoading mask3 (LoF variants only) results from {canonical_dir}...")
mask3_files = sorted(canonical_dir.glob("Mask3_*_significant_canonical.tsv"))

if len(mask3_files) == 0:
    print(f"⚠️  No mask3 files found in {canonical_dir}")
    print(f"   Looking for files matching: Mask3_*_significant_canonical.tsv")
    mask3_df = None
    best_results = None
else:
    mask3_results = []
    
    for file in mask3_files:
        # Parse filename: Mask3_{MAF}_significant_canonical.tsv
        parts = file.stem.replace("_significant_canonical", "").split("_")
        maf = parts[1]   # e.g., "0.01" or "singleton"
        
        df = pd.read_csv(file, sep='\t')
        df['MAF'] = maf
        mask3_results.append(df)
    
    # Combine all mask3 results
    mask3_df = pd.concat(mask3_results, ignore_index=True)
    print(f"Loaded {len(mask3_df)} significant associations from mask3 across {len(mask3_files)} MAF thresholds")
    
    # Get best result per gene-signature (across MAF thresholds)
    best_results = mask3_df.loc[mask3_df.groupby(['SIG', 'SYMBOL'])['LOG10P'].idxmax()].copy()
    best_results = best_results.sort_values(['SIG', 'LOG10P'], ascending=[True, False])
    
    print(f"Best results per gene-signature: {len(best_results)} associations")
    print(f"Unique genes: {best_results['SYMBOL'].nunique()}")
    print(f"Signatures with significant genes: {best_results['SIG'].nunique()}")

# ============================================================================
# CREATE MULTIPANEL FIGURE
# ============================================================================

# Improved layout: 3x4 grid with better spacing for GWAS/RVAS emphasis
fig = plt.figure(figsize=(20, 14))
# Layout: Panels A-E in columns 0-2, legend in column 3
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                      width_ratios=[1, 1, 1, 0.7], height_ratios=[1.2, 1, 1.1])

# Main title (optional - can be removed for publication)
# fig.suptitle('Genetic Associations: Lead Variants and Rare Variant Gene-Based Associations', 
#              fontsize=20, fontweight='bold', y=0.98)

# Color palette for signatures
n_sigs = 21
sig_colors = plt.cm.tab20(np.linspace(0, 1, 20))
# Add one more color for signature 20
sig_colors = np.vstack([sig_colors, plt.cm.tab20b(0.5)])
sig_color_dict = {i: sig_colors[i] for i in range(n_sigs)}

# ============================================================================
# PANEL A: Lead Variants by Signature (Top Left)
# ============================================================================

ax1 = fig.add_subplot(gs[0, 0:2])  # Top row, spans first two columns for better visibility

# Group lead variants by signature
sig_groups = loci_df.groupby('SIG_NUM')

# Create scatter plot: signature vs -log10(P-value)
# Separate novel and known loci
for sig_num in sorted(loci_df['SIG_NUM'].unique()):
    sig_data = loci_df[loci_df['SIG_NUM'] == sig_num]
    
    # Plot known loci (only for signatures where we have novelty info, i.e., Sig 5)
    if 'is_novel' in sig_data.columns:
        known_data = sig_data[sig_data['is_novel'] == False]
        if len(known_data) > 0:
            x_pos_known = sig_num + np.random.normal(0, 0.08, len(known_data))
            ax1.scatter(x_pos_known, known_data['LOG10P'], 
                        alpha=0.6, s=80, color=sig_color_dict[sig_num],
                        edgecolors='black', linewidths=0.5, marker='o',
                        label='Known (Sig 5)' if sig_num == 5 and sig_num == sorted(loci_df['SIG_NUM'].unique())[0] else '')
        
        # Plot novel loci - separate the Top 10 Unique Sig 5 loci (highlighted in green)
        novel_data = sig_data[sig_data['is_novel'] == True]
        if len(novel_data) > 0:
            # Separate Top 10 Unique loci from other novel loci
            if 'is_top10_sig5' in novel_data.columns:
                top10_data = novel_data[novel_data['is_top10_sig5'] == True]
                other_novel_data = novel_data[novel_data['is_top10_sig5'] != True]
            else:
                top10_data = pd.DataFrame()
                other_novel_data = novel_data
            
            # Plot Top 10 Unique loci with green highlighting
            if len(top10_data) > 0:
                x_pos_top10 = sig_num + np.random.normal(0, 0.08, len(top10_data))
                ax1.scatter(x_pos_top10, top10_data['LOG10P'], 
                            alpha=0.95, s=200, color='#2ecc71',  # Green color
                            edgecolors='darkgreen', linewidths=2.5, marker='*',
                            zorder=5,  # Draw on top
                            label='Top 10 Unique Sig 5 (green)' if sig_num == 5 else '')
            
            # Plot other novel loci (red star)
            if len(other_novel_data) > 0:
                x_pos_other_novel = sig_num + np.random.normal(0, 0.08, len(other_novel_data))
                ax1.scatter(x_pos_other_novel, other_novel_data['LOG10P'], 
                            alpha=0.9, s=120, color=sig_color_dict[sig_num],
                            edgecolors='red', linewidths=2, marker='*',
                            label='Novel (Sig 5)' if sig_num == 5 and len(top10_data) == 0 and sig_num == sorted(loci_df['SIG_NUM'].unique())[0] else '')
        
        # Plot other loci (no novelty info available)
        other_data = sig_data[sig_data['is_novel'].isna()]
        if len(other_data) > 0:
            x_pos_other = sig_num + np.random.normal(0, 0.08, len(other_data))
            ax1.scatter(x_pos_other, other_data['LOG10P'], 
                        alpha=0.6, s=80, color=sig_color_dict[sig_num],
                        edgecolors='black', linewidths=0.5, marker='o',
                        label=f'Sig {sig_num}' if sig_num == sorted(loci_df['SIG_NUM'].unique())[0] else '')
    else:
        # If no novelty info at all, plot all together
        x_pos = sig_num + np.random.normal(0, 0.08, len(sig_data))
        ax1.scatter(x_pos, sig_data['LOG10P'], 
                    alpha=0.6, s=80, color=sig_color_dict[sig_num],
                    edgecolors='black', linewidths=0.5,
                    label=f'Sig {sig_num}' if sig_num == sorted(loci_df['SIG_NUM'].unique())[0] else '')

# Label variants, prioritizing Top 10 Unique Sig 5 loci, then other novel, then top variants
for sig_num in sorted(loci_df['SIG_NUM'].unique()):
    sig_data = loci_df[loci_df['SIG_NUM'] == sig_num]
    if len(sig_data) > 0:
        # First, label all Top 10 Unique Sig 5 loci with green callout boxes
        if sig_num == 5 and 'is_top10_sig5' in sig_data.columns:
            top10_data = sig_data[sig_data['is_top10_sig5'] == True]
            for idx, row in top10_data.iterrows():
                gene_name = row.get('nearestgene', '')
                if pd.notna(gene_name) and gene_name != '':
                    # Use green background with dark green border for Top 10 Unique
                    ax1.annotate(gene_name, 
                                xy=(sig_num, row['LOG10P']),
                                xytext=(8, 8), textcoords='offset points',
                                fontsize=10, alpha=0.95, fontweight='bold', color='darkgreen',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='#90EE90',  # Light green
                                         alpha=0.85, edgecolor='darkgreen', linewidth=2),
                                zorder=10)  # Draw annotations on top
        
        # Then label other novel variants (Sig 5, non-Top10)
        novel_data = sig_data[sig_data['is_novel'] == True] if 'is_novel' in sig_data.columns else pd.DataFrame()
        if len(novel_data) > 0:
            if 'is_top10_sig5' in novel_data.columns:
                other_novel = novel_data[novel_data['is_top10_sig5'] != True]
            else:
                other_novel = novel_data
            if len(other_novel) > 0:
                top_other_novel = other_novel.loc[other_novel['LOG10P'].idxmax()]
                if top_other_novel['LOG10P'] > 7:
                    gene_name = top_other_novel.get('nearestgene', '')
                    if pd.notna(gene_name) and gene_name != '':
                        ax1.annotate(f'{gene_name}*',  # Asterisk indicates novel
                                    xy=(sig_num, top_other_novel['LOG10P']),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=9, alpha=0.9, fontweight='bold', color='red',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='red'))
        elif 'is_top10_sig5' not in sig_data.columns or len(sig_data[sig_data['is_top10_sig5'] == True]) == 0:
            # Label top variant if no novel variants (and no Top10)
            top_variant = sig_data.loc[sig_data['LOG10P'].idxmax()]
            if top_variant['LOG10P'] > 8:
                gene_name = top_variant.get('nearestgene', '')
                if pd.notna(gene_name) and gene_name != '':
                    ax1.annotate(gene_name, 
                                xy=(sig_num, top_variant['LOG10P']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.8, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Genome-wide significance threshold (p=5e-8)
ax1.axhline(y=-np.log10(5e-8), color='red', linestyle='--', alpha=0.7, linewidth=2,
            label='Genome-wide significance (p=5×10⁻⁸)')

ax1.set_xlabel('Signature', fontsize=13, fontweight='bold')
ax1.set_ylabel('-log₁₀(P-value)', fontsize=13, fontweight='bold')
ax1.set_title('A. Lead Variants by Signature (GWAS)\n★ = Novel Sig 5 loci; Green ★ = Top 10 Unique Sig 5 discoveries', fontsize=14, fontweight='bold', pad=10)
ax1.set_xticks(range(n_sigs))
ax1.set_xticklabels([f'Sig {i}' for i in range(n_sigs)], rotation=45, ha='right', fontsize=10)
# Set axis limits to ensure data is visible with padding
y_max = loci_df['LOG10P'].max() * 1.1 if len(loci_df) > 0 else 10
ax1.set_ylim([0, max(y_max, 10)])
ax1.set_xlim([-0.5, n_sigs - 0.5])
ax1.grid(True, alpha=0.3, axis='y')

# Create custom legend (only show if we have novelty info)
from matplotlib.lines import Line2D
if 'is_novel' in loci_df.columns and loci_df['is_novel'].notna().any():
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, label='Known Sig 5 (in component traits)', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71', 
               markersize=14, label='Top 10 Unique Sig 5 (green, highlighted)', 
               markeredgecolor='darkgreen', markeredgewidth=2.5),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markersize=12, label='Other Novel Sig 5 (red border)', 
               markeredgecolor='red', markeredgewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# ============================================================================
# PANEL B: Significant Genes by Signature (mask3) - Scatter Plot (Top Right)
# ============================================================================

ax2 = fig.add_subplot(gs[0, 2])  # Top row, third column

if best_results is not None and len(best_results) > 0:
    # Group by signature and plot
    for sig_num in sorted(best_results['SIG'].unique()):
        sig_data = best_results[best_results['SIG'] == sig_num]
        x_pos = sig_num + np.random.normal(0, 0.08, len(sig_data))  # Jitter for visibility
        ax2.scatter(x_pos, sig_data['LOG10P'], 
                    alpha=0.7, s=120, color=sig_color_dict[sig_num],
                    edgecolors='black', linewidths=1,
                    label=f'Sig {sig_num}' if len(sig_data) > 0 else '')
        
        # Label top genes
        top_gene = sig_data.loc[sig_data['LOG10P'].idxmax()]
        if top_gene['LOG10P'] > 5:
            ax2.annotate(top_gene['SYMBOL'], 
                        xy=(sig_num, top_gene['LOG10P']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Genome-wide significance threshold (p=2.5e-6 for gene-based tests)
    ax2.axhline(y=-np.log10(2.5e-6), color='red', linestyle='--', alpha=0.7, linewidth=2,
                label='Genome-wide significance (p=2.5×10⁻⁶)')
    
    ax2.set_xlabel('Signature', fontsize=13, fontweight='bold')
    ax2.set_ylabel('-log₁₀(P-value)', fontsize=13, fontweight='bold')
    ax2.set_title('B. Significant Genes by Signature (mask3: LoF Variants)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(range(n_sigs))
    ax2.set_xticklabels([f'Sig {i}' for i in range(n_sigs)], rotation=45, ha='right', fontsize=10)
    # Set axis limits to ensure data is visible
    if len(best_results) > 0:
        y_max = best_results['LOG10P'].max() * 1.15 if len(best_results) > 0 else 10
        ax2.set_ylim([0, max(y_max, 10)])
    ax2.set_xlim([-0.5, n_sigs - 0.5])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, ncol=1, framealpha=0.9)
else:
    ax2.text(0.5, 0.5, 'Mask3 data not available', 
             ha='center', va='center', fontsize=14,
             transform=ax2.transAxes)
    ax2.set_title('B. Significant Genes by Signature (mask3: LoF Variants)', fontsize=14, fontweight='bold', pad=10)

# ============================================================================
# PANEL C: Number of Lead Variants per Signature (Bottom Left)
# ============================================================================

ax3 = fig.add_subplot(gs[1, 0:2])  # Middle row, spans first two columns

sig_counts_loci = loci_df.groupby('SIG_NUM').size().sort_index()
bars1 = ax3.bar(sig_counts_loci.index, sig_counts_loci.values, 
                color=[sig_color_dict[i] for i in sig_counts_loci.index], 
                alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    if height > 0:
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xlabel('Signature', fontsize=13, fontweight='bold')
ax3.set_ylabel('Number of Lead Variants', fontsize=13, fontweight='bold')
ax3.set_title('C. Number of Lead Variants per Signature', fontsize=14, fontweight='bold', pad=10)
ax3.set_xticks(range(n_sigs))
ax3.set_xticklabels([f'Sig {i}' for i in range(n_sigs)], rotation=45, ha='right', fontsize=10)
# Set axis limits with padding
y_max = sig_counts_loci.max() * 1.1 if len(sig_counts_loci) > 0 else 1
ax3.set_ylim([0, max(y_max, 1)])
ax3.set_xlim([-0.5, n_sigs - 0.5])
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# PANEL D: Number of Significant Genes per Signature (mask3) - Bottom Right
# ============================================================================

ax4 = fig.add_subplot(gs[1, 2])  # Middle row, third column

if best_results is not None and len(best_results) > 0:
    sig_counts_genes = best_results.groupby('SIG').size().sort_index()
    bars2 = ax4.bar(sig_counts_genes.index, sig_counts_genes.values, 
                    color=[sig_color_dict[i] for i in sig_counts_genes.index], 
                    alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Signature', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Number of Significant Genes', fontsize=13, fontweight='bold')
    ax4.set_title('D. Number of Significant Genes per Signature (mask3)', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xticks(range(n_sigs))
    ax4.set_xticklabels([f'Sig {i}' for i in range(n_sigs)], rotation=45, ha='right', fontsize=10)
    # Set axis limits with padding
    y_max = sig_counts_genes.max() * 1.1 if len(sig_counts_genes) > 0 else 1
    ax4.set_ylim([0, max(y_max, 1)])
    ax4.set_xlim([-0.5, n_sigs - 0.5])
    ax4.grid(True, alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, 'Mask3 data not available', 
             ha='center', va='center', fontsize=14,
             transform=ax4.transAxes)
    ax4.set_title('D. Number of Significant Genes per Signature (mask3)', fontsize=14, fontweight='bold', pad=10)

# ============================================================================
# PANEL E: Gene Overlap - Lead Variant Genes vs Rare Variant Genes (Bottom Span)
# ============================================================================

ax5 = fig.add_subplot(gs[2, 0:3])  # Bottom row, spans first three columns

if best_results is not None and len(best_results) > 0:
    # Get unique genes from each analysis
    lead_variant_genes = set(loci_df['nearestgene'].dropna().unique())
    rare_variant_genes = set(best_results['SYMBOL'].unique())
    
    # Find overlap
    overlapping_genes = lead_variant_genes & rare_variant_genes
    lead_only = lead_variant_genes - rare_variant_genes
    rare_only = rare_variant_genes - lead_variant_genes
    
    # Create a summary table
    overlap_data = []
    for sig_num in range(n_sigs):
        # Lead variant genes for this signature
        sig_lead_genes = set(loci_df[loci_df['SIG_NUM'] == sig_num]['nearestgene'].dropna().unique())
        # Rare variant genes for this signature
        sig_rare_genes = set(best_results[best_results['SIG'] == sig_num]['SYMBOL'].unique())
        
        sig_overlap = sig_lead_genes & sig_rare_genes
        sig_lead_only = sig_lead_genes - sig_rare_genes
        sig_rare_only = sig_rare_genes - sig_lead_genes
        
        overlap_data.append({
            'Signature': sig_num,
            'Lead Variants': len(sig_lead_genes),
            'Rare Variants': len(sig_rare_genes),
            'Overlap': len(sig_overlap),
            'Lead Only': len(sig_lead_only),
            'Rare Only': len(sig_rare_only)
        })
    
    overlap_df = pd.DataFrame(overlap_data)
    
    # Create stacked bar chart
    x = np.arange(n_sigs)
    width = 0.6
    
    bars_lead = ax5.bar(x - width/3, overlap_df['Lead Only'], width/3, 
                        label='Lead Variants Only', color='#3498db', alpha=0.7, edgecolor='black')
    bars_overlap = ax5.bar(x, overlap_df['Overlap'], width/3,
                           label='Both (Overlap)', color='#e74c3c', alpha=0.7, edgecolor='black')
    bars_rare = ax5.bar(x + width/3, overlap_df['Rare Only'], width/3,
                        label='Rare Variants Only', color='#2ecc71', alpha=0.7, edgecolor='black')
    
    # Add total labels
    for i, (lead, overlap, rare) in enumerate(zip(overlap_df['Lead Only'], 
                                                   overlap_df['Overlap'], 
                                                   overlap_df['Rare Only'])):
        total = lead + overlap + rare
        if total > 0:
            ax5.text(i, total + 0.3, f'{int(total)}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax5.set_xlabel('Signature', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Number of Genes', fontsize=13, fontweight='bold')
    ax5.set_title('E. Gene Overlap: Lead Variants vs Rare Variants (mask3) by Signature', 
                  fontsize=14, fontweight='bold', pad=10)
    ax5.set_xticks(range(n_sigs))
    ax5.set_xticklabels([f'Sig {i}' for i in range(n_sigs)], rotation=45, ha='right', fontsize=10)
    # Set axis limits with padding
    max_total = overlap_df[['Lead Only', 'Overlap', 'Rare Only']].sum(axis=1).max()
    y_max = max_total * 1.15 if max_total > 0 else 1
    ax5.set_ylim([0, max(y_max, 1)])
    ax5.set_xlim([-0.5, n_sigs - 0.5])
    ax5.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Print summary
    print("\n" + "="*80)
    print("GENE OVERLAP SUMMARY")
    print("="*80)
    print(f"Total unique genes in lead variants: {len(lead_variant_genes)}")
    print(f"Total unique genes in rare variants (mask3): {len(rare_variant_genes)}")
    print(f"Overlapping genes: {len(overlapping_genes)}")
    if len(overlapping_genes) > 0:
        print(f"  Overlapping genes: {', '.join(sorted(overlapping_genes))}")
    print(f"Lead variant only: {len(lead_only)}")
    print(f"Rare variant only: {len(rare_only)}")
    
    # Show top signatures by overlap
    print("\nTop signatures by gene overlap:")
    top_overlap = overlap_df.nlargest(5, 'Overlap')
    for _, row in top_overlap.iterrows():
        if row['Overlap'] > 0:
            sig_num = int(row['Signature'])
            sig_lead_genes = set(loci_df[loci_df['SIG_NUM'] == sig_num]['nearestgene'].dropna().unique())
            sig_rare_genes = set(best_results[best_results['SIG'] == sig_num]['SYMBOL'].unique())
            sig_overlap_genes = sig_lead_genes & sig_rare_genes
            print(f"  Signature {sig_num}: {row['Overlap']} overlapping genes - {', '.join(sorted(sig_overlap_genes))}")
else:
    ax5.text(0.5, 0.5, 'Mask3 data not available for overlap analysis', 
             ha='center', va='center', fontsize=14,
             transform=ax5.transAxes)
    ax5.set_title('E. Gene Overlap: Lead Variants vs Rare Variants (mask3)', 
                  fontsize=14, fontweight='bold', pad=10)

# ============================================================================
# RIGHT PANEL: Signature Labels Legend + Top 10 Unique Sig 5 List
# ============================================================================

ax_legend = fig.add_subplot(gs[:, 3])  # Rightmost column (column 3), spans all rows
ax_legend.axis('off')

# Create signature labels table
legend_text = "Signature Categories:\n\n"
for i in range(n_sigs):
    sig_label = SIGNATURE_LABELS.get(i, f'Sig {i}')
    legend_text += f"Sig {i:2d}: {sig_label}\n"

ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
              fontsize=10, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# Add Top 10 Unique Sig 5 list below
if 'is_top10_sig5' in loci_df.columns:
    top10_sig5 = loci_df[(loci_df['SIG_NUM'] == 5) & (loci_df['is_top10_sig5'] == True)].copy()
    
    if len(top10_sig5) > 0:
        # Sort by LOG10P
        top10_sig5 = top10_sig5.sort_values('LOG10P', ascending=False)
        
        # Get gene names and rsids
        top10_list = []
        for idx, row in top10_sig5.iterrows():
            gene = row.get('nearestgene', 'N/A')
            rsid = row.get('rsid', 'N/A')
            if pd.isna(gene) or gene == '' or gene == 'N/A':
                # Try to find gene from THE_10_UNIQUE_SIG5_LOCI dict
                for r, g in THE_10_UNIQUE_SIG5_LOCI.items():
                    if (pd.notna(rsid) and str(rsid) == str(r)) or \
                       (':' in r and pd.notna(row.get('#CHR')) and pd.notna(row.get('POS')) and 
                        str(row.get('#CHR', '')) + ':' + str(row.get('POS', '')) == r):
                        gene = g
                        break
            
            gene_name = gene if pd.notna(gene) else 'N/A'
            rsid_str = str(rsid) if pd.notna(rsid) else 'N/A'
            role = BIOLOGICAL_ROLES.get(gene_name, 'Unknown function')
            top10_list.append((gene_name, rsid_str, role))
        
        # Create list text - simpler format
        top10_text = "\n\n" + "─"*40 + "\n"
        top10_text += "Top 10 Unique Sig 5 Loci\n"
        top10_text += "(Not in component trait GWAS)\n"
        top10_text += "─"*40 + "\n\n"
        
        for i, (gene, rsid, role) in enumerate(top10_list, 1):
            top10_text += f"{i:2d}. {gene} ({rsid})\n"
            top10_text += f"    {role}\n\n"
        
        ax_legend.text(0.05, 0.40, top10_text, transform=ax_legend.transAxes,
                      fontsize=9, verticalalignment='top', family='monospace',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.4, edgecolor='darkgreen', linewidth=1.5))

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save figure
output_dir = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/fig4")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "genetic_loci_visualization.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved figure to: {output_file}")

plt.show()

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nLead Variants (GWAS):")
print(f"  Total variants: {len(loci_df)}")
print(f"  Unique signatures: {loci_df['SIG_NUM'].nunique()}")
if 'is_novel' in loci_df.columns and loci_df['is_novel'].notna().any():
    # Only count Sig 5 for novelty (component traits are only for Sig 5)
    sig5_data = loci_df[loci_df['SIG_NUM'] == 5]
    if len(sig5_data) > 0:
        novel_count = (sig5_data['is_novel'] == True).sum()
        known_count = (sig5_data['is_novel'] == False).sum()
        print(f"  Signature 5 - Novel loci (not in component traits): {novel_count} ({100*novel_count/len(sig5_data):.1f}%)")
        print(f"  Signature 5 - Known loci (in component traits): {known_count} ({100*known_count/len(sig5_data):.1f}%)")
print(f"  Mean -log10(P) per signature:")
for sig_num in sorted(loci_df['SIG_NUM'].unique()):
    sig_data = loci_df[loci_df['SIG_NUM'] == sig_num]
    mean_log10p = sig_data['LOG10P'].mean()
    if 'is_novel' in sig_data.columns:
        novel_in_sig = (sig_data['is_novel'] == True).sum()
        known_in_sig = (sig_data['is_novel'] == False).sum()
        if novel_in_sig > 0 or known_in_sig > 0:
            print(f"    Signature {sig_num}: {mean_log10p:.2f} ({len(sig_data)} variants, {novel_in_sig} novel, {known_in_sig} known)")
        else:
            print(f"    Signature {sig_num}: {mean_log10p:.2f} ({len(sig_data)} variants)")
    else:
        print(f"    Signature {sig_num}: {mean_log10p:.2f} ({len(sig_data)} variants)")

if best_results is not None and len(best_results) > 0:
    print(f"\nRare Variants (mask3, gene-based):")
    print(f"  Total significant gene-signature associations: {len(best_results)}")
    print(f"  Unique genes: {best_results['SYMBOL'].nunique()}")
    print(f"  Unique signatures: {best_results['SIG'].nunique()}")
    print(f"  Mean -log10(P) per signature:")
    for sig_num in sorted(best_results['SIG'].unique()):
        sig_data = best_results[best_results['SIG'] == sig_num]
        mean_log10p = sig_data['LOG10P'].mean()
        print(f"    Signature {sig_num}: {mean_log10p:.2f} ({len(sig_data)} genes)")

# ============================================================================
# PRINT SUMMARY OF TOP 10 UNIQUE SIGNATURE 5 LOCI
# ============================================================================
if 'is_top10_sig5' in loci_df.columns:
    top10_sig5 = loci_df[(loci_df['SIG_NUM'] == 5) & (loci_df['is_top10_sig5'] == True)]
    if len(top10_sig5) > 0:
        print("\n" + "="*80)
        print("THE 10 UNIQUE SIGNATURE 5 LOCI (Highlighted in Green)")
        print("="*80)
        print("These loci are genome-wide significant in Signature 5 but NOT present")
        print("in any constituent trait GWAS (Angina, MI, Hypercholesterolemia, etc.)")
        print("\nList:")
        for idx, row in top10_sig5.sort_values('LOG10P', ascending=False).iterrows():
            rsid = row.get('rsid', 'N/A')
            gene = row.get('nearestgene', 'N/A')
            log10p = row['LOG10P']
            role = BIOLOGICAL_ROLES.get(gene, 'Unknown function')
            print(f"  {rsid:15} {gene:12} LOG10P={log10p:6.2f} - {role}")
        print(f"\nTotal found in data: {len(top10_sig5)} / 10 expected")
        if len(top10_sig5) < 10:
            missing = set(THE_10_UNIQUE_SIG5_LOCI.values()) - set(top10_sig5.get('nearestgene', []).dropna())
            if len(missing) > 0:
                print(f"  Missing genes: {', '.join(missing)}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
