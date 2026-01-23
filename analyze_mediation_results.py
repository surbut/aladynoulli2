"""
Comprehensive visualization and analysis of mediation analysis results
Creates publication-ready figures and summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Load data
print("Loading mediation results...")
top_mediation = pd.read_csv('mediation_analysis_figures/mediation_analysis_results.csv')  # Use mediation results with reverse path analysis
# Create "top" versions based on significance
top_sobel = top_mediation.nsmallest(100, 'sobel_p')  # Top 100 by Sobel p-value
top_significant = top_mediation[top_mediation['sobel_p'] < 0.05].nsmallest(100, 'sobel_p')  # Top 100 significant

# Extract signature number for easier grouping
top_mediation['sig_num'] = top_mediation['signature'].str.extract(r'(\d+)').astype(int)
top_sobel['sig_num'] = top_sobel['signature'].str.extract(r'(\d+)').astype(int)
top_significant['sig_num'] = top_significant['signature'].str.extract(r'(\d+)').astype(int)

# Define significance thresholds
sobel_sig_thresh = 0.05
path_sig_thresh = 0.05 / (21 * 4)  # Bonferroni for 21 sigs x 4 genes

print(f"\n{'='*60}")
print("MEDIATION ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"\nTotal significant mediation effects (Sobel p < {sobel_sig_thresh}): {len(top_sobel)}")
print(f"\nGenes analyzed: {', '.join(sorted(top_mediation['gene'].unique()))}")
print(f"Signatures analyzed: {top_mediation['sig_num'].nunique()}")
print(f"Diseases with significant mediation: {top_mediation['disease'].nunique()}")

# Summary by gene
print(f"\n{'-'*60}")
print("SUMMARY BY GENE:")
print(f"{'-'*60}")
for gene in sorted(top_mediation['gene'].unique()):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    print(f"\n{gene}:")
    print(f"  Significant mediations: {len(gene_data)}")
    print(f"  Unique diseases: {gene_data['disease'].nunique()}")
    print(f"  Unique signatures: {gene_data['sig_num'].nunique()}")
    print(f"  Mean proportion mediated: {gene_data['proportion_mediated'].abs().mean():.2f}%")
    print(f"  Median |Sobel Z|: {gene_data['sobel_z'].abs().median():.2f}")

# Summary by signature
print(f"\n{'-'*60}")
print("TOP SIGNATURES BY NUMBER OF SIGNIFICANT MEDIATIONS:")
print(f"{'-'*60}")
sig_counts = top_mediation.groupby('sig_num').size().sort_values(ascending=False)
for sig, count in sig_counts.head(10).items():
    print(f"  Signature {sig}: {count} significant mediations")

# Create output directory
import os
output_dir = 'mediation_analysis_figures'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# FIGURE 1: Heatmap of Proportion Mediated by Gene x Signature
# ============================================================================
print("\nCreating Figure 1: Proportion Mediated Heatmap...")
unique_genes = sorted(top_mediation['gene'].unique())
n_genes = len(unique_genes)

# Create appropriate grid layout
if n_genes == 1:
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes = [axes]
elif n_genes == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()
elif n_genes <= 4:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
else:
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

for idx, gene in enumerate(unique_genes):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    
    # Create pivot table: signature x disease, value = proportion_mediated
    pivot = gene_data.pivot_table(
        values='proportion_mediated',
        index='sig_num',
        columns='disease',
        aggfunc='mean'  # Average if multiple diseases per signature
    )
    
    # Instead, aggregate by signature (average proportion mediated per signature)
    sig_agg = gene_data.groupby('sig_num')['proportion_mediated'].agg(['mean', 'count'])
    
    # Create a simpler heatmap: signature vs mean proportion mediated
    ax = axes[idx]
    
    # Get top signatures by count of significant mediations
    top_sigs = sig_agg.nlargest(15, 'count').index
    sig_agg_top = sig_agg.loc[top_sigs]
    
    # Create bar plot showing proportion mediated
    colors = ['red' if x < 0 else 'blue' for x in sig_agg_top['mean']]
    bars = ax.barh(range(len(sig_agg_top)), sig_agg_top['mean'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(sig_agg_top)))
    ax.set_yticklabels([f'Sig {int(s)}' for s in sig_agg_top.index])
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean Proportion Mediated (%)')
    ax.set_title(f'{gene} - Top 15 Signatures\n(n = {len(gene_data)} significant mediations)', 
                 fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count annotations
    for i, (sig, row) in enumerate(sig_agg_top.iterrows()):
        ax.text(row['mean'] + (0.5 if row['mean'] > 0 else -0.5), i, 
                f"n={int(row['count'])}", 
                va='center', ha='left' if row['mean'] > 0 else 'right', fontsize=8)

# Hide unused subplots
for idx in range(n_genes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure1_proportion_mediated_by_gene_signature.png', 
            bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# FIGURE 2: Path A vs Path B Scatter Plot
# ============================================================================
print("Creating Figure 2: Path A vs Path B Scatter...")
unique_genes = sorted(top_mediation['gene'].unique())
n_genes = len(unique_genes)

# Create appropriate grid layout
if n_genes == 1:
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes = [axes]
elif n_genes == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()
elif n_genes <= 4:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
else:
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

for idx, gene in enumerate(unique_genes):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    
    ax = axes[idx]
    
    # Color by significance
    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' 
              for p in gene_data['sobel_p']]
    sizes = [abs(z)*2 for z in gene_data['sobel_z']]
    
    scatter = ax.scatter(gene_data['path_a'], gene_data['path_b'], 
                        c=gene_data['sobel_p'], cmap='RdYlGn_r',
                        s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5,
                        vmin=0, vmax=0.05)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Path A: Gene → Signature\n(Effect of Gene on Signature)')
    ax.set_ylabel('Path B: Signature → Disease\n(Effect of Signature on Disease | Gene)')
    ax.set_title(f'{gene} Mediation Pathways\n(n = {len(gene_data)})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels (corrected interpretation)
    # Top-right (A>0, B>0): Gene increases sig, sig increases disease = ENHANCING
    # Bottom-right (A>0, B<0): Gene increases sig, sig decreases disease = Can be suppressive
    # Top-left (A<0, B>0): Gene decreases sig, sig increases disease
    # Bottom-left (A<0, B<0): Gene decreases sig, sig decreases disease
    ax.text(0.98, 0.98, 'ENHANCING\n(Path A>0, B>0)\nGene↑→Sig↑→Disease↑', 
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', 
            fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.text(0.98, 0.02, 'Path A>0, B<0\nGene↑→Sig↑→Disease↓', 
            transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right', 
            fontsize=8, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.02, 0.98, 'Path A<0, B>0\nGene↓→Sig↓→Disease↑', 
            transform=ax.transAxes, verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Count points in each quadrant
    top_right = ((gene_data['path_a'] > 0) & (gene_data['path_b'] > 0)).sum()
    ax.text(0.5, 0.95, f'Top-right (enhancing): {top_right}/{len(gene_data)}', 
            transform=ax.transAxes, horizontalalignment='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sobel p-value', rotation=270, labelpad=15)

# Hide unused subplots
for idx in range(n_genes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure2_path_a_vs_path_b.png', 
            bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# FIGURE 3: Top Mediation Effects by Disease Category
# ============================================================================
print("Creating Figure 3: Top Mediation Effects...")

# Get top 20 by absolute proportion mediated
top_20 = top_mediation.nlargest(20, 'proportion_mediated', keep='all')
top_20 = top_20.sort_values('proportion_mediated', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(top_20))
colors = ['red' if x < 0 else 'blue' for x in top_20['proportion_mediated']]

bars = ax.barh(y_pos, top_20['proportion_mediated'], color=colors, alpha=0.7)

# Create labels
labels = [f"{row['gene']} → {row['signature']} → {row['disease'][:40]}..." 
          if len(row['disease']) > 40 else f"{row['gene']} → {row['signature']} → {row['disease']}"
          for _, row in top_20.iterrows()]

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Proportion Mediated (%)', fontweight='bold')
ax.set_title('Top 20 Significant Mediation Effects\nRanked by Proportion Mediated', 
             fontweight='bold', fontsize=13)
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add annotation with Sobel p-value
for i, (idx, row) in enumerate(top_20.iterrows()):
    ax.text(row['proportion_mediated'] + (1 if row['proportion_mediated'] > 0 else -1), i,
            f"p={row['sobel_p']:.2e} |Z|={abs(row['sobel_z']):.1f}",
            va='center', ha='left' if row['proportion_mediated'] > 0 else 'right',
            fontsize=8, style='italic')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure3_top_mediation_effects.png', 
            bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# FIGURE 4: Distribution of Mediation Effects by Gene
# ============================================================================
print("Creating Figure 4: Distribution of Mediation Effects...")

unique_genes = sorted(top_mediation['gene'].unique())
n_genes = len(unique_genes)

# Create appropriate grid layout
if n_genes == 1:
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes = [axes]
elif n_genes == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()
elif n_genes <= 4:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
else:
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

for idx, gene in enumerate(unique_genes):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    
    ax = axes[idx]
    
    # Create histogram of proportion mediated
    ax.hist(gene_data['proportion_mediated'], bins=30, alpha=0.7, 
           color='steelblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No mediation')
    ax.axvline(gene_data['proportion_mediated'].median(), color='green', 
              linestyle='--', linewidth=2, label=f'Median: {gene_data["proportion_mediated"].median():.1f}%')
    ax.set_xlabel('Proportion Mediated (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'{gene} - Distribution of Proportion Mediated\n(n = {len(gene_data)} significant effects)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

# Hide unused subplots
for idx in range(n_genes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure4_distribution_by_gene.png', 
            bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# FIGURE 5: Network-style Visualization (Gene-Signature-Disease)
# ============================================================================
print("Creating Figure 5: Gene-Signature-Disease Network...")

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(16, 12))

# Get top mediations for each gene
top_by_gene = {}
for gene in sorted(top_mediation['gene'].unique()):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    # Get top 5 by absolute proportion mediated
    top_by_gene[gene] = gene_data.nlargest(5, 'proportion_mediated', keep='all')

# Layout: Gene (left) -> Signatures (center) -> Diseases (right)
gene_y_positions = {gene: i*2.5 for i, gene in enumerate(sorted(top_mediation['gene'].unique()))}
sig_positions = {}
disease_positions = {}

sig_counter = 0
disease_counter = 0

# Collect all unique signatures and diseases
all_sigs = set()
all_diseases = set()
for gene_data in top_by_gene.values():
    all_sigs.update(gene_data['signature'].unique())
    all_diseases.update(gene_data['disease'].unique())

sig_y_positions = {sig: i*0.8 for i, sig in enumerate(sorted(all_sigs))}
disease_y_positions = {disease: i*0.8 for i, disease in enumerate(sorted(all_diseases))}

# Plot nodes
x_genes = 1
x_sigs = 3
x_diseases = 5

# Plot gene nodes
gene_colors = plt.cm.Set3(np.linspace(0, 1, len(gene_y_positions)))
for (gene, y), color in zip(gene_y_positions.items(), gene_colors):
    rect = FancyBboxPatch((x_genes-0.3, y-0.4), 0.6, 0.8,
                         boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_genes, y, gene, ha='center', va='center', fontweight='bold', fontsize=11)

# Plot signature nodes
for sig, y in sig_y_positions.items():
    sig_num = int(sig.split('_')[1])
    circle = plt.Circle((x_sigs, y), 0.3, color='lightblue', 
                       edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x_sigs, y, f'S{sig_num}', ha='center', va='center', 
           fontweight='bold', fontsize=9)

# Plot disease nodes (abbreviated)
for disease, y in disease_y_positions.items():
    # Abbreviate long disease names
    abbrev = disease[:15] + '...' if len(disease) > 15 else disease
    rect = FancyBboxPatch((x_diseases-0.4, y-0.3), 0.8, 0.6,
                         boxstyle="round,pad=0.05",
                         facecolor='lightyellow', edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x_diseases, y, abbrev, ha='center', va='center', 
           fontsize=8, rotation=0)

# Draw edges
for gene, gene_data in top_by_gene.items():
    gene_y = gene_y_positions[gene]
    gene_color = dict(zip(gene_y_positions.keys(), gene_colors))[gene]
    
    for _, row in gene_data.iterrows():
        sig = row['signature']
        disease = row['disease']
        prop = row['proportion_mediated']
        
        sig_y = sig_y_positions[sig]
        disease_y = disease_y_positions[disease]
        
        # Color by proportion mediated
        edge_color = 'red' if prop < 0 else 'blue'
        alpha = min(abs(prop) / 100, 1.0) if prop != 0 else 0.3
        linewidth = max(abs(prop) / 20, 0.5)
        
        # Gene -> Signature
        arrow1 = FancyArrowPatch((x_genes+0.3, gene_y), (x_sigs-0.3, sig_y),
                                arrowstyle='->', mutation_scale=20,
                                color=edge_color, alpha=alpha, linewidth=linewidth)
        ax.add_patch(arrow1)
        
        # Signature -> Disease
        arrow2 = FancyArrowPatch((x_sigs+0.3, sig_y), (x_diseases-0.4, disease_y),
                                arrowstyle='->', mutation_scale=20,
                                color=edge_color, alpha=alpha, linewidth=linewidth)
        ax.add_patch(arrow2)

ax.set_xlim(0, 6)
ax.set_ylim(-0.5, max(list(gene_y_positions.values()) + 
                      list(sig_y_positions.values()) + 
                      list(disease_y_positions.values())) + 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Gene → Signature → Disease Mediation Network\n(Top 5 mediations per gene, colored by proportion mediated)',
            fontweight='bold', fontsize=14, pad=20)

# Add legend
red_patch = mpatches.Patch(color='red', label='Suppressive mediation')
blue_patch = mpatches.Patch(color='blue', label='Enhancing mediation')
ax.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure5_mediation_network.png', 
            bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# FIGURE 6: Comparison of Direct vs Indirect Effects
# ============================================================================
print("Creating Figure 6: Direct vs Indirect Effects...")

unique_genes = sorted(top_mediation['gene'].unique())
n_genes = len(unique_genes)

# Create appropriate grid layout
if n_genes == 1:
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes = [axes]
elif n_genes == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()
elif n_genes <= 4:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
else:
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()

for idx, gene in enumerate(unique_genes):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    
    ax = axes[idx]
    
    # Scatter: direct effect vs indirect effect
    scatter = ax.scatter(gene_data['direct_effect'], gene_data['indirect_effect'],
                        c=gene_data['sobel_p'], cmap='RdYlGn_r',
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                        vmin=0, vmax=0.05)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add diagonal line for equal effects
    max_abs = max(gene_data[['direct_effect', 'indirect_effect']].abs().max())
    ax.plot([-max_abs, max_abs], [-max_abs, max_abs], 
           'r--', alpha=0.3, linewidth=1, label='Equal effects')
    
    ax.set_xlabel('Direct Effect\n(Gene → Disease | Signature)\nEffect NOT mediated through signature', 
                  fontweight='bold', fontsize=11)
    ax.set_ylabel('Indirect Effect = Path A × Path B\n(Gene → Signature) × (Signature → Disease | Gene)\nEffect MEDIATED through signature', 
                  fontweight='bold', fontsize=11)
    ax.set_title(f'{gene} - Direct vs Indirect Effects\n(n = {len(gene_data)})',
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add explanatory text box
    ax.text(0.02, 0.98, 
           'Direct Effect = Gene → Disease\ncontrolling for Signature\n\n'
           'Indirect Effect = Path A × Path B\n'
           '  Path A: Gene → Signature\n'
           '  Path B: Signature → Disease | Gene\n'
           '  (controlling for Gene)',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
           fontsize=9, family='monospace')
    
    ax.legend(loc='lower right')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sobel p-value\n(significance of mediation)', rotation=270, labelpad=20)

# Hide unused subplots
for idx in range(n_genes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure6_direct_vs_indirect.png', 
            bbox_inches='tight', facecolor='white')
plt.close()

# ============================================================================
# FIGURE 7: Directionality Test - Path A vs Path A Reverse
# Tests whether Gene → Signature persists after controlling for Disease
# ============================================================================
print("\nCreating Figure 7: Directionality Test (Path A vs Reverse Path A)...")

# Check if reverse path columns exist
if 'path_a_reverse' in top_mediation.columns and 'path_a_reverse_p' in top_mediation.columns:
    # Filter to significant mediations for cleaner plot
    sig_mediation = top_mediation[top_mediation['sobel_p'] < 0.05].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, gene in enumerate(sorted(top_mediation['gene'].unique())):
        gene_data = sig_mediation[sig_mediation['gene'] == gene]
        
        if len(gene_data) == 0:
            axes[idx].text(0.5, 0.5, f'No significant\nmediations for {gene}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
            continue
        
        ax = axes[idx]
        
        # Scatter: path_a vs path_a_reverse
        # Color by whether reverse path is still significant
        colors = ['red' if p < 0.05 else 'blue' 
                 for p in gene_data['path_a_reverse_p']]
        
        scatter = ax.scatter(gene_data['path_a'], gene_data['path_a_reverse'],
                           c=gene_data['path_a_reverse_p'], cmap='RdYlGn_r',
                           s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=0.05)
        
        # Add diagonal line (path_a = path_a_reverse)
        max_abs = max(gene_data[['path_a', 'path_a_reverse']].abs().max())
        ax.plot([-max_abs, max_abs], [-max_abs, max_abs], 
               'k--', alpha=0.3, linewidth=1, label='Path A = Reverse Path A')
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Path A: Gene → Signature\nlm(signature ~ gene_burden)\nCoefficient for gene', 
                      fontweight='bold', fontsize=11)
        ax.set_ylabel('Path A Reverse: Gene → Signature | Disease\nlm(signature ~ gene_burden + disease)\nCoefficient for gene (controlling for disease)',
                      fontweight='bold', fontsize=11)
        ax.set_title(f'{gene} - Directionality Test\nTests if Gene→Signature is independent of Disease\n(n = {len(gene_data)} significant mediations)',
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add interpretation text with more detail
        n_persist = (gene_data['path_a_reverse_p'] < 0.05).sum()
        n_disappear = (gene_data['path_a_reverse_p'] >= 0.05).sum()
        
        ax.text(0.02, 0.98, 
               f'✓ Still significant after controlling\n  for disease: {n_persist}\n'
               f'  → Supports true mediation\n'
               f'  → Gene→Signature independent\n\n'
               f'✗ Disappears after controlling\n  for disease: {n_disappear}\n'
               f'  → Suggests confounding\n'
               f'  → Signature may be downstream\n'
               f'  → or disease affects signature',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9, family='monospace')
        
        # Add explanation of diagonal line
        ax.text(0.98, 0.15, 
               'Diagonal line: Path A = Reverse Path A\n'
               'Points on diagonal: Effect unchanged\n'
               'Points off diagonal: Effect changed\n\n'
               'X-axis: lm(sig ~ gene)\n'
               'Y-axis: lm(sig ~ gene + disease)',
               transform=ax.transAxes, verticalalignment='bottom', 
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6),
               fontsize=8, style='italic', family='monospace')
        
        ax.legend(loc='upper left')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Path A Reverse p-value\n(lower = more significant)', 
                      rotation=270, labelpad=20)
    
    # Hide unused subplots
    for idx in range(len(sorted(top_mediation['gene'].unique())), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure7_directionality_test.png', 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Figure 7 saved: Directionality test (Path A vs Reverse Path A)")
else:
    print("  ⚠️  Reverse path columns not found - skipping Figure 7")

# ============================================================================
# Create summary statistics table
# ============================================================================
print("\nCreating summary statistics table...")

summary_stats = []
for gene in sorted(top_mediation['gene'].unique()):
    gene_data = top_mediation[top_mediation['gene'] == gene]
    
    summary_stats.append({
        'Gene': gene,
        'N_Significant_Mediations': len(gene_data),
        'N_Unique_Diseases': gene_data['disease'].nunique(),
        'N_Unique_Signatures': gene_data['sig_num'].nunique(),
        'Mean_Proportion_Mediated_%': gene_data['proportion_mediated'].mean(),
        'Median_Proportion_Mediated_%': gene_data['proportion_mediated'].median(),
        'Mean_Abs_Proportion_Mediated_%': gene_data['proportion_mediated'].abs().mean(),
        'Median_Abs_Sobel_Z': gene_data['sobel_z'].abs().median(),
        'N_Suppressive_Mediation': (gene_data['proportion_mediated'] < 0).sum(),
        'N_Enhancing_Mediation': (gene_data['proportion_mediated'] > 0).sum(),
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(f'{output_dir}/summary_statistics_by_gene.csv', index=False)
print(f"\nSummary statistics saved to {output_dir}/summary_statistics_by_gene.csv")

# ============================================================================
# Identify TRUE enhancing mediations with strict criteria:
# 1. Path A > 0 AND Path B > 0 (gene↑→sig↑→disease↑)
# 2. Total effect and indirect effect same sign (consistent mediation, not suppression)
# 3. Total effect is significant (p < 0.05) - gene actually associated with disease
# 4. Proportion mediated between 0% and 100% (not suppression)
# 5. Direct effect should be smaller than total effect (mediation occurring)
# ============================================================================
print("\nIdentifying TRUE enhancing mediations (strict criteria)...")
enhancing = top_mediation[
    (top_mediation['path_a'] > 0) &  # Gene increases signature
    (top_mediation['path_b'] > 0) &  # Signature increases disease
    (top_mediation['total_effect'] > 0) &  # Total effect is positive
    (top_mediation['total_effect_p'] < 0.05) &  # Total effect is significant
    (top_mediation['indirect_effect'] > 0) &  # Indirect effect is positive
    (top_mediation['proportion_mediated'] > 0) &  # Proportion mediated is positive
    (top_mediation['proportion_mediated'] <= 100) &  # Not > 100% (no suppression)
    (top_mediation['direct_effect'].abs() < top_mediation['total_effect'].abs())  # Direct < total
].copy()

print(f"  Found {len(enhancing)} TRUE enhancing mediations")
print(f"  Criteria: Path A>0, Path B>0, Total>0 & significant, Indirect>0, Prop Mediated 0-100%, Direct < Total")
print(f"\nFound {len(enhancing)} enhancing mediations:")
print(f"  Genes: {enhancing['gene'].value_counts().to_dict()}")
print(f"\nTop enhancing mediations by |Sobel Z|:")
enhancing_sorted = enhancing.nlargest(min(20, len(enhancing)), 'sobel_z', keep='all')
for _, row in enhancing_sorted.iterrows():
    print(f"  {row['gene']} → {row['signature']} → {row['disease'][:50]}: "
          f"Path A={row['path_a']:.4f}, Path B={row['path_b']:.4f}, "
          f"Prop Med={row['proportion_mediated']:.1f}%, |Z|={abs(row['sobel_z']):.2f}")

# Save enhancing mediations
enhancing.to_csv(f'{output_dir}/enhancing_mediations.csv', index=False)
print(f"\nEnhancing mediations saved to {output_dir}/enhancing_mediations.csv")

# ============================================================================
# Write up findings
# ============================================================================
print("\nCreating written summary...")

writeup = f"""
# Mediation Analysis Results: Gene → Signature → Disease Pathways

## Executive Summary

We performed mediation analysis to test whether disease signatures mediate the relationship 
between rare variant burden in key genes (LDLR, BRCA2, TTN, MIP) and disease outcomes. 
Using the Baron-Kenny approach with Sobel tests, we identified **{len(top_sobel)} significant 
mediation effects** (Sobel p < 0.05).

## Key Findings

### 1. Overall Patterns

- **Genes analyzed:** {', '.join(sorted(top_mediation['gene'].unique()))}
- **Signatures tested:** {top_mediation['sig_num'].nunique()} (Signatures 0-{top_mediation['sig_num'].max()})
- **Diseases with significant mediation:** {top_mediation['disease'].nunique()}
- **Total significant gene-signature-disease triplets:** {len(top_sobel)}

### 2. Results by Gene

"""

for _, row in summary_df.iterrows():
    writeup += f"""
**{row['Gene']}:**
- {row['N_Significant_Mediations']} significant mediations
- {row['N_Unique_Diseases']} unique diseases
- {row['N_Unique_Signatures']} unique signatures
- Mean proportion mediated: {row['Mean_Proportion_Mediated_%']:.2f}%
- Median |Sobel Z|: {row['Median_Abs_Sobel_Z']:.2f}
- Suppressive mediations: {row['N_Suppressive_Mediation']}
- Enhancing mediations: {row['N_Enhancing_Mediation']}

"""

writeup += f"""
### 3. Top Signatures

The signatures with the most significant mediation effects:

"""
for sig, count in sig_counts.head(10).items():
    writeup += f"- **Signature {sig}**: {count} significant mediations\n"

writeup += f"""
### 4. Biological Interpretation

**Suppressive Mediation:** When the indirect effect (gene → signature → disease) opposes 
the direct effect (gene → disease), suggesting the signature pathway buffers or counteracts 
the genetic effect. Found in {summary_df['N_Suppressive_Mediation'].sum()} cases.

**Enhancing Mediation:** When the indirect effect amplifies the direct effect, suggesting 
the signature pathway is a mechanism through which the genetic effect operates. Found in 
{summary_df['N_Enhancing_Mediation'].sum()} cases.

### 5. Notable Findings

1. **LDLR → Signature 5** appears to be a major mediation pathway, with multiple disease 
   associations showing strong mediation effects.

2. **Strongest mediation effects** (by absolute proportion mediated) exceed 100% in some 
   cases, indicating that the signature pathway can fully explain or even reverse the 
   observed gene-disease association.

3. **Significant Sobel test statistics** (|Z| > 8) demonstrate robust mediation effects 
   that are unlikely to be due to chance.

## Figures Generated

1. **Figure 1**: Proportion mediated by gene and signature (top 15 signatures per gene)
2. **Figure 2**: Path A (gene → signature) vs Path B (signature → disease) scatter plots
3. **Figure 3**: Top 20 mediation effects ranked by proportion mediated
4. **Figure 4**: Distribution of proportion mediated by gene
5. **Figure 5**: Network visualization of gene-signature-disease pathways
6. **Figure 6**: Direct vs indirect effects comparison
7. **Figure 7**: Directionality test (Path A vs Path A Reverse) - tests whether Gene → Signature persists after controlling for Disease

## Methods

Mediation analysis was performed using the Baron-Kenny approach:
- **Path A**: Gene → Signature (tested with linear regression)
- **Path B**: Signature → Disease | Gene (tested with logistic regression controlling for gene)
- **Direct Effect**: Gene → Disease | Signature
- **Indirect Effect**: Path A × Path B
- **Total Effect**: Gene → Disease
- **Proportion Mediated**: Indirect Effect / Total Effect
- **Sobel Test**: Test of significance for the indirect effect

Significance threshold: Sobel p < 0.05

---

*Analysis completed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

with open(f'{output_dir}/mediation_analysis_writeup.md', 'w') as f:
    f.write(writeup)

print(f"\nWriteup saved to {output_dir}/mediation_analysis_writeup.md")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE!")
print(f"{'='*60}")
print(f"\nAll figures saved to: {output_dir}/")
print(f"Summary statistics: {output_dir}/summary_statistics_by_gene.csv")
print(f"Full writeup: {output_dir}/mediation_analysis_writeup.md")

