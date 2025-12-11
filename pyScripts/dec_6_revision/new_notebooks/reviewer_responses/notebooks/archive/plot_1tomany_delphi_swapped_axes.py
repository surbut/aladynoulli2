# Create 1-to-many visualization for 1-year predictions: Our single prediction vs all Delphi ICD codes, color-coded by disease
# UPDATED: Delphi on x-axis, Aladynoulli on y-axis

if 'comparison_1yr' in locals() and len(comparison_1yr) > 0:
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 9
    
    # Get unique diseases and assign colors (use same colors as t0 plot for consistency)
    unique_diseases = sorted(comparison_1yr['Disease'].unique())
    n_diseases = len(unique_diseases)
    
    # Use a colormap with enough distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    if n_diseases > 20:
        # Extend with another colormap if needed
        colors2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors = np.vstack([colors, colors2])
    
    disease_colors = {disease: colors[i % len(colors)] for i, disease in enumerate(unique_diseases)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot diagonal line (y=x) for reference
    min_val = min(comparison_1yr['Aladynoulli_1yr'].min(), comparison_1yr['Delphi_no_gap'].min())
    max_val = max(comparison_1yr['Aladynoulli_1yr'].max(), comparison_1yr['Delphi_no_gap'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, linewidth=1, label='y=x (equal performance)')
    
    # Calculate Delphi variability statistics
    delphi_variability = []
    
    # For each disease, plot lines connecting our prediction to all Delphi points
    for disease in unique_diseases:
        disease_data = comparison_1yr[comparison_1yr['Disease'] == disease]
        our_auc = disease_data['Aladynoulli_1yr'].iloc[0]  # Same for all rows
        delphi_aucs = disease_data['Delphi_no_gap'].values
        color = disease_colors[disease]
        
        # Calculate variability metrics
        if len(delphi_aucs) > 1:
            delphi_range = delphi_aucs.max() - delphi_aucs.min()
            delphi_std = delphi_aucs.std()
            delphi_variability.append({
                'Disease': disease,
                'Range': delphi_range,
                'Std': delphi_std,
                'Min': delphi_aucs.min(),
                'Max': delphi_aucs.max(),
                'N': len(delphi_aucs)
            })
            
            # Draw shaded region showing Delphi range (behind everything) - HORIZONTAL BAND (Delphi on x-axis)
            ax.fill_between([delphi_aucs.min(), delphi_aucs.max()],
                           [our_auc - 0.005, our_auc - 0.005], 
                           [our_auc + 0.005, our_auc + 0.005],
                           color=color, alpha=0.15, zorder=0)
        
        # Draw lines from our point to each Delphi point - HORIZONTAL LINES (Delphi on x-axis)
        for delphi_auc in delphi_aucs:
            ax.plot([delphi_auc, our_auc], [our_auc, our_auc], 
                   color=color, alpha=0.3, linewidth=0.8, zorder=1)
        
        # Plot our single prediction point (larger, on diagonal)
        ax.scatter([our_auc], [our_auc], s=150, c=[color], 
                  marker='s', edgecolors='black', linewidths=1.5, 
                  label=disease, zorder=3, alpha=0.8)
        
        # Plot all Delphi points for this disease - DELPHI ON X-AXIS, ALA ON Y-AXIS
        ax.scatter(delphi_aucs, [our_auc] * len(delphi_aucs), 
                  s=80, c=[color], marker='o', 
                  edgecolors='black', linewidths=0.8, 
                  alpha=0.7, zorder=2)
        
        # Add text annotation showing range for diseases with multiple Delphi codes - SWAPPED POSITION
        if len(delphi_aucs) > 1:
            range_text = f"Δ={delphi_range:.3f}"
            ax.text((delphi_aucs.min() + delphi_aucs.max()) / 2, our_auc + 0.01,
                   range_text, fontsize=7, color=color, 
                   weight='bold', alpha=0.8, zorder=4,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=color, alpha=0.7, linewidth=1))
    
    # Labels and title - SWAPPED AXES
    ax.set_xlabel('Delphi AUC (no gap / t0)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aladynoulli AUC (1-year, washout 0)', fontsize=12, fontweight='bold')
    ax.set_title('Aladynoulli vs Delphi: 1-Year Predictions (1-to-Many Comparison)\n(Our median 1-year aggregated prediction vs all matching Delphi ICD codes, no gap)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    margin = 0.05
    ax.set_xlim([min_val - margin, max_val + margin])
    ax.set_ylim([min_val - margin, max_val + margin])
    
    # Add legend (outside plot, but limit to first 20 diseases to avoid clutter)
    legend_elements = []
    for i, disease in enumerate(unique_diseases[:20]):  # Show first 20
        color = disease_colors[disease]
        n_delphi = len(comparison_1yr[comparison_1yr['Disease'] == disease])
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                      markersize=8, markeredgecolor='black', markeredgewidth=1,
                      label=f'{disease} ({n_delphi})', linestyle='None')
        )
    if n_diseases > 20:
        legend_elements.append(
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                      markersize=8, markeredgecolor='black', markeredgewidth=1,
                      label=f'... and {n_diseases - 20} more diseases', linestyle='None')
        )
    
    # Place legend outside plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
             fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    # Add text annotation with summary stats
    wins_by_disease = comparison_1yr.groupby('Disease')['Advantage'].apply(lambda x: (x > 0).any())
    n_wins = wins_by_disease.sum()
    n_diseases_total = len(wins_by_disease)
    
    # Calculate Delphi variability summary
    if len(delphi_variability) > 0:
        variability_df = pd.DataFrame(delphi_variability)
        mean_range = variability_df['Range'].mean()
        mean_std = variability_df['Std'].mean()
        max_range = variability_df['Range'].max()
        max_range_disease = variability_df.loc[variability_df['Range'].idxmax(), 'Disease']
        
        stats_text = (f'Aladynoulli beats ≥1 Delphi ICD code: {n_wins}/{n_diseases_total} diseases ({n_wins/n_diseases_total*100:.1f}%)\n'
                     f'Delphi variability: Mean range = {mean_range:.3f}, Mean std = {mean_std:.3f}\n'
                     f'Max Delphi range: {max_range:.3f} ({max_range_disease})')
    else:
        stats_text = f'Aladynoulli beats ≥1 Delphi ICD code: {n_wins}/{n_diseases_total} diseases ({n_wins/n_diseases_total*100:.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Print variability summary
    if len(delphi_variability) > 0:
        print("\n" + "="*80)
        print("DELPHI VARIABILITY SUMMARY (1-YEAR PREDICTIONS):")
        print("="*80)
        variability_df = pd.DataFrame(delphi_variability)
        variability_df = variability_df.sort_values('Range', ascending=False)
        print(f"\nMean range across diseases: {variability_df['Range'].mean():.4f}")
        print(f"Mean std across diseases: {variability_df['Std'].mean():.4f}")
        print(f"\nTop 10 diseases by Delphi variability (range):")
        print(variability_df[['Disease', 'N', 'Range', 'Std', 'Min', 'Max']].head(10).to_string(index=False))
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / 'delphi_comparison_phecode_mapping_1yr_1tomany.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {fig_path}")
    
    plt.show()
else:
    print("⚠️  Cannot create visualization without comparison_1yr data")

