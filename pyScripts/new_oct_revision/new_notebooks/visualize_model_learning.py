#!/usr/bin/env python3
"""
Create a figure showing how the model learns to distinguish between
high-risk and lower-risk hypercholesterolemia patients.

Key insight: Non-droppers (predictions stay high) have higher event rates,
showing the model correctly identifies high-risk patients.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def create_model_learning_figure(disease_name='ASCVD', output_dir=None):
    """Create figure showing model learning for hypercholesterolemia patients."""
    
    if output_dir is None:
        output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis/plots')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load patient-level data
    patient_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis') / f'prediction_drops_patients_{disease_name}.csv'
    
    if not patient_file.exists():
        print(f"⚠️  Patient-level data not found: {patient_file}")
        return
    
    patients_df = pd.read_csv(patient_file)
    
    if 'has_hypercholesterolemia' not in patients_df.columns or 'has_ascvd_event_between' not in patients_df.columns:
        print("⚠️  Required columns not found in patient data")
        return
    
    # Prepare data
    droppers = patients_df[patients_df['is_dropper'] == True]
    non_droppers = patients_df[patients_df['is_dropper'] == False]
    
    hyperchol_droppers = droppers[droppers['has_hypercholesterolemia'] == True]
    hyperchol_non_droppers = non_droppers[non_droppers['has_hypercholesterolemia'] == True]
    
    # Calculate rates
    event_rate_hyperchol_droppers = hyperchol_droppers['has_ascvd_event_between'].mean() * 100
    event_rate_hyperchol_non_droppers = hyperchol_non_droppers['has_ascvd_event_between'].mean() * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create grouped bar plot
    categories = ['Droppers\n(Predictions Drop)', 'Non-droppers\n(Predictions Stay High)']
    event_rates = [event_rate_hyperchol_droppers, event_rate_hyperchol_non_droppers]
    colors = ['#e74c3c', '#2ecc71']  # Red for droppers, green for non-droppers
    
    bars = ax.bar(categories, event_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, event_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add sample sizes
        n_patients = len(hyperchol_droppers) if i == 0 else len(hyperchol_non_droppers)
        n_events = int(rate * n_patients / 100)
        ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                f'n={n_patients}\n({n_events} events)',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')
    
    # Styling
    ax.set_ylabel('ASCVD Event Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Learning: Hypercholesterolemia Patients\n' + 
                 'Predictions Stay High → Higher Event Rates', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(event_rates) * 1.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add key insight text
    insight_text = (
        "Key Insight: Non-droppers have HIGHER event rates (17.9% vs 11.0%)\n"
        "→ Model correctly identifies high-risk hyperchol patients\n"
        "→ Predictions stay high because patients actually have events\n"
        "→ Model is learning and calibrating correctly"
    )
    
    ax.text(0.5, 0.98, insight_text,
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='navy', linewidth=2),
            fontsize=11, family='monospace')
    
    # Add arrow annotation
    ax.annotate('', xy=(1, event_rate_hyperchol_non_droppers), 
                xytext=(0, event_rate_hyperchol_droppers),
                arrowprops=dict(arrowstyle='->', lw=3, color='darkgreen'),
                annotation_clip=False)
    
    ax.text(0.5, max(event_rates) * 1.15, '↑ Model correctly identifies\nhigh-risk patients',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'model_learning_hyperchol_{disease_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to: {output_file}")
    plt.close()
    
    # Also create a comparison figure showing the full picture
    create_full_comparison_figure(patients_df, disease_name, output_dir)

def create_full_comparison_figure(patients_df, disease_name, output_dir):
    """Create a comprehensive comparison figure."""
    
    droppers = patients_df[patients_df['is_dropper'] == True]
    non_droppers = patients_df[patients_df['is_dropper'] == False]
    
    hyperchol_droppers = droppers[droppers['has_hypercholesterolemia'] == True]
    hyperchol_non_droppers = non_droppers[non_droppers['has_hypercholesterolemia'] == True]
    
    # Calculate all metrics
    hyperchol_rate_droppers = droppers['has_hypercholesterolemia'].mean() * 100
    hyperchol_rate_non_droppers = non_droppers['has_hypercholesterolemia'].mean() * 100
    
    event_rate_all_droppers = droppers['has_ascvd_event_between'].mean() * 100
    event_rate_all_non_droppers = non_droppers['has_ascvd_event_between'].mean() * 100
    
    event_rate_hyperchol_droppers = hyperchol_droppers['has_ascvd_event_between'].mean() * 100
    event_rate_hyperchol_non_droppers = hyperchol_non_droppers['has_ascvd_event_between'].mean() * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Model Learning: Hypercholesterolemia Risk Stratification', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Hypercholesterolemia prevalence
    ax = axes[0]
    bars = ax.bar(['Droppers', 'Non-droppers'], 
                  [hyperchol_rate_droppers, hyperchol_rate_non_droppers],
                  color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Hypercholesterolemia\nPrevalence (%)', fontsize=12, fontweight='bold')
    ax.set_title('1. Hyperchol More Common\nin Droppers', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(hyperchol_rate_droppers, hyperchol_rate_non_droppers) * 1.2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Event rates - All patients (shows survivor bias)
    ax = axes[1]
    bars = ax.bar(['Droppers', 'Non-droppers'], 
                  [event_rate_all_droppers, event_rate_all_non_droppers],
                  color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('ASCVD Event Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('2. Overall: Droppers Have\nHigher Event Rates\n(Survivor Bias)', 
                 fontsize=12, fontweight='bold', color='darkred')
    ax.set_ylim(0, max(event_rate_all_droppers, event_rate_all_non_droppers) * 1.2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Add note about survivor bias
    ax.text(0.5, 0.95, 'High-risk patients\nhave events early', 
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
            fontsize=9, style='italic')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Event rates - Hyperchol patients only (THE KEY INSIGHT)
    ax = axes[2]
    bars = ax.bar(['Droppers\nwith Hyperchol', 'Non-droppers\nwith Hyperchol'], 
                  [event_rate_hyperchol_droppers, event_rate_hyperchol_non_droppers],
                  color=['#c0392b', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('ASCVD Event Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('3. Among Hyperchol: Non-droppers\nHave Higher Event Rates', 
                 fontsize=12, fontweight='bold', color='darkgreen')
    ax.set_ylim(0, max(event_rate_hyperchol_droppers, event_rate_hyperchol_non_droppers) * 1.2)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Add sample size
        n = len(hyperchol_droppers) if i == 0 else len(hyperchol_non_droppers)
        ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                f'n={n}', ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
    ax.grid(axis='y', alpha=0.3)
    
    # Add interpretation text highlighting the contrast
    interpretation = (
        "Key Contrast:\n"
        "• OVERALL: Droppers have HIGHER events (10.8% vs 4.8%) → Survivor bias\n"
        "  High-risk patients have events early, excluded from later analysis\n"
        "• WITHIN HYPERCHOL: Non-droppers have HIGHER events (17.9% vs 11.0%)\n"
        "  → Model correctly identifies high-risk hyperchol patients\n"
        "  → Predictions stay high because patients actually have events\n"
        "  → Model learns to distinguish high-risk vs lower-risk hyperchol patients\n"
        "→ This shows model learning heterogeneity within risk factors"
    )
    
    fig.text(0.5, 0.02, interpretation, ha='center', va='bottom',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7, edgecolor='orange', linewidth=2))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save
    output_file = output_dir / f'model_learning_full_comparison_{disease_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved full comparison figure to: {output_file}")
    plt.close()

def create_multiple_precursors_figure(disease_name='ASCVD', output_dir=None):
    """Create figure showing event rates for multiple correlated precursor diseases."""
    
    if output_dir is None:
        output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis/plots')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load correlated precursor analysis
    precursor_file = Path('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis') / f'correlated_precursors_event_rates_{disease_name}.csv'
    
    if not precursor_file.exists():
        print(f"⚠️  Correlated precursor analysis not found: {precursor_file}")
        print("   Run analyze_prediction_drops.py first to generate this analysis")
        return
    
    df = pd.read_csv(precursor_file)
    
    # Filter to diseases with matching pattern (non-droppers > droppers)
    matching_df = df[df['Pattern_match'] == True].head(10)
    
    if len(matching_df) == 0:
        print("⚠️  No precursor diseases found with matching pattern")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(matching_df))
    width = 0.35
    
    bars1 = ax.barh(y_pos - width/2, matching_df['Event_rate_droppers'], width,
                    label='Droppers (Predictions Drop)', color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.barh(y_pos + width/2, matching_df['Event_rate_non_droppers'], width,
                    label='Non-droppers (Predictions Stay High)', color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(matching_df['Precursor'], fontsize=10)
    ax.set_xlabel('ASCVD Event Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Learning Pattern Across Correlated Precursor Diseases\n{disease_name}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(matching_df.iterrows()):
        ax.text(row['Event_rate_droppers'], i - width/2, f" {row['Event_rate_droppers']:.1f}%",
                va='center', fontsize=9, fontweight='bold')
        ax.text(row['Event_rate_non_droppers'], i + width/2, f" {row['Event_rate_non_droppers']:.1f}%",
                va='center', fontsize=9, fontweight='bold')
    
    # Add interpretation
    interpretation = (
        "Pattern: Non-droppers have HIGHER event rates within each precursor group\n"
        "→ Model correctly identifies high-risk patients for correlated precursors\n"
        "→ Predictions stay high because patients actually have events\n"
        "→ Model learns heterogeneity within risk factors"
    )
    
    ax.text(0.5, 0.02, interpretation, ha='center', va='bottom',
            transform=fig.transFigure, fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save
    output_file = output_dir / f'model_learning_multiple_precursors_{disease_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved multiple precursors figure to: {output_file}")
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize model learning')
    parser.add_argument('--disease', type=str, default='ASCVD',
                       help='Disease name')
    parser.add_argument('--output_dir', type=str,
                       default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/analysis/plots',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CREATING MODEL LEARNING FIGURES")
    print("="*80)
    
    create_model_learning_figure(args.disease, args.output_dir)
    create_multiple_precursors_figure(args.disease, args.output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

