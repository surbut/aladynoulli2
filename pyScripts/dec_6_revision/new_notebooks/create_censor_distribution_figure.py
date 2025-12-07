#!/usr/bin/env python3
"""
Create histogram of entry and exit ages from censor_info.csv

This figure shows the distribution of enrollment ages and maximum follow-up ages,
demonstrating the variable follow-up times in the cohort.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def load_censor_data(censor_path):
    """Load censor info CSV"""
    print(f"Loading censor data from: {censor_path}")
    df = pd.read_csv(censor_path)
    
    # Check which columns we have
    print(f"Columns: {df.columns.tolist()}")
    
    # Get entry age (enrollment age)
    if 'age' in df.columns:
        entry_age = df['age'].values
    elif 'enrollment_age' in df.columns:
        entry_age = df['enrollment_age'].values
    else:
        raise ValueError("Could not find entry age column (expected 'age' or 'enrollment_age')")
    
    # Get exit age (max follow-up age)
    if 'max_censor' in df.columns:
        exit_age = df['max_censor'].values
    elif 'censor_age' in df.columns:
        exit_age = df['censor_age'].values
    else:
        raise ValueError("Could not find exit age column (expected 'max_censor' or 'censor_age')")
    
    # Calculate follow-up duration
    followup_duration = exit_age - entry_age
    
    print(f"Loaded {len(df)} patients")
    print(f"Entry age: min={entry_age.min():.1f}, max={entry_age.max():.1f}, mean={entry_age.mean():.1f}")
    print(f"Exit age: min={exit_age.min():.1f}, max={exit_age.max():.1f}, mean={exit_age.mean():.1f}")
    print(f"Follow-up duration: min={followup_duration.min():.1f}, max={followup_duration.max():.1f}, mean={followup_duration.mean():.1f}")
    
    return entry_age, exit_age, followup_duration

def create_figure(entry_age, exit_age, followup_duration, output_path, cohort_name="UK Biobank"):
    """Create histogram figure"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Entry age distribution
    ax1 = axes[0]
    ax1.hist(entry_age, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Entry Age (years)', fontweight='bold')
    ax1.set_ylabel('Number of Patients', fontweight='bold')
    ax1.set_title(f'Distribution of Entry Ages\n({cohort_name})', fontweight='bold')
    ax1.axvline(entry_age.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {entry_age.mean():.1f} years')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Exit age distribution
    ax2 = axes[1]
    ax2.hist(exit_age, bins=30, color='#A23B72', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Exit Age (years)', fontweight='bold')
    ax2.set_ylabel('Number of Patients', fontweight='bold')
    ax2.set_title(f'Distribution of Exit Ages\n({cohort_name})', fontweight='bold')
    ax2.axvline(exit_age.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {exit_age.mean():.1f} years')
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Follow-up duration distribution
    ax3 = axes[2]
    ax3.hist(followup_duration, bins=30, color='#F18F01', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Follow-up Duration (years)', fontweight='bold')
    ax3.set_ylabel('Number of Patients', fontweight='bold')
    ax3.set_title(f'Distribution of Follow-up Duration\n({cohort_name})', fontweight='bold')
    ax3.axvline(followup_duration.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {followup_duration.mean():.1f} years')
    ax3.legend()
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    print(f"\nSaving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("✓ Figure saved successfully!")
    
    return fig

def main():
    # Paths
    censor_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/censor_info.csv')
    output_dir = Path('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'censor_age_distributions.png'
    
    # Load data
    entry_age, exit_age, followup_duration = load_censor_data(censor_path)
    
    # Create figure
    fig = create_figure(entry_age, exit_age, followup_duration, output_path, cohort_name="UK Biobank")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Entry Age:")
    print(f"  Mean: {entry_age.mean():.2f} years")
    print(f"  Median: {np.median(entry_age):.2f} years")
    print(f"  Range: {entry_age.min():.0f} - {entry_age.max():.0f} years")
    print(f"\nExit Age:")
    print(f"  Mean: {exit_age.mean():.2f} years")
    print(f"  Median: {np.median(exit_age):.2f} years")
    print(f"  Range: {exit_age.min():.0f} - {exit_age.max():.0f} years")
    print(f"\nFollow-up Duration:")
    print(f"  Mean: {followup_duration.mean():.2f} years")
    print(f"  Median: {np.median(followup_duration):.2f} years")
    print(f"  Range: {followup_duration.min():.1f} - {followup_duration.max():.1f} years")
    print("="*60)
    
    plt.show()

if __name__ == '__main__':
    main()

