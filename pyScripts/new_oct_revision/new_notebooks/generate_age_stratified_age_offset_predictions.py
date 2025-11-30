#!/usr/bin/env python3
"""
Generate age-stratified age offset predictions (rolling 1-year predictions with models trained at enrollment + 0-9 years).

This script evaluates 1-year predictions using models trained at different time offsets (0-9 years after enrollment),
stratified by enrollment age groups (40-50, 50-60, 60-70, 70-80).

For each age group and offset, computes 1-year risk predictions and evaluates AUC.

Usage:
    python generate_age_stratified_age_offset_predictions.py --approach pooled_retrospective
"""

import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from evaluatetdccode import evaluate_major_diseases_rolling_1year_roc_curves


def load_data_for_batch(start_idx=0, end_idx=10000):
    """Load Y, E, pce_df, essentials for a specific batch."""
    from clust_huge_amp import subset_data
    
    base_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    
    print("Loading full data tensors...")
    Y = torch.load(base_path / 'Y_tensor.pt', weights_only=False)
    E = torch.load(base_path / 'E_matrix.pt', weights_only=False)
    G = torch.load(base_path / 'G_matrix.pt', weights_only=False)
    essentials = torch.load(base_path / 'model_essentials.pt', weights_only=False)
    
    print(f"Subsetting to batch {start_idx}-{end_idx}...")
    Y_batch, E_batch, G_batch, indices = subset_data(Y, E, G, start_index=start_idx, end_index=end_idx)
    
    # Load pce_df
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    readRDS = robjects.r['readRDS']
    pce_data = readRDS('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_df_prevent.rds')
    pce_df = pandas2ri.rpy2py(pce_data)
    
    # Subset pce_df to batch indices
    pce_df_batch = pce_df.iloc[indices].reset_index(drop=True)
    
    return Y_batch, E_batch, pce_df_batch, essentials, indices


def get_age_group_mask(ages, age_min, age_max):
    """Get boolean mask for patients in age range [age_min, age_max)"""
    return (ages >= age_min) & (ages < age_max)


def generate_age_stratified_age_offset_predictions(approach='pooled_retrospective', max_offset=9, start_idx=0, end_idx=10000):
    """
    Generate age-stratified age offset predictions for batch 0-10000.
    
    Args:
        approach: 'pooled_retrospective' (AWS/remote) or 'pooled_retrospective_local' (local)
        max_offset: Maximum offset (0-9, default 9)
        start_idx: Start index for batch (default 0)
        end_idx: End index for batch (default 10000)
    """
    print("="*80)
    print(f"GENERATING AGE-STRATIFIED AGE OFFSET PREDICTIONS")
    print("="*80)
    print(f"Approach: {approach}")
    print(f"Batch: {start_idx}-{end_idx}")
    print(f"Max offset: {max_offset}")
    print("="*80)
    
    # Load data
    Y_batch, E_batch, pce_df_batch, essentials, batch_indices = load_data_for_batch(start_idx, end_idx)
    disease_names = essentials['disease_names']
    
    # Load baseline age file for enrollment ages
    baseline_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv')
    baseline_df = pd.read_csv(baseline_path)
    
    # Get enrollment ages for the batch
    # Note: batch_indices are the original indices in the full dataset
    enrollment_ages = baseline_df.iloc[batch_indices]['age'].values
    
    print(f"\nEnrollment age range in batch: {enrollment_ages.min():.1f} - {enrollment_ages.max():.1f}")
    
    # Define age groups
    age_groups = [
        (40, 50, '40-50'),
        (50, 60, '50-60'),
        (60, 70, '60-70'),
        (70, 80, '70-80')
    ]
    
    # Determine pi file path based on approach
    if approach == 'pooled_retrospective':
        # Files from AWS run (downloaded to Dropbox)
        pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/age_offset_files')
        pi_filename_pattern = 'pi_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun.pt'
    elif approach == 'pooled_retrospective_local':
        # Files from local run
        pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox/pi_offset_using_pooled_retrospective_local')
        pi_filename_pattern = 'pi_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun_pooledall.pt'
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'pooled_retrospective' (AWS) or 'pooled_retrospective_local' (local)")
    
    # Load all pi batches for offsets 0 to max_offset
    print(f"\nLoading pi batches for offsets 0-{max_offset}...")
    pi_batches_full = []
    for k in range(max_offset + 1):
        pi_filename = pi_filename_pattern.format(k=k, start=start_idx, end=end_idx)
        pi_path = pi_base_dir / pi_filename
        
        if not pi_path.exists():
            raise FileNotFoundError(f"Pi file not found: {pi_path}")
        
        print(f"  Loading offset {k}: {pi_filename}")
        pi_batch = torch.load(pi_path, weights_only=False)
        pi_batches_full.append(pi_batch)
    
    print(f"Loaded {len(pi_batches_full)} pi batches.")
    
    # Create output directory (absolute path)
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'results' / 'age_offset' / approach / 'age_stratified'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all results
    all_auc_data = []
    
    # Process each age group
    for age_min, age_max, age_group_name in age_groups:
        print(f"\n{'='*80}")
        print(f"PROCESSING AGE GROUP: {age_group_name}")
        print(f"{'='*80}")
        
        # Get mask for this age group
        age_mask = get_age_group_mask(enrollment_ages, age_min, age_max)
        age_group_indices = np.where(age_mask)[0].tolist()
        
        n_patients = len(age_group_indices)
        print(f"Patients in age group {age_group_name}: {n_patients}")
        
        if n_patients == 0:
            print(f"Warning: No patients in age group {age_group_name}, skipping...")
            continue
        
        # Subset data to this age group
        Y_age = Y_batch[age_group_indices]
        E_age = E_batch[age_group_indices]
        pce_df_age = pce_df_batch.iloc[age_group_indices].reset_index(drop=True)
        
        # Subset pi batches to this age group
        pi_batches_age = [pi_batch[age_group_indices] for pi_batch in pi_batches_full]
        
        # Run evaluation
        print(f"\nEvaluating rolling 1-year predictions for age group {age_group_name}...")
        try:
            results = evaluate_major_diseases_rolling_1year_roc_curves(
                pi_batches_age, Y_age, E_age, disease_names, pce_df_age, 
                patient_indices=None, plot_group='ASCVD'  # Plot ASCVD by default
            )
            
            # Extract AUCs
            print(f"\nExtracting AUCs for age group {age_group_name}...")
            for disease_group, roc_data_list in results.items():
                for offset, roc_data in enumerate(roc_data_list):
                    if roc_data is not None and len(roc_data) == 4:
                        fpr, tpr, thresholds, auc_val = roc_data
                        if auc_val is not None:
                            all_auc_data.append({
                                'Age_Group': age_group_name,
                                'Disease': disease_group,
                                'Offset': offset,
                                'AUC': auc_val
                            })
            
            print(f"✓ Completed age group {age_group_name}")
            
        except Exception as e:
            print(f"✗ Error processing age group {age_group_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create results DataFrame
    if len(all_auc_data) == 0:
        print("\n⚠️  WARNING: No results generated!")
        return None
    
    auc_df = pd.DataFrame(all_auc_data)
    
    # Save AUC summary
    output_file = output_dir / f'age_stratified_age_offset_aucs_batch_{start_idx}_{end_idx}.csv'
    auc_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved AUC summary to: {output_file}")
    
    # Create pivot table for easier viewing (Disease x Offset, with Age_Group as separate column)
    # First create a multi-index pivot
    pivot_df = auc_df.pivot_table(
        index=['Age_Group', 'Disease'], 
        columns='Offset', 
        values='AUC'
    )
    pivot_file = output_dir / f'age_stratified_age_offset_aucs_pivot_batch_{start_idx}_{end_idx}.csv'
    pivot_df.to_csv(pivot_file)
    print(f"✓ Saved AUC pivot table to: {pivot_file}")
    
    # Create summary statistics table (mean, median, std, min, max, count across offsets) by age group
    summary_df = auc_df.groupby(['Age_Group', 'Disease'])['AUC'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    summary_df = summary_df.sort_values(['Age_Group', 'median'], ascending=[True, False])
    summary_file = output_dir / f'age_stratified_age_offset_aucs_summary_batch_{start_idx}_{end_idx}.csv'
    summary_df.to_csv(summary_file)
    print(f"✓ Saved AUC summary statistics to: {summary_file}")
    
    # Create plots: AUC vs Offset for each age group
    print("\nCreating plots...")
    key_diseases = ['ASCVD', 'Diabetes', 'Atrial_Fib', 'CKD', 'All_Cancers', 'Stroke', 
                    'Heart_Failure', 'Colorectal_Cancer', 'Breast_Cancer', 'Lung_Cancer']
    
    # Plot 1: AUC vs Offset for key diseases, one plot per age group
    for age_min, age_max, age_group_name in age_groups:
        age_data = auc_df[auc_df['Age_Group'] == age_group_name]
        if len(age_data) == 0:
            continue
        
        # Filter to key diseases
        age_data = age_data[age_data['Disease'].isin(key_diseases)]
        
        if len(age_data) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for disease in key_diseases:
            disease_data = age_data[age_data['Disease'] == disease].sort_values('Offset')
            if len(disease_data) > 0:
                ax.plot(disease_data['Offset'], disease_data['AUC'], 
                       marker='o', label=disease, linewidth=2, markersize=6)
        
        ax.set_xlabel('Offset (Years After Enrollment)', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax.set_title(f'AUC vs Offset: Age Group {age_group_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.4, 1.0])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(max_offset + 1))
        
        plt.tight_layout()
        plot_file = output_dir / f'auc_vs_offset_age_group_{age_group_name}_batch_{start_idx}_{end_idx}.pdf'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved plot for age group {age_group_name}: {plot_file}")
    
    # Plot 2: Compare age groups for ASCVD
    fig, ax = plt.subplots(figsize=(10, 6))
    for age_min, age_max, age_group_name in age_groups:
        age_data = auc_df[(auc_df['Age_Group'] == age_group_name) & (auc_df['Disease'] == 'ASCVD')]
        if len(age_data) > 0:
            age_data = age_data.sort_values('Offset')
            ax.plot(age_data['Offset'], age_data['AUC'], 
                   marker='o', label=f'Age {age_group_name}', linewidth=2, markersize=8)
    
    ax.set_xlabel('Offset (Years After Enrollment)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('ASCVD: AUC vs Offset by Enrollment Age Group', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(max_offset + 1))
    
    plt.tight_layout()
    plot_file = output_dir / f'ascvd_auc_vs_offset_by_age_group_batch_{start_idx}_{end_idx}.pdf'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved ASCVD comparison plot: {plot_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("AGE-STRATIFIED AGE OFFSET PREDICTIONS SUMMARY")
    print("="*80)
    
    for age_min, age_max, age_group_name in age_groups:
        age_data = auc_df[auc_df['Age_Group'] == age_group_name]
        if len(age_data) == 0:
            continue
        
        print(f"\nAge Group {age_group_name}:")
        print(f"  Top diseases by AUC at offset 0:")
        offset0_aucs = age_data[age_data['Offset'] == 0].sort_values('AUC', ascending=False)
        print(offset0_aucs.head(5)[['Disease', 'AUC']].to_string(index=False))
    
    print("\n" + "="*80)
    print("AGE-STRATIFIED AGE OFFSET PREDICTIONS COMPLETE")
    print("="*80)
    
    return auc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate age-stratified age offset predictions.")
    parser.add_argument('--approach', type=str, required=True,
                        choices=['pooled_retrospective', 'pooled_retrospective_local'],
                        help="Approach type: 'pooled_retrospective' (AWS/remote) or 'pooled_retrospective_local' (local)")
    parser.add_argument('--max_offset', type=int, default=9,
                        help="Maximum offset (0-9, default: 9)")
    parser.add_argument('--start_idx', type=int, default=0,
                        help="Start index for batch (default: 0)")
    parser.add_argument('--end_idx', type=int, default=10000,
                        help="End index for batch (default: 10000)")
    
    args = parser.parse_args()
    
    generate_age_stratified_age_offset_predictions(
        approach=args.approach,
        max_offset=args.max_offset,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )

