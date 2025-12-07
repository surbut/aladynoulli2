#!/usr/bin/env python3
"""
Generate age offset predictions (rolling 1-year predictions with models trained at enrollment + 0-9 years).

This script evaluates 1-year predictions using models trained at different time offsets:
- Offset 0: Model trained at enrollment
- Offset 1: Model trained at enrollment + 1 year
- ...
- Offset 9: Model trained at enrollment + 9 years

For each offset, computes 1-year risk predictions and evaluates AUC.

Usage:
    python generate_age_offset_predictions.py --approach pooled_retrospective  # AWS/remote version
    python generate_age_offset_predictions.py --approach pooled_retrospective_local  # Local version
"""

import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path


sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision')

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
    
    return Y_batch, E_batch, pce_df_batch, essentials

def generate_age_offset_predictions(approach='pooled_retrospective', max_offset=9, start_idx=0, end_idx=10000):
    """
    Generate age offset predictions for batch 0-10000.
    
    Args:
        approach: 'pooled_retrospective' (AWS/remote) or 'pooled_retrospective_local' (local)
        max_offset: Maximum offset (0-9, default 9)
        start_idx: Start index for batch (default 0)
        end_idx: End index for batch (default 10000)
    """
    print("="*80)
    print(f"GENERATING AGE OFFSET PREDICTIONS")
    print("="*80)
    print(f"Approach: {approach}")
    print(f"Batch: {start_idx}-{end_idx}")
    print(f"Max offset: {max_offset}")
    print("="*80)
    
    # Load data
    Y_batch, E_batch, pce_df_batch, essentials = load_data_for_batch(start_idx, end_idx)
    disease_names = essentials['disease_names']
    
    # Determine pi file path based on approach
    if approach == 'pooled_retrospective':
        # Files from AWS run (downloaded to Downloads)
        pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_offset_local_vectorized_E_corrected/')
        pi_filename_pattern = 'pi_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun_pooledall.pt'
    elif approach == 'pooled_retrospective_local':
        # Files from local run
        pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_offset_local_vectorized_E_corrected/')
        pi_filename_pattern = 'pi_enroll_fixedphi_age_offset_{k}_sex_{start}_{end}_try2_withpcs_newrun_pooledall.pt'
    else:
        raise ValueError(f"Unknown approach: {approach}. Use 'pooled_retrospective' (AWS) or 'pooled_retrospective_local' (local) or 'pooled_enrollment' (local)")
    
    # Load all pi batches for offsets 0 to max_offset
    print(f"\nLoading pi batches for offsets 0-{max_offset}...")
    pi_batches = []
    for k in range(max_offset + 1):
        pi_filename = pi_filename_pattern.format(k=k, start=start_idx, end=end_idx)
        pi_path = pi_base_dir / pi_filename
        
        if not pi_path.exists():
            raise FileNotFoundError(f"Pi file not found: {pi_path}")
        
        print(f"  Loading offset {k}: {pi_filename}")
        pi_batch = torch.load(pi_path, weights_only=False)
        pi_batches.append(pi_batch)
    
    print(f"Loaded {len(pi_batches)} pi batches.")
    
    # Create output directory (absolute path relative to script location)
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / 'results' / 'age_offset' / approach
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load washout AUC for Year 0 (full dataset result)
    washout_file = script_dir.parent / 'results' / 'washout' / 'pooled_retrospective' / 'washout_0yr_results.csv'
    washout_auc_year0 = None
    if washout_file.exists():
        washout_df = pd.read_csv(washout_file)
        ascvd_row = washout_df[washout_df['Disease'] == 'ASCVD']
        if len(ascvd_row) > 0:
            washout_auc_year0 = ascvd_row.iloc[0]['AUC']
            print(f"\nLoaded washout 0-year AUC (full dataset): {washout_auc_year0:.3f}")
            print(f"  Will use this for Year 0 label in plot (instead of batch-specific AUC)")
    
    # Run evaluation
    print("\nEvaluating rolling 1-year predictions...")
    results = evaluate_major_diseases_rolling_1year_roc_curves(
        pi_batches, Y_batch, E_batch, disease_names, pce_df_batch, 
        patient_indices=None, plot_group=None  # Don't plot yet, we'll do it manually
    )
    
    # Manually create the plot with updated Year 0 label
    if 'ASCVD' in results and washout_auc_year0 is not None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc as sklearn_auc
        
        # Check PCE/PREVENT availability
        pce_available = 'pce_goff_fuull' in pce_df_batch.columns
        prevent_available = 'prevent_impute' in pce_df_batch.columns
        
        plt.figure(figsize=(8, 6))
        
        # Plot all years
        for k, roc_data in enumerate(results['ASCVD']):
            if roc_data is not None and len(roc_data) == 4:
                fpr, tpr, thresholds, auc_val = roc_data
                if fpr is not None:
                    if k == 0:
                        # Use washout AUC for Year 0 label
                        plt.plot(fpr, tpr, label=f'Aladynoulli 1-year (Year {k}, AUC={washout_auc_year0:.3f})')
                    else:
                        plt.plot(fpr, tpr, label=f'Aladynoulli 1-year (Year {k}, AUC={auc_val:.3f})')
        
        # Plot PCE/PREVENT (recompute for offset 0)
        if pce_available or prevent_available:
            # Get ASCVD disease indices
            major_diseases = {
                'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                          'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified']
            }
            disease_indices = []
            unique_indices = set()
            for disease in major_diseases['ASCVD']:
                indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
                for idx in indices:
                    if idx not in unique_indices:
                        disease_indices.append(idx)
                        unique_indices.add(idx)
            
            pce_preds_10yr = []
            prevent_preds_10yr = []
            outcomes_10yr = []
            
            for i in range(len(pce_df_batch)):
                age = pce_df_batch.iloc[i]['age']
                t_enroll = int(age - 30)
                t_start = t_enroll + 0  # Offset 0
                if t_start < 0 or t_start >= pi_batches[0].shape[2]:
                    continue
                
                # 10-year outcome
                event_10yr = 0
                end_time_10yr = min(t_start + 10, Y_batch.shape[2])
                for d_idx in disease_indices:
                    if d_idx >= Y_batch.shape[1]: continue
                    if torch.any(Y_batch[i, d_idx, t_start:end_time_10yr] > 0):
                        event_10yr = 1
                        break
                
                if pce_available and pd.notna(pce_df_batch.iloc[i]['pce_goff_fuull']):
                    pce_preds_10yr.append(pce_df_batch.iloc[i]['pce_goff_fuull'])
                    outcomes_10yr.append(event_10yr)
                if prevent_available and pd.notna(pce_df_batch.iloc[i]['prevent_impute']):
                    prevent_preds_10yr.append(pce_df_batch.iloc[i]['prevent_impute'])
            
            if pce_available and len(pce_preds_10yr) == len(outcomes_10yr) and len(pce_preds_10yr) > 0 and len(np.unique(outcomes_10yr)) > 1:
                fpr_pce, tpr_pce, _ = roc_curve(outcomes_10yr, pce_preds_10yr)
                auc_pce = sklearn_auc(fpr_pce, tpr_pce)
                plt.plot(fpr_pce, tpr_pce, '--', label=f'PCE 10-year (Enrollment, AUC={auc_pce:.3f})')
            
            if prevent_available and len(prevent_preds_10yr) == len(outcomes_10yr) and len(prevent_preds_10yr) > 0 and len(np.unique(outcomes_10yr)) > 1:
                fpr_prev, tpr_prev, _ = roc_curve(outcomes_10yr, prevent_preds_10yr)
                auc_prev = sklearn_auc(fpr_prev, tpr_prev)
                plt.plot(fpr_prev, tpr_prev, ':', label=f'PREVENT 10-year (Enrollment, AUC={auc_prev:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if pce_available or prevent_available:
            plt.title('ROC Curves for ASCVD: Aladynoulli 1-year (All Offsets) vs PCE/PREVENT 10-year (Enrollment Only)')
        else:
            plt.title('ROC Curves for ASCVD: Aladynoulli 1-year')
        plt.legend()
        plt.grid(True)
        
        # Save to output directory
        plot_file = output_dir / 'roc_curves_ASCVD.pdf'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved ASCVD ROC plot to: {plot_file}")
        print(f"  Note: Year 0 label uses washout AUC ({washout_auc_year0:.3f}) from full dataset")
        plt.show()
    
    # Extract AUCs and save to CSV
    print("\nExtracting AUCs...")
    auc_data = []
    for disease_group, roc_data_list in results.items():
        for offset, roc_data in enumerate(roc_data_list):
            if roc_data is not None and len(roc_data) == 4:
                fpr, tpr, thresholds, auc_val = roc_data
                if auc_val is not None:
                    auc_data.append({
                        'Disease': disease_group,
                        'Offset': offset,
                        'AUC': auc_val
                    })
    
    auc_df = pd.DataFrame(auc_data)
    
    # Save AUC summary
    output_file = output_dir / f'age_offset_aucs_batch_{start_idx}_{end_idx}.csv'
    auc_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved AUC summary to: {output_file}")
    
    # Create pivot table for easier viewing
    pivot_df = auc_df.pivot(index='Disease', columns='Offset', values='AUC')
    pivot_file = output_dir / f'age_offset_aucs_pivot_batch_{start_idx}_{end_idx}.csv'
    pivot_df.to_csv(pivot_file)
    print(f"✓ Saved AUC pivot table to: {pivot_file}")
    
    # Create summary statistics table (mean, median, std, min, max, count across offsets)
    summary_df = auc_df.groupby('Disease')['AUC'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
    summary_df = summary_df.sort_values('median', ascending=False)
    summary_file = output_dir / f'age_offset_aucs_summary_batch_{start_idx}_{end_idx}.csv'
    summary_df.to_csv(summary_file)
    print(f"✓ Saved AUC summary statistics to: {summary_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("AGE OFFSET PREDICTIONS SUMMARY")
    print("="*80)
    print(f"\nTop diseases by AUC at offset 0:")
    offset0_aucs = auc_df[auc_df['Offset'] == 0].sort_values('AUC', ascending=False)
    print(offset0_aucs.head(10).to_string(index=False))
    
    print(f"\nAUC change from offset 0 to {max_offset}:")
    for disease in auc_df['Disease'].unique():
        offset0_auc = auc_df[(auc_df['Disease'] == disease) & (auc_df['Offset'] == 0)]['AUC'].values
        offset_max_auc = auc_df[(auc_df['Disease'] == disease) & (auc_df['Offset'] == max_offset)]['AUC'].values
        if len(offset0_auc) > 0 and len(offset_max_auc) > 0:
            change = offset_max_auc[0] - offset0_auc[0]
            print(f"  {disease:25s} Offset 0: {offset0_auc[0]:.3f}, Offset {max_offset}: {offset_max_auc[0]:.3f}, Change: {change:+.3f}")
    
    print("\n" + "="*80)
    print("AGE OFFSET PREDICTIONS COMPLETE")
    print("="*80)
    
    return results, auc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate age offset predictions.")
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
    
    generate_age_offset_predictions(
        approach=args.approach,
        max_offset=args.max_offset,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )