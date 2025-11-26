#!/usr/bin/env python3
"""
Run Leave-One-Out (LOO) vs Full Pooled and AWS vs Local validation comparisons.

This script:
1. Compares LOO validation (phi trained excluding one batch) vs Full Pooled (phi trained on all batches)
   for excluded batches (0, 6, 15, 17, 18, 20, 24, 34, 35, 37)
2. Compares AWS retrospective models vs Local retrospective models for batches 0-10

Usage:
    python run_loo_aws_validation.py --n_bootstraps 100
"""

import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from fig5utils import (
    evaluate_major_diseases_wsex_with_bootstrap_dynamic,
    evaluate_major_diseases_wsex_with_bootstrap
)

# Load model essentials
def load_essentials():
    """Load model essentials including disease names"""
    essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
    essentials = torch.load(essentials_path, weights_only=False)
    return essentials

def load_model():
    """Load and initialize model"""
    from quick_model_dummy import load_model_essentials, AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    Y, E, G, essentials = load_model_essentials()
    
    # Load a checkpoint to get G_with_sex
    ckpt = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt', weights_only=False)
    G_with_sex = ckpt['G']
    
    # Initialize model (dummy, will load state_dict later)
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=10000,  # Dummy
        D=Y.shape[1],
        T=Y.shape[2],
        K=20,
        P=G_with_sex.shape[1],
        init_sd_scaler=1e-1,
        G=G_with_sex,
        Y=Y[:10000],  # Dummy
        genetic_scale=1,
        W=0,
        R=0,
        prevalence_t=essentials['prevalence_t'],
        signature_references=torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/reference_trajectories.pt')['signature_refs'],
        healthy_reference=True,
        disease_names=essentials['disease_names']
    )
    
    # Initialize with psi and clusters
    initial_psi = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_psi_400k.pt', weights_only=False)
    initial_clusters = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt', weights_only=False)
    model.initialize_params(true_psi=initial_psi)
    model.clusters = initial_clusters
    
    return model, essentials

def extract_aucs_from_results(results_list):
    """Extract AUCs from results list into a dictionary by batch and disease"""
    aucs_by_batch = {}
    for result in results_list:
        batch_idx = result['batch_idx']
        if batch_idx not in aucs_by_batch:
            aucs_by_batch[batch_idx] = {}
        for disease, metrics in result.items():
            if disease not in ['batch_idx', 'analysis_type'] and isinstance(metrics, dict):
                if 'auc' in metrics:
                    aucs_by_batch[batch_idx][disease] = metrics['auc']
    return aucs_by_batch

def compare_results(group1_aucs, group2_aucs, title, output_dir):
    """Compare two groups of AUCs per batch and save results"""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    
    all_differences = []
    comparison_data = []
    
    for batch_idx in sorted(set(list(group1_aucs.keys()) + list(group2_aucs.keys()))):
        group1_batch = group1_aucs.get(batch_idx, {})
        group2_batch = group2_aucs.get(batch_idx, {})
        
        common_diseases = set(group1_batch.keys()) & set(group2_batch.keys())
        
        if not common_diseases:
            continue
        
        differences = []
        for disease in sorted(common_diseases):
            group1_auc = group1_batch[disease]
            group2_auc = group2_batch[disease]
            diff = group2_auc - group1_auc  # group2 - group1
            differences.append(diff)
            all_differences.append(diff)
            
            comparison_data.append({
                'Batch': batch_idx,
                'Disease': disease,
                'Group1_AUC': group1_auc,
                'Group2_AUC': group2_auc,
                'Difference': diff,
                'Abs_Difference': abs(diff)
            })
    
    # Save comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    if not comparison_df.empty:
        output_file = output_dir / f"{title.replace(' ', '_').replace('-', '_')}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved comparison to: {output_file}")
        
        # Print summary
        print(f"\nSummary Statistics:")
        print(f"  Mean difference: {comparison_df['Difference'].mean():.4f}")
        print(f"  Median difference: {comparison_df['Difference'].median():.4f}")
        print(f"  Std difference: {comparison_df['Difference'].std():.4f}")
        print(f"  Max absolute difference: {comparison_df['Abs_Difference'].max():.4f}")
        print(f"  Diseases with |diff| < 0.01: {(comparison_df['Abs_Difference'] < 0.01).sum()}/{len(comparison_df)}")
    
    return comparison_df

def run_loo_validation(excluded_batches, model, Y_full, E_full, pce_df_full, disease_names, n_bootstraps, output_dir):
    """Run LOO vs Full Pooled validation"""
    print("\n" + "="*80)
    print("LEAVE-ONE-OUT VALIDATION")
    print("="*80)
    
    loo_10yr_results = []
    loo_30yr_results = []
    loo_static_10yr_results = []
    
    full_pooled_10yr_results = []
    full_pooled_30yr_results = []
    full_pooled_static_10yr_results = []
    
    for batch_idx in excluded_batches:
        start_idx = batch_idx * 10000
        end_idx = (batch_idx + 1) * 10000
        
        print(f"\nProcessing batch {batch_idx}: {start_idx} to {end_idx}")
        
        pce_df_subset = pce_df_full[start_idx:end_idx].copy().reset_index(drop=True)
        Y_batch = Y_full[start_idx:end_idx]
        E_batch = E_full[start_idx:end_idx]
        
        # LOO checkpoint
        loo_ckpt_path = f'/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/leave_one_out_validation/batch_{batch_idx}/model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        try:
            print(f"  Loading LOO checkpoint...")
            loo_ckpt = torch.load(loo_ckpt_path, weights_only=False)
            model.load_state_dict(loo_ckpt['model_state_dict'])
            model.Y = torch.tensor(Y_batch, dtype=torch.float32)
            model.N = Y_batch.shape[0]
            
            print(f"  Evaluating LOO - 10yr...")
            loo_10yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10, patient_indices=None
            )
            loo_10yr['batch_idx'] = batch_idx
            loo_10yr_results.append(loo_10yr)
            
            print(f"  Evaluating LOO - 30yr...")
            loo_30yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=30, patient_indices=None
            )
            loo_30yr['batch_idx'] = batch_idx
            loo_30yr_results.append(loo_30yr)
            
            print(f"  Evaluating LOO - Static 10yr...")
            loo_static_10yr = evaluate_major_diseases_wsex_with_bootstrap(
                model=model, Y_100k=Y_batch, E_100k=E_batch,
                disease_names=disease_names, pce_df=pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10
            )
            loo_static_10yr['batch_idx'] = batch_idx
            loo_static_10yr_results.append(loo_static_10yr)
            
        except FileNotFoundError:
            print(f"  ⚠ LOO checkpoint not found: {loo_ckpt_path}")
        except Exception as e:
            print(f"  ⚠ Error processing LOO checkpoint: {e}")
        
        # Full pooled checkpoint
        full_pooled_ckpt_path = f'/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        try:
            print(f"  Loading Full Pooled checkpoint...")
            full_ckpt = torch.load(full_pooled_ckpt_path, weights_only=False)
            model.load_state_dict(full_ckpt['model_state_dict'])
            model.Y = torch.tensor(Y_batch, dtype=torch.float32)
            model.N = Y_batch.shape[0]
            
            print(f"  Evaluating Full Pooled - 10yr...")
            full_10yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10, patient_indices=None
            )
            full_10yr['batch_idx'] = batch_idx
            full_pooled_10yr_results.append(full_10yr)
            
            print(f"  Evaluating Full Pooled - 30yr...")
            full_30yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=30, patient_indices=None
            )
            full_30yr['batch_idx'] = batch_idx
            full_pooled_30yr_results.append(full_30yr)
            
            print(f"  Evaluating Full Pooled - Static 10yr...")
            full_static_10yr = evaluate_major_diseases_wsex_with_bootstrap(
                model=model, Y_100k=Y_batch, E_100k=E_batch,
                disease_names=disease_names, pce_df=pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10
            )
            full_static_10yr['batch_idx'] = batch_idx
            full_pooled_static_10yr_results.append(full_static_10yr)
            
        except FileNotFoundError:
            print(f"  ⚠ Full pooled checkpoint not found: {full_pooled_ckpt_path}")
        except Exception as e:
            print(f"  ⚠ Error processing full pooled checkpoint: {e}")
    
    # Extract AUCs and compare
    loo_10yr_aucs = extract_aucs_from_results(loo_10yr_results)
    loo_30yr_aucs = extract_aucs_from_results(loo_30yr_results)
    loo_static_10yr_aucs = extract_aucs_from_results(loo_static_10yr_results)
    
    full_10yr_aucs = extract_aucs_from_results(full_pooled_10yr_results)
    full_30yr_aucs = extract_aucs_from_results(full_pooled_30yr_results)
    full_static_10yr_aucs = extract_aucs_from_results(full_pooled_static_10yr_results)
    
    print(f"\n{'='*80}")
    print("LOO VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"LOO - 10yr: {len(loo_10yr_results)} batches")
    print(f"LOO - 30yr: {len(loo_30yr_results)} batches")
    print(f"Full Pooled - 10yr: {len(full_pooled_10yr_results)} batches")
    print(f"Full Pooled - 30yr: {len(full_pooled_30yr_results)} batches")
    
    # Compare results
    compare_results(loo_10yr_aucs, full_10yr_aucs, "LOO_vs_FullPooled_10yr", output_dir)
    compare_results(loo_30yr_aucs, full_30yr_aucs, "LOO_vs_FullPooled_30yr", output_dir)
    compare_results(loo_static_10yr_aucs, full_static_10yr_aucs, "LOO_vs_FullPooled_Static10yr", output_dir)
    
    return loo_10yr_results, loo_30yr_results, loo_static_10yr_results, full_pooled_10yr_results, full_pooled_30yr_results, full_pooled_static_10yr_results

def run_aws_validation(aws_batches, model, Y_full, E_full, pce_df_full, disease_names, n_bootstraps, output_dir):
    """Run AWS vs Local validation"""
    print("\n" + "="*80)
    print("AWS vs LOCAL VALIDATION")
    print("="*80)
    
    aws_10yr_results = []
    aws_30yr_results = []
    aws_static_10yr_results = []
    
    local_10yr_results = []
    local_30yr_results = []
    local_static_10yr_results = []
    
    for batch_idx in aws_batches:
        start_idx = batch_idx * 10000
        end_idx = (batch_idx + 1) * 10000
        
        print(f"\nProcessing batch {batch_idx}: {start_idx} to {end_idx}")
        
        pce_df_subset = pce_df_full[start_idx:end_idx].copy().reset_index(drop=True)
        Y_batch = Y_full[start_idx:end_idx]
        E_batch = E_full[start_idx:end_idx]
        
        # AWS checkpoint
        aws_ckpt_path = f'/Users/sarahurbut/Downloads/aws_first_10_batches_models/model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        try:
            print(f"  Loading AWS checkpoint...")
            aws_ckpt = torch.load(aws_ckpt_path, weights_only=False)
            model.load_state_dict(aws_ckpt['model_state_dict'])
            model.Y = torch.tensor(Y_batch, dtype=torch.float32)
            model.N = Y_batch.shape[0]
            
            print(f"  Evaluating AWS - 10yr...")
            aws_10yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10, patient_indices=None
            )
            aws_10yr['batch_idx'] = batch_idx
            aws_10yr_results.append(aws_10yr)
            
            print(f"  Evaluating AWS - 30yr...")
            aws_30yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=30, patient_indices=None
            )
            aws_30yr['batch_idx'] = batch_idx
            aws_30yr_results.append(aws_30yr)
            
            print(f"  Evaluating AWS - Static 10yr...")
            aws_static_10yr = evaluate_major_diseases_wsex_with_bootstrap(
                model=model, Y_100k=Y_batch, E_100k=E_batch,
                disease_names=disease_names, pce_df=pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10
            )
            aws_static_10yr['batch_idx'] = batch_idx
            aws_static_10yr_results.append(aws_static_10yr)
            
        except FileNotFoundError:
            print(f"  ⚠ AWS checkpoint not found: {aws_ckpt_path}")
        except Exception as e:
            print(f"  ⚠ Error processing AWS checkpoint: {e}")
        
        # Local checkpoint
        local_ckpt_path = f'/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
        
        try:
            print(f"  Loading Local checkpoint...")
            local_ckpt = torch.load(local_ckpt_path, weights_only=False)
            model.load_state_dict(local_ckpt['model_state_dict'])
            model.Y = torch.tensor(Y_batch, dtype=torch.float32)
            model.N = Y_batch.shape[0]
            
            print(f"  Evaluating Local - 10yr...")
            local_10yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10, patient_indices=None
            )
            local_10yr['batch_idx'] = batch_idx
            local_10yr_results.append(local_10yr)
            
            print(f"  Evaluating Local - 30yr...")
            local_30yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
                model, Y_batch, E_batch, disease_names, pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=30, patient_indices=None
            )
            local_30yr['batch_idx'] = batch_idx
            local_30yr_results.append(local_30yr)
            
            print(f"  Evaluating Local - Static 10yr...")
            local_static_10yr = evaluate_major_diseases_wsex_with_bootstrap(
                model=model, Y_100k=Y_batch, E_100k=E_batch,
                disease_names=disease_names, pce_df=pce_df_subset,
                n_bootstraps=n_bootstraps, follow_up_duration_years=10
            )
            local_static_10yr['batch_idx'] = batch_idx
            local_static_10yr_results.append(local_static_10yr)
            
        except FileNotFoundError:
            print(f"  ⚠ Local checkpoint not found: {local_ckpt_path}")
        except Exception as e:
            print(f"  ⚠ Error processing local checkpoint: {e}")
    
    # Extract AUCs and compare
    aws_10yr_aucs = extract_aucs_from_results(aws_10yr_results)
    aws_30yr_aucs = extract_aucs_from_results(aws_30yr_results)
    aws_static_10yr_aucs = extract_aucs_from_results(aws_static_10yr_results)
    
    local_10yr_aucs = extract_aucs_from_results(local_10yr_results)
    local_30yr_aucs = extract_aucs_from_results(local_30yr_results)
    local_static_10yr_aucs = extract_aucs_from_results(local_static_10yr_results)
    
    print(f"\n{'='*80}")
    print("AWS VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"AWS - 10yr: {len(aws_10yr_results)} batches")
    print(f"AWS - 30yr: {len(aws_30yr_results)} batches")
    print(f"Local - 10yr: {len(local_10yr_results)} batches")
    print(f"Local - 30yr: {len(local_30yr_results)} batches")
    
    # Compare results
    compare_results(aws_10yr_aucs, local_10yr_aucs, "AWS_vs_Local_10yr", output_dir)
    compare_results(aws_30yr_aucs, local_30yr_aucs, "AWS_vs_Local_30yr", output_dir)
    compare_results(aws_static_10yr_aucs, local_static_10yr_aucs, "AWS_vs_Local_Static10yr", output_dir)
    
    return aws_10yr_results, aws_30yr_results, aws_static_10yr_results, local_10yr_results, local_30yr_results, local_static_10yr_results

def main():
    parser = argparse.ArgumentParser(description="Run LOO and AWS validation comparisons.")
    parser.add_argument('--n_bootstraps', type=int, default=100,
                        help='Number of bootstrap iterations (default: 100)')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks/results/validation/',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LOO AND AWS VALIDATION")
    print("="*80)
    
    # Load essentials
    print("\nLoading model and essentials...")
    model, essentials = load_model()
    disease_names = essentials['disease_names']
    
    # Load data
    print("Loading data tensors...")
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt', weights_only=False)
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt', weights_only=False)
    pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
    
    # LOO validation
    excluded_batches = [0, 6, 15, 17, 18, 20, 24, 34, 35, 37]
    run_loo_validation(excluded_batches, model, Y_full, E_full, pce_df_full, disease_names, args.n_bootstraps, output_dir)
    
    # AWS validation
    aws_batches = list(range(11))  # Batches 0-10
    run_aws_validation(aws_batches, model, Y_full, E_full, pce_df_full, disease_names, args.n_bootstraps, output_dir)
    
    print("\n" + "="*80)
    print("ALL VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")

if __name__ == '__main__':
    main()


