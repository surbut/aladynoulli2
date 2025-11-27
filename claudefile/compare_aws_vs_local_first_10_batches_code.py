from fig5utils import *
import pandas as pd
import numpy as np
import torch

# Load full pce_df
pce_df_full = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
disease_names = essentials['disease_names']

# Storage for results - AWS vs Local (both retrospective pooled)
aws_retrospective_10yr_results = []
aws_retrospective_30yr_results = []
local_retrospective_10yr_results = []
local_retrospective_30yr_results = []

# Load full tensors once (shared across both analyses)
if 'Y_full' not in globals():
    Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
if 'E_full' not in globals():
    E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_enrollment_full.pt')

# Loop through first 10 batches (0-9)
for batch_idx in range(10):
    start_idx = batch_idx * 10000
    end_idx = (batch_idx + 1) * 10000
    
    print(f"\n{'='*80}")
    print(f"Processing batch {batch_idx}: {start_idx} to {end_idx}")
    print(f"{'='*80}")
    
    # Get pce_df subset for this batch
    pce_df_subset = pce_df_full[start_idx:end_idx].copy().reset_index(drop=True)
    
    # Extract batch from full tensors (shared for both analyses)
    Y_batch = Y_full[start_idx:end_idx]
    E_batch = E_full[start_idx:end_idx]
    
    # ===== AWS RETROSPECTIVE POOLED =====
    aws_retrospective_ckpt_path = f'/Users/sarahurbut/Downloads/aws_first_10_batches_models/model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
    
    try:
        print(f"\n--- AWS Retrospective Pooled ---")
        aws_ckpt = torch.load(aws_retrospective_ckpt_path, weights_only=False)
        model.load_state_dict(aws_ckpt['model_state_dict'])
        
        # Update model.Y and model.N so forward() uses correct patients
        model.Y = torch.tensor(Y_batch, dtype=torch.float32)
        model.N = Y_batch.shape[0]
       
        # 10-year predictions
        print(f"AWS Retrospective - 10 year predictions...")
        aws_10yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
            model, Y_batch, E_batch, disease_names, pce_df_subset, 
            n_bootstraps=100, follow_up_duration_years=10, patient_indices=None
        )
        aws_10yr['batch_idx'] = batch_idx
        aws_10yr['source'] = 'aws_retrospective'
        aws_retrospective_10yr_results.append(aws_10yr)
        
        # 30-year predictions
        print(f"AWS Retrospective - 30 year predictions...")
        aws_30yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
            model, Y_batch, E_batch, disease_names, pce_df_subset, 
            n_bootstraps=100, follow_up_duration_years=30, patient_indices=None
        )
        aws_30yr['batch_idx'] = batch_idx
        aws_30yr['source'] = 'aws_retrospective'
        aws_retrospective_30yr_results.append(aws_30yr)
        
    except FileNotFoundError:
        print(f"AWS retrospective checkpoint not found: {aws_retrospective_ckpt_path}")
    except Exception as e:
        print(f"Error processing AWS retrospective checkpoint {batch_idx}: {e}")
        import traceback
        traceback.print_exc()
    
    # ===== LOCAL RETROSPECTIVE POOLED =====
    local_retrospective_ckpt_path = f'/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt'
    
    try:
        print(f"\n--- Local Retrospective Pooled ---")
        local_ckpt = torch.load(local_retrospective_ckpt_path, weights_only=False)
        model.load_state_dict(local_ckpt['model_state_dict'])
        
        # Update model.Y and model.N so forward() uses correct patients
        model.Y = torch.tensor(Y_batch, dtype=torch.float32)
        model.N = Y_batch.shape[0]
       
        # 10-year predictions
        print(f"Local Retrospective - 10 year predictions...")
        local_10yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
            model, Y_batch, E_batch, disease_names, pce_df_subset, 
            n_bootstraps=100, follow_up_duration_years=10, patient_indices=None
        )
        local_10yr['batch_idx'] = batch_idx
        local_10yr['source'] = 'local_retrospective'
        local_retrospective_10yr_results.append(local_10yr)
        
        # 30-year predictions
        print(f"Local Retrospective - 30 year predictions...")
        local_30yr = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
            model, Y_batch, E_batch, disease_names, pce_df_subset, 
            n_bootstraps=100, follow_up_duration_years=30, patient_indices=None
        )
        local_30yr['batch_idx'] = batch_idx
        local_30yr['source'] = 'local_retrospective'
        local_retrospective_30yr_results.append(local_30yr)
        
    except FileNotFoundError:
        print(f"Local retrospective checkpoint not found: {local_retrospective_ckpt_path}")
    except Exception as e:
        print(f"Error processing local retrospective checkpoint {batch_idx}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*80}")
print("Completed processing first 10 batches!")
print(f"{'='*80}")
print(f"AWS Retrospective - 10yr: {len(aws_retrospective_10yr_results)} batches")
print(f"AWS Retrospective - 30yr: {len(aws_retrospective_30yr_results)} batches")
print(f"Local Retrospective - 10yr: {len(local_retrospective_10yr_results)} batches")
print(f"Local Retrospective - 30yr: {len(local_retrospective_30yr_results)} batches")

# Aggregate and compare
if aws_retrospective_10yr_results and local_retrospective_10yr_results:
    print(f"\n{'='*80}")
    print("COMPARISON: AWS vs Local Retrospective Pooled (10-year)")
    print(f"{'='*80}")
    
    # Aggregate AWS results
    aws_10yr_agg = compute_aggregated_cis(aws_retrospective_10yr_results, "AWS Retrospective 10yr")
    local_10yr_agg = compute_aggregated_cis(local_retrospective_10yr_results, "Local Retrospective 10yr")
    
    # Compare
    comparison_10yr = pd.DataFrame({
        'AWS_AUC': aws_10yr_agg['auc'],
        'Local_AUC': local_10yr_agg['auc'],
        'Difference': aws_10yr_agg['auc'] - local_10yr_agg['auc']
    })
    comparison_10yr = comparison_10yr.sort_values('Difference', ascending=False)
    print(comparison_10yr.round(4))
    print(f"\nMean difference: {comparison_10yr['Difference'].mean():.4f}")
    print(f"Max difference: {comparison_10yr['Difference'].max():.4f}")
    print(f"Min difference: {comparison_10yr['Difference'].min():.4f}")

if aws_retrospective_30yr_results and local_retrospective_30yr_results:
    print(f"\n{'='*80}")
    print("COMPARISON: AWS vs Local Retrospective Pooled (30-year)")
    print(f"{'='*80}")
    
    # Aggregate AWS results
    aws_30yr_agg = compute_aggregated_cis(aws_retrospective_30yr_results, "AWS Retrospective 30yr")
    local_30yr_agg = compute_aggregated_cis(local_retrospective_30yr_results, "Local Retrospective 30yr")
    
    # Compare
    comparison_30yr = pd.DataFrame({
        'AWS_AUC': aws_30yr_agg['auc'],
        'Local_AUC': local_30yr_agg['auc'],
        'Difference': aws_30yr_agg['auc'] - local_30yr_agg['auc']
    })
    comparison_30yr = comparison_30yr.sort_values('Difference', ascending=False)
    print(comparison_30yr.round(4))
    print(f"\nMean difference: {comparison_30yr['Difference'].mean():.4f}")
    print(f"Max difference: {comparison_30yr['Difference'].max():.4f}")
    print(f"Min difference: {comparison_30yr['Difference'].min():.4f}")




