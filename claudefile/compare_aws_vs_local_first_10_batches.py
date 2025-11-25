#!/usr/bin/env python
"""
Compare AWS vs Local results for first 10 batches (retrospective pooled)
Compares 10-year and 30-year predictions
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os

# Paths
aws_models_dir = '/Users/sarahurbut/Downloads/aws_first_10_batches_models/'
local_base_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_predictions_fixedphi_RETROSPECTIVE_pooled/'

# Load essentials for evaluation
essentials_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/model_essentials.pt'
essentials = torch.load(essentials_path, weights_only=False)
disease_names = essentials['disease_names']

# Load Y tensor for evaluation
Y_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt'
Y = torch.load(Y_path, weights_only=False)

# Load covariates for enrollment ages
covariates_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh_withpcs.csv'
fh_processed = pd.read_csv(covariates_path)
enrollment_ages = fh_processed['age'].values[:100000]  # First 10 batches
enroll_times = (enrollment_ages - 30).astype(int)
enroll_times = np.clip(enroll_times, 0, Y.shape[2] - 1)

# Batch ranges
batches = [
    (0, 10000),
    (10000, 20000),
    (20000, 30000),
    (30000, 40000),
    (40000, 50000),
    (50000, 60000),
    (60000, 70000),
    (70000, 80000),
    (80000, 90000),
    (90000, 100000)
]

def evaluate_predictions(pi, Y_batch, enroll_times_batch, offset_years=10):
    """Evaluate predictions at enrollment + offset_years"""
    aucs = []
    disease_names_list = []
    
    for d in range(Y_batch.shape[1]):
        # Get predictions at enrollment + offset_years
        preds = []
        actuals = []
        
        for i, enroll_time in enumerate(enroll_times_batch):
            pred_time = enroll_time + offset_years
            if pred_time < Y_batch.shape[2] and enroll_time >= 0:
                # Exclude prevalent cases
                if torch.any(Y_batch[i, d, :enroll_time] > 0):
                    continue
                
                preds.append(pi[i, d, enroll_time].item())
                # Check if event occurred in next year after prediction time
                if pred_time + 1 < Y_batch.shape[2]:
                    actuals.append(torch.any(Y_batch[i, d, enroll_time:pred_time+1]).item())
                else:
                    actuals.append(0)
        
        if len(np.unique(actuals)) > 1 and len(preds) > 0:
            auc = roc_auc_score(actuals, preds)
            aucs.append(auc)
            disease_names_list.append(disease_names[d])
    
    return aucs, disease_names_list

print("="*80)
print("COMPARING AWS vs LOCAL - First 10 Batches (Retrospective Pooled)")
print("="*80)

all_aws_10yr_aucs = {}
all_local_10yr_aucs = {}
all_aws_30yr_aucs = {}
all_local_30yr_aucs = {}

for start_idx, end_idx in batches:
    print(f"\n{'='*80}")
    print(f"BATCH: {start_idx} to {end_idx}")
    print(f"{'='*80}")
    
    # Load AWS model
    aws_model_path = os.path.join(aws_models_dir, f'model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
    if not os.path.exists(aws_model_path):
        print(f"⚠️  AWS model not found: {aws_model_path}")
        continue
    
    aws_model = torch.load(aws_model_path, weights_only=False)
    aws_E = aws_model['E']
    
    # Load local model
    local_model_path = os.path.join(local_base_path, f'model_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
    if not os.path.exists(local_model_path):
        print(f"⚠️  Local model not found: {local_model_path}")
        continue
    
    local_model = torch.load(local_model_path, weights_only=False)
    local_E = local_model['E']
    
    # Load predictions (pi files)
    aws_pi_path = os.path.join(aws_models_dir.replace('models', 'predictions'), f'pi_enroll_fixedphi_sex_{start_idx}_{end_idx}.pt')
    # Actually, we need to load pi from somewhere - let me check if we can compute it or need to download
    
    # For now, let's compare the model states
    print(f"\nComparing model states...")
    print(f"AWS E shape: {aws_E.shape}")
    print(f"Local E shape: {local_E.shape}")
    
    # Compare E matrices
    if torch.equal(aws_E, local_E):
        print("✓ E matrices match!")
    else:
        max_diff = (aws_E - local_E).abs().max().item()
        mean_diff = (aws_E - local_E).abs().mean().item()
        print(f"✗ E matrices differ: max={max_diff:.6f}, mean={mean_diff:.6f}")

print("\n" + "="*80)
print("NOTE: To fully compare predictions, we need the pi files.")
print("Would you like to also download the pi files?")
print("="*80)


