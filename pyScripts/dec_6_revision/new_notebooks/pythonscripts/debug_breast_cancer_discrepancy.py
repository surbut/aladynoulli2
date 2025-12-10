"""
Debug script to understand why Breast Cancer AUC differs between washout and rolling functions.
"""
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from evaluatetdccode import evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
from evaluatetdccode_MU import evaluate_major_diseases_rolling_1year_roc_curves

# Load the same data used in compare_pi_tensors.py
print("Loading data...")
pi_2 = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/pi_enroll_fixedphi_sex_FULL.pt')
Y_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
E_full = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_matrix_corrected.pt')
disease_names = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv')

# Use first 10k patients
batch_size = 10000
pi_batch = pi_2[:batch_size]
Y_batch = Y_full[:batch_size]
E_batch = E_full[:batch_size]

# Load pce_df
pce_df_full_original = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_prevent_full.csv')
pce_df_full = pce_df_full_original.copy()
if 'Sex' in pce_df_full.columns and pce_df_full['Sex'].dtype == 'object':
    pce_df_full['sex'] = pce_df_full['Sex'].map({'Female': 0, 'Male': 1}).astype(int)
elif 'sex' in pce_df_full.columns:
    pce_df_full['Sex'] = pce_df_full['sex'].map({0: 'Female', 1: 'Male'}).astype(str)

pce_df_batch = pce_df_full.iloc[:batch_size].reset_index(drop=True)

# Create E_enrollment_batch (all zeros, not used)
E_enrollment_batch = torch.zeros_like(E_batch)

print(f"Data loaded: {len(pce_df_batch)} patients")
print(f"Pi shape: {pi_batch.shape}")
print(f"Y shape: {Y_batch.shape}")

# Find Breast Cancer disease indices
breast_cancer_names = ['Breast cancer [female]', 'Malignant neoplasm of female breast']
disease_indices = []
for disease in breast_cancer_names:
    indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    disease_indices.extend(indices)
disease_indices = list(set(disease_indices))
disease_indices = [idx for idx in disease_indices if idx <= pi_batch.shape[1] - 1]
print(f"\nBreast Cancer disease indices: {disease_indices}")

# Filter for females
mask_female = (pce_df_batch['sex'] == 0)
female_indices = np.where(mask_female)[0]
print(f"Female patients: {len(female_indices)}/{len(pce_df_batch)}")

# Manual calculation to see what's different
print("\n" + "="*80)
print("MANUAL CALCULATION - WASHOUT FUNCTION LOGIC")
print("="*80)

# Washout function logic
current_pi_auc = pi_batch[female_indices]
current_Y_100k_auc = Y_batch[female_indices]
current_pce_df_auc = pce_df_batch.iloc[female_indices].reset_index(drop=True)
current_N_auc = len(female_indices)

risks_washout = []
outcomes_washout = []
n_prevalent_excluded_washout = 0
n_invalid_t_start_washout = 0

for i in range(current_N_auc):
    age = current_pce_df_auc.iloc[i]['age']
    t_enroll = int(age - 30)
    t_start = t_enroll + 0  # start_offset = 0
    t_end = t_start + 1  # follow_up_duration_years = 1
    
    if t_start < 0 or t_start >= current_pi_auc.shape[2]:
        n_invalid_t_start_washout += 1
        continue
    
    # Check prevalent
    prevalent = False
    for d_idx in disease_indices:
        if d_idx >= current_Y_100k_auc.shape[1]:
            continue
        if torch.any(current_Y_100k_auc[i, d_idx, :t_start] > 0):
            prevalent = True
            break
    if prevalent:
        n_prevalent_excluded_washout += 1
        continue
    
    # Risk
    pi_diseases = current_pi_auc[i, disease_indices, t_start]
    yearly_risk = 1 - torch.prod(1 - pi_diseases)
    risks_washout.append(yearly_risk.item())
    
    # Outcome
    end_time = min(t_end, current_Y_100k_auc.shape[2])
    if end_time <= t_start:
        continue
    event = 0
    for d_idx in disease_indices:
        if d_idx >= current_Y_100k_auc.shape[1]:
            continue
        if torch.any(current_Y_100k_auc[i, d_idx, t_start:end_time] > 0):
            event = 1
            break
    outcomes_washout.append(event)

print(f"Valid patients (washout): {len(risks_washout)}")
print(f"Prevalent excluded: {n_prevalent_excluded_washout}")
print(f"Invalid t_start: {n_invalid_t_start_washout}")
print(f"Events: {sum(outcomes_washout)}/{len(outcomes_washout)} ({sum(outcomes_washout)/len(outcomes_washout)*100:.2f}%)")
if len(risks_washout) > 0:
    auc_washout = roc_auc_score(outcomes_washout, risks_washout)
    print(f"AUC (washout logic): {auc_washout:.6f}")

print("\n" + "="*80)
print("MANUAL CALCULATION - ROLLING FUNCTION LOGIC")
print("="*80)

# Rolling function logic
T = pi_batch.shape[2]
risks_rolling = []
outcomes_rolling = []
n_prevalent_excluded_rolling = 0
n_invalid_t_start_rolling = 0

for i in female_indices:  # i is original index
    age = pce_df_batch.iloc[i]['age']
    t_enroll = int(age - 30)
    t_start = t_enroll + 0  # k = 0 for offset 0
    
    if t_start < 0 or t_start >= T:
        n_invalid_t_start_rolling += 1
        continue
    
    # Check prevalent
    prevalent = False
    for d_idx in disease_indices:
        if d_idx >= Y_batch.shape[1]:
            continue
        if torch.any(Y_batch[i, d_idx, :t_start] > 0):
            prevalent = True
            break
    if prevalent:
        n_prevalent_excluded_rolling += 1
        continue
    
    # Risk
    pi_diseases = pi_batch[0, disease_indices, t_start]  # Wait, this is wrong! Should be pi_batch[i, ...]
    # Actually, rolling function uses pi_batches[k][i, ...] where pi_batches[0] = pi_batch
    # So it should be: pi_batch[i, disease_indices, t_start]
    pi_diseases = pi_batch[i, disease_indices, t_start]
    yearly_risk = 1 - torch.prod(1 - pi_diseases)
    risks_rolling.append(yearly_risk.item())
    
    # Outcome
    end_time = min(t_start + 1, Y_batch.shape[2])
    event = 0
    for d_idx in disease_indices:
        if d_idx >= Y_batch.shape[1]:
            continue
        if torch.any(Y_batch[i, d_idx, t_start:end_time] > 0):
            event = 1
            break
    outcomes_rolling.append(event)

print(f"Valid patients (rolling): {len(risks_rolling)}")
print(f"Prevalent excluded: {n_prevalent_excluded_rolling}")
print(f"Invalid t_start: {n_invalid_t_start_rolling}")
print(f"Events: {sum(outcomes_rolling)}/{len(outcomes_rolling)} ({sum(outcomes_rolling)/len(outcomes_rolling)*100:.2f}%)")
if len(risks_rolling) > 0:
    auc_rolling = roc_auc_score(outcomes_rolling, risks_rolling)
    print(f"AUC (rolling logic): {auc_rolling:.6f}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print(f"Washout AUC: {auc_washout:.6f}")
print(f"Rolling AUC: {auc_rolling:.6f}")
print(f"Difference: {auc_washout - auc_rolling:.6f}")
print(f"\nWashout patients: {len(risks_washout)}, Rolling patients: {len(risks_rolling)}")
print(f"Washout events: {sum(outcomes_washout)}, Rolling events: {sum(outcomes_rolling)}")

# Check if patient sets are the same
if len(risks_washout) == len(risks_rolling):
    print("\nChecking if same patients...")
    # Compare risks and outcomes
    risks_match = np.allclose(risks_washout, risks_rolling, rtol=1e-5)
    outcomes_match = (np.array(outcomes_washout) == np.array(outcomes_rolling)).all()
    print(f"Risks match: {risks_match}")
    print(f"Outcomes match: {outcomes_match}")
    
    if not risks_match:
        diff_idx = np.where(~np.isclose(risks_washout, risks_rolling, rtol=1e-5))[0]
        print(f"\nFirst 5 differing risk indices: {diff_idx[:5]}")
        for idx in diff_idx[:5]:
            print(f"  Index {idx}: washout={risks_washout[idx]:.8f}, rolling={risks_rolling[idx]:.8f}, diff={risks_washout[idx]-risks_rolling[idx]:.8f}")
    
    if not outcomes_match:
        diff_idx = np.where(np.array(outcomes_washout) != np.array(outcomes_rolling))[0]
        print(f"\nDiffering outcome indices: {len(diff_idx)}")
        for idx in diff_idx[:10]:
            print(f"  Index {idx}: washout={outcomes_washout[idx]}, rolling={outcomes_rolling[idx]}")

