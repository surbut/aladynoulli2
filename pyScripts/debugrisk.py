from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any 

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering  # Add this import

def load_model_essentials(base_path='/Users/sarahurbut/Dropbox/data_for_running/'):
    """
    Load all essential components
    """
    print("Loading components...")
    
    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt')
    E = torch.load(base_path + 'E_matrix.pt')
    G = torch.load(base_path + 'G_matrix.pt')
    
    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt')
    
    print("Loaded all components successfully!")
    
    return Y, E, G, essentials



def plot_roc_curve(y_true, y_pred, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=label)

def load_model_essentials(base_path='/Users/sarahurbut/Dropbox/data_for_running/'):
    """
    Load all essential components
    """
    print("Loading components...")
    
    # Load large matrices
    Y = torch.load(base_path + 'Y_tensor.pt')
    E = torch.load(base_path + 'E_matrix.pt')
    G = torch.load(base_path + 'G_matrix.pt')
    
    # Load other components
    essentials = torch.load(base_path + 'model_essentials.pt')
    
    print("Loaded all components successfully!")
    
    return Y, E, G, essentials



def plot_roc_curve(y_true, y_pred, label):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=label)



def compare_with_pce(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff_fuull'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()



def compare_with_prevent(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction
    """
    our_10yr_risks = []
    actual_10yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Get mean risks across patients for calibration
    predicted_risk_2d = pi.mean(axis=0)  # Shape: [D, T]
    observed_risk_2d = model.Y.numpy().mean(axis=0)  # Shape: [D, T]
    
    # Sort and get LOESS calibration curve
    pred_flat = predicted_risk_2d.flatten()
    obs_flat = observed_risk_2d.flatten()
    sort_idx = np.argsort(pred_flat)
    smoothed = lowess(obs_flat[sort_idx], pred_flat[sort_idx], frac=0.3)
    
    # Apply calibration to all predictions using interpolation
    pi_calibrated = np.interp(pi.flatten(), smoothed[:, 0], smoothed[:, 1]).reshape(pi.shape)
    
    # Calculate 10-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Only use predictions at enrollment time
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk first
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        
        # Convert to 10-year risk
        risk = 1 - (1 - yearly_risk)**10
        our_10yr_risks.append(risk)
        
        # Still look at actual events over 10 years
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+10]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_10yr.append(actual.item())
   
    # Rest of the function remains the same
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['prevent_impute'].values[:len(our_10yr_risks)]
    
    # Calculate ROC AUCs
    our_auc = roc_auc_score(actual_10yr, our_10yr_risks)
    pce_auc = roc_auc_score(actual_10yr, pce_risks)
    
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Our model: {our_auc:.3f}")
    print(f"PREVENT: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr, our_10yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr, pce_risks, label=f'PREVENT (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()



def evaluate_major_diseases_wsex_with_bootstrap_return_risks_too(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10):
    """
    Same as evaluate_major_diseases_wsex but adds bootstrap CIs for AUC.
    Uses exact same time logic and event counting as original.
    Also prints sex-stratified AUCs (except for sex-specific diseases) and ASCVD AUCs for patients with pre-existing RA or breast cancer.
    """

    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],# Sex-specific
        'Prostate_Cancer': ['Cancer of prostate'], # Sex-specific
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    # For ASCVD analysis with pre-existing conditions
    pre_existing_conditions = {
        'RA': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }

    results = {}
    
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")
    
    with torch.no_grad():
        pi, _, _ = model.forward()
        
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True) 

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} ({follow_up_duration_years}-Year Outcome, 1-Year Score)...")
        
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                 if idx not in unique_indices:
                      disease_indices.append(idx)
                      unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
             print(f"No valid matching disease indices found for {disease_group}.")
             results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 
                                     'ci_lower': np.nan, 'ci_upper': np.nan}
             continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'

        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]

        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                 print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                 results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0,
                                         'ci_lower': np.nan, 'ci_upper': np.nan}
                 continue
        
        if len(int_indices_pce) == 0: 
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce] 
            current_N_auc = len(int_indices_pce)

            # Pre-allocate tensors for ALL patients
            risks_auc = torch.zeros(current_N_auc, device=pi.device)
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device)
            processed_indices_auc_final = [] 

            n_prevalent_excluded = 0
            # For ASCVD analysis with pre-existing conditions
            if disease_group == 'ASCVD':
                pre_existing_indices = {}
                for condition, condition_names in pre_existing_conditions.items():
                    indices = []
                    for name in condition_names:
                        matches = [i for i, dname in enumerate(disease_names) if name.lower() in dname.lower()]
                        indices.extend(matches)
                    pre_existing_indices[condition] = list(set(indices))
                # Precompute pre-existing flags for all patients
                pre_existing_flags = {cond: np.zeros(current_N_auc, dtype=bool) for cond in pre_existing_indices}
                for i in range(current_N_auc):
                    t_enroll = int(current_pce_df_auc.iloc[i]['age'] - 30)
                    for cond, idxs in pre_existing_indices.items():
                        for idx in idxs:
                            if idx < current_Y_100k_auc.shape[1] and torch.any(current_Y_100k_auc[i, idx, :t_enroll] > 0):
                                pre_existing_flags[cond][i] = True
                                break

            one_year_risks = []
            ten_year_risks = []
            person_indices = []

            for i in range(current_N_auc): 
                age = current_pce_df_auc.iloc[i]['age'] 
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue

                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue 

                # Store risk for ALL valid enrollment times
                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                risks_auc[i] = yearly_risk

                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2]) 
                if end_time <= t_enroll: continue
                # Check for events and store outcome
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): 
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                # Only add to processed indices if we'll use for AUC
                processed_indices_auc_final.append(i) 

                # 1-year risk (at enrollment + 1)
                pi_diseases_1yr = current_pi_auc[i, disease_indices, t_enroll + 1]
                one_year_risk = 1 - torch.prod(1 - pi_diseases_1yr).item()

                # 10-year risk (cumulative)
                yearly_risks = []
                for t in range(1, follow_up_duration_years + 1):
                    pi_diseases = current_pi_auc[i, disease_indices, t_enroll + t]
                    yearly_risk = 1 - torch.prod(1 - pi_diseases)
                    yearly_risks.append(yearly_risk.item())
                survival_prob = np.prod([1 - r for r in yearly_risks])
                ten_year_risk = 1 - survival_prob

                one_year_risks.append(one_year_risk)
                ten_year_risks.append(ten_year_risk)
                person_indices.append(int_indices_pce[i])

            if not processed_indices_auc_final:
                 auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                 ci_lower = np.nan; ci_upper = np.nan
            else:
                 # Get risks/outcomes only for AUC calculation
                 risks_np = risks_auc[processed_indices_auc_final].cpu().numpy()
                 outcomes_np = outcomes_auc[processed_indices_auc_final].cpu().numpy()
                 n_processed = len(outcomes_np)
                 
                 if disease_group in ["Bipolar_Disorder", "Depression"]:
                    df = pd.DataFrame({
                        "risk": risks_np,
                        "outcome": outcomes_np
                    })
                    df.to_csv(f"debug_{disease_group}.csv", index=False)
                 
                 # Calculate AUC using roc_curve + auc consistently
                 if len(np.unique(outcomes_np)) > 1:
                      fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                      auc_score = auc(fpr, tpr)
                      # Bootstrap CI calculation using same method
                      aucs = []
                      for _ in range(n_bootstraps):
                          indices = np.random.choice(len(risks_np), size=len(risks_np), replace=True)
                          if len(np.unique(outcomes_np[indices])) > 1:
                              fpr_boot, tpr_boot, _ = roc_curve(outcomes_np[indices], risks_np[indices])
                              bootstrap_auc = auc(fpr_boot, tpr_boot)
                              aucs.append(bootstrap_auc)
                      if aucs:
                          ci_lower = np.percentile(aucs, 2.5)
                          ci_upper = np.percentile(aucs, 97.5)
                      else:
                          ci_lower = ci_upper = np.nan
                 else:
                      auc_score = np.nan
                      ci_lower = ci_upper = np.nan
                      print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                 # Calculate events using ALL outcomes
                 n_events = int(torch.sum(outcomes_auc).item())
                 event_rate = (n_events / current_N_auc * 100)
        
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'risk_df': pd.DataFrame({
                'person_index': person_indices,
                'one_year_risk': one_year_risks,
                'ten_year_risk': ten_year_risks
            })
        }
        
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)") 
        print(f"Events ({follow_up_duration_years}-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {current_N_auc} individuals)") 
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")

        # Sex-stratified analysis (except for sex-specific diseases)
        if disease_group not in ['Breast_Cancer', 'Prostate_Cancer'] and len(int_indices_pce) > 0 and n_processed > 0:
            print("\n   Sex-stratified analysis:")
            # Map processed indices to sex (string values)
            processed_sexes = [current_pce_df_auc.iloc[i]['Sex'] for i in processed_indices_auc_final]
            for sex in ['Female', 'Male']:
                sex_indices = [j for j, s in enumerate(processed_sexes) if s == sex]
                if len(sex_indices) > 0:
                    sex_risks = risks_auc[sex_indices].cpu().numpy()
                    sex_outcomes = outcomes_auc[sex_indices].cpu().numpy()
                    if len(np.unique(sex_outcomes)) > 1:
                        fpr, tpr, _ = roc_curve(sex_outcomes, sex_risks)
                        sex_auc = auc(fpr, tpr)
                    else:
                        sex_auc = np.nan
                    sex_events = int(np.sum(sex_outcomes))
                    print(f"   {sex}: AUC = {sex_auc:.3f}, Events = {sex_events}/{len(sex_indices)}")
                else:
                    print(f"   {sex}: No data.")

        # Pre-existing condition analysis for ASCVD
        if disease_group == 'ASCVD' and len(int_indices_pce) > 0 and n_processed > 0:
            print("\n   ASCVD risk in patients with pre-existing conditions:")
            for cond in pre_existing_conditions.keys():
                cond_indices = [i for i in processed_indices_auc_final if pre_existing_flags[cond][i]]
                if len(cond_indices) > 0:
                    cond_risks = risks_auc[cond_indices].cpu().numpy()
                    cond_outcomes = outcomes_auc[cond_indices].cpu().numpy()
                    if len(np.unique(cond_outcomes)) > 1:
                        fpr, tpr, _ = roc_curve(cond_outcomes, cond_risks)
                        cond_auc = auc(fpr, tpr)
                    else:
                        cond_auc = np.nan
                    cond_events = int(np.sum(cond_outcomes))
                    print(f"   {cond}: AUC = {cond_auc:.3f}, Events = {cond_events}/{len(cond_indices)}")

    print(f"\nSummary of Results (Prospective {follow_up_duration_years}-Year Outcome, 1-Year Score, Sex-Adjusted):") 
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)

    return results

major_diseases = { # Example, ADD ALL YOURS
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis'], 'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'], 'CKD': ['Chronic renal failure [CKD]'],
    'All_Cancers': ['Colon cancer', 'Breast cancer [female]', 'Cancer of prostate'], 
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction'], 'Heart_Failure': ['Congestive heart failure (CHF) NOS'],
    'Pneumonia': ['Pneumonia'], 'COPD': ['Chronic airway obstruction'], 'Osteoporosis': ['Osteoporosis NOS'],
    'Anemia': ['Iron deficiency anemias'], 'Colorectal_Cancer': ['Colon cancer'],
    'Breast_Cancer': ['Breast cancer [female]'], 'Prostate_Cancer': ['Cancer of prostate'], 
    'Lung_Cancer': ['Cancer of bronchus; lung'], 'Bladder_Cancer': ['Malignant neoplasm of bladder'],
    'Secondary_Cancer': ['Secondary malignant neoplasm'], 'Depression': ['Major depressive disorder'],
    'Anxiety': ['Anxiety disorder'], 'Bipolar_Disorder': ['Bipolar'], 'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
    'Psoriasis': ['Psoriasis vulgaris'], 'Ulcerative_Colitis': ['Ulcerative colitis'], 'Crohns_Disease': ['Regional enteritis'],
    'Asthma': ['Asthma'], 'Parkinsons': ["Parkinson's disease"], 'Multiple_Sclerosis': ['Multiple sclerosis'],
    'Thyroid_Disorders': ['Hypothyroidism NOS']
}
disease_mapping = { # Example, ADD ALL YOURS
     'ASCVD': ['heart_disease', 'heart_disease.1'], 'Stroke': ['stroke', 'stroke.1'],
     'Diabetes': ['diabetes', 'diabetes.1'], 'Breast_Cancer': ['breast_cancer', 'breast_cancer.1'],
     'Prostate_Cancer': ['prostate_cancer', 'prostate_cancer.1'], 'Lung_Cancer': ['lung_cancer', 'lung_cancer.1'],
     'Colorectal_Cancer': ['bowel_cancer', 'bowel_cancer.1'], 'Depression': [], 'Osteoporosis': [], 
     'Parkinsons': ['parkinsons', 'parkinsons.1'], 'COPD': [], 'Anemia': [], 'CKD': [], 
     'Heart_Failure': ['heart_disease', 'heart_disease.1'], 'Pneumonia': [], 'Atrial_Fib': [], 
     'Bladder_Cancer': [], 'Secondary_Cancer': [], 'Anxiety': [], 'Bipolar_Disorder': [], 
     'Rheumatoid_Arthritis': [], 'Psoriasis': [], 'Ulcerative_Colitis': [], 'Crohns_Disease': [], 
     'Asthma': [], 'Multiple_Sclerosis': [], 'Thyroid_Disorders': []
 }




def evaluate_major_diseases_wsex_with_bootstrap_dynamic_return_risks_too(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10, patient_indices=None):
    """
    Evaluate dynamic 10-year risk for each patient using Aladynoulli model, with bootstrap CIs for AUC.
    For each patient, at each year after enrollment, use the model to get the risk for that year.
    The cumulative 10-year risk is 1 - prod(1 - yearly_risks).
    If patient_indices is provided, subset all data to those indices.
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, auc

    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
        'Prostate_Cancer': ['Cancer of prostate'],
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    results = {}
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")

    # Subset all data if patient_indices is provided
    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    with torch.no_grad():
        pi, _, _ = model.forward()
    if patient_indices is not None:
        pi = pi[patient_indices]
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True)

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} (Dynamic 10-Year Risk)...")
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
            print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
                continue
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce]
            current_N_auc = len(int_indices_pce)
            risks_auc = np.zeros(current_N_auc)
            outcomes_auc = np.zeros(current_N_auc)
            processed_indices_auc_final = []
            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll + follow_up_duration_years >= current_pi_auc.shape[2]:
                    continue
                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue
                # Collect yearly risks for years 1 to 10 after enrollment
                yearly_risks = []
                for t in range(1, follow_up_duration_years + 1):
                    pi_diseases = current_pi_auc[i, disease_indices, t_enroll + t]
                    yearly_risk = 1 - torch.prod(1 - pi_diseases)
                    yearly_risks.append(yearly_risk.item())
                # Compute cumulative 10-year risk, but might be a problem if diseaes occurs, so we should really do td cox
                survival_prob = np.prod([1 - r for r in yearly_risks])
                ten_year_risk = 1 - survival_prob
                risks_auc[i] = ten_year_risk
                # Outcome: did any event occur in the 10 years after enrollment? is this ok because not everyone will have 10 years of follow-up?
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2])
                event_found = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0):
                        outcomes_auc[i] = 1
                        event_found = True
                        break
                processed_indices_auc_final.append(i)
            if not processed_indices_auc_final:
                auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                ci_lower = np.nan; ci_upper = np.nan
            else:
                risks_np = risks_auc[processed_indices_auc_final]
                outcomes_np = outcomes_auc[processed_indices_auc_final]
                n_processed = len(outcomes_np)
                if len(np.unique(outcomes_np)) > 1:
                    fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                    auc_score = auc(fpr, tpr)
                    aucs = []
                    for _ in range(n_bootstraps):
                        indices = np.random.choice(len(risks_np), size=len(risks_np), replace=True)
                        if len(np.unique(outcomes_np[indices])) > 1:
                            fpr_boot, tpr_boot, _ = roc_curve(outcomes_np[indices], risks_np[indices])
                            bootstrap_auc = auc(fpr_boot, tpr_boot)
                            aucs.append(bootstrap_auc)
                    if aucs:
                        ci_lower = np.percentile(aucs, 2.5)
                        ci_upper = np.percentile(aucs, 97.5)
                    else:
                        ci_lower = ci_upper = np.nan
                else:
                    auc_score = np.nan
                    ci_lower = ci_upper = np.nan
                    print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                n_events = int(np.sum(outcomes_np))
                event_rate = (n_events / n_processed * 100) if n_processed > 0 else 0.0
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)")
        print(f"Events (10-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {n_processed} individuals)")
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (Dynamic 10-Year Risk, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)
    return results




def evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=1, patient_indices=None):
    """
    Evaluate 1-year risk for each patient using Aladynoulli model, with bootstrap CIs for AUC.
    For each patient, use the model to get the risk for the first year after enrollment.
    If patient_indices is provided, subset all data to those indices.
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, auc

    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
        'Prostate_Cancer': ['Cancer of prostate'],
        'Lung_Cancer': ['Cancer of bronchus; lung'],
        'Bladder_Cancer': ['Malignant neoplasm of bladder'],
        'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
        'Depression': ['Major depressive disorder'],
        'Anxiety': ['Anxiety disorder'],
        'Bipolar_Disorder': ['Bipolar'],
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Psoriasis': ['Psoriasis vulgaris'],
        'Ulcerative_Colitis': ['Ulcerative colitis'],
        'Crohns_Disease': ['Regional enteritis'],
        'Asthma': ['Asthma'],
        'Parkinsons': ["Parkinson's disease"],
        'Multiple_Sclerosis': ['Multiple sclerosis'],
        'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
    }

    results = {}
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")

    # Subset all data if patient_indices is provided
    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    with torch.no_grad():
        pi, _, _ = model.forward()
    if patient_indices is not None:
        pi = pi[patient_indices]
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for evaluation cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N]
        pce_df = pce_df.iloc[:min_N]
        Y_100k = Y_100k[:min_N]
    pce_df = pce_df.reset_index(drop=True)

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} (1-Year Risk)...")
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        max_model_disease_idx = pi.shape[1] - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
            print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
                continue
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce]
            current_N_auc = len(int_indices_pce)
            risks_auc = np.zeros(current_N_auc)
            outcomes_auc = np.zeros(current_N_auc)
            processed_indices_auc_final = []
            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                if t_enroll < 0 or t_enroll + 1 >= current_pi_auc.shape[2]:
                    continue
                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_enroll] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue
                # Get 1-year risk directly from model
                pi_diseases = current_pi_auc[i, disease_indices, t_enroll + 1]  # +1 for first year after enrollment
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                risks_auc[i] = yearly_risk.item()
                # Outcome: did any event occur in the first year after enrollment?
                end_time = min(t_enroll + 1, current_Y_100k_auc.shape[2])
                event_found = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0):
                        outcomes_auc[i] = 1
                        event_found = True
                        break
                processed_indices_auc_final.append(i)
            if not processed_indices_auc_final:
                auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                ci_lower = np.nan; ci_upper = np.nan
            else:
                risks_np = risks_auc[processed_indices_auc_final]
                outcomes_np = outcomes_auc[processed_indices_auc_final]
                n_processed = len(outcomes_np)
                if len(np.unique(outcomes_np)) > 1:
                    fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                    auc_score = auc(fpr, tpr)
                    aucs = []
                    for _ in range(n_bootstraps):
                        indices = np.random.choice(len(risks_np), size=len(risks_np), replace=True)
                        if len(np.unique(outcomes_np[indices])) > 1:
                            fpr_boot, tpr_boot, _ = roc_curve(outcomes_np[indices], risks_np[indices])
                            bootstrap_auc = auc(fpr_boot, tpr_boot)
                            aucs.append(bootstrap_auc)
                    if aucs:
                        ci_lower = np.percentile(aucs, 2.5)
                        ci_upper = np.percentile(aucs, 97.5)
                    else:
                        ci_lower = ci_upper = np.nan
                else:
                    auc_score = np.nan
                    ci_lower = ci_upper = np.nan
                    print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
                n_events = int(np.sum(outcomes_np))
                event_rate = (n_events / n_processed * 100) if n_processed > 0 else 0.0
        results[disease_group] = {
            'auc': auc_score,
            'n_events': n_events,
            'event_rate': event_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)")
        print(f"Events (1-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {n_processed} individuals)")
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (1-Year Risk, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)
    return results


def get_major_disease_1_10year_auc(Y_full, FH_processed, train_indices, test_indices, disease_mapping, major_diseases, disease_names, aladynoulli_1yr_risk_train, aladynoulli_1yr_risk_test, follow_up_duration_years=10):
    """
    Get 10-year AUC for major diseases.
    """
    