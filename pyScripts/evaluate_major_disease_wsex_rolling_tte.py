
def evaluate_major_diseases_wsex_with_bootstrap_dynamic_rolling_tte(
    pi_batches, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10, patient_indices=None):
    """
    Evaluate dynamic 10-year risk for each patient using rolling, leakage-free predictions from pi_batches.
    For each patient, at each year after enrollment, use the prediction from the model trained for that offset.
    The cumulative 10-year risk is 1 - prod(1 - yearly_risks), where each yearly_risk is from the correct pi_batch.
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
    #if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    #if 'sex' not in pce_df.columns: raise ValueError("'sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")

    # Subset all data if patient_indices is provided
    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    N = len(pce_df)
    D = pi_batches[0].shape[1]
    T = pi_batches[0].shape[2]
    years_to_use = len(pi_batches)

    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group} (Dynamic 10-Year Risk, Rolling)...")
        disease_indices = []
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                if idx not in unique_indices:
                    disease_indices.append(idx)
                    unique_indices.add(idx)
        max_model_disease_idx = D - 1
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
            print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue

        

                # --- Robust sex/sex column handling ---
        sex_col = None
        for col in pce_df.columns:
            if col.lower() == "sex":
                sex_col = col
                break
        if sex_col is None:
            raise ValueError("No 'sex' or 'Sex' column found in pce_df")

        target_sex = None
        if disease_group == 'Breast_Cancer':
            target_sex = 'female'
        elif disease_group == 'Prostate_Cancer':
            target_sex = 'male'

        if target_sex:
            # Normalize column to string for comparison
            col_vals = pce_df[sex_col]
            if col_vals.dtype == object or isinstance(col_vals.iloc[0], str):
                mask_pce = col_vals.str.lower() == target_sex
            else:
                # Assume 0=female, 1=male (change if your convention is different)
                mask_pce = (col_vals == (0 if target_sex == 'female' else 1))
        else:
            mask_pce = pd.Series(True, index=pce_df.index)
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
                if t_enroll < 0 or t_enroll + years_to_use >= T:
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
                # Collect yearly risks for years 1 to years_to_use after enrollment, using rolling pi_batches
                yearly_risks = []
                for k in range(years_to_use):
                    if t_enroll + k < T:
                        # Check if event already occurred before this year
                        event_before_this_year = False
                        for d_idx in disease_indices:
                            if d_idx >= current_Y_100k_auc.shape[1]: continue
                            if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:t_enroll+k] > 0):
                                event_before_this_year = True
                                break
                        
                        if event_before_this_year:
                            break  # Stop collecting predictions after first event
                        
                        pi_diseases = pi_batches[k][int_indices_pce[i], disease_indices, t_enroll + k]
                        yearly_risk = 1 - torch.prod(1 - pi_diseases)
                        yearly_risks.append(yearly_risk.item())

                
                # Calculate cumulative risk from collected yearly risks
                if yearly_risks:
                    survival_prob = np.prod([1 - r for r in yearly_risks])
                    ten_year_risk = 1 - survival_prob
                else:
                    ten_year_risk = 0.0
                risks_auc[i] = ten_year_risk
                # Outcome: did any event occur in the 10 years after enrollment?
                end_time = min(t_enroll + years_to_use, current_Y_100k_auc.shape[2])
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
    print(f"\nSummary of Results (Dynamic 10-Year Risk, Rolling, Censored at First Event, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)
    return results