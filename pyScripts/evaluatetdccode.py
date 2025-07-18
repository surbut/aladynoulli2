import torch

def evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
    pi, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, 
    follow_up_duration_years=1, patient_indices=None, start_offset=0
):
    """
    Evaluate 1-year risk for each patient using pre-computed pi values, with bootstrap CIs for AUC.
    For each patient, use the pi tensor to get the risk for the first year after enrollment.
    If patient_indices is provided, subset all data to those indices.
    
    Args:
        pi: Pre-computed tensor of disease probabilities (shape: [n_patients, n_diseases, n_time_points])
        Y_100k: Outcome tensor
        E_100k: Event tensor  
        disease_names: List of disease names
        pce_df: DataFrame with patient characteristics including 'Sex' and 'age'
        n_bootstraps: Number of bootstrap samples for CI calculation
        follow_up_duration_years: Follow-up duration in years
        patient_indices: Optional indices to subset the data
        start_offset: Start offset for the evaluation window
    """
    import numpy as np
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
    if 'sex' not in pce_df.columns: raise ValueError("'sex' column not found in pce_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")

    # Subset all data if patient_indices is provided
    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)
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
        #print(f"\nEvaluating {disease_group} (1-Year Risk)...")
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
            #print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'c_index': np.nan}
            continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 0
        elif disease_group == 'Prostate_Cancer': target_sex = 1
        mask_pce = (pce_df['sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        #print("int_indices_pce:", int_indices_pce, type(int_indices_pce))
        #   print("len(int_indices_pce):", len(int_indices_pce))
        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'c_index': np.nan}
                continue
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
            c_index = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce]
            current_N_auc = len(int_indices_pce)
            risks_auc = torch.zeros(current_N_auc, device=pi.device)
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device)
            processed_indices_auc_final = [] 
            # For C-index
            age_enrolls = []
            age_at_events = []
            event_indicators = []
            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                t_start = t_enroll + start_offset
                t_end = t_start + follow_up_duration_years
                if t_start < 0 or t_start >= current_pi_auc.shape[2]:
                    continue
                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_start] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue
                # Store risk for ALL valid enrollment times
                pi_diseases = current_pi_auc[i, disease_indices, t_start]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                risks_auc[i] = yearly_risk
                end_time = min(t_end, current_Y_100k_auc.shape[2]) 
                if end_time <= t_start: continue
                # --- C-index: Find time-to-event and event indicator ---
                age_enroll = t_start + 30
                age_at_event = end_time + 30 - 1
                event = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    event_times = torch.where(current_Y_100k_auc[i, d_idx, t_start:end_time] > 0)[0]
                    if len(event_times) > 0:
                        this_event_age = t_start + event_times[0].item() + 30
                        if this_event_age < age_at_event:
                            age_at_event = this_event_age
                            event = 1
                age_enrolls.append(age_enroll)
                age_at_events.append(age_at_event)
                event_indicators.append(event)
                # --- Outcome: Check event in next follow_up_duration_years ---
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_start:end_time] > 0): 
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                processed_indices_auc_final.append(i) 
            if not processed_indices_auc_final:
                 auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                 ci_lower = np.nan; ci_upper = np.nan
                 c_index = np.nan
            else:
                 # Get risks/outcomes only for AUC calculation
                 risks_np = risks_auc[processed_indices_auc_final].cpu().numpy()
                 outcomes_np = outcomes_auc[processed_indices_auc_final].cpu().numpy()
                 n_processed = len(outcomes_np)
                 # For C-index, filter to processed indices
                 age_enrolls_np = np.array(age_enrolls)
                 age_at_events_np = np.array(age_at_events)
                 event_indicators_np = np.array(event_indicators)
                 durations = age_at_events_np - age_enrolls_np
                 # Calculate C-index
                 from lifelines.utils import concordance_index
                 try:
                     c_index = concordance_index(durations, risks_np, event_indicators_np)
                 except Exception as e:
                     #print(f"C-index calculation failed: {e}")
                     c_index = np.nan
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
            'c_index': c_index
        }
        #print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)") 
        #print(f"C-index: {c_index:.3f} (calculated on {n_processed} individuals)")
        #print(f"Events ({follow_up_duration_years}-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {current_N_auc} individuals)") 
       # print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (1-Year Risk, Sex-Adjusted, Offset={start_offset}):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10} {'C-index':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        c_index_str = f"{res['c_index']:.3f}" if not np.isnan(res['c_index']) else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str:<10} {c_index_str}")
    print("-" * 80)
    return results


def evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end(
    pi, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, 
    follow_up_duration_years=1, patient_indices=None, start_offset=0
):
    """
    Evaluate 1-year risk for each patient using pre-computed pi values, with bootstrap CIs for AUC.
    For each patient, use the pi tensor to get the risk for the first year after enrollment.
    If patient_indices is provided, subset all data to those indices.
    
    Args:
        pi: Pre-computed tensor of disease probabilities (shape: [n_patients, n_diseases, n_time_points])
        Y_100k: Outcome tensor
        E_100k: Event tensor  
        disease_names: List of disease names
        pce_df: DataFrame with patient characteristics including 'Sex' and 'age'
        n_bootstraps: Number of bootstrap samples for CI calculation
        follow_up_duration_years: Follow-up duration in years
        patient_indices: Optional indices to subset the data
        start_offset: Start offset for the evaluation window
    """
    import numpy as np
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
        #print(f"\nEvaluating {disease_group} (1-Year Risk)...")
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
            #print(f"No valid matching disease indices found for {disease_group}.")
            results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'c_index': np.nan}
            continue

        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        #print("int_indices_pce:", int_indices_pce, type(int_indices_pce))
        #   print("len(int_indices_pce):", len(int_indices_pce))
        if target_sex:
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} individuals in cohort")
            if len(int_indices_pce) == 0:
                print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0, 'ci_lower': np.nan, 'ci_upper': np.nan, 'c_index': np.nan}
                continue
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
            ci_lower = np.nan; ci_upper = np.nan
            c_index = np.nan
        else:
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce]
            current_N_auc = len(int_indices_pce)
            risks_auc = torch.zeros(current_N_auc, device=pi.device)
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device)
            processed_indices_auc_final = [] 
            # For C-index
            age_enrolls = []
            age_at_events = []
            event_indicators = []
            n_prevalent_excluded = 0
            for i in range(current_N_auc):
                age = current_pce_df_auc.iloc[i]['age']
                t_enroll = int(age - 30)
                t_start = t_enroll + start_offset
                t_end = t_start + follow_up_duration_years
                if t_start < 0 or t_start >= current_pi_auc.shape[2]:
                    continue
                # INCIDENT DISEASE FILTER: Only for single-disease outcomes
                if len(disease_indices) == 1:
                    prevalent = False
                    for d_idx in disease_indices:
                        if d_idx >= current_Y_100k_auc.shape[1]:
                            continue
                        if torch.any(current_Y_100k_auc[i, d_idx, :t_start] > 0):
                            prevalent = True
                            break
                    if prevalent:
                        n_prevalent_excluded += 1
                        continue
                # Store risk for ALL valid enrollment times
                pi_diseases = current_pi_auc[i, disease_indices, t_start]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                risks_auc[i] = yearly_risk
                end_time = min(t_end, current_Y_100k_auc.shape[2]) 
                if end_time <= t_start: continue
                # --- C-index: Find time-to-event and event indicator ---
                age_enroll = t_start + 30
                age_at_event = end_time + 30 - 1
                event = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    event_times = torch.where(current_Y_100k_auc[i, d_idx, t_start:end_time] > 0)[0]
                    if len(event_times) > 0:
                        this_event_age = t_start + event_times[0].item() + 30
                        if this_event_age < age_at_event:
                            age_at_event = this_event_age
                            event = 1
                age_enrolls.append(age_enroll)
                age_at_events.append(age_at_event)
                event_indicators.append(event)
                # --- Outcome: Check event in next follow_up_duration_years ---
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_start:end_time] > 0): 
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                processed_indices_auc_final.append(i) 
            if not processed_indices_auc_final:
                 auc_score = np.nan; n_events = 0; event_rate = 0.0; n_processed = 0
                 ci_lower = np.nan; ci_upper = np.nan
                 c_index = np.nan
            else:
                 # Get risks/outcomes only for AUC calculation
                 risks_np = risks_auc[processed_indices_auc_final].cpu().numpy()
                 outcomes_np = outcomes_auc[processed_indices_auc_final].cpu().numpy()
                 n_processed = len(outcomes_np)
                 # For C-index, filter to processed indices
                 age_enrolls_np = np.array(age_enrolls)
                 age_at_events_np = np.array(age_at_events)
                 event_indicators_np = np.array(event_indicators)
                 durations = age_at_events_np - age_enrolls_np
                 # Calculate C-index
                 from lifelines.utils import concordance_index
                 try:
                     c_index = concordance_index(durations, risks_np, event_indicators_np)
                 except Exception as e:
                     #print(f"C-index calculation failed: {e}")
                     c_index = np.nan
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
            'c_index': c_index
        }
        #print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)") 
        #print(f"C-index: {c_index:.3f} (calculated on {n_processed} individuals)")
        #print(f"Events ({follow_up_duration_years}-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {current_N_auc} individuals)") 
       # print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (1-Year Risk, Sex-Adjusted, Offset={start_offset}):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10} {'C-index':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        c_index_str = f"{res['c_index']:.3f}" if not np.isnan(res['c_index']) else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str:<10} {c_index_str}")
    print("-" * 80)
    return results
#

def dynamic_aladynoulli_auc_for_preexisting_dynamic_1year(
    pi_offset, Y_100k, E_100k, disease_names, pce_df, preexisting_group, 
    n_bootstraps=100, follow_up_duration_years=1, start_offset=0
):
    """
    Compute dynamic 1-year ASCVD AUC for patients with pre-existing RA or breast cancer.
    preexisting_group: 'Rheumatoid_Arthritis' or 'Breast_Cancer'
    """
    preexisting_diseases = {
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }
    ascvd_diseases = [
        'Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease',
        'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'
    ]
    # Find indices for pre-existing
    preexisting_indices = []
    for disease in preexisting_diseases[preexisting_group]:
        preexisting_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    # Find patients with pre-existing at enrollment
    preexisting_patients = []
    for i, row in enumerate(pce_df.itertuples()):
        age = row.age
        t_enroll = int(age - 30)
        if t_enroll < 0 or t_enroll >= Y_100k.shape[2]:
            continue
        for d_idx in preexisting_indices:
            if torch.any(Y_100k[i, d_idx, :t_enroll] > 0):
                preexisting_patients.append(i)
                break
    if not preexisting_patients:
        print(f"No patients with {preexisting_group} at enrollment.")
        return None
    results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end(
        pi_offset, Y_100k, E_100k, disease_names, pce_df, 
        n_bootstraps=n_bootstraps, 
        follow_up_duration_years=follow_up_duration_years, 
        start_offset=start_offset,
        patient_indices=preexisting_patients
    )
    return results['ASCVD']



def dynamic_aladynoulli_auc_for_preexisting(model, Y_100k, E_100k, disease_names, pce_df, preexisting_group, n_bootstraps=100, follow_up_duration_years=10):
    """
    Compute dynamic 10-year ASCVD AUC for patients with pre-existing RA or breast cancer.
    preexisting_group: 'Rheumatoid_Arthritis' or 'Breast_Cancer'
    """
    # Get indices for pre-existing condition
    preexisting_diseases = {
        'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast']
    }
    ascvd_diseases = [
        'Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease',
        'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'
    ]
    # Find indices for pre-existing
    preexisting_indices = []
    for disease in preexisting_diseases[preexisting_group]:
        preexisting_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    # Find indices for ASCVD
    ascvd_indices = []
    for disease in ascvd_diseases:
        ascvd_indices += [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
    # Find patients with pre-existing at enrollment
    preexisting_patients = []
    for i, row in enumerate(pce_df.itertuples()):
        age = row.age
        t_enroll = int(age - 30)
        if t_enroll < 0 or t_enroll >= Y_100k.shape[2]:
            continue
        for d_idx in preexisting_indices:
            if torch.any(Y_100k[i, d_idx, :t_enroll] > 0):
                preexisting_patients.append(i)
                break
    if not preexisting_patients:
        print(f"No patients with {preexisting_group} at enrollment.")
        return ModuleNotFoundError
    # Use dynamic logic for ASCVD only
    results = evaluate_major_diseases_wsex_with_bootstrap_dynamic(
        model, Y_100k, E_100k, disease_names, pce_df,
        n_bootstraps=n_bootstraps, follow_up_duration_years=follow_up_duration_years,
        patient_indices=preexisting_patients
    )
    return results['ASCVD']




def evaluate_major_diseases_wsex_with_bootstrap_dynamic(model, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10, patient_indices=None):
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

def evaluate_major_diseases_wsex_with_bootstrap_dynamic_rolling(
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
                        pi_diseases = pi_batches[k][int_indices_pce[i], disease_indices, t_enroll + k]
                        yearly_risk = 1 - torch.prod(1 - pi_diseases)
                        yearly_risks.append(yearly_risk.item())
                survival_prob = np.prod([1 - r for r in yearly_risks])
                ten_year_risk = 1 - survival_prob
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

def evaluate_major_diseases_rolling_1year_roc_curves(
    pi_batches, Y_100k, E_100k, disease_names, pce_df, patient_indices=None, plot_group=None):
    """
    For each offset (year), compute and return the ROC curve (FPR, TPR, thresholds, AUC) for 1-year risk prediction for each major disease group,
    using the appropriate pi_batch for that offset. Optionally plot the ROC curves for a selected disease group.
    For ASCVD at offset 0 (enrollment), also plot PCE/PREVENT ROC curves for 10-year risk prediction.
    Returns a dictionary: {disease_group: [ (fpr, tpr, thresholds, auc) for each offset ]}
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

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

    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    N = len(pce_df)
    D = pi_batches[0].shape[1]
    T = pi_batches[0].shape[2]
    years_to_use = len(pi_batches)

    results = {group: [] for group in major_diseases}

    # Check if PCE/PREVENT clinical variables are available (only at enrollment)
    pce_available = 'pce_goff_fuull' in pce_df.columns
    prevent_available = 'prevent_impute' in pce_df.columns
    
    if pce_available or prevent_available:
        print(f"PCE available: {pce_available}, PREVENT available: {prevent_available}")
        # Check how many patients have valid PCE/PREVENT scores (not NaN)
        if pce_available:
            pce_valid_count = pce_df['pce_goff_fuull'].notna().sum()
            print(f"Patients with valid PCE scores: {pce_valid_count}/{len(pce_df)} ({pce_valid_count/len(pce_df)*100:.1f}%)")
        if prevent_available:
            prevent_valid_count = pce_df['prevent_impute'].notna().sum()
            print(f"Patients with valid PREVENT scores: {prevent_valid_count}/{len(pce_df)} ({prevent_valid_count/len(pce_df)*100:.1f}%)")
    else:
        print("No PCE or PREVENT clinical variables found in the data.")

    # Store PCE/PREVENT ROC data for ASCVD 10-year risk (only at offset 0)
    pce_roc_curve_10yr_offset0 = None
    prevent_roc_curve_10yr_offset0 = None

    for disease_group, disease_list in major_diseases.items():
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
            results[disease_group] = [None for _ in range(years_to_use)]
            continue
        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        
        for k in range(years_to_use):
            preds = []
            outcomes = []
            pce_preds_10yr = []
            prevent_preds_10yr = []
            outcomes_10yr = []
            
            for i in int_indices_pce:
                age = pce_df.iloc[i]['age']
                t_enroll = int(age - 30)
                t_start = t_enroll + k
                if t_start < 0 or t_start >= T:
                    continue
                # Exclude prevalent cases at t_start
                prevalent = False
                for d_idx in disease_indices:
                    if d_idx >= Y_100k.shape[1]:
                        continue
                    if torch.any(Y_100k[i, d_idx, :t_start] > 0):
                        prevalent = True
                        break
                if prevalent:
                    continue
                # Prediction for this year (1-year risk)
                pi_diseases = pi_batches[k][i, disease_indices, t_start]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                preds.append(yearly_risk.item())
                # Outcome: did any event occur in the next year?
                event = 0
                end_time = min(t_start + 1, Y_100k.shape[2])
                for d_idx in disease_indices:
                    if d_idx >= Y_100k.shape[1]: continue
                    if torch.any(Y_100k[i, d_idx, t_start:end_time] > 0):
                        event = 1
                        break
                outcomes.append(event)
                
                # PCE/PREVENT predictions for 10-year risk (only at offset 0)
                if disease_group == 'ASCVD' and k == 0 and (pce_available or prevent_available):
                    if pce_available and pd.notna(pce_df.iloc[i]['pce_goff_fuull']):
                        pce_preds_10yr.append(pce_df.iloc[i]['pce_goff_fuull'])
                    if prevent_available and pd.notna(pce_df.iloc[i]['prevent_impute']):
                        prevent_preds_10yr.append(pce_df.iloc[i]['prevent_impute'])
                    
                    # 10-year outcome: did any event occur in the next 10 years?
                    event_10yr = 0
                    end_time_10yr = min(t_start + 10, Y_100k.shape[2])
                    for d_idx in disease_indices:
                        if d_idx >= Y_100k.shape[1]: continue
                        if torch.any(Y_100k[i, d_idx, t_start:end_time_10yr] > 0):
                            event_10yr = 1
                            break
                    outcomes_10yr.append(event_10yr)
            
            if disease_group == 'ASCVD':
                print(f"Offset {k}: Included patients: {len(preds)}, 1-year Events: {sum(outcomes)}")
                if k == 0 and (pce_available or prevent_available):
                    print(f"  PCE patients: {len(pce_preds_10yr)}, PREVENT patients: {len(prevent_preds_10yr)}, 10-year Events: {sum(outcomes_10yr)}")
            
            if len(np.unique(outcomes)) > 1:
                fpr, tpr, thresholds = roc_curve(outcomes, preds)
                auc_val = auc(fpr, tpr)
            else:
                fpr, tpr, thresholds, auc_val = None, None, None, None
            results[disease_group].append((fpr, tpr, thresholds, auc_val))
            
            # Compute PCE/PREVENT ROC for ASCVD 10-year risk (only at offset 0)
            if disease_group == 'ASCVD' and k == 0 and len(outcomes_10yr) > 0 and len(np.unique(outcomes_10yr)) > 1:
                if pce_available and len(pce_preds_10yr) == len(outcomes_10yr) and len(pce_preds_10yr) > 0:
                    fpr_pce, tpr_pce, _ = roc_curve(outcomes_10yr, pce_preds_10yr)
                    auc_pce = auc(fpr_pce, tpr_pce)
                    pce_roc_curve_10yr_offset0 = (fpr_pce, tpr_pce, auc_pce)
                if prevent_available and len(prevent_preds_10yr) == len(outcomes_10yr) and len(prevent_preds_10yr) > 0:
                    fpr_prev, tpr_prev, _ = roc_curve(outcomes_10yr, prevent_preds_10yr)
                    auc_prev = auc(fpr_prev, tpr_prev)
                    prevent_roc_curve_10yr_offset0 = (fpr_prev, tpr_prev, auc_prev)

    # Optionally plot for a selected group
    if plot_group is not None and plot_group in results:
        plt.figure(figsize=(8,6))
        for k, (fpr, tpr, thresholds, auc_val) in enumerate(results[plot_group]):
            if fpr is not None:
                plt.plot(fpr, tpr, label=f'Aladynoulli 1-year (Year {k}, AUC={auc_val:.3f})')
        
        # Plot PCE/PREVENT for ASCVD 10-year risk (only at offset 0)
        if plot_group == 'ASCVD' and (pce_available or prevent_available):
            if pce_roc_curve_10yr_offset0 is not None:
                fpr_pce, tpr_pce, auc_pce = pce_roc_curve_10yr_offset0
                plt.plot(fpr_pce, tpr_pce, '--', label=f'PCE 10-year (Enrollment, AUC={auc_pce:.3f})')
            if prevent_roc_curve_10yr_offset0 is not None:
                fpr_prev, tpr_prev, auc_prev = prevent_roc_curve_10yr_offset0
                plt.plot(fpr_prev, tpr_prev, ':', label=f'PREVENT 10-year (Enrollment, AUC={auc_prev:.3f})')
        
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if pce_available or prevent_available:
            plt.title(f'ROC Curves for {plot_group}: Aladynoulli 1-year (All Offsets) vs PCE/PREVENT 10-year (Enrollment Only)')
        else:
            plt.title(f'ROC Curves for {plot_group}: Aladynoulli 1-year')
        plt.legend()
        plt.grid(True)
        
        # Save to PDF
        plt.savefig(f'roc_curves_{plot_group}.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    return results


def evaluate_major_diseases_rolling_1year_roc_curves_highres(
    pi_batches, Y_100k, E_100k, disease_names, pce_df, patient_indices=None, plot_group=None):
    """
    For each offset (year), compute and return the ROC curve (FPR, TPR, thresholds, AUC) for 1-year risk prediction for each major disease group,
    using the appropriate pi_batch for that offset. Optionally plot the ROC curves for a selected disease group.
    For ASCVD at each offset, also plot PCE/PREVENT ROC curves for 1-year risk prediction.
    Returns a dictionary: {disease_group: [ (aladynoulli: (fpr, tpr, thresholds, auc), pce: (fpr, tpr, auc), prevent: (fpr, tpr, auc)) for each offset ]}
    """
    import numpy as np
    import torch
    import pandas as pd
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

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

    if patient_indices is not None:
        Y_100k = Y_100k[patient_indices]
        E_100k = E_100k[patient_indices]
        pce_df = pce_df.iloc[patient_indices].reset_index(drop=True)

    N = len(pce_df)
    D = pi_batches[0].shape[1]
    T = pi_batches[0].shape[2]
    years_to_use = len(pi_batches)

    results = {group: [] for group in major_diseases}

    # Check if PCE/PREVENT clinical variables are available (only at enrollment)
    pce_available = 'pce_goff_fuull' in pce_df.columns
    prevent_available = 'prevent_impute' in pce_df.columns
    
    if pce_available or prevent_available:
        print(f"PCE available: {pce_available}, PREVENT available: {prevent_available}")
        if pce_available:
            pce_valid_count = pce_df['pce_goff_fuull'].notna().sum()
            print(f"Patients with valid PCE scores: {pce_valid_count}/{len(pce_df)} ({pce_valid_count/len(pce_df)*100:.1f}%)")
        if prevent_available:
            prevent_valid_count = pce_df['prevent_impute'].notna().sum()
            print(f"Patients with valid PREVENT scores: {prevent_valid_count}/{len(pce_df)} ({prevent_valid_count/len(pce_df)*100:.1f}%)")
    else:
        print("No PCE or PREVENT clinical variables found in the data.")

    for disease_group, disease_list in major_diseases.items():
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
            results[disease_group] = [None for _ in range(years_to_use)]
            continue
        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'
        mask_pce = (pce_df['Sex'] == target_sex) if target_sex else pd.Series(True, index=pce_df.index)
        int_indices_pce = np.where(mask_pce)[0]
        
        for k in range(years_to_use):
            preds = []
            outcomes = []
            pce_preds_1yr = []
            prevent_preds_1yr = []
            outcomes_1yr = []
            
            for i in int_indices_pce:
                age = pce_df.iloc[i]['age']
                t_enroll = int(age - 30)
                t_start = t_enroll + k
                if t_start < 0 or t_start >= T:
                    continue
                # Exclude prevalent cases at t_start
                prevalent = False
                for d_idx in disease_indices:
                    if d_idx >= Y_100k.shape[1]:
                        continue
                    if torch.any(Y_100k[i, d_idx, :t_start] > 0):
                        prevalent = True
                        break
                if prevalent:
                    continue
                # Prediction for this year (1-year risk)
                pi_diseases = pi_batches[k][i, disease_indices, t_start]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                preds.append(yearly_risk.item())
                # Outcome: did any event occur in the next year?
                event = 0
                end_time = min(t_start + 1, Y_100k.shape[2])
                for d_idx in disease_indices:
                    if d_idx >= Y_100k.shape[1]: continue
                    if torch.any(Y_100k[i, d_idx, t_start:end_time] > 0):
                        event = 1
                        break
                outcomes.append(event)
                # PCE/PREVENT predictions for 1-year risk at this offset
                if pce_available and pd.notna(pce_df.iloc[i]['pce_goff_fuull']):
                    pce_preds_1yr.append(pce_df.iloc[i]['pce_goff_fuull'])
                if prevent_available and pd.notna(pce_df.iloc[i]['prevent_impute']):
                    prevent_preds_1yr.append(pce_df.iloc[i]['prevent_impute'])
                outcomes_1yr.append(event)
            # Compute ROC/AUC for Aladynoulli
            if len(np.unique(outcomes)) > 1:
                fpr, tpr, thresholds = roc_curve(outcomes, preds)
                auc_val = auc(fpr, tpr)
            else:
                fpr, tpr, thresholds, auc_val = None, None, None, None
            # Compute ROC/AUC for PCE
            if pce_available and len(pce_preds_1yr) == len(outcomes_1yr) and len(pce_preds_1yr) > 0 and len(np.unique(outcomes_1yr)) > 1:
                fpr_pce, tpr_pce, _ = roc_curve(outcomes_1yr, pce_preds_1yr)
                auc_pce = auc(fpr_pce, tpr_pce)
                pce_roc = (fpr_pce, tpr_pce, auc_pce)
            else:
                pce_roc = (None, None, None)
            # Compute ROC/AUC for PREVENT
            if prevent_available and len(prevent_preds_1yr) == len(outcomes_1yr) and len(prevent_preds_1yr) > 0 and len(np.unique(outcomes_1yr)) > 1:
                fpr_prev, tpr_prev, _ = roc_curve(outcomes_1yr, prevent_preds_1yr)
                auc_prev = auc(fpr_prev, tpr_prev)
                prevent_roc = (fpr_prev, tpr_prev, auc_prev)
            else:
                prevent_roc = (None, None, None)
            # Store all three in results
            results[disease_group].append({
                'aladynoulli': (fpr, tpr, thresholds, auc_val),
                'pce': pce_roc,
                'prevent': prevent_roc
            })
            if disease_group == 'ASCVD':
                print(f"Offset {k}: Included patients: {len(preds)}, 1-year Events: {sum(outcomes)}")
                if pce_available:
                    print(f"  PCE patients: {len(pce_preds_1yr)}, 1-year Events: {sum(outcomes_1yr)}")
                if prevent_available:
                    print(f"  PREVENT patients: {len(prevent_preds_1yr)}, 1-year Events: {sum(outcomes_1yr)}")
    # Optionally plot for a selected group
    if plot_group is not None and plot_group in results:
        plt.figure(figsize=(10,8))
        for k, res in enumerate(results[plot_group]):
            fpr, tpr, _, auc_val = res['aladynoulli']
            if fpr is not None:
                plt.plot(fpr, tpr, label=f'Aladynoulli (Year {k}, AUC={auc_val:.3f})')
            fpr_pce, tpr_pce, auc_pce = res['pce']
            if fpr_pce is not None:
                plt.plot(fpr_pce, tpr_pce, '--', label=f'PCE (Year {k}, AUC={auc_pce:.3f})')
            fpr_prev, tpr_prev, auc_prev = res['prevent']
            if fpr_prev is not None:
                plt.plot(fpr_prev, tpr_prev, ':', label=f'PREVENT (Year {k}, AUC={auc_prev:.3f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'1-Year ROC Curves for {plot_group} at Each Offset')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'roc_curves_{plot_group}_all_years.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    return results

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

def evaluate_major_diseases_wsex_with_bootstrap_pi_input(
    pi, Y_100k, E_100k, disease_names, pce_df, n_bootstraps=100, follow_up_duration_years=10, static_10yr_from_1yr=False):
    """
    Same as evaluate_major_diseases_wsex but adds bootstrap CIs for AUC.
    If static_10yr_from_1yr=True, computes 10-year risk as 1 - (1 - 1yr_pi)^10 using only the 1-year pi at enrollment.
    Compares this static 10-year risk to observed 10-year outcome (event in 10 years after enrollment).
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
        print(f"\nEvaluating {disease_group} (10-Year Outcome, Static 1-Year Pi at Enrollment: {static_10yr_from_1yr})...")
        
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

            # Use lists for all per-patient results
            risks_auc = []
            outcomes_auc = []
            age_enrolls = []
            age_at_events = []
            event_indicators = []
            n_prevalent_excluded = 0

            # (ASCVD pre-existing logic stays as before)

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
                        continue  # Skip this patient for this disease group

                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                end_time = min(t_enroll + follow_up_duration_years, current_Y_100k_auc.shape[2]) 
                if end_time <= t_enroll: continue

                # --- C-index: Find time-to-event and event indicator ---
                age_enroll = t_enroll + 30
                age_at_event = end_time + 30 - 1
                event = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    event_times = torch.where(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0)[0]
                    if len(event_times) > 0:
                        this_event_age = t_enroll + event_times[0].item() + 30
                        if this_event_age < age_at_event:
                            age_at_event = this_event_age
                            event = 1

                # --- Outcome: Check event in next follow_up_duration_years ---
                outcome = 0
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): 
                        outcome = 1
                        break

                # Only add to lists if included
                risks_auc.append(yearly_risk.item() if hasattr(yearly_risk, 'item') else float(yearly_risk))
                outcomes_auc.append(outcome)
                age_enrolls.append(age_enroll)
                age_at_events.append(age_at_event)
                event_indicators.append(event)

            n_processed = len(risks_auc)
            if n_processed == 0:
                auc_score = np.nan; n_events = 0; event_rate = 0.0; ci_lower = np.nan; ci_upper = np.nan; c_index = np.nan
            else:
                risks_np = np.array(risks_auc)
                outcomes_np = np.array(outcomes_auc)
                age_enrolls_np = np.array(age_enrolls)
                age_at_events_np = np.array(age_at_events)
                event_indicators_np = np.array(event_indicators)
                durations = age_at_events_np - age_enrolls_np
                from lifelines.utils import concordance_index
                try:
                    c_index = concordance_index(durations, -risks_np, event_indicators_np)
                except Exception as e:
                    print(f"C-index calculation failed: {e}")
                    c_index = np.nan
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
            'ci_upper': ci_upper,
            'c_index': c_index
        }
        print(f"AUC: {auc_score:.3f} ({ci_lower:.3f}-{ci_upper:.3f}) (calculated on {n_processed} individuals)")
        print(f"Events (10-Year in Eval Cohort): {n_events} ({event_rate:.1f}%) (from {n_processed} individuals)")
        print(f"Excluded {n_prevalent_excluded} prevalent cases for {disease_group}.")
    print(f"\nSummary of Results (Static 10-Year Risk from 1-Year Pi at Enrollment: {static_10yr_from_1yr}, Sex-Adjusted):")
    print("-" * 80)
    print(f"{'Disease Group':<20} {'AUC':<25} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 80)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f} ({res['ci_lower']:.3f}-{res['ci_upper']:.3f})" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<25} {res['n_events']:<10d} {rate_str}")
    print("-" * 80)
    return results
