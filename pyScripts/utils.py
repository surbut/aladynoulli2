from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import torch

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def plot_training_evolution(history_tuple):
    losses, gradient_history = history_tuple
    
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.yscale('log')
    plt.legend()
    
    # Plot lambda gradients
    plt.subplot(1, 3, 2)
    lambda_norms = [torch.norm(g).item() for g in gradient_history['lambda_grad']]
    plt.plot(lambda_norms, label='Lambda gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Lambda Gradient Evolution')
    plt.legend()
    
    # Plot phi gradients
    plt.subplot(1, 3, 3)
    phi_norms = [torch.norm(g).item() for g in gradient_history['phi_grad']]
    plt.plot(phi_norms, label='Phi gradients')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient norm')
    plt.title('Phi Gradient Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_disease_lambda_alignment(model):
    """
    Plot lambda values aligned with disease occurrences for selected patients
    """
    # Find patients with specific diseases and their diagnosis times
    disease_idx = 112  # MI
    patients_with_disease = []
    diagnosis_times = []
    
    for patient in range(model.Y.shape[0]):
        diag_time = torch.where(model.Y[patient, disease_idx])[0]
        if len(diag_time) > 0:
            patients_with_disease.append(patient)
            diagnosis_times.append(diag_time[0].item())
    
    # Sample a few patients
    n_samples = min(5, len(patients_with_disease))
    sample_indices = np.random.choice(len(patients_with_disease), n_samples, replace=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    time_points = np.arange(model.T)
    
    # Find signature that most strongly associates with this disease
    psi_disease = model.psi[:, disease_idx].detach()
    sig_idx = torch.argmax(psi_disease).item()
    
    # Plot for each sampled patient
    for idx in sample_indices:
        patient = patients_with_disease[idx]
        diag_time = diagnosis_times[idx]
        
        # Plot lambda (detached)
        lambda_values = torch.softmax(model.lambda_[patient].detach(), dim=0)[sig_idx]
        ax.plot(time_points, lambda_values.numpy(),
                alpha=0.5, label=f'Patient {patient}')
        
        # Mark diagnosis time
        ax.axvline(x=diag_time, linestyle=':', alpha=0.3)
    
    ax.set_title(f'Lambda Values for Signature {sig_idx} (Most Associated with MI)\nVertical Lines Show Diagnosis Times')
    ax.set_xlabel('Time')
    ax.set_ylabel('Lambda (proportion)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_theta_differences(model, diseases=None, signatures=None):
    """
    Plot theta distributions for diagnosed vs non-diagnosed patients
    
    Parameters:
    model: The trained model (can be enrollment-constrained or full-data)
    diseases: List of disease indices to plot, default [112, 67, 127, 10, 17, 114]
    signatures: List of signature indices for each disease, default [5, 7, 0, 17, 19, 5]
    """
    if diseases is None:
        diseases = [112, 67, 127, 10, 17, 114]
    if signatures is None:
        signatures = [5, 7, 0, 17, 19, 5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (d, sig) in enumerate(zip(diseases, signatures)):
        ax = axes[i]
        
        # Get diagnosis times
        diagnosis_mask = model.Y[:, d, :].bool()
        diagnosed = torch.where(diagnosis_mask)[0]
        
        # Get thetas
        pi, theta, phi_prob = model.forward()
        
        # Plot distributions
        diagnosed_theta = theta[diagnosis_mask, sig].detach().numpy()
        others_theta = theta[~diagnosis_mask, sig].detach().numpy()
        
        ax.hist(diagnosed_theta, alpha=0.5, label='At diagnosis', density=True)
        ax.hist(others_theta, alpha=0.5, label='Others', density=True)
        
        ax.set_title(f'Disease {d} (sig {sig})')
        ax.set_xlabel('Theta')
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.show()




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


def plot_signature_temporal_patterns(model, disease_names, n_top=10, selected_signatures=None):
    """Show temporal patterns of top diseases for each signature"""
    phi = model.phi.detach().numpy()
    prevalence_logit = model.logit_prev_t.detach().numpy()
    import os
    phi_centered = np.zeros_like(phi)
    for k in range(phi.shape[0]):
        for d in range(phi.shape[1]):
            phi_centered[k, d, :] = phi[k, d, :] - prevalence_logit[d, :]
    
    phi_avg = phi_centered.mean(axis=2)
    
    if selected_signatures is None:
        selected_signatures = range(phi_avg.shape[0])
    
    n_sigs = len(selected_signatures)
    fig, axes = plt.subplots(n_sigs, 1, figsize=(15, 5*n_sigs))
    if n_sigs == 1:
        axes = [axes]
    
    for i, k in enumerate(selected_signatures):
        scores = phi_avg[k, :]
        top_indices = np.argsort(scores)[-n_top:][::-1]
        
        ax = axes[i]
        for idx in top_indices:
            temporal_pattern = phi[k, idx, :]
            disease_name = disease_names[idx]
            ax.plot(temporal_pattern, label=disease_name)
        
        ax.set_title(f'Signature {k} - Top Disease Temporal Patterns')
        ax.set_xlabel('Time')
        ax.set_ylabel('Phi Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
   



def compare_with_pce_using_enrollment_time(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using the model's predicted trajectory
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
    
    # Calculate 10-year risks using the full trajectory
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 10 >= model.T:
            continue
            
        # Calculate cumulative risk from enrollment to 10 years later
        max_t = min(enroll_time + 10, model.T - 1)
        p_not_disease = 1.0
        for t in range(enroll_time, max_t+1):
            for d_idx in ascvd_indices:
                p_not_disease *= (1 - pi_calibrated[patient_idx, d_idx, t])
        
        risk = 1 - p_not_disease
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


def compare_with_pce_one_year(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 1-year predictions using single timepoint prediction
    """
    our_1yr_risks = []
    actual_1yr = []

    # Get predictions
    pi = model.forward()[0].detach().numpy()
    
    # Apply calibration as before (optional)
    pi_calibrated = pi  # Or apply calibration if desired
    
    # Calculate 1-year risks using only enrollment time prediction
    for patient_idx, row in enumerate(pce_df.itertuples()):
        enroll_time = int(row.age - 30)
        if enroll_time + 1 >= model.T:
            continue
            
        # Only use predictions at enrollment time for ASCVD indices
        pi_ascvd = pi_calibrated[patient_idx, ascvd_indices, enroll_time]
        
        # Calculate 1-year risk (combine across ASCVD diseases)
        yearly_risk = 1 - np.prod(1 - pi_ascvd)
        our_1yr_risks.append(yearly_risk)
        
        # Look at actual events over 1 year
        Y_ascvd = model.Y[patient_idx, ascvd_indices, enroll_time:enroll_time+1]
        actual = torch.any(torch.any(Y_ascvd, dim=0))
        actual_1yr.append(actual.item())
   
    # Convert to arrays
    our_1yr_risks = np.array(our_1yr_risks)
    actual_1yr = np.array(actual_1yr)
    
    # Calculate ROC AUC
    our_auc = roc_auc_score(actual_1yr, our_1yr_risks)
    
    print(f"\nROC AUC for 1-year prediction from enrollment:")
    print(f"Our model: {our_auc:.3f}")
    
    # No PCE comparison for 1-year risk (PCE is designed for 10-year risk)
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_1yr, our_1yr_risks, label=f'Our Model (AUC={our_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for 1-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return our_auc, our_1yr_risks, actual_1yr


import numpy as np
import torch
from sklearn.metrics import roc_curve, auc

import torch
from sklearn.metrics import roc_curve, auc

def evaluate_major_diseases(model, Y_100k, E_100k, disease_names, pce_df, Y_full,enrollment_df):  # Add Y_full parameter
    """
    Evaluate model performance on major diseases, using full Y for event rates
    """
    """
    Evaluate model performance on major diseases
    
    Parameters:
    - model: trained model
    - Y_100k: disease status matrix (PyTorch tensor)
    - E_100k: event times matrix (PyTorch tensor)
    - disease_names: list of disease names
    - pce_df: DataFrame with patient characteristics
    """
    # Define major diseases to evaluate
    major_diseases = {
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
              'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
    'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'],
    'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
    # Add to major_diseases dictionary:
    'All_Cancers': ['Colon cancer', 
                'Malignant neoplasm of rectum, rectosigmoid junction, and anus',
                'Cancer of bronchus; lung',
                'Breast cancer [female]',
                'Malignant neoplasm of female breast',
                'Cancer of prostate',
                'Malignant neoplasm of bladder',
                'Secondary malignant neoplasm',
                'Secondary malignancy of lymph nodes',
                'Secondary malignancy of respiratory organs',
                'Secondary malignant neoplasm of digestive systems',
                'Secondary malignant neoplasm of liver',
                'Secondary malignancy of bone'],
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
    'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
    'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
    'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
    'Hip_Fracture': ['Hip fracture'],  # if this exact term exists in disease_names
    'Osteoporosis': ['Osteoporosis NOS'],
    'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
    'Alzheimer': ['Alzheimer disease and other dementias'],
    'Esophageal_Cancer': ['Cancer of esophagus'],  # adjust if different in disease_names
    'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
    'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
    'Prostate_Cancer': ['Cancer of prostate'],
    'Lung_Cancer': ['Cancer of bronchus; lung'],
    'Bladder_Cancer': ['Malignant neoplasm of bladder'],
    'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 
                        'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
    'Depression': ['Major depressive disorder'],
    'Anxiety': ['Anxiety disorder'],
    'Bipolar_Disorder': ['Bipolar'],
    'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
    'Psoriasis': ['Psoriasis vulgaris'],
    'Ulcerative_Colitis': ['Ulcerative colitis'],
    'Crohns_Disease': ['Regional enteritis'],
    'Asthma': ['Asthma'],
    #'Allergic_Rhinitis': ['Allergic rhinitis'],
    # Additional common conditions
    'Parkinsons': ["Parkinson's disease"],
    'Multiple_Sclerosis': ['Multiple sclerosis'],
    #'Sleep_Apnea': ['Sleep apnea'],
    #'Glaucoma': ['Glaucoma', 'Primary open angle glaucoma'],
    #'Cataract': ['Cataract', 'Senile cataract'],
    'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
}


    
    # Get model predictions
    with torch.no_grad():
        pi, _, _ = model.forward()
    
    results = {}
    
    # For each disease group
    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group}...")
        
        # Find disease indices
        disease_indices = []
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            disease_indices.extend(indices)
        
        if not disease_indices:
            print(f"No matching diseases found for {disease_group}")
            continue
            
        # Get predictions at enrollment time
        N = len(pce_df)
        risks = torch.zeros(N, device=pi.device)
        outcomes = torch.zeros(N, device=pi.device)
        
        # Calculate risks and outcomes
        for i in range(N):
            age = pce_df.iloc[i]['age']
            t_enroll = int(age - 30)  # Convert age to time index
            
            if t_enroll >= pi.shape[2]:
                continue
                
            # Get prediction at enrollment
            pi_diseases = pi[i, disease_indices, t_enroll]
            yearly_risk = 1 - torch.prod(1 - pi_diseases)
            risks[i] = 1 - (1 - yearly_risk)**10  # 10-year risk
            
            # Check for actual events in next 10 years
            end_time = min(t_enroll + 10, Y_100k.shape[2])
            for d_idx in disease_indices:
                if torch.any(Y_100k[i, d_idx, t_enroll:end_time] > 0):
                    outcomes[i] = 1
                    break
        
        # Convert to numpy for sklearn metrics
        risks_np = risks.cpu().numpy()
        outcomes_np = outcomes.cpu().numpy()
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
        auc_score = auc(fpr, tpr)
        
        # NEW: Calculate event rate using full Y tensor
        full_outcomes = torch.zeros(Y_full.shape[0], device=Y_full.device)
        for i in range(Y_full.shape[0]):
            age_at_enrollment = enrollment_df.iloc[i]['age']
            t_enroll = int(age_at_enrollment - 30)  # Convert age to time index
            end_time = min(t_enroll + 10, Y_full.shape[2])
            for d_idx in disease_indices:
                if torch.any(Y_full[i, d_idx, t_enroll:end_time] > 0):
                    full_outcomes[i] = 1
                    break
        
        full_event_rate = (full_outcomes.mean() * 100).item()
        full_event_count = int(full_outcomes.sum().item())
        
        results[disease_group] = {
            'auc': auc_score,
            'n_events': full_event_count,  # Use full data count
            'event_rate': full_event_rate  # Use full data rate
        }
        

        
        print(f"AUC: {auc_score:.3f}")
        print(f"Events: {int(outcomes.sum().item())} ({outcomes.mean()*100:.1f}%)")
    
    # Print summary table
    print("\nSummary of Results:")
    print("-" * 50)
    print(f"{'Disease Group':<15} {'AUC':<8} {'Events':<8} {'Rate':<8}")
    print("-" * 50)
    for group, res in results.items():
        print(f"{group:<15} {res['auc']:.3f}   {res['n_events']:<8d} {res['event_rate']:.1f}%")
    
    return results



def evaluate_major_diseases_wsex(model, Y_100k, E_100k, disease_names, pce_df, Y_full, enrollment_df):
    """
    Evaluate model performance on major diseases, using full Y for event rates 
    and handling sex-specific diseases correctly. 
    FIX 2: Uses integer positional indices consistently after filtering.
    
    Parameters are the same as before. Assumes alignment between pce_df rows (0..N-1) and Y_100k/pi rows (0..N-1),
    and alignment between enrollment_df rows (0..M-1) and Y_full rows (0..M-1).
    """
    major_diseases = {
        'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
                  'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
        'Diabetes': ['Type 2 diabetes'],
        'Atrial_Fib': ['Atrial fibrillation and flutter'],
        'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
        'All_Cancers': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus', 'Cancer of bronchus; lung', 'Breast cancer [female]', 'Malignant neoplasm of female breast', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver', 'Secondary malignancy of bone'],
        'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
        'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
        'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
        'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
        'Osteoporosis': ['Osteoporosis NOS'],
        'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
        'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
        'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'], # Sex-specific
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

    # --- Input Validation ---
    if 'Sex' not in pce_df.columns: raise ValueError("'Sex' column not found in pce_df")
    if 'sex' not in enrollment_df.columns: raise ValueError("'Sex' column not found in enrollment_df")
    if 'age' not in pce_df.columns: raise ValueError("'age' column not found in pce_df")
    if 'age' not in enrollment_df.columns: raise ValueError("'age' column not found in enrollment_df")

    with torch.no_grad():
        pi, _, _ = model.forward()
        
    N_pi = pi.shape[0]
    N_pce = len(pce_df)
    N_y100k = Y_100k.shape[0]
    N_yfull = Y_full.shape[0]
    N_enroll = len(enrollment_df)

    # Ensure alignment for AUC calculation cohort
    if not (N_pi == N_pce == N_y100k):
        print(f"Warning: Size mismatch for AUC cohort. pi: {N_pi}, pce_df: {N_pce}, Y_100k: {N_y100k}. Using minimum size.")
        min_N_auc = min(N_pi, N_pce, N_y100k)
        pi = pi[:min_N_auc]
        pce_df = pce_df.iloc[:min_N_auc]
        Y_100k = Y_100k[:min_N_auc]
        N_auc_cohort = min_N_auc
    else:
        N_auc_cohort = N_pce

    # Ensure alignment for Rate calculation cohort
    if not (N_yfull == N_enroll):
        print(f"Warning: Size mismatch for Rate cohort. Y_full: {N_yfull}, enrollment_df: {N_enroll}. Using minimum size.")
        min_N_rate = min(N_yfull, N_enroll)
        Y_full = Y_full[:min_N_rate]
        enrollment_df = enrollment_df.iloc[:min_N_rate]
        N_rate_cohort = min_N_rate
    else:
        N_rate_cohort = N_enroll
        
    # Reset index after potential slicing to ensure 0-based sequential index for iloc
    pce_df = pce_df.reset_index(drop=True)
    enrollment_df = enrollment_df.reset_index(drop=True)

    results = {}
    
    # --- Main Loop ---
    for disease_group, disease_list in major_diseases.items():
        print(f"\nEvaluating {disease_group}...")
        
        # --- Get Disease Indices ---
        disease_indices = []
        # ... (same logic as before to find indices, check bounds against pi.shape[1]) ...
        unique_indices = set()
        for disease in disease_list:
            indices = [i for i, name in enumerate(disease_names) if disease.lower() in name.lower()]
            for idx in indices:
                 if idx not in unique_indices:
                      disease_indices.append(idx)
                      unique_indices.add(idx)
        
        max_model_disease_idx = pi.shape[1] - 1
        original_indices_count = len(disease_indices)
        disease_indices = [idx for idx in disease_indices if idx <= max_model_disease_idx]
        if not disease_indices:
             print(f"No valid matching disease indices found for {disease_group} within model output bounds.")
             results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0}
             continue

        # --- Sex Filtering ---
        target_sex = None
        if disease_group == 'Breast_Cancer': target_sex = 'Female'
        elif disease_group == 'Prostate_Cancer': target_sex = 'Male'

        # Get boolean masks based on the potentially trimmed and re-indexed DataFrames
        mask_pce = pd.Series(True, index=pce_df.index)
        mask_enroll = pd.Series(True, index=enrollment_df.index)
        
        if target_sex:
            mask_pce = (pce_df['Sex'] == target_sex)
            mask_enroll = (enrollment_df['sex'] == target_sex)
            # Find integer positions (iloc indices) where mask is True
            int_indices_pce = np.where(mask_pce)[0]
            int_indices_enroll = np.where(mask_enroll)[0]
            print(f"Filtering for {target_sex}: Found {len(int_indices_pce)} in AUC cohort, {len(int_indices_enroll)} in Rate cohort")
            if len(int_indices_pce) == 0 or len(int_indices_enroll) == 0:
                 print(f"Warning: No individuals found for target sex '{target_sex}'. Skipping.")
                 results[disease_group] = {'auc': np.nan, 'n_events': 0, 'event_rate': 0.0}
                 continue
        else:
            # Use all integer indices if not sex-specific
            int_indices_pce = np.arange(N_auc_cohort)
            int_indices_enroll = np.arange(N_rate_cohort)

        # --- Calculate AUC (using integer positions) ---
        if len(int_indices_pce) == 0:
            auc_score = np.nan; n_events_auc = 0; n_processed_auc = 0; outcomes_np = np.array([]) # Handle empty case
        else:
            # Slice tensors and DataFrame using the integer positions
            current_pi_auc = pi[int_indices_pce]
            current_Y_100k_auc = Y_100k[int_indices_pce]
            current_pce_df_auc = pce_df.iloc[int_indices_pce] # Use iloc with integer positions
            current_N_auc = len(int_indices_pce)

            risks_auc = torch.zeros(current_N_auc, device=pi.device)
            outcomes_auc = torch.zeros(current_N_auc, device=pi.device)
            processed_count_auc = 0

            # Iterate based on the length of the filtered integer indices
            for i in range(current_N_auc):
                # Access DataFrame row using iloc with relative index i
                age = current_pce_df_auc.iloc[i]['age'] 
                t_enroll = int(age - 30)

                if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue

                # Access tensors using relative index i
                pi_diseases = current_pi_auc[i, disease_indices, t_enroll]
                yearly_risk = 1 - torch.prod(1 - pi_diseases)
                # --- MODIFICATION: Evaluate 1-year risk ---
                risks_auc[i] = yearly_risk # Use 1-year risk directly

                # --- MODIFICATION: Check event in next 1 year ---
                end_time = min(t_enroll + 10, current_Y_100k_auc.shape[2]) # Look only 1 year ahead
                if end_time <= t_enroll: continue
                
                event_found_auc = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_100k_auc.shape[1]: continue
                    if torch.any(current_Y_100k_auc[i, d_idx, t_enroll:end_time] > 0): # Check t_enroll to end_time
                        outcomes_auc[i] = 1
                        event_found_auc = True
                        break
                processed_count_auc += 1 # Increment count if this iteration was valid

            # Calculate AUC based on processed data
            if processed_count_auc == 0:
                 auc_score = np.nan; outcomes_np = np.array([])
            else:
                 # Only use results from processed indices - NOTE: this slicing might be tricky if indices are sparse
                 # It's simpler to create new lists and convert at the end
                 valid_risks_list = []
                 valid_outcomes_list = []
                 temp_risks_cpu = risks_auc.cpu().numpy()
                 temp_outcomes_cpu = outcomes_auc.cpu().numpy()

                 # Re-iterate to gather valid pairs (safer than complex slicing)
                 processed_indices_auc_final = [] # Store indices relative to the loop (0 to current_N_auc-1)
                 for i in range(current_N_auc):
                     age = current_pce_df_auc.iloc[i]['age'] 
                     t_enroll = int(age - 30)
                     if t_enroll < 0 or t_enroll >= current_pi_auc.shape[2]: continue
                     end_time = min(t_enroll + 1, current_Y_100k_auc.shape[2])
                     if end_time <= t_enroll: continue
                     processed_indices_auc_final.append(i)

                 if not processed_indices_auc_final:
                      auc_score = np.nan; outcomes_np = np.array([])
                 else:
                      risks_np = temp_risks_cpu[processed_indices_auc_final]
                      outcomes_np = temp_outcomes_cpu[processed_indices_auc_final]

                      if len(np.unique(outcomes_np)) > 1:
                           fpr, tpr, _ = roc_curve(outcomes_np, risks_np)
                           auc_score = auc(fpr, tpr)
                      else:
                           auc_score = np.nan
                           print(f"Warning: Only one class present ({np.unique(outcomes_np)}) for AUC.")
            n_processed_auc = len(outcomes_np) # Number used for final AUC calc

        # --- Calculate Event Rate/Count (using integer positions) ---
        if len(int_indices_enroll) == 0:
            full_event_rate = 0.0; full_event_count = 0; num_processed_for_rate = 0 # Handle empty case
        else:
            # Slice tensors and DataFrame using the integer positions
            current_Y_full_rate = Y_full[int_indices_enroll]
            current_enrollment_df_rate = enrollment_df.iloc[int_indices_enroll] # Use iloc
            current_N_rate = len(int_indices_enroll)

            full_outcomes_rate = torch.zeros(current_N_rate, device=Y_full.device)
            processed_count_rate = 0

            # Iterate based on the length of the filtered integer indices
            for i in range(current_N_rate):
                # Access DataFrame row using iloc with relative index i
                age_at_enrollment = current_enrollment_df_rate.iloc[i]['age'] 
                t_enroll = int(age_at_enrollment - 30)

                if t_enroll < 0 or t_enroll >= current_Y_full_rate.shape[2]: continue
                
                # --- MODIFICATION: Check event in next 1 year ---
                end_time = min(t_enroll + 10, current_Y_full_rate.shape[2]) # Look only 1 year ahead
                if end_time <= t_enroll: continue

                event_found_rate = False
                for d_idx in disease_indices:
                    if d_idx >= current_Y_full_rate.shape[1]: continue
                    if torch.any(current_Y_full_rate[i, d_idx, t_enroll:end_time] > 0): # Check t_enroll to end_time
                        full_outcomes_rate[i] = 1
                        event_found_rate = True
                        break
                processed_count_rate += 1 # Increment count if this iteration was valid
            
            # Calculate rate/count based on processed data
            if processed_count_rate == 0:
                 full_event_rate = 0.0; full_event_count = 0
            else:
                 # Similar to AUC, safer to collect valid outcomes
                 valid_outcomes_rate_list = []
                 temp_outcomes_rate_cpu = full_outcomes_rate.cpu().numpy()
                 processed_indices_rate_final = []
                 for i in range(current_N_rate):
                     age_at_enrollment = current_enrollment_df_rate.iloc[i]['age'] 
                     t_enroll = int(age_at_enrollment - 30)
                     if t_enroll < 0 or t_enroll >= current_Y_full_rate.shape[2]: continue
                     end_time = min(t_enroll + 1, current_Y_full_rate.shape[2])
                     if end_time <= t_enroll: continue
                     processed_indices_rate_final.append(i)
                 
                 if not processed_indices_rate_final:
                      full_event_rate = 0.0; full_event_count = 0
                 else:
                      full_outcomes_valid = temp_outcomes_rate_cpu[processed_indices_rate_final]
                      full_event_count = int(np.sum(full_outcomes_valid))
                      # Rate is based on the number actually processed
                      full_event_rate = (full_event_count / processed_count_rate * 100) if processed_count_rate > 0 else 0.0 
            num_processed_for_rate = processed_count_rate

        # Store results
        results[disease_group] = {
            'auc': auc_score,
            'n_events': full_event_count,
            'event_rate': full_event_rate
        }
        
        print(f"AUC (1-Year): {auc_score if not np.isnan(auc_score) else 'N/A'} (calculated on {n_processed_auc} individuals)")
        print(f"Events (1-Year, Full Cohort, Filtered): {full_event_count} ({full_event_rate:.1f}%) (calculated on {num_processed_for_rate} individuals)")

    # Print summary table
    print("\nSummary of Results (Prospective 1-Year, Sex-Adjusted):")
    # ... (rest of printing code is the same) ...
    print("-" * 60)
    print(f"{'Disease Group':<20} {'AUC':<8} {'Events':<10} {'Rate (%)':<10}")
    print("-" * 60)
    for group, res in results.items():
        auc_str = f"{res['auc']:.3f}" if not np.isnan(res['auc']) else "N/A"
        rate_str = f"{res['event_rate']:.1f}" if res['event_rate'] is not None else "N/A"
        print(f"{group:<20} {auc_str:<8} {res['n_events']:<10d} {rate_str}")
    print("-" * 60)

    return results
# Usage:
#results = evaluate_major_diseases(model, Y_100k, E_100k, disease_names, pce_df)

def compare_with_pce_filtered(model, pce_df, ascvd_indices=[111,112,113,114,115,116]):
    """
    Compare 10-year predictions using single timepoint prediction, handling missing PCE values
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
   
    our_10yr_risks = np.array(our_10yr_risks)
    actual_10yr = np.array(actual_10yr)
    pce_risks = pce_df['pce_goff'].values[:len(our_10yr_risks)]

    # Get indices of non-missing PCE values
    non_missing_idx = ~np.isnan(pce_risks)

    # Filter all arrays to only include non-missing cases
    our_10yr_risks_filtered = our_10yr_risks[non_missing_idx]
    actual_10yr_filtered = actual_10yr[non_missing_idx]
    pce_risks_filtered = pce_risks[non_missing_idx]

    # Calculate ROC AUCs on filtered data
    our_auc = roc_auc_score(actual_10yr_filtered, our_10yr_risks_filtered)
    pce_auc = roc_auc_score(actual_10yr_filtered, pce_risks_filtered)

    # Print results with sample size info
    n_total = len(our_10yr_risks)
    n_complete = len(our_10yr_risks_filtered)
    print(f"\nROC AUC Comparison (10-year prediction from enrollment):")
    print(f"Sample size: {n_complete}/{n_total} ({n_complete/n_total*100:.1f}% complete cases)")
    print(f"Our model: {our_auc:.3f}")
    print(f"PCE: {pce_auc:.3f}")
    
    plt.figure(figsize=(8,6))
    plot_roc_curve(actual_10yr_filtered, our_10yr_risks_filtered, label=f'Our Model (AUC={our_auc:.3f})')
    plot_roc_curve(actual_10yr_filtered, pce_risks_filtered, label=f'PCE (AUC={pce_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 10-year ASCVD Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

    return our_10yr_risks_filtered, actual_10yr_filtered, pce_risks_filtered 



def compare_clusters_across_biobanks(mgb_checkpoint, aou_checkpoint, ukb_checkpoint, disease_names_all):
    """
    Compare cluster assignments across biobanks, handling different disease sets
    """
    import pandas as pd
    import numpy as np
    
    # Create initial dataframes for each biobank with their diseases
    mgb_df = pd.DataFrame({
        'Disease': disease_names_all[:len(mgb_checkpoint['clusters'])],
        'MGB_cluster': mgb_checkpoint['clusters']
    })
    
    aou_df = pd.DataFrame({
        'Disease': disease_names_all[:len(aou_checkpoint['clusters'])],
        'AoU_cluster': aou_checkpoint['clusters']
    })
    
    ukb_df = pd.DataFrame({
        'Disease': disease_names_all[:len(ukb_checkpoint['clusters'])],
        'UKB_cluster': ukb_checkpoint['clusters']
    })
    
    # Merge dataframes on Disease column
    df = mgb_df.merge(aou_df, on='Disease', how='outer')\
               .merge(ukb_df, on='Disease', how='outer')
    
    print("Number of diseases in each biobank:")
    print(f"MGB: {len(mgb_df)}")
    print(f"AoU: {len(aou_df)}")
    print(f"UKB: {len(ukb_df)}")
    print(f"Total unique diseases: {len(df)}")
    
    # Print cluster sizes for each biobank
    print("\nCluster sizes in each biobank:")
    for col in ['MGB_cluster', 'AoU_cluster', 'UKB_cluster']:
        if col in df.columns:
            print(f"\n{col.split('_')[0]}:")
            print(df[col].value_counts().sort_index())
    
    # Find common diseases across all biobanks
    common_diseases = df.dropna(subset=['MGB_cluster', 'AoU_cluster', 'UKB_cluster'])
    print(f"\nNumber of diseases common to all biobanks: {len(common_diseases)}")
    
    # Create heatmap for common diseases
    if len(common_diseases) > 0:
        plt.figure(figsize=(15, 10))
        
        n_diseases = len(common_diseases)
        disease_list = common_diseases['Disease'].tolist()
        
        # Create binary matrices for co-clustering
        mgb_cocluster = np.zeros((n_diseases, n_diseases))
        aou_cocluster = np.zeros((n_diseases, n_diseases))
        ukb_cocluster = np.zeros((n_diseases, n_diseases))
        
        for i in range(n_diseases):
            for j in range(n_diseases):
                mgb_cocluster[i,j] = common_diseases['MGB_cluster'].iloc[i] == common_diseases['MGB_cluster'].iloc[j]
                aou_cocluster[i,j] = common_diseases['AoU_cluster'].iloc[i] == common_diseases['AoU_cluster'].iloc[j]
                ukb_cocluster[i,j] = common_diseases['UKB_cluster'].iloc[i] == common_diseases['UKB_cluster'].iloc[j]
        
        # Average co-clustering across biobanks
        avg_cocluster = (mgb_cocluster + aou_cocluster + ukb_cocluster) / 3
        
        # Plot heatmap
        sns.heatmap(avg_cocluster, 
                    xticklabels=disease_list,
                    yticklabels=disease_list,
                    cmap='YlOrRd')
        plt.title('Disease Co-clustering Consistency Across Biobanks\n(Common Diseases Only)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Find most consistent disease pairs
        print("\nMost consistently co-clustered diseases (among common diseases):")
        consistent_pairs = []
        for i in range(n_diseases):
            for j in range(i+1, n_diseases):
                consistency = avg_cocluster[i,j]
                if consistency > 0.66:  # Co-clustered in at least 2 biobanks
                    consistent_pairs.append((disease_list[i], disease_list[j], consistency))
        
        consistent_pairs.sort(key=lambda x: x[2], reverse=True)
        for d1, d2, score in consistent_pairs[:10]:
            print(f"{d1} - {d2}: {score:.2f}")
    
    return df

# Function to look at specific disease clusters
def examine_disease_clusters(df, disease_of_interest):
    """
    Examine clusters containing a specific disease across biobanks
    """
    print(f"\nClusters containing {disease_of_interest}:")
    
    for biobank in ['MGB', 'AoU', 'UKB']:
        col = f'{biobank}_cluster'
        if col in df.columns:
            # Get the cluster number for the disease of interest
            disease_cluster = df[df['Disease'] == disease_of_interest][col].iloc[0]
            if pd.notna(disease_cluster):  # Check if disease exists in this biobank
                cohort_diseases = df[df[col] == disease_cluster]['Disease'].tolist()
                print(f"\n{biobank} cluster {disease_cluster}:")
                print(cohort_diseases)
            else:
                print(f"\n{biobank}: Disease not present")

# Use the functions:
