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
            age_at_enrollment = enrollment_df.iloc[i]['f.21022.0.0']
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

# Usage:

# results = evaluate_major_diseases(model, Y_100k, E_100k, disease_names, pce_df)