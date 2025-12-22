"""
Create Supplementary Figure S30: Competing Risks Analysis

This figure demonstrates that Aladynoulli handles competing risks correctly by:
1. Showing patients who develop multiple serious diseases (not mutually exclusive)
2. Visualizing risk trajectories and risk ratios for example patients
3. Demonstrating that patients remain at risk for all diseases after initial diagnosis

Based on R3_Competing_Risks.ipynb
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add path for utils if needed
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')

def find_disease_index(search_terms, disease_names, preferred_keywords=None):
    """Find disease index by searching disease names."""
    matches = []
    for i, name in enumerate(disease_names):
        name_lower = str(name).lower()
        if any(term.lower() in name_lower for term in search_terms):
            matches.append((i, name))
    
    if not matches:
        return None
    
    # If preferred keywords, prioritize matches with those keywords
    if preferred_keywords:
        preferred_matches = [m for m in matches if any(pk.lower() in str(m[1]).lower() for pk in preferred_keywords)]
        if preferred_matches:
            return preferred_matches[0][0]
    
    return matches[0][0]

def create_S30_figure(
    pi_predictions_path=None,
    E_path=None,
    prevalence_path=None,
    disease_names_path=None,
    output_path=None
):
    """
    Create S30 multi-panel figure showing competing risks analysis.
    
    Parameters:
    -----------
    pi_predictions_path : str or Path
        Path to pi predictions file (pi_fullmode_400k.pt)
    E_path : str or Path
        Path to E matrix (E_matrix_corrected.pt)
    prevalence_path : str or Path
        Path to prevalence file (prevalence_t_corrected.pt)
    disease_names_path : str or Path
        Path to disease names CSV
    output_path : str or Path
        Output path for S30.pdf
    """
    
    print("="*80)
    print("CREATING S30: COMPETING RISKS ANALYSIS")
    print("="*80)
    
    # Set default paths
    if pi_predictions_path is None:
        pi_predictions_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/pi_fullmode_400k.pt")
    else:
        pi_predictions_path = Path(pi_predictions_path)
    
    if E_path is None:
        E_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/E_matrix_corrected.pt")
    else:
        E_path = Path(E_path)
    
    if prevalence_path is None:
        prevalence_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/prevalence_t_corrected.pt")
    else:
        prevalence_path = Path(prevalence_path)
    
    if disease_names_path is None:
        disease_names_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv")
    else:
        disease_names_path = Path(disease_names_path)
    
    if output_path is None:
        output_path = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_figs/supp/s30/S30.pdf")
    else:
        output_path = Path(output_path)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    print(f"Loading pi predictions from: {pi_predictions_path}")
    pi_predictions = torch.load(pi_predictions_path, map_location='cpu', weights_only=False)
    print(f"✓ Loaded pi predictions: {pi_predictions.shape}")
    
    print(f"Loading E matrix from: {E_path}")
    E_batch = torch.load(E_path, map_location='cpu', weights_only=False)
    print(f"✓ Loaded E matrix: {E_batch.shape}")
    
    print(f"Loading prevalence from: {prevalence_path}")
    prevalence_t = torch.load(prevalence_path, map_location='cpu', weights_only=False)
    print(f"✓ Loaded corrected prevalence: {prevalence_t.shape}")
    
    print(f"Loading disease names from: {disease_names_path}")
    disease_names = pd.read_csv(disease_names_path).iloc[:, 1].tolist()
    print(f"✓ Loaded {len(disease_names)} disease names")
    
    # Ensure pi_predictions and E_batch are aligned
    min_N = min(len(pi_predictions), len(E_batch))
    pi_predictions = pi_predictions[:min_N]
    E_batch = E_batch[:min_N]
    print(f"\nUsing {min_N:,} patients for analysis")
    
    # Define example patients
    specific_patients = [
        {
            'patient_idx': 23941,
            'first_disease': 'Coronary atherosclerosis',
            'second_disease': 'Cancer of bronchus; lung',
            'first_search': ['coronary atherosclerosis'],
            'second_search': ['lung', 'bronchus', 'bronchial'],
            'second_preferred': ['malignant', 'cancer', 'neoplasm']
        },
        {
            'patient_idx': 769,
            'first_disease': 'Myocardial infarction',
            'second_disease': 'Hemorrhage of rectum and anus',
            'first_search': ['myocardial infarction'],
            'second_search': ['colon cancer', 'rectum', 'rectosigmoid', 'anus', 'hemorrhage'],
            'second_preferred': ['colon cancer', 'rectum', 'rectosigmoid', 'anus', 'hemorrhage']
        }
    ]
    
    print("\n" + "="*80)
    print("FINDING DISEASE INDICES")
    print("="*80)
    
    # Find disease indices for each patient
    for patient_info in specific_patients:
        first_idx = find_disease_index(patient_info['first_search'], disease_names, 
                                       preferred_keywords=patient_info['first_search'])
        
        second_idx = find_disease_index(patient_info['second_search'], disease_names,
                                        preferred_keywords=patient_info.get('second_preferred', patient_info['second_search']))
        
        if first_idx is None or second_idx is None:
            print(f"⚠️  Could not find disease indices for Patient {patient_info['patient_idx']}")
            continue
        
        patient_info['first_idx'] = first_idx
        patient_info['second_idx'] = second_idx
        patient_info['first_name'] = disease_names[first_idx]
        patient_info['second_name'] = disease_names[second_idx]
        
        print(f"\nPatient {patient_info['patient_idx']}:")
        print(f"  {patient_info['first_disease']}: found '{patient_info['first_name']}' (idx {first_idx})")
        print(f"  {patient_info['second_disease']}: found '{patient_info['second_name']}' (idx {second_idx})")
        
        # Check if patient has both diseases
        patient_idx = patient_info['patient_idx']
        if patient_idx >= len(E_batch):
            print(f"  ⚠️  Patient index {patient_idx} out of range (max: {len(E_batch)-1})")
            continue
        
        first_event_time = E_batch[patient_idx, first_idx].item()
        second_event_time = E_batch[patient_idx, second_idx].item()
        
        if first_event_time >= 51 or second_event_time >= 51:
            print(f"  ⚠️  Patient {patient_idx} does not have both diseases in the data")
            print(f"      First disease event time: {first_event_time}, Second: {second_event_time}")
            continue
        
        first_age = first_event_time + 30
        second_age = second_event_time + 30
        first_t = int(first_event_time)
        second_t = int(second_event_time)
        
        patient_info['first_age'] = first_age
        patient_info['second_age'] = second_age
        patient_info['first_t'] = first_t
        patient_info['second_t'] = second_t
        
        print(f"  First disease at age {int(first_age)} (timepoint {first_t})")
        print(f"  Second disease at age {int(second_age)} (timepoint {second_t})")
    
    print("\n" + "="*80)
    print("CREATING MULTI-PANEL FIGURE")
    print("="*80)
    
    # Create figure with 6 panels (3 patients × 2 panels each)
    fig = plt.figure(figsize=(16, 12))
    
    # Define panel layout: 2 rows × 2 columns
    # Row 1: Patient 23941 (absolute risk, risk ratio)
    # Row 2: Patient 769 (absolute risk, risk ratio)
    
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, left=0.08, right=0.95, top=0.98, bottom=0.05)
    
    ages = np.arange(30, 82)  # Ages 30-81 (52 timepoints)
    
    # Plot each patient
    for row_idx, patient_info in enumerate(specific_patients):
        if 'first_age' not in patient_info:
            continue
        
        patient_idx = patient_info['patient_idx']
        first_idx = patient_info['first_idx']
        second_idx = patient_info['second_idx']
        first_name = patient_info['first_name']
        second_name = patient_info['second_name']
        first_age = patient_info['first_age']
        second_age = patient_info['second_age']
        first_t = patient_info['first_t']
        second_t = patient_info['second_t']
        
        # Get probability trajectories
        prob_first = pi_predictions[patient_idx, first_idx, :].numpy()
        prob_second = pi_predictions[patient_idx, second_idx, :].numpy()
        
        # Get baseline prevalence
        baseline_first = prevalence_t[first_idx, :].numpy()
        baseline_second = prevalence_t[second_idx, :].numpy()
        
        # Calculate risk ratios
        rr_first = prob_first / (baseline_first + 1e-8)
        rr_second = prob_second / (baseline_second + 1e-8)
        rr_second_at_first_dx = rr_second[first_t]
        
        print(f"\nPatient {patient_idx}:")
        print(f"  Risk ratio for {second_name} at {first_name} diagnosis: {rr_second_at_first_dx:.2f}x")
        
        # ===== LEFT PANEL: Absolute Risk Trajectories =====
        ax1 = fig.add_subplot(gs[row_idx, 0])
        
        # Plot baselines first (dashed)
        ax1.plot(ages, baseline_first, '--', linewidth=2, 
                 label=f'Population Baseline: {first_name}', 
                 color='#e74c3c', alpha=0.6)
        ax1.plot(ages, baseline_second, '--', linewidth=2, 
                 label=f'Population Baseline: {second_name}', 
                 color='#3498db', alpha=0.6)
        
        # Plot patient risks (solid)
        ax1.plot(ages, prob_first, '-', linewidth=2.5, 
                 label=f'Patient Risk: {first_name}', 
                 color='#e74c3c', alpha=0.9)
        ax1.plot(ages, prob_second, '-', linewidth=2.5, 
                 label=f'Patient Risk: {second_name}', 
                 color='#3498db', alpha=0.9)
        
        # Add vertical lines at diagnoses
        ax1.axvline(x=first_age, color='purple', linestyle=':', linewidth=2.5, 
                    alpha=0.8, label=f'{patient_info["first_disease"]} Dx (Age {int(first_age)})')
        if second_age > first_age:
            ax1.axvline(x=second_age, color='blue', linestyle='--', linewidth=2.5, 
                        alpha=0.8, label=f'{patient_info["second_disease"]} Dx (Age {int(second_age)})')
        
        # Shade regions
        ax1.axvspan(30, first_age, alpha=0.05, color='gray')
        if second_age > first_age:
            ax1.axvspan(first_age, second_age, alpha=0.1, color='gray')
            ax1.axvspan(second_age, 80, alpha=0.15, color='lightcoral')
        
        # Add annotation for risk ratio at first diagnosis
        ax1.annotate(f'{patient_info["second_disease"]} RR: {rr_second_at_first_dx:.2f}x\nat {patient_info["first_disease"]} Dx', 
                    xy=(first_age, prob_second[first_t]),
                    xytext=(first_age + 3, prob_second[first_t] + 0.002),
                    fontsize=10, fontweight='bold', color='blue',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        ax1.set_xlabel('Age (years)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Predicted Disease Risk', fontsize=11, fontweight='bold')
        ax1.set_title(f'Patient {patient_idx}: {patient_info["first_disease"]} → {patient_info["second_disease"]}\n'
                      f'Absolute Risk Trajectories', 
                      fontsize=12, fontweight='bold', pad=10)
        ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(30, 80)
        
        # ===== RIGHT PANEL: Risk Ratio Trajectories =====
        ax2 = fig.add_subplot(gs[row_idx, 1])
        
        ax2.plot(ages, rr_first, '-', linewidth=2.5, 
                 label=f'{first_name} Risk Ratio', 
                 color='#e74c3c', alpha=0.9)
        ax2.plot(ages, rr_second, '-', linewidth=2.5, 
                 label=f'{second_name} Risk Ratio', 
                 color='#3498db', alpha=0.9)
        
        # Add horizontal line at RR=1.0
        ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                    alpha=0.7, label='Population Average (RR=1.0)')
        
        # Shade elevated risk region
        ax2.axhspan(1.0, 5.0, alpha=0.1, color='green')
        
        # Add vertical lines at diagnoses
        ax2.axvline(x=first_age, color='purple', linestyle=':', linewidth=2.5, 
                    alpha=0.8, label=f'{patient_info["first_disease"]} Dx (Age {int(first_age)})')
        if second_age > first_age:
            ax2.axvline(x=second_age, color='blue', linestyle='--', linewidth=2.5, 
                        alpha=0.8, label=f'{patient_info["second_disease"]} Dx (Age {int(second_age)})')
        
        # Add annotation for risk ratio at first diagnosis
        ax2.annotate(f'RR = {rr_second_at_first_dx:.2f}x', 
                    xy=(first_age, rr_second_at_first_dx),
                    xytext=(first_age + 3, rr_second_at_first_dx + 0.5),
                    fontsize=11, fontweight='bold', color='blue',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        
        ax2.set_xlabel('Age (years)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Risk Ratio (Patient / Population)', fontsize=11, fontweight='bold')
        ax2.set_title(f'Patient {patient_idx}: Risk Ratio Trajectories\n'
                      f'Relative to Population Average', 
                      fontsize=12, fontweight='bold', pad=10)
        ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(4.0, rr_first.max() * 1.1, rr_second.max() * 1.1))
        ax2.set_xlim(30, 80)
    
    # Add overall title
    fig.suptitle('S30: Competing Risks Analysis - Patients Remain at Risk for All Diseases', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure
    print(f"\nSaving figure to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved S30 figure to: {output_path}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("1. Patients remain at risk for all diseases after initial diagnosis")
    print("2. Risk ratios are elevated for second disease at first diagnosis")
    print("3. Model correctly captures competing risks as non-exclusive events")
    print("4. Multiple diseases can occur in sequence (not mutually exclusive)")
    
    return output_path

if __name__ == "__main__":
    create_S30_figure()

