#!/usr/bin/env python3
"""
Interpret the Counterintuitive Signature 3 Pattern

The plot shows:
- RA → MI: Sig 3 increases from ~0.02 to ~0.05 (moderate)
- RA (no MI): Sig 3 increases from ~0.025 to ~0.18 (very high!)

This is counterintuitive - why would Sig 3 be HIGHER in patients who DON'T develop MI?

Possible explanations:
1. Sig 3 is a "protective" signature (higher = lower MI risk)
2. Sig 3 represents RA disease activity itself (not MI risk)
3. Sig 3 is a compensatory mechanism that prevents MI
4. The groups are not well-matched (confounding)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data
from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model


def investigate_signature_3_pattern(
    transition_disease_name='Rheumatoid arthritis',
    target_disease_name='myocardial infarction',
    signature_idx=3,
    years_before=10
):
    """
    Investigate what Signature 3 represents and why it's higher in non-progressors
    """
    
    print("="*80)
    print("INVESTIGATING SIGNATURE 3 PATTERN")
    print("="*80)
    print(f"Transition: {transition_disease_name} → {target_disease_name}")
    print(f"Signature: {signature_idx}")
    print("="*80)
    
    # Load UKB data
    print("\n1. Loading UKB data...")
    Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
    
    if isinstance(Y_ukb, np.ndarray):
        Y_ukb = torch.from_numpy(Y_ukb)
    if isinstance(thetas_ukb, torch.Tensor):
        thetas_ukb = thetas_ukb.numpy()
    
    # Find disease indices
    transition_idx = None
    target_idx = None
    
    for i, name in enumerate(disease_names_ukb):
        if transition_disease_name.lower() in name.lower():
            transition_idx = i
            print(f"   Found transition disease: {name} (index {i})")
        if target_disease_name.lower() in name.lower():
            target_idx = i
            print(f"   Found target disease: {name} (index {i})")
    
    if transition_idx is None or target_idx is None:
        print("❌ Could not find diseases")
        return None
    
    # ============================================================================
    # 1. WHAT IS SIGNATURE 3? (Check phi associations)
    # ============================================================================
    print("\n" + "="*80)
    print("2. WHAT DOES SIGNATURE 3 REPRESENT?")
    print("="*80)
    
    # Try to load phi (signature-disease associations)
    # This would tell us which diseases Signature 3 is associated with
    print("\n   Note: To fully understand Signature 3, we need phi (signature-disease associations)")
    print("   This would show which diseases Signature 3 is most associated with")
    
    # ============================================================================
    # 2. COMPARE SIGNATURE 3 LEVELS IN DIFFERENT GROUPS
    # ============================================================================
    print("\n" + "="*80)
    print("3. COMPARING SIGNATURE 3 LEVELS ACROSS GROUPS")
    print("="*80)
    
    # Group 1: RA patients who develop MI
    ra_mi_patients = []
    for patient_id in range(Y_ukb.shape[0]):
        ra_occurrences = torch.where(Y_ukb[patient_id, transition_idx, :] > 0)[0]
        mi_occurrences = torch.where(Y_ukb[patient_id, target_idx, :] > 0)[0]
        
        if len(ra_occurrences) > 0 and len(mi_occurrences) > 0:
            ra_age = ra_occurrences[0].item() + 30
            mi_age = mi_occurrences[0].item() + 30
            if mi_age > ra_age:  # MI after RA
                # Get Sig 3 level at RA diagnosis
                ra_idx = ra_occurrences[0].item()
                sig3_at_ra = thetas_ukb[patient_id, signature_idx, ra_idx]
                ra_mi_patients.append({
                    'patient_id': patient_id,
                    'ra_age': ra_age,
                    'mi_age': mi_age,
                    'sig3_at_ra': sig3_at_ra,
                    'sig3_avg_pre_mi': np.mean(thetas_ukb[patient_id, signature_idx, max(0, mi_age-30-years_before):mi_age-30])
                })
    
    # Group 2: RA patients who DON'T develop MI
    ra_no_mi_patients = []
    for patient_id in range(Y_ukb.shape[0]):
        ra_occurrences = torch.where(Y_ukb[patient_id, transition_idx, :] > 0)[0]
        mi_occurrences = torch.where(Y_ukb[patient_id, target_idx, :] > 0)[0]
        
        if len(ra_occurrences) > 0 and len(mi_occurrences) == 0:
            ra_age = ra_occurrences[0].item() + 30
            # Get Sig 3 level at RA diagnosis
            ra_idx = ra_occurrences[0].item()
            sig3_at_ra = thetas_ukb[patient_id, signature_idx, ra_idx]
            # Get average Sig 3 in years after RA (equivalent window)
            end_idx = min(ra_idx + years_before, Y_ukb.shape[2])
            sig3_avg_post_ra = np.mean(thetas_ukb[patient_id, signature_idx, ra_idx:end_idx])
            ra_no_mi_patients.append({
                'patient_id': patient_id,
                'ra_age': ra_age,
                'sig3_at_ra': sig3_at_ra,
                'sig3_avg_post_ra': sig3_avg_post_ra
            })
    
    print(f"\n   Group 1 (RA → MI): {len(ra_mi_patients)} patients")
    print(f"   Group 2 (RA, no MI): {len(ra_no_mi_patients)} patients")
    
    if len(ra_mi_patients) > 0 and len(ra_no_mi_patients) > 0:
        # Compare Signature 3 levels
        sig3_at_ra_mi = [p['sig3_at_ra'] for p in ra_mi_patients]
        sig3_at_ra_no_mi = [p['sig3_at_ra'] for p in ra_no_mi_patients]
        
        sig3_pre_mi = [p['sig3_avg_pre_mi'] for p in ra_mi_patients]
        sig3_post_ra = [p['sig3_avg_post_ra'] for p in ra_no_mi_patients]
        
        print(f"\n   Signature 3 at RA diagnosis:")
        print(f"      RA → MI:     {np.mean(sig3_at_ra_mi):.4f} ± {np.std(sig3_at_ra_mi):.4f}")
        print(f"      RA (no MI):  {np.mean(sig3_at_ra_no_mi):.4f} ± {np.std(sig3_at_ra_no_mi):.4f}")
        print(f"      Difference:  {np.mean(sig3_at_ra_no_mi) - np.mean(sig3_at_ra_mi):.4f}")
        
        print(f"\n   Signature 3 in years after RA (or before MI):")
        print(f"      RA → MI:     {np.mean(sig3_pre_mi):.4f} ± {np.std(sig3_pre_mi):.4f}")
        print(f"      RA (no MI):  {np.mean(sig3_post_ra):.4f} ± {np.std(sig3_post_ra):.4f}")
        print(f"      Difference:  {np.mean(sig3_post_ra) - np.mean(sig3_pre_mi):.4f}")
        
        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(sig3_post_ra, sig3_pre_mi)
        print(f"\n   T-test: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"   ✅ Significant difference (p < 0.05)")
        else:
            print(f"   ⚠️  Not statistically significant")
    
    # ============================================================================
    # 3. CHECK FOR CONFOUNDING: Age, other diseases, etc.
    # ============================================================================
    print("\n" + "="*80)
    print("4. CHECKING FOR CONFOUNDING FACTORS")
    print("="*80)
    
    if len(ra_mi_patients) > 0 and len(ra_no_mi_patients) > 0:
        # Compare ages
        ages_ra_mi = [p['ra_age'] for p in ra_mi_patients]
        ages_ra_no_mi = [p['ra_age'] for p in ra_no_mi_patients]
        
        print(f"\n   Age at RA diagnosis:")
        print(f"      RA → MI:     {np.mean(ages_ra_mi):.1f} ± {np.std(ages_ra_mi):.1f} years")
        print(f"      RA (no MI):  {np.mean(ages_ra_no_mi):.1f} ± {np.std(ages_ra_no_mi):.1f} years")
        print(f"      Difference:  {np.mean(ages_ra_no_mi) - np.mean(ages_ra_mi):.1f} years")
        
        # Check other diseases
        print(f"\n   Checking other disease prevalences...")
        # This would require checking Y matrix for other diseases
    
    # ============================================================================
    # 4. INTERPRETATION
    # ============================================================================
    print("\n" + "="*80)
    print("5. POSSIBLE INTERPRETATIONS")
    print("="*80)
    
    print("""
    The counterintuitive pattern (Sig 3 higher in non-progressors) could mean:
    
    1. **PROTECTIVE SIGNATURE**: Signature 3 might represent a protective mechanism
       - Higher Sig 3 → Lower MI risk
       - Could be anti-inflammatory, cardioprotective, or compensatory
       
    2. **RA DISEASE ACTIVITY**: Signature 3 might represent RA disease activity itself
       - Higher Sig 3 = More active RA
       - But active RA doesn't necessarily lead to MI
       - Could be that well-controlled RA (low Sig 3) allows other pathways to MI
       
    3. **COMPENSATORY MECHANISM**: Signature 3 might be upregulated to compensate
       - In response to RA, body upregulates Sig 3
       - This prevents MI in some patients
       - In others, compensation fails → MI occurs
       
    4. **CONFOUNDING**: Groups might differ in other ways
       - Age differences
       - Treatment differences
       - Other comorbidities
       - Follow-up time differences
       
    5. **SIGNATURE INTERPRETATION**: Need to check what diseases Sig 3 is associated with
       - Load phi matrix to see Sig 3's disease associations
       - This would clarify what biological process Sig 3 represents
    """)
    
    return {
        'ra_mi_patients': ra_mi_patients,
        'ra_no_mi_patients': ra_no_mi_patients
    }


if __name__ == "__main__":
    results = investigate_signature_3_pattern()

