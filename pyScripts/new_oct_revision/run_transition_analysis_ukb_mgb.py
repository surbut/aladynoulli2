#!/usr/bin/env python3
"""
Run Transition Analysis on Both UKB and MGB

This script runs the same transition analysis (e.g., Rheumatoid arthritis → MI)
on both UKB and MGB data to demonstrate reproducibility of transition patterns.

Example:
    python run_transition_analysis_ukb_mgb.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from plot_transition_deviations import plot_bc_to_mi_progression
from pathway_discovery import load_full_data
from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
from find_disease_in_cohort import find_disease_flexible


def run_transition_analysis_both_cohorts(
    transition_disease_name='Rheumatoid arthritis',
    target_disease_name='myocardial infarction',
    years_before=10,
    age_tolerance=5,
    min_followup=5,
    mgb_model_path='/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt',
    output_dir='transition_analysis_ukb_mgb'
):
    """
    Run transition analysis on both UKB and MGB cohorts
    
    Parameters:
    -----------
    transition_disease_name : str
        Precursor disease (e.g., 'Rheumatoid arthritis', 'Breast cancer')
    target_disease_name : str
        Target disease (e.g., 'myocardial infarction')
    years_before : int
        Years before target disease to analyze
    age_tolerance : int
        Age matching tolerance
    min_followup : int
        Minimum follow-up time
    mgb_model_path : str
        Path to MGB model file
    output_dir : str
        Output directory for results
    """
    
    print("="*80)
    print("TRANSITION ANALYSIS: UKB vs MGB")
    print("="*80)
    print(f"Precursor: {transition_disease_name}")
    print(f"Target: {target_disease_name}")
    print(f"Years before: {years_before}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ============================================================================
    # 1. LOAD UKB DATA
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING UKB DATA")
    print("="*80)
    
    Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
    
    # Convert to torch if needed
    if isinstance(Y_ukb, np.ndarray):
        Y_ukb = torch.from_numpy(Y_ukb)
    if isinstance(thetas_ukb, torch.Tensor):
        thetas_ukb = thetas_ukb.numpy()
    
    print(f"✅ UKB data loaded:")
    print(f"   Y shape: {Y_ukb.shape}")
    print(f"   Thetas shape: {thetas_ukb.shape}")
    print(f"   Diseases: {len(disease_names_ukb)}")
    
    # Check if diseases exist in UKB
    print(f"\n🔍 Checking if diseases exist in UKB...")
    transition_matches_ukb = find_disease_flexible(transition_disease_name, disease_names_ukb, verbose=True)
    target_matches_ukb = find_disease_flexible(target_disease_name, disease_names_ukb, verbose=True)
    
    if not transition_matches_ukb:
        print(f"❌ ERROR: Could not find '{transition_disease_name}' in UKB!")
        return None
    if not target_matches_ukb:
        print(f"❌ ERROR: Could not find '{target_disease_name}' in UKB!")
        return None
    
    # Use best match
    transition_name_ukb = transition_matches_ukb[0][1]
    target_name_ukb = target_matches_ukb[0][1]
    print(f"\n✅ Using UKB disease names:")
    print(f"   Transition: '{transition_name_ukb}'")
    print(f"   Target: '{target_name_ukb}'")
    
    # ============================================================================
    # 2. RUN UKB TRANSITION ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: RUNNING UKB TRANSITION ANALYSIS")
    print("="*80)
    
    ukb_results = plot_bc_to_mi_progression(
        transition_disease_name=transition_name_ukb,  # Use matched name
        target_disease_name=target_name_ukb,  # Use matched name
        Y=Y_ukb,
        thetas=thetas_ukb,
        disease_names=disease_names_ukb,
        years_before=years_before,
        age_tolerance=age_tolerance,
        min_followup=min_followup,
        save_plots=True
    )
    
    if ukb_results is None:
        print("❌ UKB transition analysis failed")
        return None
    
    # Save UKB plots with cohort prefix
    if 'figure' in ukb_results:
        ukb_fig = ukb_results['figure']
        ukb_save_path = os.path.join(output_dir, f'ukb_{transition_disease_name.lower().replace(" ", "_")}_to_{target_disease_name.lower().replace(" ", "_")}_progression.png')
        ukb_fig.savefig(ukb_save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved UKB plot: {ukb_save_path}")
    
    # ============================================================================
    # 3. LOAD MGB DATA
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 3: LOADING MGB DATA")
    print("="*80)
    
    Y_mgb, thetas_mgb, disease_names_mgb, _ = load_mgb_data_from_model(mgb_model_path)
    
    # Convert to torch if needed
    if isinstance(Y_mgb, np.ndarray):
        Y_mgb = torch.from_numpy(Y_mgb)
    if isinstance(thetas_mgb, torch.Tensor):
        thetas_mgb = thetas_mgb.numpy()
    
    print(f"✅ MGB data loaded:")
    print(f"   Y shape: {Y_mgb.shape}")
    print(f"   Thetas shape: {thetas_mgb.shape}")
    print(f"   Diseases: {len(disease_names_mgb)}")
    
    # Check if diseases exist in MGB
    print(f"\n🔍 Checking if diseases exist in MGB...")
    transition_matches_mgb = find_disease_flexible(transition_disease_name, disease_names_mgb, verbose=True)
    target_matches_mgb = find_disease_flexible(target_disease_name, disease_names_mgb, verbose=True)
    
    if not transition_matches_mgb:
        print(f"❌ ERROR: Could not find '{transition_disease_name}' in MGB!")
        return None
    if not target_matches_mgb:
        print(f"❌ ERROR: Could not find '{target_disease_name}' in MGB!")
        return None
    
    # Use best match
    transition_name_mgb = transition_matches_mgb[0][1]
    target_name_mgb = target_matches_mgb[0][1]
    print(f"\n✅ Using MGB disease names:")
    print(f"   Transition: '{transition_name_mgb}'")
    print(f"   Target: '{target_name_mgb}'")
    
    # ============================================================================
    # 4. RUN MGB TRANSITION ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 4: RUNNING MGB TRANSITION ANALYSIS")
    print("="*80)
    
    mgb_results = plot_bc_to_mi_progression(
        transition_disease_name=transition_name_mgb,  # Use matched name
        target_disease_name=target_name_mgb,  # Use matched name
        Y=Y_mgb,
        thetas=thetas_mgb,
        disease_names=disease_names_mgb,
        years_before=years_before,
        age_tolerance=age_tolerance,
        min_followup=min_followup,
        save_plots=True
    )
    
    if mgb_results is None:
        print("❌ MGB transition analysis failed")
        return None
    
    # Save MGB plots with cohort prefix
    if 'figure' in mgb_results:
        mgb_fig = mgb_results['figure']
        mgb_save_path = os.path.join(output_dir, f'mgb_{transition_disease_name.lower().replace(" ", "_")}_to_{target_disease_name.lower().replace(" ", "_")}_progression.png')
        mgb_fig.savefig(mgb_save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved MGB plot: {mgb_save_path}")
    
    # ============================================================================
    # 5. COMPARE RESULTS
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 5: COMPARING UKB vs MGB RESULTS")
    print("="*80)
    
    print("\n📊 SUMMARY STATISTICS:")
    print("-" * 80)
    
    # Compare sample sizes
    # The function returns 'matched_pairs', 'progressor_deviations', and 'non_progressor_deviations'
    ukb_n_progressors = len(ukb_results.get('progressor_deviations', []))
    ukb_n_non_progressors = len(ukb_results.get('non_progressor_deviations', []))
    ukb_n_matched = len(ukb_results.get('matched_pairs', []))
    
    mgb_n_progressors = len(mgb_results.get('progressor_deviations', []))
    mgb_n_non_progressors = len(mgb_results.get('non_progressor_deviations', []))
    mgb_n_matched = len(mgb_results.get('matched_pairs', []))
    
    print(f"\nSample Sizes:")
    print(f"  UKB: {ukb_n_progressors} progressors, {ukb_n_non_progressors} non-progressors")
    print(f"       {ukb_n_matched} matched pairs")
    print(f"  MGB: {mgb_n_progressors} progressors, {mgb_n_non_progressors} non-progressors")
    print(f"       {mgb_n_matched} matched pairs")
    
    # Compare signature patterns (if available)
    # Check for both possible return formats (handle numpy arrays correctly)
    ukb_prog_avg = ukb_results.get('prog_avg')
    if ukb_prog_avg is None:
        ukb_prog_avg = ukb_results.get('with_target_avg')
    
    ukb_np_avg = ukb_results.get('np_avg')
    if ukb_np_avg is None:
        ukb_np_avg = ukb_results.get('without_target_avg')
    
    mgb_prog_avg = mgb_results.get('prog_avg')
    if mgb_prog_avg is None:
        mgb_prog_avg = mgb_results.get('with_target_avg')
    
    mgb_np_avg = mgb_results.get('np_avg')
    if mgb_np_avg is None:
        mgb_np_avg = mgb_results.get('without_target_avg')
    
    if ukb_prog_avg is not None and mgb_prog_avg is not None:
        ukb_avg = ukb_prog_avg  # Shape: (K, T)
        mgb_avg = mgb_prog_avg
        
        print(f"\nSignature Trajectories:")
        print(f"  UKB shape: {ukb_avg.shape}")
        print(f"  MGB shape: {mgb_avg.shape}")
        
        # Compare signature deviations at key timepoints
        if ukb_avg.shape[0] == mgb_avg.shape[0]:  # Same number of signatures
            K = ukb_avg.shape[0]
            print(f"\n  Comparing {K} signatures...")
            
            # Calculate correlation for each signature
            signature_correlations = []
            for k in range(K):
                if ukb_avg.shape[1] == mgb_avg.shape[1]:  # Same timepoints
                    corr = np.corrcoef(ukb_avg[k, :], mgb_avg[k, :])[0, 1]
                    signature_correlations.append(corr)
            
            if len(signature_correlations) > 0:
                avg_corr = np.mean(signature_correlations)
                print(f"  Average signature trajectory correlation: {avg_corr:.3f}")
                
                # Find most similar signatures
                top_corrs = np.argsort(signature_correlations)[::-1][:5]
                print(f"\n  Top 5 most similar signatures (by trajectory correlation):")
                for i, k in enumerate(top_corrs):
                    print(f"    Signature {k}: correlation = {signature_correlations[k]:.3f}")
                
                # Show actual values for Signature 3 (the counterintuitive one)
                sig3_idx = 3
                if sig3_idx < K:
                    print(f"\n  📊 SIGNATURE 3 DETAILED COMPARISON:")
                    print(f"  {'-'*80}")
                    
                    # Get non-progressor averages too
                    ukb_np_avg = ukb_results.get('np_avg')
                    if ukb_np_avg is None:
                        ukb_np_avg = ukb_results.get('without_target_avg')
                    
                    mgb_np_avg = mgb_results.get('np_avg')
                    if mgb_np_avg is None:
                        mgb_np_avg = mgb_results.get('without_target_avg')
                    
                    if ukb_np_avg is not None and mgb_np_avg is not None:
                        # Progressors
                        ukb_prog_sig3 = ukb_prog_avg[sig3_idx, :]
                        mgb_prog_sig3 = mgb_prog_avg[sig3_idx, :]
                        
                        # Non-progressors
                        ukb_np_sig3 = ukb_np_avg[sig3_idx, :]
                        mgb_np_sig3 = mgb_np_avg[sig3_idx, :]
                        
                        # Mean values
                        ukb_prog_mean = np.mean(ukb_prog_sig3)
                        ukb_np_mean = np.mean(ukb_np_sig3)
                        mgb_prog_mean = np.mean(mgb_prog_sig3)
                        mgb_np_mean = np.mean(mgb_np_sig3)
                        
                        print(f"\n  UKB:")
                        print(f"    Progressors (RA → MI):     mean = {ukb_prog_mean:+.4f}")
                        print(f"    Non-progressors (RA only): mean = {ukb_np_mean:+.4f}")
                        print(f"    Difference (NP - Prog):     {ukb_np_mean - ukb_prog_mean:+.4f}")
                        ukb_pattern = "NP > Prog" if ukb_np_mean > ukb_prog_mean else "Prog > NP"
                        print(f"    Pattern: {ukb_pattern}")
                        
                        print(f"\n  MGB:")
                        print(f"    Progressors (RA → MI):     mean = {mgb_prog_mean:+.4f}")
                        print(f"    Non-progressors (RA only): mean = {mgb_np_mean:+.4f}")
                        print(f"    Difference (NP - Prog):     {mgb_np_mean - mgb_prog_mean:+.4f}")
                        mgb_pattern = "NP > Prog" if mgb_np_mean > mgb_prog_mean else "Prog > NP"
                        print(f"    Pattern: {mgb_pattern}")
                        
                        # Check if pattern is consistent
                        pattern_match = (ukb_np_mean > ukb_prog_mean) == (mgb_np_mean > mgb_prog_mean)
                        print(f"\n  Pattern Consistency: {'✅ SAME' if pattern_match else '❌ DIFFERENT'}")
                        
                        # Absolute level comparison
                        print(f"\n  Absolute Levels (Progressors):")
                        print(f"    UKB: {ukb_prog_mean:+.4f}")
                        print(f"    MGB: {mgb_prog_mean:+.4f}")
                        print(f"    Ratio (MGB/UKB): {mgb_prog_mean/ukb_prog_mean:.2f}" if ukb_prog_mean != 0 else "    Ratio: N/A")
                        
                        print(f"\n  Absolute Levels (Non-progressors):")
                        print(f"    UKB: {ukb_np_mean:+.4f}")
                        print(f"    MGB: {mgb_np_mean:+.4f}")
                        print(f"    Ratio (MGB/UKB): {mgb_np_mean/ukb_np_mean:.2f}" if ukb_np_mean != 0 else "    Ratio: N/A")
    
    # ============================================================================
    # 6. CREATE SIDE-BY-SIDE COMPARISON PLOT
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 6: CREATING SIDE-BY-SIDE COMPARISON")
    print("="*80)
    
    if ukb_prog_avg is not None and mgb_prog_avg is not None:
        create_comparison_plot(
            ukb_results, mgb_results,
            transition_disease_name, target_disease_name,
            output_dir
        )
    
    print("\n" + "="*80)
    print("✅ TRANSITION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    
    return {
        'ukb_results': ukb_results,
        'mgb_results': mgb_results,
        'output_dir': output_dir
    }


def create_comparison_plot(ukb_results, mgb_results, 
                          transition_disease_name, target_disease_name,
                          output_dir):
    """
    Create side-by-side comparison plot of UKB vs MGB transition patterns
    """
    
    # Handle both return formats (handle numpy arrays correctly)
    ukb_with = ukb_results.get('prog_avg')
    if ukb_with is None:
        ukb_with = ukb_results.get('with_target_avg')  # (K, T)
    
    ukb_without = ukb_results.get('np_avg')
    if ukb_without is None:
        ukb_without = ukb_results.get('without_target_avg')
    
    mgb_with = mgb_results.get('prog_avg')
    if mgb_with is None:
        mgb_with = mgb_results.get('with_target_avg')
    
    mgb_without = mgb_results.get('np_avg')
    if mgb_without is None:
        mgb_without = mgb_results.get('without_target_avg')
    
    # Get sample sizes
    ukb_n_prog = len(ukb_results.get('progressor_deviations', []))
    ukb_n_np = len(ukb_results.get('non_progressor_deviations', []))
    mgb_n_prog = len(mgb_results.get('progressor_deviations', []))
    mgb_n_np = len(mgb_results.get('non_progressor_deviations', []))
    
    if ukb_with is None or mgb_with is None:
        print("⚠️  Cannot create comparison plot (missing data)")
        return
    
    K = ukb_with.shape[0]  # Number of signatures
    T = ukb_with.shape[1]  # Number of timepoints
    
    # Create time axis (years before target disease)
    time_axis = np.arange(-T, 0) + 1  # e.g., [-9, -8, ..., -1, 0]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{transition_disease_name} → {target_disease_name}: UKB vs MGB Comparison', 
                 fontsize=16, fontweight='bold')
    
    # UKB: With target (progressors)
    ax1 = axes[0, 0]
    for k in range(K):
        ax1.plot(time_axis, ukb_with[k, :], label=f'Sig {k}', alpha=0.7, linewidth=2)
    ax1.set_title(f'UKB: {transition_disease_name} → {target_disease_name}\n(n={ukb_n_prog})', 
                  fontweight='bold')
    ax1.set_xlabel('Years before MI')
    ax1.set_ylabel('Signature Deviation')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # UKB: Without target (non-progressors)
    ax2 = axes[0, 1]
    for k in range(K):
        ax2.plot(time_axis, ukb_without[k, :], label=f'Sig {k}', alpha=0.7, linewidth=2)
    ax2.set_title(f'UKB: {transition_disease_name} (no {target_disease_name})\n(n={ukb_n_np})', 
                  fontweight='bold')
    ax2.set_xlabel('Years before equivalent age')
    ax2.set_ylabel('Signature Deviation')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # MGB: With target (progressors)
    ax3 = axes[1, 0]
    if mgb_with.shape[0] == K and mgb_with.shape[1] == T:
        for k in range(K):
            ax3.plot(time_axis, mgb_with[k, :], label=f'Sig {k}', alpha=0.7, linewidth=2)
    ax3.set_title(f'MGB: {transition_disease_name} → {target_disease_name}\n(n={mgb_n_prog})', 
                  fontweight='bold')
    ax3.set_xlabel('Years before MI')
    ax3.set_ylabel('Signature Deviation')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # MGB: Without target (non-progressors)
    ax4 = axes[1, 1]
    if mgb_without.shape[0] == K and mgb_without.shape[1] == T:
        for k in range(K):
            ax4.plot(time_axis, mgb_without[k, :], label=f'Sig {k}', alpha=0.7, linewidth=2)
    ax4.set_title(f'MGB: {transition_disease_name} (no {target_disease_name})\n(n={mgb_n_np})', 
                  fontweight='bold')
    ax4.set_xlabel('Years before equivalent age')
    ax4.set_ylabel('Signature Deviation')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, 
                            f'ukb_mgb_comparison_{transition_disease_name.lower().replace(" ", "_")}_to_{target_disease_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Run transition analysis for Rheumatoid arthritis → MI
    results = run_transition_analysis_both_cohorts(
        transition_disease_name='Rheumatoid arthritis',
        target_disease_name='myocardial infarction',
        years_before=10,
        age_tolerance=5,
        min_followup=5
    )
    
    print("\n✅ Analysis complete!")

