"""
Investigate: What drives Signature 5 elevation in Pathway 1 if precursors are low?

Hypothesis test:
- If sig5 ≈ precursor levels → it's just lower-but-present precursors
- If sig5 >> precursor levels → there's additional biology (subclinical, uncoded, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathway_discovery import load_full_data
import pickle as pkl

def investigate_sig5_vs_precursors(target_disease="myocardial infarction", output_dir="output_10yr"):
    """
    For each pathway, compare Signature 5 deviation to precursor disease prevalence.
    Specifically test whether Pathway 1's modest sig5 rise is consistent with its precursor levels.
    """
    print("="*80)
    print("INVESTIGATING SIGNATURE 5 vs PRECURSOR DISEASE LEVELS")
    print("="*80)
    
    # Load data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Load pathway results
    results_file = f'{output_dir}/complete_analysis_results.pkl'
    with open(results_file, 'rb') as f:
        results = pkl.load(f)
    
    pathway_data = results['pathway_data_dev']
    patients = pathway_data['patients']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    
    # Define precursor diseases that Signature 5 captures
    precursor_diseases = [
        'coronary atherosclerosis',
        'hypercholesterolemia', 
        'angina',
        'hypertension',
        'diabetes',
        'obesity'
    ]
    
    # Find precursor indices
    precursor_indices = {}
    for prec_name in precursor_diseases:
        for i, name in enumerate(disease_names):
            if prec_name.lower() in name.lower():
                precursor_indices[prec_name] = i
                break
    
    print(f"\nPrecursor disease indices: {precursor_indices}")
    
    # For each pathway, calculate:
    # 1. Average Signature 5 deviation (ages 60-70)
    # 2. Average precursor burden (sum of normalized precursor prevalences)
    # 3. General population Signature 5 at same time window
    # 4. General population precursor prevalence
    # 5. Ratio: sig5_deviation / precursor_deviation_from_pop
    
    pathway_ratios = {}
    population_reference = np.mean(thetas, axis=0)  # [K, T]
    
    for pathway_id in range(n_pathways):
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        n_patients = len(pathway_patients)
        
        print(f"\n{'='*60}")
        print(f"PATHWAY {pathway_id} (n={n_patients})")
        print('='*60)
        
        # Calculate Signature 5 trajectory
        pathway_patient_ids = [p['patient_id'] for p in pathway_patients]
        pathway_thetas = thetas[pathway_patient_ids, :, :]
        population_reference = np.mean(thetas, axis=0)
        
        # Signature 5 trajectory (last 10 years before MI)
        sig5_deviation = []
        for patient_info in pathway_patients:
            patient_id = patient_info['patient_id']
            age_at_mi = patient_info['age_at_disease']
            mi_time_idx = age_at_mi - 30
            
            if mi_time_idx >= 10:
                # Get 10 years before MI
                start_idx = mi_time_idx - 10
                patient_sig5 = pathway_thetas[pathway_patient_ids.index(patient_id), 5, start_idx:mi_time_idx]
                pop_sig5 = population_reference[5, start_idx:mi_time_idx]
                
                sig5_dev = np.mean(patient_sig5 - pop_sig5)
                sig5_deviation.append(sig5_dev)
        
        avg_sig5_deviation = np.mean(sig5_deviation)
        print(f"Average Signature 5 deviation: {avg_sig5_deviation:+.4f}")
        
        # Calculate precursor burden vs POPULATION
        # Get average MI age for this pathway
        avg_mi_ages = [p['age_at_disease'] for p in pathway_patients]
        avg_mi_age = np.mean(avg_mi_ages)
        avg_mi_time_idx = int(avg_mi_age - 30)
        avg_mi_time_idx = max(0, min(avg_mi_time_idx, 51))
        
        # Population precursor prevalence at this age
        pop_precursor_prevalence = {}
        for prec_name, prec_idx in precursor_indices.items():
            # Population prevalence of this precursor up to avg_mi_time_idx
            pop_window = Y[:, prec_idx, :avg_mi_time_idx]
            pop_count = (pop_window.sum(dim=1) > 0).sum().item()
            pop_prevalence = pop_count / Y.shape[0]
            pop_precursor_prevalence[prec_name] = pop_prevalence
        
        # Calculate pathway burden deviation from population
        precursor_prevalence = {}
        precursor_burden = 0
        for prec_name, prec_idx in precursor_indices.items():
            count = 0
            for patient_info in pathway_patients:
                patient_id = patient_info['patient_id']
                age_at_mi = patient_info['age_at_disease']
                cutoff_idx = age_at_mi - 30
                
                if cutoff_idx > 0:
                    if Y[patient_id, prec_idx, :cutoff_idx].sum() > 0:
                        count += 1
            
            prevalence = count / n_patients if n_patients > 0 else 0
            precursor_prevalence[prec_name] = prevalence
            precursor_burden += prevalence
        
        # Normalize by number of precursors
        pathway_burden_normalized = precursor_burden / len(precursor_indices)
        pop_burden_normalized = sum(pop_precursor_prevalence.values()) / len(precursor_indices)
        
        # Deviation from population burden
        precursor_deviation = pathway_burden_normalized - pop_burden_normalized
        
        # Population Signature 5 at same time window
        pop_sig5_at_age = population_reference[5, avg_mi_time_idx]
        
        print(f"\nPathway precursor burden (normalized): {pathway_burden_normalized:.4f}")
        print(f"Population precursor burden (normalized): {pop_burden_normalized:.4f}")
        print(f"Precursor deviation from population: {precursor_deviation:+.4f}")
        print(f"\nPopulation Signature 5 at age {avg_mi_age:.1f}: {pop_sig5_at_age:.4f}")
        print(f"\nPrecursor prevalences (pathway vs population):")
        for prec_name in precursor_indices.keys():
            pw_prev = precursor_prevalence[prec_name]
            pop_prev = pop_precursor_prevalence[prec_name]
            print(f"  {prec_name}: pathway {pw_prev*100:.1f}%, population {pop_prev*100:.1f}%")
        
        # Calculate Z-scores (relative to population)
        # Population values at this age
        pop_sig5_absolute = population_reference[5, avg_mi_time_idx]
        pop_precursor_absolute = pop_burden_normalized
        
        # Pathway values
        pathway_sig5_absolute = pop_sig5_absolute + avg_sig5_deviation
        pathway_precursor_absolute = pop_precursor_absolute + precursor_deviation
        
        # Z-scores (fold-change from population)
        sig5_zscore = (pathway_sig5_absolute - pop_sig5_absolute) / (pop_sig5_absolute + 1e-6)
        precursor_zscore = (pathway_precursor_absolute - pop_precursor_absolute) / (pop_precursor_absolute + 1e-6)
        
        # Ratio of z-scores (on same scale)
        ratio = sig5_zscore / (precursor_zscore + 1e-6)
        pathway_ratios[pathway_id] = {
            'sig5_dev': avg_sig5_deviation,
            'pathway_burden': pathway_burden_normalized,
            'pop_burden': pop_burden_normalized,
            'precursor_deviation': precursor_deviation,
            'ratio': ratio,
            'sig5_zscore': sig5_zscore,
            'precursor_zscore': precursor_zscore,
            'individual_prevalences': precursor_prevalence,
            'pop_prevalences': pop_precursor_prevalence
        }
        
        print(f"\nRelative deviation from population:")
        print(f"  Sig5: {sig5_zscore:.2%} above population")
        print(f"  Precursors: {precursor_zscore:.2%} above population")
        print(f"\nRatio (Sig5 z-score / Precursor z-score): {ratio:.4f}")
        print("  Interpretation:")
        print(f"  - If ratio ≈ 1: Sig5 and precursors deviate from population proportionally")
        print(f"  - If ratio >> 1: Sig5 deviates from population MORE than precursors (additional biology?)")
        print(f"  - If ratio << 1: Sig5 deviates from population LESS than precursors (unexpected)")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left: Scatter plot - Precursor z-score vs Sig5 z-score (on same scale!)
    ax1 = axes[0]
    pathway_ids = list(pathway_ratios.keys())
    sig5_zscores = [pathway_ratios[pw]['sig5_zscore'] for pw in pathway_ids]
    precursor_zscores = [pathway_ratios[pw]['precursor_zscore'] for pw in pathway_ids]
    
    scatter = ax1.scatter(precursor_zscores, sig5_zscores, s=200, c=pathway_ids, 
                         cmap='tab10', edgecolors='black', linewidth=2)
    
    # Add labels
    for pw_id, sig5_z, prec_z in zip(pathway_ids, sig5_zscores, precursor_zscores):
        ax1.annotate(f'Pathway {pw_id}', (prec_z, sig5_z), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Precursor Z-score (relative to population)', fontsize=12)
    ax1.set_ylabel('Signature 5 Z-score (relative to population)', fontsize=12)
    ax1.set_title('Signature 5 vs Precursors (Normalized Scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal reference line (y = x) - now both axes on same scale!
    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()
    lim_min = min(x_min, y_min)
    lim_max = max(x_max, y_max)
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 
             'r--', alpha=0.5, label='y = x (sig5 z ≈ precursor z)')
    ax1.legend()
    ax1.set_xlim(lim_min, lim_max)
    ax1.set_ylim(lim_min, lim_max)
    
    # Right: Bar plot of ratios
    ax2 = axes[1]
    ratios = [pathway_ratios[pw]['ratio'] for pw in pathway_ids]
    bars = ax2.bar(pathway_ids, ratios, color=plt.cm.tab10(range(len(pathway_ids))), 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for pw, bar in zip(pathway_ids, bars):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pathway_ratios[pw]["ratio"]:.2f}',
                ha='center', va='bottom' if bar.get_height() > 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Pathway', fontsize=12)
    ax2.set_ylabel('Ratio (sig5_dev / precursor_burden)', fontsize=12)
    ax2.set_title('Signature 5 "Excess" Beyond Precursor Levels', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                label='Expected if sig5 ≈ precursors')
    ax2.legend()
    
    plt.tight_layout()
    filename = f'{output_dir}/sig5_vs_precursors_investigation.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved investigation plot: {filename}")
    plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Pathway':<10} {'Sig5 Z':<12} {'Prec Z':<12} {'Ratio':<12} {'Interpretation'}")
    print('-' * 90)
    for pw in pathway_ids:
        data = pathway_ratios[pw]
        ratio = data['ratio']
        
        if ratio > 1.5:
            interp = "sig5 deviates >> precursors (additional biology?)"
        elif ratio > 1.0:
            interp = "sig5 deviates > precursors"
        elif ratio > 0.7:
            interp = "sig5 ≈ precursor deviation (expected)"
        else:
            interp = "sig5 deviates < precursors"
        
        print(f"{pw:<10} {data['sig5_zscore']:<12.2%} {data['precursor_zscore']:<12.2%} "
              f"{ratio:<12.2f} {interp}")
    
    return pathway_ratios

if __name__ == "__main__":
    results = investigate_sig5_vs_precursors()
