"""
Analyze What Signature 5 Picks Up in Each MI Pathway

This script addresses the key question: What does noulli (the model) pick up on 
in pathways where patients don't have high precursor disease prevalence?

Signature 5 = Ischemic Cardiovascular (coronary atherosclerosis, hypercholesterolemia, angina)

Key Insight: Even without diagnosed precursor diseases, signature patterns can detect 
subclinical cardiovascular risk that hasn't yet crossed the clinical threshold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from pathway_discovery import load_full_data
from pathway_interrogation import interrogate_disease_pathways
import pickle
import json

def analyze_signature5_by_pathway(target_disease="myocardial infarction", 
                                   output_dir="output_10yr"):
    """
    Analyze what signature 5 detects in each pathway, especially pathways with low precursor prevalence
    """
    print("="*80)
    print("ANALYZING SIGNATURE 5 IN EACH MI PATHWAY")
    print("="*80)
    
    # Load data
    Y, thetas, disease_names, processed_ids = load_full_data()
    
    # Load pathway results
    results_file = f'{output_dir}/complete_analysis_results.pkl'
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    pathway_data = results['pathway_data_dev']
    patients = pathway_data['patients']
    target_disease_idx = pathway_data['target_disease_idx']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    
    print(f"\nAnalyzing {n_pathways} pathways...")
    
    # Define key precursor diseases
    precursor_diseases = [
        'coronary atherosclerosis',
        'hypercholesterolemia',
        'angina',
        'hypertension',
        'diabetes',
        'obesity'
    ]
    
    # Find precursor disease indices
    precursor_indices = {}
    for prec_name in precursor_diseases:
        for i, name in enumerate(disease_names):
            if prec_name.lower() in name.lower():
                precursor_indices[prec_name] = i
                break
    
    print(f"Found precursor disease indices:")
    for prec_name, idx in precursor_indices.items():
        print(f"  {prec_name}: index {idx}")
    
    # Analyze each pathway
    pathway_analysis = {}
    
    for pathway_id in range(n_pathways):
        print(f"\n{'='*60}")
        print(f"PATHWAY {pathway_id}")
        print('='*60)
        
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        n_pathway_patients = len(pathway_patients)
        
        print(f"Number of patients: {n_pathway_patients}")
        
        # Calculate precursor disease prevalence
        precursor_prevalence = {}
        for prec_name, prec_idx in precursor_indices.items():
            count = 0
            for patient_info in pathway_patients:
                patient_id = patient_info['patient_id']
                age_at_mi = patient_info['age_at_disease']
                cutoff_idx = age_at_mi - 30
                
                if cutoff_idx > 0:
                    if Y[patient_id, prec_idx, :cutoff_idx].sum() > 0:
                        count += 1
            
            prevalence = count / n_pathway_patients if n_pathway_patients > 0 else 0
            precursor_prevalence[prec_name] = {
                'count': count,
                'prevalence': prevalence
            }
        
        print(f"\nPrecursor Disease Prevalence (BEFORE MI):")
        for prec_name, stats in precursor_prevalence.items():
            print(f"  {prec_name}: {stats['count']} ({stats['prevalence']*100:.1f}%)")

        # Add reference population prevalence at comparable age (up to average MI age for this pathway)
        # Compute average MI age in this pathway and corresponding time index
        if n_pathway_patients > 0:
            avg_mi_age_pathway = np.mean([p['age_at_disease'] for p in pathway_patients])
        else:
            avg_mi_age_pathway = 65.0
        avg_mi_time_idx = int(max(1, min(Y.shape[2], avg_mi_age_pathway - 30)))

        # For each precursor, compute reference prevalence in the full population up to this age
        # (any diagnosis prior to avg MI age, not restricted to MI patients)
        for prec_name, prec_idx in precursor_indices.items():
            try:
                # Y is torch tensor: [N, D, T]
                # Slice all patients, this disease, up to avg_mi_time_idx
                window = Y[:, prec_idx, :avg_mi_time_idx]
                # Patients with any diagnosis prior to that age
                ref_count = (window.sum(dim=1) > 0).sum().item()
                ref_prev = ref_count / Y.shape[0]
            except Exception:
                # Fallback (in case Y is numpy)
                window = np.array(Y[:, prec_idx, :avg_mi_time_idx])
                ref_count = (window.sum(axis=1) > 0).sum()
                ref_prev = ref_count / Y.shape[0]

            # Store alongside pathway prevalence
            pathway_prev = precursor_prevalence[prec_name]['prevalence']
            precursor_prevalence[prec_name]['ref_prevalence_at_age'] = ref_prev
            precursor_prevalence[prec_name]['delta_prevalence'] = pathway_prev - ref_prev

        # Print reference and delta for clarity
        print(f"\nReference population prevalence up to average MI age (\u2248 {avg_mi_age_pathway:.1f}y):")
        for prec_name, stats in precursor_prevalence.items():
            ref_pct = stats.get('ref_prevalence_at_age', 0) * 100
            delta_pct = stats.get('delta_prevalence', 0) * 100
            print(f"  {prec_name}: ref {ref_pct:.1f}%, pathway {stats['prevalence']*100:.1f}% (\u0394 {delta_pct:+.1f} pp)")
        
        # Calculate signature 5 trajectory
        pathway_patient_ids = [p['patient_id'] for p in pathway_patients]
        pathway_thetas = thetas[pathway_patient_ids, :, :]
        population_reference = np.mean(thetas, axis=0)
        
        # Calculate mean signature 5 loading for this pathway at each age
        sig5_values = []
        for t in range(thetas.shape[2]):
            pathway_mean_sig5 = np.mean(pathway_thetas[:, 5, t])
            population_mean_sig5 = population_reference[5, t]
            deviation = pathway_mean_sig5 - population_mean_sig5
            sig5_values.append({
                'age': 30 + t,
                'pathway_mean': pathway_mean_sig5,
                'population_mean': population_mean_sig5,
                'deviation': deviation
            })
        
        # Calculate timepoint closest to MI (age 70 for average)
        # Focus on last 10 years before MI
        sig5_values_last10 = [v for v in sig5_values if 60 <= v['age'] <= 70]
        avg_deviation = np.mean([v['deviation'] for v in sig5_values_last10])
        max_deviation = max([abs(v['deviation']) for v in sig5_values_last10])
        
        print(f"\nSignature 5 Deviations (ages 60-70):")
        print(f"  Average deviation: {avg_deviation:+.4f}")
        print(f"  Max deviation: {max_deviation:.4f}")
        
        # Store results
        pathway_analysis[pathway_id] = {
            'n_patients': n_pathway_patients,
            'precursor_prevalence': precursor_prevalence,
            'sig5_deviation': avg_deviation,
            'sig5_max_deviation': max_deviation,
            'sig5_trajectory': sig5_values
        }
        
        # Calculate total precursor burden
        total_precursor_pct = sum([p['prevalence'] for p in precursor_prevalence.values()])
        avg_precursor_pct = total_precursor_pct / len(precursor_prevalence)
        pathway_analysis[pathway_id]['avg_precursor_prevalence'] = avg_precursor_pct
        
        print(f"\nSummary:")
        print(f"  Average precursor prevalence: {avg_precursor_pct*100:.1f}%")
        print(f"  Signature 5 deviation: {avg_deviation:+.4f}")
        
        # Interpret the finding
        if avg_precursor_pct < 0.15:  # Low precursor prevalence
            if abs(avg_deviation) > 0.02:  # But signature 5 is still elevated
                print(f"\n  ⚠️  KEY FINDING: Low precursor prevalence ({avg_precursor_pct*100:.1f}%)")
                print(f"     BUT signature 5 still elevated ({avg_deviation:+.4f})")
                print(f"     INTERPRETATION: Subclinical cardiovascular risk detected!")
            else:
                print(f"\n  ✓ Low precursor prevalence ({avg_precursor_pct*100:.1f}%)")
                print(f"     and low signature 5 deviation ({avg_deviation:+.4f})")
                print(f"     INTERPRETATION: Minimal cardiovascular history")
        else:  # High precursor prevalence
            if avg_deviation > 0:
                print(f"\n  ✓ High precursor prevalence ({avg_precursor_pct*100:.1f}%)")
                print(f"     and elevated signature 5 ({avg_deviation:+.4f})")
                print(f"     INTERPRETATION: Classic atherosclerosis pathway")
    
    # Create visualization
    create_signature5_analysis_plot(pathway_analysis, output_dir, target_disease)
    
    return pathway_analysis

def create_signature5_analysis_plot(pathway_analysis, output_dir, target_disease):
    """
    Create a comprehensive plot showing signature 5 vs precursor prevalence
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Top row: Scatter plot of precursor prevalence vs signature 5 deviation
    ax1 = fig.add_subplot(gs[0, :])
    
    pathway_ids = []
    avg_precursor = []
    sig5_deviation = []
    patient_counts = []
    
    for pathway_id, data in pathway_analysis.items():
        pathway_ids.append(f"Pathway {pathway_id}")
        avg_precursor.append(data['avg_precursor_prevalence'])
        sig5_deviation.append(data['sig5_deviation'])
        patient_counts.append(data['n_patients'])
    
    scatter = ax1.scatter(avg_precursor, sig5_deviation, s=[c/50 for c in patient_counts], 
                         c=range(len(pathway_ids)), cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, (pct, dev, pid) in enumerate(zip(avg_precursor, sig5_deviation, pathway_ids)):
        ax1.annotate(f'{pid}', xy=(pct, dev), xytext=(5, 5), 
                    textcoords='offset points', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Average Precursor Disease Prevalence (%)', fontsize=12)
    ax1.set_ylabel('Signature 5 Deviation (Δ Proportion)', fontsize=12)
    ax1.set_title(f'Signature 5 vs Precursor Prevalence by Pathway\n(Point size = number of patients)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Middle row: Signature 5 trajectories over time
    ax2 = fig.add_subplot(gs[1, :])
    
    colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(pathway_analysis)))
    
    for pathway_id, data in pathway_analysis.items():
        trajectory = data['sig5_trajectory']
        ages = [v['age'] for v in trajectory]
        deviations = [v['deviation'] for v in trajectory]
        
        ax2.plot(ages, deviations, linewidth=2.5, marker='o', markersize=4,
                label=f"Pathway {pathway_id} (n={data['n_patients']})", 
                color=colors[pathway_id], alpha=0.8)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Age', fontsize=12)
    ax2.set_ylabel('Signature 5 Deviation from Population Mean', fontsize=12)
    ax2.set_title('Signature 5 Trajectories by Pathway', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Bottom row: Detailed precursor breakdown for each pathway
    for i, (pathway_id, data) in enumerate(pathway_analysis.items()):
        if i < 2:  # Only plot first 2 pathways in details
            ax = fig.add_subplot(gs[2, i])
            
            precursor_names = list(data['precursor_prevalence'].keys())
            prevalences = [data['precursor_prevalence'][name]['prevalence']*100 for name in precursor_names]
            
            bars = ax.barh(range(len(precursor_names)), prevalences, color=colors[pathway_id], alpha=0.6)
            
            ax.set_yticks(range(len(precursor_names)))
            ax.set_yticklabels(precursor_names, fontsize=9)
            ax.set_xlabel('Prevalence (%)', fontsize=10)
            ax.set_title(f'Pathway {pathway_id} Precursor Prevalence\n(n={data["n_patients"]} patients)', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for j, (prev, bar) in enumerate(zip(prevalences, bars)):
                ax.text(prev + 1, j, f'{prev:.1f}%', va='center', fontsize=9)
    
    plt.suptitle(f'Signature 5 Analysis: What Does the Model Detect?\n{target_disease.title()}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save plot
    filename = f'{output_dir}/signature5_analysis_{target_disease.replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved signature 5 analysis plot: {filename}")
    plt.close()

def create_prevalence_barplots(pathway_analysis, output_dir, target_disease):
    """
    Create per-pathway barplots comparing precursor prevalence vs reference-at-age and the deviation.
    One subplot per pathway (assumes up to 4 pathways).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_pathways = len(pathway_analysis)
    n_cols = 2 if n_pathways > 2 else n_pathways
    n_rows = int(np.ceil(n_pathways / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows), squeeze=False)
    fig.suptitle(f'Precursor Prevalence vs Reference (Age-Matched): {target_disease.title()}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Ensure consistent precursor ordering from the first pathway
    first_key = sorted(pathway_analysis.keys())[0]
    precursor_names = list(pathway_analysis[first_key]['precursor_prevalence'].keys())

    # Compute a common y-axis limit across all subplots (use both pathway and reference %)
    global_max_pct = 0.0
    first_key = sorted(pathway_analysis.keys())[0]
    precursor_names = list(pathway_analysis[first_key]['precursor_prevalence'].keys())
    for _pid in sorted(pathway_analysis.keys()):
        _pdata = pathway_analysis[_pid]
        _prev_dict = _pdata['precursor_prevalence']
        _pathway_vals = [ _prev_dict[name]['prevalence']*100 for name in precursor_names ]
        _ref_vals = [ _prev_dict[name].get('ref_prevalence_at_age', 0)*100 for name in precursor_names ]
        global_max_pct = max(global_max_pct, max(_pathway_vals + _ref_vals) if (_pathway_vals or _ref_vals) else 0.0)
    # Add a small headroom
    common_ylim = max(5.0, np.ceil(global_max_pct * 1.1))

    for idx, pathway_id in enumerate(sorted(pathway_analysis.keys())):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        pdata = pathway_analysis[pathway_id]
        prev_dict = pdata['precursor_prevalence']

        # Collect values in consistent order
        pathway_vals = [prev_dict[name]['prevalence']*100 for name in precursor_names]
        ref_vals = [prev_dict[name].get('ref_prevalence_at_age', 0)*100 for name in precursor_names]
        #delta_vals = [prev - ref for prev, ref in zip(pathway_vals, ref_vals)]

        x = np.arange(len(precursor_names))
        width = 0.35

        # Side-by-side bars for pathway vs reference
        bars1 = ax.bar(x - width/2, pathway_vals, width, label='Pathway (pre-MI %)', color='#4C78A8', alpha=0.85)
        bars2 = ax.bar(x + width/2, ref_vals, width, label='Reference (age-matched %)', color='#F58518', alpha=0.85)

        # Twin axis for deviation bars (pp)
        #ax2 = ax.twinx()
        #bars3 = ax2.bar(x, delta_vals, width*0.6, label='Δ (pp)', color='#54A24B', alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(precursor_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Prevalence (%)')
        #ax2.set_ylabel('Δ Prevalence (pp)', color='#54A24B')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, common_ylim)
        ax.set_title(f'Pathway {pathway_id} (n={pdata["n_patients"]})', fontsize=12, fontweight='bold')

        # Legends
        lines, labels = [], []
        for b in [bars1, bars2]:
            lines.append(b)
        labels.extend(['Pathway (pre-MI %)', 'Reference (age-matched %)'])
        l1 = ax.legend(lines, labels, loc='upper left', fontsize=9)
        ax.add_artist(l1)
        #ax2.legend([bars3], ['Δ (pp)'], loc='upper right', fontsize=9)

        # Annotate delta bars
        #for i, b in enumerate(bars3):
        #    val = delta_vals[i]
        #    ax2.text(b.get_x() + b.get_width()/2, b.get_height(), f'{val:+.1f}',
        #             ha='center', va='bottom' if val >= 0 else 'top', fontsize=8, color='#2F4B7C')

    # Hide any empty subplots
    total_axes = n_rows * n_cols
    for j in range(len(pathway_analysis), total_axes):
        r = j // n_cols
        c = j % n_cols
        axes[r][c].axis('off')

    plt.tight_layout()
    out_path = f"{output_dir}/prevalence_comparison_by_pathway.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved prevalence comparison barplots: {out_path}")
    plt.close()

if __name__ == "__main__":
    analysis = analyze_signature5_by_pathway()

