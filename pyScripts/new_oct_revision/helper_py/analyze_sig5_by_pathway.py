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
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.proportion import proportion_confint
import pickle
import json
import os

def analyze_signature5_by_pathway(target_disease="myocardial infarction", 
                                   output_dir="output_10yr",
                                   fh_carrier_path=None):
    """
    Analyze what signature 5 detects in each pathway, especially pathways with low precursor prevalence
    
    Parameters:
    -----------
    target_disease : str
        Target disease name
    output_dir : str
        Output directory for results
    fh_carrier_path : str, optional
        Path to FH carrier file. If None, tries default path.
        Expected format: tab-separated file with 'IID' or 'eid' column
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
    
    # Load FH carrier data
    fh_carrier_set = set()
    if fh_carrier_path is None:
        fh_carrier_path = '/Users/sarahurbut/Downloads/out/ukb_exome_450k_fh.carrier.txt'
    
    try:
        if os.path.exists(fh_carrier_path):
            print(f"\nLoading FH carrier data from: {fh_carrier_path}")
            fh = pd.read_csv(fh_carrier_path, sep='\t', dtype={'IID': int}, low_memory=False)
            
            # Robustly pick eid column
            if 'IID' not in fh.columns:
                cand = [c for c in fh.columns if c.lower() in ('eid','id','ukb_eid','participant_id')]
                if len(cand) > 0:
                    fh = fh.rename(columns={cand[0]: 'IID'})
                else:
                    raise ValueError(f"No EID column found in {fh_carrier_path}")
            
            fh_carriers = fh[['IID']].drop_duplicates()
            fh_carrier_set = set(fh_carriers['IID'].astype(int).tolist())
            print(f"  ✅ Loaded {len(fh_carrier_set):,} FH carriers")
        else:
            print(f"\n⚠️  FH carrier file not found at: {fh_carrier_path}")
            print(f"  Skipping FH carrier analysis")
    except Exception as e:
        print(f"\n⚠️  Error loading FH carrier data: {e}")
        print(f"  Skipping FH carrier analysis")
    
    # Create eid to carrier mapping
    eids = processed_ids.astype(int)
    is_carrier = np.isin(eids, list(fh_carrier_set)) if len(fh_carrier_set) > 0 else None
    
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
        
        # Calculate FH carrier prevalence for this pathway
        fh_carrier_stats = None
        if is_carrier is not None and n_pathway_patients > 0:
            pathway_patient_ids_list = [p['patient_id'] for p in pathway_patients]
            pathway_carrier_status = is_carrier[pathway_patient_ids_list]
            
            n_carriers = pathway_carrier_status.sum()
            n_noncarriers = len(pathway_carrier_status) - n_carriers
            carrier_prop = n_carriers / n_pathway_patients if n_pathway_patients > 0 else 0
            
            # Compare to overall population prevalence
            overall_carrier_prop = is_carrier.mean()
            carrier_enrichment = carrier_prop / overall_carrier_prop if overall_carrier_prop > 0 else np.nan
            
            fh_carrier_stats = {
                'n_carriers': int(n_carriers),
                'n_noncarriers': int(n_noncarriers),
                'carrier_proportion': carrier_prop,
                'overall_carrier_proportion': overall_carrier_prop,
                'carrier_enrichment': carrier_enrichment
            }
            
            print(f"\nFH Carrier Prevalence:")
            print(f"  Carriers: {n_carriers}/{n_pathway_patients} ({carrier_prop*100:.2f}%)")
            print(f"  Non-carriers: {n_noncarriers}/{n_pathway_patients} ({1-carrier_prop:.2f}%)")
            print(f"  Overall population carrier rate: {overall_carrier_prop*100:.2f}%")
            print(f"  Enrichment ratio: {carrier_enrichment:.2f}x")
        
        # Store results
        pathway_analysis[pathway_id] = {
            'n_patients': n_pathway_patients,
            'precursor_prevalence': precursor_prevalence,
            'sig5_deviation': avg_deviation,
            'sig5_max_deviation': max_deviation,
            'sig5_trajectory': sig5_values,
            'fh_carrier_stats': fh_carrier_stats
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
    
    # Print FH carrier summary across all pathways
    if any(data.get('fh_carrier_stats') is not None for data in pathway_analysis.values()):
        print("\n" + "="*80)
        print("FH CARRIER PREVALENCE SUMMARY ACROSS PATHWAYS")
        print("="*80)
        print(f"{'Pathway':<10} {'N':<8} {'Carriers':<12} {'Carrier %':<12} {'Enrichment':<12} {'95% CI':<20}")
        print("-" * 100)
        
        pathway_ids_with_fh = sorted([pid for pid in pathway_analysis.keys() 
                                     if pathway_analysis[pid].get('fh_carrier_stats') is not None])
        
        for pathway_id in pathway_ids_with_fh:
            data = pathway_analysis[pathway_id]
            fh_stats = data.get('fh_carrier_stats')
            if fh_stats is not None:
                # Calculate confidence interval
                ci = proportion_confint(fh_stats['n_carriers'], data['n_patients'], 
                                      method='wilson', alpha=0.05)
                ci_str = f"[{ci[0]*100:.2f}, {ci[1]*100:.2f}]"
                
                print(f"Pathway {pathway_id:<5} {data['n_patients']:<8} "
                      f"{fh_stats['n_carriers']}/{fh_stats['n_noncarriers']:<10} "
                      f"{fh_stats['carrier_proportion']*100:>6.2f}%     "
                      f"{fh_stats['carrier_enrichment']:>6.2f}x     {ci_str}")
        
        # Statistical comparison between pathways
        if len(pathway_ids_with_fh) >= 2:
            print("\n" + "="*80)
            print("STATISTICAL COMPARISONS BETWEEN PATHWAYS")
            print("="*80)
            
            # Compare each pathway to the others (Fisher's exact test)
            for i, pid1 in enumerate(pathway_ids_with_fh):
                for pid2 in pathway_ids_with_fh[i+1:]:
                    stats1 = pathway_analysis[pid1]['fh_carrier_stats']
                    stats2 = pathway_analysis[pid2]['fh_carrier_stats']
                    
                    # 2x2 contingency table
                    table = [[stats1['n_carriers'], stats1['n_noncarriers']],
                            [stats2['n_carriers'], stats2['n_noncarriers']]]
                    
                    try:
                        # Fisher's exact test
                        odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
                        
                        print(f"\nPathway {pid1} vs Pathway {pid2}:")
                        print(f"  Carrier rates: {stats1['carrier_proportion']*100:.2f}% vs {stats2['carrier_proportion']*100:.2f}%")
                        print(f"  Odds Ratio: {odds_ratio:.3f}")
                        print(f"  Fisher's exact p-value: {p_value:.4e}")
                        
                        if p_value < 0.05:
                            print(f"  ✓ Significant difference (p < 0.05)")
                        else:
                            print(f"  Not significant (p >= 0.05)")
                    except Exception as e:
                        print(f"\nPathway {pid1} vs Pathway {pid2}: Could not compute test ({e})")
    
    # Create visualization
    create_signature5_analysis_plot(pathway_analysis, output_dir, target_disease)
    
    # Create FH carrier visualization if available
    if any(data.get('fh_carrier_stats') is not None for data in pathway_analysis.values()):
        create_fh_carrier_plot(pathway_analysis, output_dir, target_disease)
        # Create comprehensive comparison plot
        create_comprehensive_pathway_comparison_plot(pathway_analysis, output_dir, target_disease)
    
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

def create_fh_carrier_plot(pathway_analysis, output_dir, target_disease):
    """
    Create a plot showing FH carrier prevalence across pathways
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    pathway_ids = sorted([pid for pid in pathway_analysis.keys() 
                         if pathway_analysis[pid].get('fh_carrier_stats') is not None])
    
    if len(pathway_ids) == 0:
        return
    
    # Left plot: Carrier proportion by pathway
    ax1 = axes[0]
    
    carrier_props = []
    carrier_labels = []
    n_carriers_list = []
    n_total_list = []
    enrichment_list = []
    colors_list = []
    
    color_map = cm.get_cmap('tab10')
    max_pathway_id = max(pathway_ids) if pathway_ids else 0
    
    for i, pid in enumerate(pathway_ids):
        fh_stats = pathway_analysis[pid]['fh_carrier_stats']
        carrier_props.append(fh_stats['carrier_proportion'] * 100)
        carrier_labels.append(f"Pathway {pid}")
        n_carriers_list.append(fh_stats['n_carriers'])
        n_total_list.append(pathway_analysis[pid]['n_patients'])
        enrichment_list.append(fh_stats['carrier_enrichment'])
        # Use pathway_id for consistent coloring with other plots
        colors_list.append(color_map(pid / max(1, max_pathway_id)))
    
    bars = ax1.bar(range(len(pathway_ids)), carrier_props, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add overall population rate as reference line
    if len(pathway_ids) > 0:
        overall_rate = pathway_analysis[pathway_ids[0]]['fh_carrier_stats']['overall_carrier_proportion'] * 100
        ax1.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall population ({overall_rate:.2f}%)', alpha=0.7)
    
    ax1.set_xticks(range(len(pathway_ids)))
    ax1.set_xticklabels(carrier_labels, fontsize=11, fontweight='bold')
    ax1.set_ylabel('FH Carrier Prevalence (%)', fontsize=12)
    ax1.set_title('FH Carrier Prevalence by Pathway', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=10)
    
    # Add value labels on bars
    for i, (bar, prop, n_car, n_tot) in enumerate(zip(bars, carrier_props, n_carriers_list, n_total_list)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prop:.2f}%\n(n={n_car}/{n_tot})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Right plot: Enrichment ratio
    ax2 = axes[1]
    
    bars2 = ax2.bar(range(len(pathway_ids)), enrichment_list, color=colors_list, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
               label='No enrichment (1.0x)', alpha=0.7)
    
    ax2.set_xticks(range(len(pathway_ids)))
    ax2.set_xticklabels(carrier_labels, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Enrichment Ratio (vs Population)', fontsize=12)
    ax2.set_title('FH Carrier Enrichment by Pathway', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    
    # Add value labels on bars
    for i, (bar, enrich) in enumerate(zip(bars2, enrichment_list)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{enrich:.2f}x',
                ha='center', va='bottom' if height > 1.0 else 'top', 
                fontsize=10, fontweight='bold')
    
    plt.suptitle(f'FH Carrier Prevalence Across {target_disease.title()} Pathways', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{output_dir}/fh_carrier_prevalence_by_pathway_{target_disease.replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved FH carrier prevalence plot: {filename}")
    plt.close()

def create_comprehensive_pathway_comparison_plot(pathway_analysis, output_dir, target_disease):
    """
    Create a comprehensive plot comparing FH carrier prevalence, Signature 5 deviation, 
    and precursor disease prevalence across pathways
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, 
                         height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    pathway_ids = sorted([pid for pid in pathway_analysis.keys() 
                         if pathway_analysis[pid].get('fh_carrier_stats') is not None])
    
    if len(pathway_ids) == 0:
        return
    
    color_map = cm.get_cmap('tab10')
    max_pathway_id = max(pathway_ids) if pathway_ids else 1
    colors = [color_map(pid / max(1, max_pathway_id)) for pid in pathway_ids]
    
    # Extract data
    fh_carrier_props = []
    sig5_deviations = []
    avg_precursor_prevs = []
    n_patients_list = []
    fh_enrichments = []
    
    for pid in pathway_ids:
        data = pathway_analysis[pid]
        fh_stats = data.get('fh_carrier_stats')
        fh_carrier_props.append(fh_stats['carrier_proportion'] * 100)
        sig5_deviations.append(data['sig5_deviation'])
        avg_precursor_prevs.append(data['avg_precursor_prevalence'] * 100)
        n_patients_list.append(data['n_patients'])
        fh_enrichments.append(fh_stats['carrier_enrichment'])
    
    # --- ROW 1: Scatter plots comparing different metrics ---
    
    # (1,1) FH Carrier Prevalence vs Signature 5 Deviation
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(fh_carrier_props, sig5_deviations, 
                          s=[n/30 for n in n_patients_list], 
                          c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    for i, pid in enumerate(pathway_ids):
        ax1.annotate(f'P{pid}', (fh_carrier_props[i], sig5_deviations[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    ax1.set_xlabel('FH Carrier Prevalence (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Signature 5 Deviation', fontsize=12, fontweight='bold')
    ax1.set_title('FH Carriers vs Sig 5 Deviation', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # (1,2) FH Carrier Prevalence vs Precursor Prevalence
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(fh_carrier_props, avg_precursor_prevs,
                          s=[n/30 for n in n_patients_list],
                          c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    for i, pid in enumerate(pathway_ids):
        ax2.annotate(f'P{pid}', (fh_carrier_props[i], avg_precursor_prevs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    ax2.set_xlabel('FH Carrier Prevalence (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Precursor Prevalence (%)', fontsize=12, fontweight='bold')
    ax2.set_title('FH Carriers vs Precursor Disease', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # (1,3) Signature 5 Deviation vs Precursor Prevalence
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(sig5_deviations, avg_precursor_prevs,
                          s=[n/30 for n in n_patients_list],
                          c=colors, alpha=0.7, edgecolors='black', linewidth=2)
    for i, pid in enumerate(pathway_ids):
        ax3.annotate(f'P{pid}', (sig5_deviations[i], avg_precursor_prevs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Signature 5 Deviation', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Avg Precursor Prevalence (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Sig 5 Deviation vs Precursor Disease', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # --- ROW 2: Bar charts for each metric ---
    
    # (2,1) FH Carrier Prevalence
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(range(len(pathway_ids)), fh_carrier_props, 
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    if len(pathway_ids) > 0:
        overall_rate = pathway_analysis[pathway_ids[0]]['fh_carrier_stats']['overall_carrier_proportion'] * 100
        ax4.axhline(y=overall_rate, color='red', linestyle='--', linewidth=2, 
                   label=f'Population ({overall_rate:.2f}%)', alpha=0.7)
    ax4.set_xticks(range(len(pathway_ids)))
    ax4.set_xticklabels([f'Pathway {pid}' for pid in pathway_ids], fontsize=10, fontweight='bold')
    ax4.set_ylabel('FH Carrier Prevalence (%)', fontsize=11, fontweight='bold')
    ax4.set_title('FH Carrier Prevalence by Pathway', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=9)
    for i, (bar, prop) in enumerate(zip(bars4, fh_carrier_props)):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{prop:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # (2,2) Signature 5 Deviation
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(range(len(pathway_ids)), sig5_deviations,
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_xticks(range(len(pathway_ids)))
    ax5.set_xticklabels([f'Pathway {pid}' for pid in pathway_ids], fontsize=10, fontweight='bold')
    ax5.set_ylabel('Signature 5 Deviation', fontsize=11, fontweight='bold')
    ax5.set_title('Signature 5 Deviation by Pathway', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for i, (bar, dev) in enumerate(zip(bars5, sig5_deviations)):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{dev:+.4f}', ha='center', 
                va='bottom' if dev > 0 else 'top', fontsize=9, fontweight='bold')
    
    # (2,3) Average Precursor Prevalence
    ax6 = fig.add_subplot(gs[1, 2])
    bars6 = ax6.bar(range(len(pathway_ids)), avg_precursor_prevs,
                    color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_xticks(range(len(pathway_ids)))
    ax6.set_xticklabels([f'Pathway {pid}' for pid in pathway_ids], fontsize=10, fontweight='bold')
    ax6.set_ylabel('Avg Precursor Prevalence (%)', fontsize=11, fontweight='bold')
    ax6.set_title('Average Precursor Disease Prevalence', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    for i, (bar, prev) in enumerate(zip(bars6, avg_precursor_prevs)):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{prev:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- ROW 3: Summary table and correlations ---
    
    # (3,1-2) Create correlation matrix visualization
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Create correlation data
    import pandas as pd
    corr_data = pd.DataFrame({
        'FH_Carrier_Prev': fh_carrier_props,
        'Sig5_Deviation': sig5_deviations,
        'Precursor_Prev': avg_precursor_prevs,
        'FH_Enrichment': fh_enrichments
    })
    
    corr_matrix = corr_data.corr()
    
    # Create heatmap using seaborn style
    im = ax7.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1, interpolation='nearest')
    ax7.set_xticks(range(len(corr_matrix.columns)))
    ax7.set_yticks(range(len(corr_matrix.columns)))
    ax7.set_xticklabels(corr_matrix.columns, fontsize=10, rotation=45, ha='right')
    ax7.set_yticklabels(corr_matrix.columns, fontsize=10)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            # Use white text for dark backgrounds, black for light
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax7.text(j, i, f'{val:.3f}',
                    ha="center", va="center", color=text_color, 
                    fontsize=11, fontweight='bold')
    
    ax7.set_title('Correlation Matrix: FH Carriers, Sig 5, and Precursor Disease', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=10)
    
    # (3,3) Summary statistics table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create summary table
    table_data = []
    for i, pid in enumerate(pathway_ids):
        table_data.append([
            f'Pathway {pid}',
            f'{n_patients_list[i]:,}',
            f'{fh_carrier_props[i]:.2f}%',
            f'{fh_enrichments[i]:.2f}x',
            f'{sig5_deviations[i]:+.4f}',
            f'{avg_precursor_prevs[i]:.1f}%'
        ])
    
    table = ax8.table(cellText=table_data,
                     colLabels=['Pathway', 'N', 'FH %', 'FH Enr.', 'Sig5 Dev', 'Prec. %'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style pathway rows
    for i, pid in enumerate(pathway_ids):
        row_idx = i + 1
        color = colors[i] if i < len(colors) else 'white'
        for j in range(6):
            table[(row_idx, j)].set_facecolor(color)
            table[(row_idx, j)].set_alpha(0.3)
    
    ax8.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle(f'Comprehensive Pathway Comparison: FH Carriers, Signature 5, and Precursor Disease\n{target_disease.title()}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'{output_dir}/comprehensive_pathway_comparison_{target_disease.replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive pathway comparison plot: {filename}")
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

