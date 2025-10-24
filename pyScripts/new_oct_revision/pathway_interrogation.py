#!/usr/bin/env python3
"""
Pathway Interrogation Script
Analyzes discovered pathways to understand what distinguishes them
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def interrogate_disease_pathways(pathway_data, Y, thetas, disease_names):
    """
    Interrogate what distinguishes the different pathways to the same disease
    """
    print(f"=== INTERROGATING PATHWAYS TO {pathway_data['target_disease'].upper()} ===")
    
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    method = pathway_data['method']
    
    # 1. Calculate pathway statistics
    print(f"\n1. PATHWAY STATISTICS:")
    unique_labels, counts = np.unique(pathway_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"   Pathway {label}: {count} patients ({count/len(patients)*100:.1f}%)")
    
    # 2. Calculate signature trajectories for each pathway
    print(f"\n2. CALCULATING SIGNATURE TRAJECTORIES:")
    N, K, T = thetas.shape
    ages = np.arange(30, 30 + T)
    
    pathway_trajectories = {}
    for pathway_id in unique_labels:
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        
        # Get trajectories for this pathway
        pathway_traj_list = [p['trajectory'] for p in pathway_patients]
        pathway_trajectories[pathway_id] = {
            'patients': pathway_patients,
            'n_patients': len(pathway_patients),
            'trajectories': pathway_traj_list
        }
        
        # Calculate mean and std trajectories
        mean_trajectory = np.mean(pathway_traj_list, axis=0)  # [K, T]
        std_trajectory = np.std(pathway_traj_list, axis=0)    # [K, T]
        
        pathway_trajectories[pathway_id]['mean_trajectory'] = mean_trajectory
        pathway_trajectories[pathway_id]['std_trajectory'] = std_trajectory
        
        print(f"   Pathway {pathway_id}: {len(pathway_patients)} patients")
    
    # 3. Find most discriminating signatures
    print(f"\n3. MOST DISCRIMINATING SIGNATURES:")
    
    # Calculate population reference (all patients with this disease)
    all_patients_trajectories = [p['trajectory'] for p in patients]
    population_reference = np.mean(all_patients_trajectories, axis=0)  # [K, T]
    
    signature_discrimination = []
    for sig_idx in range(K):
        # Get average signature loading over time for each pathway
        pathway_avg_loadings = []
        for pathway_id in unique_labels:
            if pathway_id in pathway_trajectories:
                # Average loading over time
                pathway_avg = np.mean(pathway_trajectories[pathway_id]['mean_trajectory'][sig_idx, :])
                pathway_avg_loadings.append(pathway_avg)
        
        if len(pathway_avg_loadings) > 1:
            # Calculate variance between pathways
            between_var = np.var(pathway_avg_loadings)
            
            # Calculate within-pathway variance
            within_var = 0
            total_patients = 0
            for pathway_id in unique_labels:
                if pathway_id in pathway_trajectories:
                    pathway_patients = pathway_trajectories[pathway_id]['patients']
                    pathway_sig_values = []
                    for patient_info in pathway_patients:
                        # Average loading over time for this patient
                        patient_sig_avg = np.mean(patient_info['trajectory'][sig_idx, :])
                        pathway_sig_values.append(patient_sig_avg)
                    
                    within_var += np.var(pathway_sig_values) * len(pathway_patients)
                    total_patients += len(pathway_patients)
            
            within_var = within_var / total_patients if total_patients > 0 else 0
            discrimination = between_var / (within_var + 1e-8)
            signature_discrimination.append(discrimination)
        else:
            signature_discrimination.append(0)
    
    # Get top discriminating signatures
    top_sigs = np.argsort(signature_discrimination)[::-1][:5]
    print(f"   Top 5 discriminating signatures:")
    for i, sig_idx in enumerate(top_sigs):
        print(f"     {i+1}. Signature {sig_idx}: Score = {signature_discrimination[sig_idx]:.4f}")
    
    # 4. Analyze disease patterns by pathway - LOOKING AT PRE-DISEASE HISTORY
    print(f"\n4. DISEASE PATTERNS BY PATHWAY (PRE-TARGET DISEASE):")
    
    target_disease_idx = pathway_data['target_disease_idx']
    
    # For each pathway, find what other diseases they had BEFORE the target disease
    pathway_disease_patterns = {}
    for pathway_id in unique_labels:
        if pathway_id in pathway_trajectories:
            pathway_patients = pathway_trajectories[pathway_id]['patients']
            
            # Get disease counts for this pathway BEFORE target disease onset
            pathway_diseases = {}
            for disease_idx in range(Y.shape[1]):
                if disease_idx != target_disease_idx:
                    disease_count = 0
                    for patient_info in pathway_patients:
                        patient_id = patient_info['patient_id']
                        age_at_target = patient_info['age_at_disease']
                        cutoff_idx = age_at_target - 30  # Time index before target disease
                        
                        if cutoff_idx > 0:
                            # Count if they had this disease BEFORE target disease
                            if Y[patient_id, disease_idx, :cutoff_idx].sum() > 0:
                                disease_count += 1
                    
                    if disease_count > 0:
                        pathway_diseases[disease_names[disease_idx]] = disease_count
            
            # Sort by frequency
            pathway_diseases = dict(sorted(pathway_diseases.items(), key=lambda x: x[1], reverse=True))
            pathway_disease_patterns[pathway_id] = pathway_diseases
            
            print(f"   Pathway {pathway_id} top PRE-disease conditions:")
            for i, (disease, count) in enumerate(list(pathway_diseases.items())[:10]):
                pct = count / len(pathway_patients) * 100
                print(f"     {i+1}. {disease}: {count} patients ({pct:.1f}%)")
    
    # 4b. Find diseases that DIFFERENTIATE pathways (chi-square test)
    print(f"\n4b. DISEASES THAT DIFFERENTIATE PATHWAYS:")
    
    # Create contingency tables for each disease
    disease_differentiation = []
    for disease_idx in range(Y.shape[1]):
        if disease_idx != target_disease_idx:
            # Count patients with this disease in each pathway (pre-target disease)
            pathway_counts = []
            for pathway_id in unique_labels:
                if pathway_id in pathway_trajectories:
                    pathway_patients = pathway_trajectories[pathway_id]['patients']
                    count = 0
                    for patient_info in pathway_patients:
                        patient_id = patient_info['patient_id']
                        age_at_target = patient_info['age_at_disease']
                        cutoff_idx = age_at_target - 30
                        if cutoff_idx > 0 and Y[patient_id, disease_idx, :cutoff_idx].sum() > 0:
                            count += 1
                    pathway_counts.append(count)
            
            # Only test if disease is present in at least one pathway
            if sum(pathway_counts) > 10:
                # Calculate chi-square statistic manually (variance in prevalence)
                pathway_sizes = [len(pathway_trajectories[pid]['patients']) for pid in unique_labels if pid in pathway_trajectories]
                prevalences = [pathway_counts[i]/pathway_sizes[i] if pathway_sizes[i] > 0 else 0 for i in range(len(pathway_counts))]
                
                if len(prevalences) > 1:
                    variance = np.var(prevalences)
                    disease_differentiation.append({
                        'disease': disease_names[disease_idx],
                        'variance': variance,
                        'prevalences': prevalences,
                        'counts': pathway_counts
                    })
    
    # Sort by variance in prevalence
    disease_differentiation.sort(key=lambda x: x['variance'], reverse=True)
    
    print(f"   Top 15 diseases that differentiate pathways (by variance in prevalence):")
    for i, disease_info in enumerate(disease_differentiation[:15]):
        print(f"     {i+1}. {disease_info['disease']}")
        for pathway_id in unique_labels:
            if pathway_id < len(disease_info['prevalences']):
                prev = disease_info['prevalences'][pathway_id] * 100
                count = disease_info['counts'][pathway_id]
                print(f"        Pathway {pathway_id}: {count} patients ({prev:.1f}%)")
    
    # 5. Create visualizations
    print(f"\n5. CREATING PATHWAY VISUALIZATIONS:")
    
    # Calculate reference signatures (population average)
    sig_refs = population_reference  # [K, T] - reference signatures over time
    
    # Plot signature trajectory deviations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Signature Trajectory Deviations by Pathway: {pathway_data["target_disease"]}', 
                 fontsize=16, fontweight='bold')
    
    # Top discriminating signatures
    top_sigs_to_plot = top_sigs[-4:]  # Top 4
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for idx, sig_idx in enumerate(top_sigs_to_plot):
        ax = axes[idx//2, idx%2]
        
        for pathway_id in unique_labels:
            if pathway_id in pathway_trajectories:
                pathway_data_plot = pathway_trajectories[pathway_id]
                pathway_mean_traj = pathway_data_plot['mean_trajectory'][sig_idx, :]
                pathway_std_traj = pathway_data_plot['std_trajectory'][sig_idx, :]
                
                # Calculate deviation from reference
                reference_traj = sig_refs[sig_idx, :]
                deviation_traj = pathway_mean_traj - reference_traj
                deviation_std = pathway_std_traj
                
                ax.plot(ages, deviation_traj, color=colors[pathway_id], 
                        linewidth=3, label=f'Pathway {pathway_id} (n={pathway_data_plot["n_patients"]})')
                ax.fill_between(ages, deviation_traj - deviation_std, deviation_traj + deviation_std,
                                color=colors[pathway_id], alpha=0.2)
        
        # Add reference line at 0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Reference')
        
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel(f'Signature {sig_idx} Deviation from Reference', fontsize=12)
        ax.set_title(f'Signature {sig_idx} Deviations (Score: {signature_discrimination[sig_idx]:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot pathway size distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pathway sizes
    ax1.bar(unique_labels, counts, color=colors)
    ax1.set_xlabel('Pathway ID', fontsize=12)
    ax1.set_ylabel('Number of Patients', fontsize=12)
    ax1.set_title('Pathway Size Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Age at disease onset by pathway
    pathway_ages = {}
    for pathway_id in unique_labels:
        pathway_patients = pathway_trajectories[pathway_id]['patients']
        ages = [p['age_at_disease'] for p in pathway_patients]
        pathway_ages[pathway_id] = ages
    
    ax2.boxplot([pathway_ages[pid] for pid in unique_labels], 
                labels=[f'Pathway {pid}' for pid in unique_labels])
    ax2.set_xlabel('Pathway ID', fontsize=12)
    ax2.set_ylabel('Age at Disease Onset', fontsize=12)
    ax2.set_title('Age at Disease Onset by Pathway', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Create stacked bar plot of ALL signature deviations (years prior to event per cluster)
    print(f"\n6. CREATING STACKED SIGNATURE DEVIATION PLOTS:")
    
    # Calculate signature deviations for the 5 years before disease onset
    pre_disease_deviations = {}
    
    for pathway_id in unique_labels:
        if pathway_id in pathway_trajectories:
            pathway_patients = pathway_trajectories[pathway_id]['patients']
            
            # Get 5-year pre-disease trajectories for this pathway
            pre_disease_trajectories = []
            for patient_info in pathway_patients:
                trajectory = patient_info['trajectory']
                age_at_disease = patient_info['age_at_disease']
                
                # Get 5 years before disease
                cutoff_idx = age_at_disease - 30
                lookback_idx = max(0, cutoff_idx - 5)
                
                if cutoff_idx > 5:
                    pre_disease_traj = trajectory[:, lookback_idx:cutoff_idx]
                    pre_disease_trajectories.append(pre_disease_traj)
            
            if pre_disease_trajectories:
                # Average across patients and time (5 years before disease)
                avg_pre_disease = np.mean(pre_disease_trajectories, axis=(0, 2))  # Average across patients and time
                reference_pre_disease = np.mean([sig_refs[:, lookback_idx:cutoff_idx] for _ in range(len(pre_disease_trajectories))], axis=(0, 2))
                deviation = avg_pre_disease - reference_pre_disease
                pre_disease_deviations[pathway_id] = deviation
    
    # Create stacked bar plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Stacked bar plot of signature deviations
    ax1 = axes[0]
    
    pathway_ids = list(pre_disease_deviations.keys())
    signature_ids = list(range(len(pre_disease_deviations[pathway_ids[0]])))
    
    # Create stacked bars
    bottom = np.zeros(len(pathway_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(signature_ids)))
    
    for sig_idx in signature_ids:
        deviations = [pre_disease_deviations[pid][sig_idx] for pid in pathway_ids]
        bars = ax1.bar(pathway_ids, deviations, bottom=bottom, label=f'Sig {sig_idx}', color=colors[sig_idx], alpha=0.7)
        bottom += deviations
    
    ax1.set_xlabel('Pathway ID', fontsize=12)
    ax1.set_ylabel('Cumulative Signature Deviation from Reference', fontsize=12)
    ax1.set_title('Stacked Signature Deviations: 5 Years Before Disease Onset', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add legend for top 10 most variable signatures
    signature_variances = [np.var([pre_disease_deviations[pid][sig_idx] for pid in pathway_ids]) for sig_idx in signature_ids]
    top_var_sigs = np.argsort(signature_variances)[::-1][:10]
    
    legend_elements = [plt.Rectangle((0,0),1,1, color=colors[sig_idx], alpha=0.7, label=f'Sig {sig_idx}') 
                      for sig_idx in top_var_sigs]
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Right plot: Individual signature deviations (top 10 most variable)
    ax2 = axes[1]
    
    x_pos = np.arange(len(pathway_ids))
    width = 0.08
    
    for i, sig_idx in enumerate(top_var_sigs):
        deviations = [pre_disease_deviations[pid][sig_idx] for pid in pathway_ids]
        ax2.bar(x_pos + i*width, deviations, width, label=f'Sig {sig_idx}', color=colors[sig_idx], alpha=0.7)
    
    ax2.set_xlabel('Pathway ID', fontsize=12)
    ax2.set_ylabel('Signature Deviation from Reference', fontsize=12)
    ax2.set_title('Top 10 Most Variable Signature Deviations', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos + width * 4.5)
    ax2.set_xticklabels(pathway_ids)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of signature deviations
    print(f"\nSummary of signature deviations (5 years before disease):")
    for pathway_id in pathway_ids:
        total_dev = np.sum(np.abs(pre_disease_deviations[pathway_id]))
        print(f"  Pathway {pathway_id}: Total absolute deviation = {total_dev:.3f}")
        top_3_sigs = np.argsort(np.abs(pre_disease_deviations[pathway_id]))[::-1][:3]
        print(f"    Top 3 signatures: {[(sig_idx, pre_disease_deviations[pathway_id][sig_idx]) for sig_idx in top_3_sigs]}")
    
    return {
        'pathway_trajectories': pathway_trajectories,
        'signature_discrimination': signature_discrimination,
        'top_sigs': top_sigs,
        'pathway_disease_patterns': pathway_disease_patterns,
        'sig_refs': sig_refs,
        'pre_disease_deviations': pre_disease_deviations
    }

def analyze_prs_by_pathway(pathway_data, processed_ids, prs_file_path=None):
    """
    Analyze polygenic risk scores (PRS) by pathway to understand genetic differences
    """
    print(f"\n=== ANALYZING POLYGENIC RISK SCORES BY PATHWAY ===")
    
    if prs_file_path is None:
        # Try to find PRS file
        possible_paths = [
            '/Users/sarahurbut/aladynoulli2/pyScripts/prs_with_eid.csv',
            #'/Users/sarahurbut/aladynoulli2/data_for_running/prs_scores.csv',
            #'/Users/sarahurbut/aladynoulli2/pyScripts/big_stuff/all_patient_genetics.csv',
            #'/Users/sarahurbut/aladynoulli2/pyScripts/prs_scores.csv'
        ]
        
        prs_file_path = None
        for path in possible_paths:
            try:
                import os
                if os.path.exists(path):
                    prs_file_path = path
                    break
            except:
                continue
    
    if prs_file_path is None:
        print("❌ PRS file not found. Please specify the path to your PRS scores file.")
        print("Expected format: CSV with columns 'PatientID' and various PRS scores")
        return None
    
    try:
        import pandas as pd
        prs_data = pd.read_csv(prs_file_path)
        print(f"✅ Loaded PRS data: {prs_data.shape}")
        print(f"Available PRS columns: {list(prs_data.columns)}")
    except Exception as e:
        print(f"❌ Error loading PRS data: {e}")
        return None
    
    # Get patient IDs from pathway analysis
    patients = pathway_data['patients']
    pathway_patient_ids = [p['patient_id'] for p in patients]
    
    # Map pathway patient IDs to eids using processed_ids
    pathway_eids = [processed_ids[pid] for pid in pathway_patient_ids if pid < len(processed_ids)]
    
    print(f"Pathway patient IDs: {pathway_patient_ids[:5]}...")
    print(f"Corresponding eids: {pathway_eids[:5]}...")
    
    # Filter PRS data to pathway patients using actual eids
    pathway_prs = prs_data[prs_data['PatientID'].isin(pathway_eids)].copy()
    
    if len(pathway_prs) == 0:
        print("❌ No PRS data found for pathway patients")
        print(f"Looking for eids: {pathway_eids[:10]}...")
        print(f"PRS PatientID sample: {prs_data['PatientID'].head().tolist()}")
        return None
    
    print(f"Found PRS data for {len(pathway_prs)} pathway patients")
    
    # Create mapping: eid -> pathway_id
    eid_to_pathway = {}
    for i, pid in enumerate(pathway_patient_ids):
        if pid < len(processed_ids):
            eid = processed_ids[pid]
            eid_to_pathway[eid] = patients[i]['pathway']
    
    # Add pathway labels to PRS data
    pathway_prs['pathway'] = pathway_prs['PatientID'].map(eid_to_pathway)
    pathway_prs = pathway_prs.dropna(subset=['pathway'])
    
    # Get PRS columns (exclude PatientID and pathway)
    prs_columns = [col for col in pathway_prs.columns if col not in ['PatientID', 'pathway']]
    
    print(f"Analyzing {len(prs_columns)} PRS scores across {pathway_prs['pathway'].nunique()} pathways")
    
    # Analyze PRS differences by pathway
    pathway_prs_analysis = {}
    
    for pathway_id in pathway_prs['pathway'].unique():
        pathway_prs_subset = pathway_prs[pathway_prs['pathway'] == pathway_id]
        
        pathway_prs_analysis[pathway_id] = {
            'n_patients': len(pathway_prs_subset),
            'prs_means': {},
            'prs_stds': {}
        }
        
        for prs_col in prs_columns:
            prs_values = pathway_prs_subset[prs_col].dropna()
            if len(prs_values) > 0:
                pathway_prs_analysis[pathway_id]['prs_means'][prs_col] = prs_values.mean()
                pathway_prs_analysis[pathway_id]['prs_stds'][prs_col] = prs_values.std()
    
    # Find PRS that differentiate pathways
    print(f"\nPRS DIFFERENCES BY PATHWAY:")
    
    # Calculate variance in PRS means across pathways
    prs_variance = {}
    for prs_col in prs_columns:
        pathway_means = []
        for pathway_id in pathway_prs['pathway'].unique():
            if prs_col in pathway_prs_analysis[pathway_id]['prs_means']:
                pathway_means.append(pathway_prs_analysis[pathway_id]['prs_means'][prs_col])
        
        if len(pathway_means) > 1:
            prs_variance[prs_col] = np.var(pathway_means)
    
    # Sort by variance (most discriminating PRS)
    sorted_prs = sorted(prs_variance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 most discriminating PRS scores:")
    for i, (prs_name, variance) in enumerate(sorted_prs[:10]):
        print(f"\n{i+1}. {prs_name} (variance: {variance:.4f}):")
        
        # Collect data for statistical testing
        pathway_data_for_test = []
        pathway_labels_for_test = []
        
        for pathway_id in sorted(pathway_prs['pathway'].unique()):
            if prs_name in pathway_prs_analysis[pathway_id]['prs_means']:
                mean_val = pathway_prs_analysis[pathway_id]['prs_means'][prs_name]
                std_val = pathway_prs_analysis[pathway_id]['prs_stds'][prs_name]
                n_patients = pathway_prs_analysis[pathway_id]['n_patients']
                print(f"   Pathway {pathway_id}: {mean_val:.3f} ± {std_val:.3f} (n={n_patients})")
                
                # Get actual PRS values for this pathway
                pathway_subset = pathway_prs[pathway_prs['pathway'] == pathway_id]
                prs_values = pathway_subset[prs_name].dropna()
                pathway_data_for_test.extend(prs_values.tolist())
                pathway_labels_for_test.extend([pathway_id] * len(prs_values))
        
        # Perform statistical tests
        if len(pathway_data_for_test) > 0 and len(set(pathway_labels_for_test)) > 1:
            from scipy import stats
            
            # ANOVA test
            pathway_groups = []
            for pathway_id in sorted(pathway_prs['pathway'].unique()):
                if prs_name in pathway_prs_analysis[pathway_id]['prs_means']:
                    pathway_subset = pathway_prs[pathway_prs['pathway'] == pathway_id]
                    prs_values = pathway_subset[prs_name].dropna()
                    if len(prs_values) > 0:
                        pathway_groups.append(prs_values.tolist())
            
            if len(pathway_groups) >= 2:
                # ANOVA
                f_stat, p_value = stats.f_oneway(*pathway_groups)
                print(f"   ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
                
                # Post-hoc pairwise t-tests
                pathway_ids = sorted(pathway_prs['pathway'].unique())
                significant_pairs = []
                for i, pathway_id1 in enumerate(pathway_ids):
                    for pathway_id2 in pathway_ids[i+1:]:
                        if (prs_name in pathway_prs_analysis[pathway_id1]['prs_means'] and 
                            prs_name in pathway_prs_analysis[pathway_id2]['prs_means']):
                            
                            group1 = pathway_prs[pathway_prs['pathway'] == pathway_id1][prs_name].dropna()
                            group2 = pathway_prs[pathway_prs['pathway'] == pathway_id2][prs_name].dropna()
                            
                            if len(group1) > 0 and len(group2) > 0:
                                t_stat, p_val = stats.ttest_ind(group1, group2)
                                if p_val < 0.05:
                                    significant_pairs.append(f"Pathway {pathway_id1} vs {pathway_id2}: p={p_val:.4f}")
                
                if significant_pairs:
                    print(f"   Significant pairwise differences:")
                    for pair in significant_pairs[:3]:  # Show top 3
                        print(f"     {pair}")
                else:
                    print(f"   No significant pairwise differences (p<0.05)")
    
    # Create visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PRS Differences by Pathway', fontsize=16, fontweight='bold')
    
    # Plot top 4 most discriminating PRS
    for i, (prs_name, variance) in enumerate(sorted_prs[:4]):
        ax = axes[i//2, i%2]
        
        pathway_means = []
        pathway_stds = []
        pathway_ids = []
        
        for pathway_id in sorted(pathway_prs['pathway'].unique()):
            if prs_name in pathway_prs_analysis[pathway_id]['prs_means']:
                pathway_means.append(pathway_prs_analysis[pathway_id]['prs_means'][prs_name])
                pathway_stds.append(pathway_prs_analysis[pathway_id]['prs_stds'][prs_name])
                pathway_ids.append(pathway_id)
        
        bars = ax.bar(pathway_ids, pathway_means, yerr=pathway_stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Pathway ID')
        ax.set_ylabel(f'{prs_name} PRS Score')
        ax.set_title(f'{prs_name} by Pathway (variance: {variance:.4f})')
        ax.grid(True, alpha=0.3)
        
        # Color bars by pathway
        colors = plt.cm.tab10(np.linspace(0, 1, len(pathway_ids)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'pathway_prs_analysis': pathway_prs_analysis,
        'prs_variance': prs_variance,
        'sorted_prs': sorted_prs
    }

def analyze_granular_diseases_by_pathway(pathway_data, Y, disease_names, min_prevalence=0.01):
    """
    Analyze more granular disease patterns by pathway, including rare diseases
    """
    print(f"\n=== ANALYZING GRANULAR DISEASE PATTERNS BY PATHWAY ===")
    print(f"Including diseases with ≥{min_prevalence*100:.1f}% prevalence in at least one pathway")
    
    patients = pathway_data['patients']
    pathway_labels = pathway_data['pathway_labels']
    target_disease_idx = pathway_data['target_disease_idx']
    unique_labels = np.unique(pathway_labels)
    
    # Calculate pathway trajectories
    pathway_trajectories = {}
    for pathway_id in unique_labels:
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        pathway_trajectories[pathway_id] = {'patients': pathway_patients}
    
    # Find diseases that differentiate pathways (including rare ones)
    print(f"\nDISEASES THAT DIFFERENTIATE PATHWAYS (including rare diseases):")
    
    # Create contingency tables for each disease
    disease_differentiation = []
    for disease_idx in range(Y.shape[1]):
        if disease_idx != target_disease_idx:
            # Count patients with this disease in each pathway (pre-target disease)
            pathway_counts = []
            pathway_sizes = []
            
            for pathway_id in unique_labels:
                if pathway_id in pathway_trajectories:
                    pathway_patients = pathway_trajectories[pathway_id]['patients']
                    count = 0
                    for patient_info in pathway_patients:
                        patient_id = patient_info['patient_id']
                        age_at_target = patient_info['age_at_disease']
                        cutoff_idx = age_at_target - 30
                        if cutoff_idx > 0 and Y[patient_id, disease_idx, :cutoff_idx].sum() > 0:
                            count += 1
                    
                    pathway_counts.append(count)
                    pathway_sizes.append(len(pathway_patients))
            
            # Calculate prevalences
            prevalences = [pathway_counts[i]/pathway_sizes[i] if pathway_sizes[i] > 0 else 0 
                          for i in range(len(pathway_counts))]
            
            # Only include diseases with sufficient prevalence in at least one pathway
            max_prevalence = max(prevalences) if prevalences else 0
            
            if max_prevalence >= min_prevalence:
                # Calculate variance in prevalence
                variance = np.var(prevalences)
                disease_differentiation.append({
                    'disease': disease_names[disease_idx],
                    'variance': variance,
                    'prevalences': prevalences,
                    'counts': pathway_counts,
                    'max_prevalence': max_prevalence
                })
    
    # Sort by variance in prevalence
    disease_differentiation.sort(key=lambda x: x['variance'], reverse=True)
    
    print(f"Found {len(disease_differentiation)} diseases with sufficient prevalence")
    print(f"Top 20 diseases that differentiate pathways (including rare diseases):")
    
    for i, disease_info in enumerate(disease_differentiation[:20]):
        print(f"\n{i+1}. {disease_info['disease']} (max prevalence: {disease_info['max_prevalence']*100:.1f}%):")
        for pathway_id in unique_labels:
            if pathway_id < len(disease_info['prevalences']):
                prev = disease_info['prevalences'][pathway_id] * 100
                count = disease_info['counts'][pathway_id]
                print(f"   Pathway {pathway_id}: {count} patients ({prev:.1f}%)")
    
    # Analyze disease categories
    print(f"\n=== DISEASE CATEGORY ANALYSIS ===")
    
    # Define disease categories (you can expand this)
    disease_categories = {
        'cardiovascular': ['hypertension', 'coronary', 'angina', 'atrial', 'heart', 'cardiac', 'stroke', 'cerebrovascular'],
        'metabolic': ['diabetes', 'obesity', 'hypercholesterolemia', 'metabolic', 'insulin', 'glucose'],
        'rheumatologic': ['arthritis', 'rheumatoid', 'lupus', 'psoriasis', 'arthropathy', 'musculoskeletal'],
        'neoplastic': ['cancer', 'neoplasm', 'malignant', 'tumor', 'carcinoma', 'lymphoma', 'leukemia'],
        'respiratory': ['asthma', 'copd', 'respiratory', 'pneumonia', 'bronchitis', 'airway'],
        'gastrointestinal': ['diverticulosis', 'hernia', 'gastrointestinal', 'ulcer', 'colitis', 'crohn'],
        'neurological': ['parkinson', 'alzheimer', 'dementia', 'seizure', 'epilepsy', 'migraine'],
        'infectious': ['infection', 'sepsis', 'pneumonia', 'uti', 'hepatitis', 'tuberculosis'],
        'renal': ['renal', 'kidney', 'nephritis', 'dialysis', 'creatinine', 'proteinuria'],
        'endocrine': ['thyroid', 'adrenal', 'pituitary', 'hormone', 'endocrine'],
        'ophthalmic': ['cataract', 'glaucoma', 'retinopathy', 'eye', 'vision'],
        'dermatologic': ['skin', 'dermatitis', 'eczema', 'psoriasis', 'rash']
    }
    
    # Analyze each category
    for category_name, keywords in disease_categories.items():
        category_diseases = []
        for disease_info in disease_differentiation:
            disease_name = disease_info['disease'].lower()
            if any(keyword in disease_name for keyword in keywords):
                category_diseases.append(disease_info)
        
        if category_diseases:
            print(f"\n{category_name.upper()} DISEASES:")
            # Sort by variance
            category_diseases.sort(key=lambda x: x['variance'], reverse=True)
            
            for i, disease_info in enumerate(category_diseases[:5]):  # Top 5 in category
                print(f"  {i+1}. {disease_info['disease']} (variance: {disease_info['variance']:.4f})")
                for pathway_id in unique_labels:
                    if pathway_id < len(disease_info['prevalences']):
                        prev = disease_info['prevalences'][pathway_id] * 100
                        print(f"     Pathway {pathway_id}: {prev:.1f}%")
    
    return {
        'disease_differentiation': disease_differentiation,
        'disease_categories': disease_categories
    }

def compare_pathway_methods(pathway_data_avg, pathway_data_traj, Y, thetas, disease_names):
    """Compare results from both clustering methods"""
    print("=== COMPARING CLUSTERING METHODS ===")
    
    # Interrogate both methods
    print("\n1. AVERAGE LOADING METHOD:")
    results_avg = interrogate_disease_pathways(pathway_data_avg, Y, thetas, disease_names)
    
    print("\n2. TRAJECTORY SIMILARITY METHOD:")
    results_traj = interrogate_disease_pathways(pathway_data_traj, Y, thetas, disease_names)
    
    # Compare pathway sizes
    print("\n3. PATHWAY SIZE COMPARISON:")
    print("Average Loading Method:")
    unique_labels_avg, counts_avg = np.unique(pathway_data_avg['pathway_labels'], return_counts=True)
    for label, count in zip(unique_labels_avg, counts_avg):
        print(f"  Pathway {label}: {count} patients")
    
    print("Trajectory Similarity Method:")
    unique_labels_traj, counts_traj = np.unique(pathway_data_traj['pathway_labels'], return_counts=True)
    for label, count in zip(unique_labels_traj, counts_traj):
        print(f"  Pathway {label}: {count} patients")
    
    return results_avg, results_traj

if __name__ == "__main__":
    print("Pathway Interrogation Script")
    print("This script analyzes discovered pathways to understand what distinguishes them")
    print("Run pathway_discovery.py first to discover pathways, then use this script to analyze them.")
