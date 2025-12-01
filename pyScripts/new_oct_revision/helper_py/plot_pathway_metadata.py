import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pathway_metadata_differences(
    patient_metadata_file,
    pathway_data,
    processed_ids,
    save_filename='pathway_metadata_comparison.png'
):
    """Plot age at first diagnosis and total codes across pathway clusters"""
    
    # Load metadata
    metadata = pd.read_csv(patient_metadata_file)
    
    # Get patients from pathway data
    patients = pathway_data['patients']
    target_disease = pathway_data['target_disease']
    
    # Group patients by pathway
    pathway_patients = {}
    for p in patients:
        pathway_id = p['pathway']
        if pathway_id not in pathway_patients:
            pathway_patients[pathway_id] = []
        pathway_patients[pathway_id].append(p['patient_id'])
    
    n_pathways = len(pathway_patients)
    pathway_ids = sorted(pathway_patients.keys())
    
    print("="*80)
    print(f"METADATA ANALYSIS BY PATHWAY CLUSTER: {target_disease}")
    print("="*80)
    
    # Collect metadata for each pathway
    pathway_metadata = {}
    for pathway_id in pathway_ids:
        patient_ids = pathway_patients[pathway_id]
        eids = [processed_ids[pid] for pid in patient_ids]
        pathway_meta = metadata[metadata['eid'].isin(eids)]
        
        if len(pathway_meta) > 0:
            pathway_metadata[pathway_id] = {
                'total_codes': pathway_meta['total_codes'].values,
                'min_diag': pathway_meta['min_diag'].values,
                'n_patients': len(pathway_meta)
            }
    
    # Print statistics
    for pathway_id in pathway_ids:
        if pathway_id in pathway_metadata:
            data = pathway_metadata[pathway_id]
            print(f"\nPathway {pathway_id} (n={data['n_patients']}):")
            print(f"  Total codes: {np.mean(data['total_codes']):.1f} ± {np.std(data['total_codes']):.1f}")
            print(f"  Age: {np.mean(data['min_diag']):.1f} ± {np.std(data['min_diag']):.1f}")
    
    # Create boxplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pathway_colors = plt.cm.Set2(np.linspace(0, 1, n_pathways))
    
    codes_data = [pathway_metadata[pid]['total_codes'] for pid in pathway_ids if pid in pathway_metadata]
    age_data = [pathway_metadata[pid]['min_diag'] for pid in pathway_ids if pid in pathway_metadata]
    labels = [f'P{pid}\n(n={pathway_metadata[pid]["n_patients"]})' for pid in pathway_ids if pid in pathway_metadata]
    
    # Box plot 1: Total codes
    bp = axes[0].boxplot(codes_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], pathway_colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Total Diagnostic Codes')
    axes[0].set_title('Total Diagnostic Codes by Pathway')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot 2: Age
    bp = axes[1].boxplot(age_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], pathway_colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel('Age at First Diagnosis')
    axes[1].set_title('Age at First Diagnosis by Pathway')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'{target_disease}: Metadata Comparison Across Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as '{save_filename}'")
    plt.show()
    
    # ANOVA tests
    from scipy.stats import f_oneway
    f_stat_codes, p_val_codes = f_oneway(*codes_data)
    f_stat_age, p_val_age = f_oneway(*age_data)
    
    print(f"\nANOVA Results:")
    print(f"  Total codes: F={f_stat_codes:.3f}, p={p_val_codes:.4f}")
    print(f"  Age: F={f_stat_age:.3f}, p={p_val_age:.4f}")
    
    return pathway_metadata
