"""
Investigate ALL signatures in Pathway 1 to understand what drives the elevated MI risk.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathway_discovery import load_full_data
import pickle as pkl

def investigate_all_signatures(target_disease="myocardial infarction", output_dir="output_10yr"):
    """
    Analyze ALL 21 signatures across all pathways to see which ones are elevated in Pathway 1.
    """
    print("="*80)
    print("INVESTIGATING ALL SIGNATURES BY PATHWAY")
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
    population_reference = np.mean(thetas, axis=0)  # [K, T]
    
    print(f"\nAnalyzing {n_pathways} pathways with {len(thetas)} patients")
    
    # For each pathway, calculate average signature deviations (ages 60-70, or last 10 years before MI)
    signature_deviations = {pw: [] for pw in range(n_pathways)}
    
    for pathway_id in range(n_pathways):
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        n_patients = len(pathway_patients)
        
        print(f"\n{'='*60}")
        print(f"PATHWAY {pathway_id} (n={n_patients})")
        print('='*60)
        
        # For each signature
        sig_deviations = []
        
        for sig_idx in range(21):
            # Calculate pathway mean signature (last 10 years before MI)
            sig_deviations_for_patients = []
            
            for patient_info in pathway_patients:
                patient_id = patient_info['patient_id']
                age_at_mi = patient_info['age_at_disease']
                mi_time_idx = age_at_mi - 30
                
                if mi_time_idx >= 10:
                    start_idx = mi_time_idx - 10
                    patient_sig = thetas[patient_id, sig_idx, start_idx:mi_time_idx]
                    pop_sig = population_reference[sig_idx, start_idx:mi_time_idx]
                    
                    sig_dev = np.mean(patient_sig - pop_sig)
                    sig_deviations_for_patients.append(sig_dev)
            
            if sig_deviations_for_patients:
                avg_sig_dev = np.mean(sig_deviations_for_patients)
                sig_deviations.append(avg_sig_dev)
            else:
                sig_deviations.append(0.0)
        
        signature_deviations[pathway_id] = np.array(sig_deviations)
        
        # Print top 5 most elevated signatures
        top_5_idx = np.argsort(signature_deviations[pathway_id])[::-1][:5]
        print(f"Top 5 elevated signatures:")
        for rank, idx in enumerate(top_5_idx, 1):
            print(f"  {rank}. Signature {idx}: {signature_deviations[pathway_id][idx]:+.4f}")
    
    # Create visualization comparing Pathway 1 to others
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sig_indices = np.arange(21)
    
    # Top-left: Pathway 1 signature deviations
    ax1 = axes[0, 0]
    pw1_deviations = signature_deviations[1]
    colors = ['#4C78A8' if d > 0 else '#F58518' for d in pw1_deviations]
    bars = ax1.bar(sig_indices, pw1_deviations, color=colors, edgecolor='black', linewidth=1)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Signature Index', fontsize=12)
    ax1.set_ylabel('Deviation from Population', fontsize=12)
    ax1.set_title('Pathway 1: ALL Signature Deviations', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels for non-zero signatures
    for sig_idx, bar in enumerate(bars):
        if abs(pw1_deviations[sig_idx]) > 0.01:
            h = bar.get_height()
            ax1.text(sig_idx, h + 0.003 if h > 0 else h - 0.008,
                    f'{h:.3f}', ha='center', va='bottom' if h > 0 else 'top',
                    fontsize=8, rotation=90)
    
    # Top-right: Compare Pathway 1 vs others
    ax2 = axes[0, 1]
    x = np.arange(21)
    width = 0.15
    
    for pw in [0, 1, 2, 3]:
        offset = (pw - 1.5) * width
        deviations = signature_deviations[pw]
        colors_by_pw = ['blue', 'green', 'orange', 'red']
        ax2.bar(x + offset, deviations, width, label=f'Pathway {pw}',
               color=colors_by_pw[pw], alpha=0.7, edgecolor='black')
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Signature Index', fontsize=12)
    ax2.set_ylabel('Deviation from Population', fontsize=12)
    ax2.set_title('Signature Deviations: All Pathways', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Bottom-left: Top 10 signatures ranked by elevation in Pathway 1
    ax3 = axes[1, 0]
    pw1_top10_idx = np.argsort(pw1_deviations)[::-1][:10]
    pw1_top10_vals = pw1_deviations[pw1_top10_idx]
    colors_top10 = ['#4C78A8' if d > 0 else '#F58518' for d in pw1_top10_vals]
    
    bars = ax3.barh(range(10), pw1_top10_vals, color=colors_top10, edgecolor='black', linewidth=1)
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([f'Sig {pw1_top10_idx[i]}' for i in range(10)])
    ax3.set_xlabel('Deviation from Population', fontsize=12)
    ax3.set_title('Pathway 1: Top 10 Most Elevated Signatures', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pw1_top10_vals)):
        ax3.text(val + 0.003 if val > 0 else val - 0.008, i,
                f'{val:.3f}', ha='left' if val > 0 else 'right', va='center',
                fontsize=9, fontweight='bold')
    
    # Bottom-right: Compare Pathway 1's top signatures across all pathways
    ax4 = axes[1, 1]
    top_sigs_to_compare = pw1_top10_idx[:5]  # Top 5 signatures in Pathway 1
    
    x = np.arange(len(top_sigs_to_compare))
    width = 0.15
    
    for pw in [0, 1, 2, 3]:
        offset = (pw - 1.5) * width
        deviations = [signature_deviations[pw][sig_idx] for sig_idx in top_sigs_to_compare]
        colors_by_pw = ['blue', 'green', 'orange', 'red']
        ax4.bar(x + offset, deviations, width, label=f'Pathway {pw}',
               color=colors_by_pw[pw], alpha=0.7, edgecolor='black')
    
    ax4.axhline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'Sig {idx}' for idx in top_sigs_to_compare])
    ax4.set_ylabel('Deviation from Population', fontsize=12)
    ax4.set_title("Pathway 1's Top Signatures Across All Pathways", fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{output_dir}/all_signatures_by_pathway.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {filename}")
    plt.close()
    
    # Print summary for Pathway 1
    print(f"\n{'='*80}")
    print("PATHWAY 1 SIGNATURE ANALYSIS")
    print(f"{'='*80}")
    print(f"\nElevated signatures (deviation > 0.01):")
    for sig_idx in range(21):
        if pw1_deviations[sig_idx] > 0.01:
            print(f"  Signature {sig_idx}: {pw1_deviations[sig_idx]:+.4f}")
    
    print(f"\nDepressed signatures (deviation < -0.01):")
    for sig_idx in range(21):
        if pw1_deviations[sig_idx] < -0.01:
            print(f"  Signature {sig_idx}: {pw1_deviations[sig_idx]:+.4f}")
    
    return signature_deviations

if __name__ == "__main__":
    results = investigate_all_signatures()



