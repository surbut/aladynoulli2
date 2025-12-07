#!/usr/bin/env python3
"""
Run Complete Pathway Analysis - Deviation Method Only

This script runs the complete pathway analysis pipeline using ONLY the 
deviation-from-reference method (10-year lookback). This is the most 
biologically meaningful method as it removes age effects.

Includes:
- Pathway discovery using deviation method
- Most distinctive diseases
- Medication differences
- PRS differences
- Granular disease patterns
- Disease sequence analysis (doesn't require pathways)

Outputs all results to saved files.
"""

import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/pythonscripts/new_oct_revision')
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/helper_py')
from pathway_discovery import load_full_data, discover_disease_pathways
from pathway_interrogation import interrogate_disease_pathways, analyze_prs_by_pathway, analyze_granular_diseases_by_pathway
from medication_integration import integrate_medications_with_pathways, visualize_medication_pathway_integration
from disease_sequence_analysis import analyze_disease_sequences_for_target
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import io
import sys

class TeeOutput:
    """Capture stdout to both console and file"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        self.file.close()

def create_signature_deviation_plots(pathway_data, thetas, output_dir, lookback_years=10):
    """
    Create stacked signature deviation plots AND line plots for each pathway
    """
    patients = pathway_data['patients']
    target_disease = pathway_data['target_disease']
    n_pathways = len(np.unique([p['pathway'] for p in patients]))
    K, T = thetas.shape[1], thetas.shape[2]
    
    # Get population reference
    population_reference = np.mean(thetas, axis=0)  # (K, T)
    
    # Get patient global IDs for each pathway
    pathway_patients = {}
    for pathway_id in range(n_pathways):
        pathway_patients[pathway_id] = [p['patient_id'] for p in patients if p['pathway'] == pathway_id]
    
    # Calculate average deviation for each pathway
    time_diff_by_cluster = np.zeros((n_pathways, K, T))
    
    for pathway_id in range(n_pathways):
        patient_ids = pathway_patients[pathway_id]
        pathway_thetas = thetas[patient_ids, :, :]
        
        for k in range(K):
            for t in range(T):
                pathway_mean = np.mean(pathway_thetas[:, k, t])
                time_diff_by_cluster[pathway_id, k, t] = pathway_mean - population_reference[k, t]
    
    # Create stacked area plot with distinct colors for all signatures
    if K <= 20:
        sig_colors = cm.get_cmap('tab20')(np.linspace(0, 1, K))
    else:
        # For 21 signatures, use tab20 + one color from tab20b
        colors_20 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors_b = cm.get_cmap('tab20b')(np.linspace(0, 1, 20))
        sig_colors = np.vstack([colors_20, colors_b[0:1]])  # Take first color from tab20b for 21st
        sig_colors = sig_colors[:K]  # In case K > 21
    
    # STACKED PLOT (for reference)
    fig, axes = plt.subplots(n_pathways, 1, figsize=(12, 4*n_pathways))
    if n_pathways == 1:
        axes = [axes]
    
    fig.suptitle(f'Signature Deviations from Reference: {target_disease}\n({lookback_years}-Year Lookback)', 
                 fontsize=16, fontweight='bold')
    
    time_points = np.linspace(30, 81, T)
    
    for i in range(n_pathways):
        ax = axes[i]
        n_patients = len(pathway_patients[i])
        pathway_deviations = time_diff_by_cluster[i, :, :]
        
        cumulative = np.zeros(T)
        
        for sig_idx in range(K):
            sig_values = pathway_deviations[sig_idx, :]
            ax.fill_between(time_points, cumulative, cumulative + sig_values, 
                           color=sig_colors[sig_idx], alpha=0.95, label=f'Sig {sig_idx}')
            cumulative += sig_values
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.set_title(f'Pathway {i} (n={n_patients})', fontweight='bold', fontsize=13)
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Signature Deviation from Reference', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    filename = f'{output_dir}/signature_deviations_{target_disease.replace(" ", "_")}_10yr_stacked.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved stacked deviation plot: {filename}")
    
    # LINE PLOT (much clearer for individual signature interpretation)
    fig2, axes2 = plt.subplots(n_pathways, 1, figsize=(14, 5*n_pathways))
    if n_pathways == 1:
        axes2 = [axes2]
    
    fig2.suptitle(f'Individual Signature Deviations by Pathway: {target_disease}\n({lookback_years}-Year Lookback)', 
                  fontsize=16, fontweight='bold')
    
    for i in range(n_pathways):
        ax2 = axes2[i]
        n_patients = len(pathway_patients[i])
        pathway_deviations = time_diff_by_cluster[i, :, :]
        
        # Plot each signature as a line
        for sig_idx in range(K):
            sig_values = pathway_deviations[sig_idx, :]
            ax2.plot(time_points, sig_values, 
                    color=sig_colors[sig_idx], 
                    linewidth=2.0, 
                    marker='o', 
                    markersize=4,
                    label=f'Sig {sig_idx}',
                    alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.set_title(f'Pathway {i} (n={n_patients})', fontweight='bold', fontsize=13)
        ax2.set_xlabel('Age', fontsize=12)
        ax2.set_ylabel('Deviation from Population Mean (Δ Proportion, θ)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        if i == 0:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=3)
    
    plt.tight_layout()
    filename2 = f'{output_dir}/signature_deviations_{target_disease.replace(" ", "_")}_10yr_line.pdf'
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved line deviation plot: {filename2}")

def run_deviation_only_analysis(target_disease, n_pathways=4, output_dir='pathway_analysis_output', lookback_years=10):
    """
    Run complete pathway analysis using ONLY the deviation-from-reference method
    
    Parameters:
    - target_disease: Name of disease to analyze (e.g., "myocardial infarction")
    - n_pathways: Number of pathways to discover
    - output_dir: Directory to save outputs
    - lookback_years: Number of years to look back before disease onset (default: 10)
    """
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up output capture
    log_file = f"{output_dir}/complete_analysis_log.txt"
    tee = TeeOutput(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("="*80)
        print(f"COMPLETE PATHWAY ANALYSIS: {target_disease.upper()}")
        print(f"Method: Deviation-from-Reference ({lookback_years}-year lookback)")
        print("="*80)
        
        # Step 1: Load full data
        print("\n1. LOADING FULL DATASET")
        Y, thetas, disease_names, processed_ids = load_full_data()
        
        # Step 2: Discover pathways using deviation method
        print(f"\n2. DISCOVERING PATHWAYS TO {target_disease.upper()}")
        print(f"Using Deviation-from-Reference Method ({lookback_years}-year lookback)")
        
        pathway_data_dev = discover_disease_pathways(
            target_disease, Y, thetas, disease_names, 
            n_pathways=n_pathways, method='deviation_from_reference',
            lookback_years=lookback_years
        )
        
        if pathway_data_dev is None:
            print(f"❌ Could not discover pathways for {target_disease}")
            return None
        
        # Step 3: Interrogate pathways (includes signature deviations, disease patterns, etc.)
        print(f"\n3. INTERROGATING PATHWAYS")
        print("   This step includes:")
        print("   - Most discriminating signatures")
        print("   - Disease prevalence patterns")
        print("   - Signature deviation trajectories by pathway")
        results_dev = interrogate_disease_pathways(
            pathway_data_dev, Y, thetas, disease_names, output_dir=output_dir
        )
        
        # Create stacked signature deviation plots
        print(f"\n3b. CREATING SIGNATURE DEVIATION PLOTS")
        create_signature_deviation_plots(pathway_data_dev, thetas, output_dir, lookback_years)
        
        # Step 4: Analyze medications
        print(f"\n4. ANALYZING MEDICATION DIFFERENCES BY PATHWAY")
        medication_results_dev = integrate_medications_with_pathways(
            pathway_data_dev, Y, thetas, disease_names, processed_ids
        )
        
        if medication_results_dev is not None:
            visualize_medication_pathway_integration(medication_results_dev, pathway_data_dev)
        
        # Step 5: Analyze PRS differences
        print(f"\n5. ANALYZING PRS DIFFERENCES BY PATHWAY")
        prs_results_dev = analyze_prs_by_pathway(pathway_data_dev, processed_ids, output_dir=output_dir)
        
        # Step 6: Analyze granular disease patterns
        print(f"\n6. ANALYZING GRANULAR DISEASE PATTERNS")
        granular_results_dev = analyze_granular_diseases_by_pathway(pathway_data_dev, Y, disease_names)
        
        # Step 7: Analyze disease sequences (DISABLED - signature transitions preferred)
        # print(f"\n7. ANALYZING DISEASE SEQUENCES FROM GRANULAR ICD-10 DATA")
        # print("(This doesn't require pathways - analyzes sequences before target disease)")
        
        sequence_results = None
        
        # DISABLED: ICD-10 sequence analysis - using signature transitions instead
        # Set DISABLE_SEQUENCE_ANALYSIS = False to enable ICD-10 sequence analysis
        DISABLE_SEQUENCE_ANALYSIS = True
        
        if not DISABLE_SEQUENCE_ANALYSIS:
            try:
                # Map disease names to ICD-10 codes (expanded list)
                icd10_mapping = {
                    'myocardial infarction': ['I21', 'I22'],
                    'breast cancer': ['C50'],
                    'diabetes': ['E10', 'E11'],
                    'stroke': ['I63', 'I64'],
                    'coronary atherosclerosis': ['I25'],
                    'depression': ['F32', 'F33'],
                    # Cardiovascular
                    'atrial fibrillation': ['I48'],
                    'heart failure': ['I50'],
                    'angina': ['I20'],
                    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15', 'I16'],
                    # Cancer
                    'lung cancer': ['C34'],
                    'colorectal cancer': ['C18', 'C19', 'C20'],
                    'prostate cancer': ['C61'],
                    # Other
                    'chronic kidney disease': ['N18'],
                    'copd': ['J44'],
                    'asthma': ['J45', 'J46']
                }
                
                target_lower = target_disease.lower()
                target_icd_codes = None
                
                # Find matching ICD-10 codes
                for disease_key, icd_codes in icd10_mapping.items():
                    if disease_key in target_lower or target_lower in disease_key:
                        target_icd_codes = icd_codes
                        break
                
                if target_icd_codes:
                    print(f"Found ICD-10 codes for {target_disease}: {target_icd_codes}")
                    # Patient ICD data from RDS, mapping from CSV
                    mapping_file_path = '/Users/sarahurbut/aladynoulli2/detailed_icdtophe.csv'
                    sequence_results = analyze_disease_sequences_for_target(
                        target_disease, target_icd_codes, thetas, processed_ids,
                        icd_file_path=None,  # Will use default RDS path
                        mapping_file_path=mapping_file_path  # Use detailed mapping
                    )
                else:
                    print(f"⚠️  No ICD-10 mapping found for {target_disease}")
                    print(f"   Add mapping to run sequence analysis")
            
            except Exception as e:
                print(f"⚠️  Could not run sequence analysis: {e}")
                print(f"   This is optional - main pathway analysis is complete")
        
        # Step 8: Save all results
        print(f"\n8. SAVING RESULTS")
        
        # Save complete results as pickle
        results_dict = {
            'pathway_data_dev': pathway_data_dev,
            'results_dev': results_dev,
            'medication_results_dev': medication_results_dev,
            'prs_results_dev': prs_results_dev,
            'granular_results_dev': granular_results_dev,
            'sequence_results': sequence_results
        }
        
        results_file = f"{output_dir}/complete_analysis_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"   Saved complete results to: {results_file}")
        
        # Save summary to text file
        summary_file = f"{output_dir}/analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PATHWAY ANALYSIS SUMMARY: {target_disease.upper()}\n")
            f.write(f"Method: Deviation-from-Reference (10-year lookback)\n")
            f.write("="*80 + "\n\n")
            
            # Pathway sizes
            patients = pathway_data_dev['patients']
            unique_labels, counts = np.unique([p['pathway'] for p in patients], return_counts=True)
            
            f.write("PATHWAY SIZES:\n")
            f.write("-"*80 + "\n")
            for label, count in zip(unique_labels, counts):
                pct = count / len(patients) * 100
                f.write(f"Pathway {label}: {count:5d} patients ({pct:5.1f}%)\n")
            
            f.write("\n")
        
        print(f"   Saved summary to: {summary_file}")
        
        print(f"\n✅ Complete pathway analysis for {target_disease} finished!")
        print(f"   All results saved to: {output_dir}/")
        print(f"   Complete log saved to: {log_file}")
        
        return results_dict
        
    finally:
        # Restore stdout and close file
        sys.stdout = tee.stdout
        tee.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run pathway analysis using deviation method')
    parser.add_argument('--disease', type=str, default='myocardial infarction',
                        help='Target disease name (default: myocardial infarction)')
    parser.add_argument('--n_pathways', type=int, default=4,
                        help='Number of pathways (default: 4)')
    parser.add_argument('--output_dir', type=str, default='pathway_analysis_output',
                        help='Output directory (default: pathway_analysis_output)')
    
    args = parser.parse_args()
    
    results = run_deviation_only_analysis(
        args.disease, 
        n_pathways=args.n_pathways,
        output_dir=args.output_dir
    )
