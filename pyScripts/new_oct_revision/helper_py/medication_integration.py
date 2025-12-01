#!/usr/bin/env python3
"""
Medication Integration Script
Integrates long-term medication data with signature pathway analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_medication_data(gp_scripts_path=None):
    """Load and process medication data"""
    if gp_scripts_path is None:
        gp_scripts_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/gp_scripts.txt'
    
    print(f"Loading medication data from {gp_scripts_path}...")
    
    # Load GP scripts data
    try:
        gp_scripts = pd.read_csv(gp_scripts_path, sep='\t', low_memory=False)
        print(f"✅ Loaded {len(gp_scripts):,} prescription records")
        print(f"   From {gp_scripts['eid'].nunique():,} unique patients")
        print(f"   Covering {gp_scripts['read_2'].nunique():,} unique medications")
        
        return gp_scripts
    except Exception as e:
        print(f"❌ Error loading medication data: {e}")
        return None

def integrate_medications_with_pathways(pathway_data, Y, thetas, disease_names, processed_ids, gp_scripts_path=None):
    """
    Integrate long-term medication data with signature pathway analysis
    Uses the identify_long_term_medications.py logic for meaningful BNF categories
    
    Parameters:
    - pathway_data: Results from pathway discovery
    - Y: Binary event matrix
    - thetas: Signature loadings
    - disease_names: List of disease names
    - processed_ids: Array of eids corresponding to the first 400K patients
    - gp_scripts_path: Path to GP scripts data
    """
    print(f"=== INTEGRATING LONG-TERM MEDICATIONS WITH SIGNATURE PATHWAYS ===")
    
    # Load medication data
    gp_scripts = load_medication_data(gp_scripts_path)
    if gp_scripts is None:
        return None
    
    # Import the long-term medication analysis functions
    import sys
    sys.path.append('/Users/sarahurbut/dtwin_noulli_clean/scripts')
    from identify_long_term_medications import analyze_prescription_duration, identify_systematic_long_term_drugs, analyze_drug_categories
    
    print("Analyzing long-term medication patterns...")
    
    # Analyze prescription duration patterns (5+ prescriptions over 5+ years)
    patterns_df = analyze_prescription_duration(gp_scripts, min_prescriptions=5, min_years=5)
    if patterns_df is None:
        print("❌ No long-term medication patterns found")
        return None
    
    # Identify systematic long-term drugs (100+ patients, 3+ years average)
    systematic_drugs = identify_systematic_long_term_drugs(patterns_df, min_patients=100, min_avg_duration=3)
    
    # Analyze drug categories
    category_summary = analyze_drug_categories(systematic_drugs, patterns_df)
    
    # Create BNF category mapping
    bnf_categories = {
        '01': 'Gastro-intestinal system',
        '02': 'Cardiovascular system', 
        '03': 'Respiratory system',
        '04': 'Central nervous system',
        '05': 'Infections',
        '06': 'Endocrine system',
        '07': 'Obstetrics, gynaecology and urinary-tract disorders',
        '08': 'Malignant disease and immunosuppression',
        '09': 'Nutrition and blood',
        '10': 'Musculoskeletal and joint diseases',
        '11': 'Eye',
        '12': 'Ear, nose and oropharynx',
        '13': 'Skin',
        '14': 'Immunological products and vaccines',
        '15': 'Anaesthesia'
    }
    
    # Get patient IDs from pathway analysis
    patients = pathway_data['patients']
    pathway_patient_ids = [p['patient_id'] for p in patients]
    
    print(f"Pathway patient IDs range: {min(pathway_patient_ids)} to {max(pathway_patient_ids)}")
    print(f"Sample pathway patient IDs: {pathway_patient_ids[:10]}")
    
    # Check what eid range we have in medication data
    print(f"Medication eid range: {gp_scripts['eid'].min()} to {gp_scripts['eid'].max()}")
    print(f"Sample medication eids: {sorted(gp_scripts['eid'].unique())[:10]}")
    
    # Map pathway patient IDs to medication data using processed_ids
    # Patient index i in our analysis corresponds to processed_ids[i] = actual eid
    # Y and thetas don't contain eids - they're just ordered arrays where row i = patient i
    print(f"Using processed IDs for mapping (have {len(processed_ids)} processed IDs)")
    
    # Map pathway patient IDs to eids: patient_index → processed_ids[patient_index] = eid
    pathway_to_eid = {}
    for pathway_patient_id in pathway_patient_ids:
        if pathway_patient_id < len(processed_ids):
            pathway_to_eid[pathway_patient_id] = processed_ids[pathway_patient_id]
        else:
            print(f"Warning: pathway_patient_id {pathway_patient_id} exceeds available processed IDs")
    
    print(f"Successfully mapped {len(pathway_to_eid)} pathway patients to eids")
    print(f"Sample mappings: {dict(list(pathway_to_eid.items())[:5])}")
    
    # Get the corresponding eids for our pathway patients
    pathway_eids = list(pathway_to_eid.values())
    
    # Filter medication data to patients in our pathway analysis
    pathway_medications = gp_scripts[gp_scripts['eid'].isin(pathway_eids)]
    
    print(f"Found {len(pathway_medications)} medication records for pathway patients")
    print(f"Unique patients with medication data: {pathway_medications['eid'].nunique()}")
    
    if len(pathway_medications) == 0:
        print("❌ No medication data found for pathway patients")
        return None
    
    # Create reverse mapping: eid -> pathway_patient_id
    eid_to_pathway = {eid: pid for pid, eid in pathway_to_eid.items()}
    
    # Analyze long-term medications for each pathway using systematic drugs
    print(f"\nAnalyzing long-term medication patterns for {len(patients)} patients with {pathway_data['target_disease']}")
    
    pathway_medication_analysis = {}
    pathway_labels = pathway_data['pathway_labels']
    unique_labels = np.unique(pathway_labels)
    
    for pathway_id in unique_labels:
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        pathway_patient_ids_pathway = [p['patient_id'] for p in pathway_patients]
        
        # Get eids for this pathway
        pathway_eids_pathway = [pathway_to_eid[pid] for pid in pathway_patient_ids_pathway if pid in pathway_to_eid]
        
        # Get long-term medication patterns for this pathway
        pathway_patterns = patterns_df[patterns_df['eid'].isin(pathway_eids_pathway)]
        
        if len(pathway_patterns) == 0:
            pathway_medication_analysis[pathway_id] = {
                'n_patients': len(pathway_patients),
                'n_patients_with_meds': 0,
                'long_term_medications': {},
                'medication_diversity': 0,
                'total_prescriptions': 0,
                'bnf_categories': {}
            }
            continue
        
        # Filter to only systematic long-term drugs (meaningful names)
        pathway_systematic_patterns = pathway_patterns[pathway_patterns['drug_name'].isin(systematic_drugs.index)]
        
        if len(pathway_systematic_patterns) == 0:
            # Fallback to all long-term medications
            long_term_med_counts = pathway_patterns['drug_name'].value_counts()
            patient_med_diversity = pathway_patterns.groupby('eid')['drug_name'].nunique()
            pathway_patterns_for_categories = pathway_patterns
        else:
            # Use systematic drugs only
            long_term_med_counts = pathway_systematic_patterns['drug_name'].value_counts()
            patient_med_diversity = pathway_systematic_patterns.groupby('eid')['drug_name'].nunique()
            pathway_patterns_for_categories = pathway_systematic_patterns
        
        # Analyze by BNF categories
        pathway_patterns_copy = pathway_patterns_for_categories.copy()
        pathway_patterns_copy['bnf_category'] = pathway_patterns_copy['bnf_code'].str[:2]
        pathway_patterns_copy['category_name'] = pathway_patterns_copy['bnf_category'].map(bnf_categories)
        
        bnf_category_counts = pathway_patterns_copy['category_name'].value_counts()
        
        # Get top systematic drugs for this pathway (from systematic_drugs)
        pathway_systematic_drugs = systematic_drugs[systematic_drugs.index.isin(pathway_patterns['drug_name'])]
        
        pathway_medication_analysis[pathway_id] = {
            'n_patients': len(pathway_patients),
            'n_patients_with_meds': pathway_patterns['eid'].nunique(),
            'long_term_medications': long_term_med_counts.head(10).to_dict(),
            'medication_diversity': patient_med_diversity.mean() if len(patient_med_diversity) > 0 else 0,
            'total_prescriptions': len(pathway_patterns),
            'bnf_categories': bnf_category_counts.head(10).to_dict(),
            'systematic_drugs': pathway_systematic_drugs
        }
        
        print(f"Pathway {pathway_id}: {len(pathway_patients)} patients, {pathway_patterns['eid'].nunique()} with long-term meds")
        print(f"  Long-term medications: {len(long_term_med_counts)} unique drugs")
        print(f"  Average medication diversity: {pathway_medication_analysis[pathway_id]['medication_diversity']:.2f}")
        
        # Show meaningful drug names (not BNF codes)
        top_drugs = list(long_term_med_counts.head(3).index)
        meaningful_drugs = [drug for drug in top_drugs if not drug.startswith(('b', 'a', 'c')) or len(drug) < 10]
        if meaningful_drugs:
            print(f"  Top 3 long-term meds: {meaningful_drugs}")
        else:
            print(f"  Top 3 long-term meds: {top_drugs}")
        
        print(f"  Top 3 BNF categories: {list(bnf_category_counts.head(3).index)}")
        
        # Show systematic drugs for this pathway
        if len(pathway_systematic_drugs) > 0:
            print(f"  Systematic drugs in pathway: {len(pathway_systematic_drugs)}")
            top_systematic = pathway_systematic_drugs.head(3)
            for drug_name, row in top_systematic.iterrows():
                print(f"    • {drug_name}: {row['n_patients']} patients, {row['avg_duration']:.1f} years avg")
    
    # 5. Find pathway-specific medication patterns (differential analysis)
    print(f"\n=== PATHWAY-SPECIFIC MEDICATION PATTERNS ===")
    
    # Calculate medication prevalence by pathway
    pathway_med_prevalence = {}
    for pathway_id in unique_labels:
        if pathway_id in pathway_medication_analysis:
            pathway_data = pathway_medication_analysis[pathway_id]
            pathway_med_prevalence[pathway_id] = pathway_data['long_term_medications']
    
    # Find drugs that are over-represented in each pathway
    all_drugs = set()
    for pathway_id in unique_labels:
        if pathway_id in pathway_med_prevalence:
            all_drugs.update(pathway_med_prevalence[pathway_id].keys())
    
    print(f"Analyzing {len(all_drugs)} unique medications across {len(unique_labels)} pathways")
    
    # For each pathway, find drugs that are over-represented
    for pathway_id in unique_labels:
        if pathway_id in pathway_med_prevalence:
            pathway_drugs = pathway_med_prevalence[pathway_id]
            pathway_size = pathway_medication_analysis[pathway_id]['n_patients']
            
            print(f"\nPathway {pathway_id} MEDICATION PATTERNS:")
            print(f"  Total patients: {pathway_size}")
            
            # Calculate prevalence in this pathway vs others
            pathway_specific_drugs = []
            for drug_name in pathway_drugs.keys():  # All drugs in this pathway
                drug_count_this_pathway = pathway_drugs[drug_name]
                drug_prevalence_this_pathway = drug_count_this_pathway / pathway_size
                
                # Calculate average prevalence in other pathways
                other_pathway_prevalences = []
                for other_pathway_id in unique_labels:
                    if other_pathway_id != pathway_id and other_pathway_id in pathway_med_prevalence:
                        other_pathway_size = pathway_medication_analysis[other_pathway_id]['n_patients']
                        if drug_name in pathway_med_prevalence[other_pathway_id]:
                            other_count = pathway_med_prevalence[other_pathway_id][drug_name]
                            other_prevalence = other_count / other_pathway_size
                            other_pathway_prevalences.append(other_prevalence)
                        else:
                            other_pathway_prevalences.append(0)
                
                if other_pathway_prevalences:
                    avg_other_prevalence = np.mean(other_pathway_prevalences)
                    fold_enrichment = drug_prevalence_this_pathway / (avg_other_prevalence + 1e-8)
                    
                    # More lenient threshold: 1.2x enrichment OR at least 5% prevalence
                    if fold_enrichment > 1.2 or drug_prevalence_this_pathway > 0.05:
                        pathway_specific_drugs.append((
                            drug_name, 
                            drug_prevalence_this_pathway, 
                            avg_other_prevalence, 
                            fold_enrichment,
                            drug_count_this_pathway
                        ))
            
            # Sort by fold enrichment
            pathway_specific_drugs.sort(key=lambda x: x[3], reverse=True)
            
            # Show top 10 pathway-differentiating drugs
            print(f"\n  Top 10 differentiating medications (ranked by fold enrichment):")
            for i, (drug_name, this_prev, other_prev, fold_enrich, count) in enumerate(pathway_specific_drugs[:10]):
                print(f"    {i+1}. {drug_name}:")
                print(f"       This pathway: {count} patients ({this_prev*100:.1f}%)")
                print(f"       Other pathways: {other_prev*100:.1f}% average")
                print(f"       Fold enrichment: {fold_enrich:.2f}x")
            
            if len(pathway_specific_drugs) == 0:
                print(f"  No differentiating medications found")
            else:
                print(f"\n  Found {len(pathway_specific_drugs)} total differentiating medications")
    
    return {
        'pathway_medication_analysis': pathway_medication_analysis,
        'pathway_medications': pathway_medications,
        'pathway_to_eid': pathway_to_eid,
        'eid_to_pathway': eid_to_pathway
    }

def visualize_medication_pathway_integration(medication_results, pathway_data):
    """Create visualizations for medication-pathway integration"""
    print(f"\n=== CREATING MEDICATION-PATHWAY VISUALIZATIONS ===")
    
    pathway_medication_analysis = medication_results['pathway_medication_analysis']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Medication Patterns by Pathway: {pathway_data["target_disease"]}', 
                 fontsize=16, fontweight='bold')
    
    pathway_ids = list(pathway_medication_analysis.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(pathway_ids)))
    
    # 1. Patients with medication data by pathway
    ax1 = axes[0, 0]
    n_patients = [pathway_medication_analysis[pid]['n_patients'] for pid in pathway_ids]
    n_with_meds = [pathway_medication_analysis[pid]['n_patients_with_meds'] for pid in pathway_ids]
    
    x = np.arange(len(pathway_ids))
    width = 0.35
    
    ax1.bar(x - width/2, n_patients, width, label='Total Patients', alpha=0.7)
    ax1.bar(x + width/2, n_with_meds, width, label='With Medication Data', alpha=0.7)
    ax1.set_xlabel('Pathway ID')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Patients with Medication Data by Pathway')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Pathway {pid}' for pid in pathway_ids])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Medication diversity by pathway
    ax2 = axes[0, 1]
    med_diversity = [pathway_medication_analysis[pid]['medication_diversity'] for pid in pathway_ids]
    ax2.bar(pathway_ids, med_diversity, color=colors)
    ax2.set_xlabel('Pathway ID')
    ax2.set_ylabel('Average Medication Diversity')
    ax2.set_title('Medication Diversity by Pathway')
    ax2.grid(True, alpha=0.3)
    
    # 3. Total prescriptions by pathway
    ax3 = axes[0, 2]
    total_prescriptions = [pathway_medication_analysis[pid]['total_prescriptions'] for pid in pathway_ids]
    ax3.bar(pathway_ids, total_prescriptions, color=colors)
    ax3.set_xlabel('Pathway ID')
    ax3.set_ylabel('Total Prescriptions')
    ax3.set_title('Total Prescriptions by Pathway')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top BNF categories across pathways (heatmap)
    ax4 = axes[1, 0]
    
    # Collect all unique BNF categories across pathways
    all_categories = set()
    for pid in pathway_ids:
        all_categories.update(pathway_medication_analysis[pid]['bnf_categories'].keys())
    
    # Create BNF category count matrix
    category_matrix = []
    category_names = []
    for category in list(all_categories)[:10]:  # Top 10 categories
        category_names.append(category[:15] + '...' if len(category) > 15 else category)
        category_counts = []
        for pid in pathway_ids:
            count = pathway_medication_analysis[pid]['bnf_categories'].get(category, 0)
            category_counts.append(count)
        category_matrix.append(category_counts)
    
    if category_matrix:
        category_matrix = np.array(category_matrix)
        im = ax4.imshow(category_matrix, cmap='Blues', aspect='auto')
        ax4.set_xticks(range(len(pathway_ids)))
        ax4.set_xticklabels([f'Pathway {pid}' for pid in pathway_ids])
        ax4.set_yticks(range(len(category_names)))
        ax4.set_yticklabels(category_names, fontsize=8)
        ax4.set_title('Top BNF Categories by Pathway')
        plt.colorbar(im, ax=ax4, label='Number of Patients')
    
    # 5. Pathway size distribution
    ax5 = axes[1, 1]
    pathway_sizes = [pathway_medication_analysis[pid]['n_patients'] for pid in pathway_ids]
    ax5.pie(pathway_sizes, labels=[f'Pathway {pid}' for pid in pathway_ids], 
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax5.set_title('Pathway Size Distribution')
    
    # 6. Medication coverage rate
    ax6 = axes[1, 2]
    coverage_rates = []
    for pid in pathway_ids:
        total_patients = pathway_medication_analysis[pid]['n_patients']
        patients_with_meds = pathway_medication_analysis[pid]['n_patients_with_meds']
        coverage_rate = (patients_with_meds / total_patients * 100) if total_patients > 0 else 0
        coverage_rates.append(coverage_rate)
    
    ax6.bar(pathway_ids, coverage_rates, color=colors)
    ax6.set_xlabel('Pathway ID')
    ax6.set_ylabel('Medication Coverage Rate (%)')
    ax6.set_title('Medication Data Coverage by Pathway')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== MEDICATION INTEGRATION SUMMARY ===")
    print(f"Target Disease: {pathway_data['target_disease']}")
    print(f"Total Pathways: {len(pathway_ids)}")
    print(f"Total Patients: {sum(pathway_medication_analysis[pid]['n_patients'] for pid in pathway_ids)}")
    print(f"Patients with Medication Data: {sum(pathway_medication_analysis[pid]['n_patients_with_meds'] for pid in pathway_ids)}")
    
    for pid in pathway_ids:
        analysis = pathway_medication_analysis[pid]
        print(f"\nPathway {pid}:")
        print(f"  Patients: {analysis['n_patients']}")
        print(f"  With meds: {analysis['n_patients_with_meds']}")
        print(f"  Coverage: {analysis['n_patients_with_meds']/analysis['n_patients']*100:.1f}%")
        print(f"  Medication diversity: {analysis['medication_diversity']:.2f}")
        print(f"  Total prescriptions: {analysis['total_prescriptions']}")

if __name__ == "__main__":
    print("Medication Integration Script")
    print("This script integrates medication data with discovered pathways")
    print("Run pathway_discovery.py and pathway_interrogation.py first.")
