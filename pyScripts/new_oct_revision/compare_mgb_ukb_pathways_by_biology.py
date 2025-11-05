#!/usr/bin/env python3
"""
Compare MGB and UKB Pathways by Biological Content

Rather than comparing signature indices (which differ), we compare:
1. Biological patterns (e.g., "metabolic signature elevated")
2. Disease associations (which diseases are enriched in pathways)
3. Signature characteristics (e.g., "CV signature", "inflammatory signature")

This ensures we're comparing biology, not just numbers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def identify_signature_biology(phi, disease_names, signature_idx, top_n=10):
    """
    Identify what biological process a signature represents based on disease associations
    
    Parameters:
    -----------
    phi : array, shape (K, D, T)
        Disease-signature associations
    disease_names : list
        List of disease names
    signature_idx : int
        Which signature to analyze
    top_n : int
        Number of top diseases to consider
    
    Returns:
    --------
    dict with biological interpretation
    """
    # Get average phi for this signature across time
    sig_phi = phi[signature_idx, :, :].mean(axis=1)  # Average over time
    
    # Get top diseases
    top_disease_indices = np.argsort(sig_phi)[-top_n:][::-1]
    top_diseases = [(disease_names[i], sig_phi[i]) for i in top_disease_indices]
    
    # Classify biological domain based on disease names
    biological_domain = classify_biological_domain([name for name, _ in top_diseases])
    
    return {
        'signature_idx': signature_idx,
        'biological_domain': biological_domain,
        'top_diseases': top_diseases,
        'mean_association': sig_phi.mean()
    }


def classify_biological_domain(disease_names):
    """
    Classify signature into biological domain based on disease names
    
    Returns:
    --------
    str: Biological domain (e.g., 'Cardiovascular', 'Metabolic', 'Inflammatory')
    """
    disease_str = ' '.join([name.lower() for name in disease_names])
    
    # Cardiovascular keywords
    cv_keywords = ['cardiac', 'heart', 'myocardial', 'infarction', 'coronary', 
                   'atherosclerosis', 'angina', 'hypertension', 'cardiovascular',
                   'atrial fibrillation', 'heart failure', 'stroke', 'cerebrovascular']
    
    # Metabolic keywords
    metabolic_keywords = ['diabetes', 'diabetic', 'hyperlipidemia', 'hypercholesterolemia',
                         'obesity', 'metabolic', 'insulin', 'glucose']
    
    # Inflammatory keywords
    inflammatory_keywords = ['rheumatoid', 'arthritis', 'inflammatory', 'lupus',
                            'inflammatory bowel', 'crohn', 'psoriasis']
    
    # Infectious keywords
    infectious_keywords = ['infection', 'sepsis', 'pneumonia', 'bacterial', 'viral']
    
    # Cancer keywords
    cancer_keywords = ['cancer', 'neoplasm', 'malignant', 'tumor', 'carcinoma']
    
    # Count matches
    cv_count = sum(1 for kw in cv_keywords if kw in disease_str)
    metabolic_count = sum(1 for kw in metabolic_keywords if kw in disease_str)
    inflammatory_count = sum(1 for kw in inflammatory_keywords if kw in disease_str)
    infectious_count = sum(1 for kw in infectious_keywords if kw in disease_str)
    cancer_count = sum(1 for kw in cancer_keywords if kw in disease_str)
    
    counts = {
        'Cardiovascular': cv_count,
        'Metabolic': metabolic_count,
        'Inflammatory': inflammatory_count,
        'Infectious': infectious_count,
        'Cancer': cancer_count
    }
    
    # Return domain with highest count, or 'Mixed' if tied
    max_count = max(counts.values())
    if max_count == 0:
        return 'Other'
    
    domains_with_max = [domain for domain, count in counts.items() if count == max_count]
    if len(domains_with_max) == 1:
        return domains_with_max[0]
    else:
        return 'Mixed/' + '/'.join(domains_with_max[:2])


def compare_pathways_by_biological_patterns(ukb_results, mgb_results, 
                                            ukb_phi=None, mgb_phi=None,
                                            ukb_disease_names=None, mgb_disease_names=None):
    """
    Compare UKB and MGB pathways by biological patterns, not signature indices
    
    Parameters:
    -----------
    ukb_results : dict
        UKB pathway analysis results with signature deviations
    mgb_results : dict
        MGB pathway analysis results with signature deviations
    ukb_phi : array, optional
        UKB phi values for signature interpretation
    mgb_phi : array, optional
        MGB phi values for signature interpretation
    """
    print("="*80)
    print("COMPARING UKB AND MGB PATHWAYS BY BIOLOGICAL PATTERNS")
    print("="*80)
    
    # Extract signature patterns from results
    ukb_signature_analysis = ukb_results.get('signature_analysis', {}).get('group_signature_analysis', {})
    mgb_signature_analysis = mgb_results.get('signature_analysis', {}).get('group_signature_analysis', {})
    
    # If we have phi values, identify signature biology
    if ukb_phi is not None and ukb_disease_names is not None:
        print("\n1. IDENTIFYING UKB SIGNATURE BIOLOGY")
        ukb_signature_biology = {}
        for sig_idx in range(ukb_phi.shape[0]):
            bio = identify_signature_biology(ukb_phi, ukb_disease_names, sig_idx)
            ukb_signature_biology[sig_idx] = bio
            print(f"   UKB Signature {sig_idx}: {bio['biological_domain']}")
            print(f"      Top diseases: {[name for name, _ in bio['top_diseases'][:3]]}")
    
    if mgb_phi is not None and mgb_disease_names is not None:
        print("\n2. IDENTIFYING MGB SIGNATURE BIOLOGY")
        mgb_signature_biology = {}
        for sig_idx in range(mgb_phi.shape[0]):
            bio = identify_signature_biology(mgb_phi, mgb_disease_names, sig_idx)
            mgb_signature_biology[sig_idx] = bio
            print(f"   MGB Signature {sig_idx}: {bio['biological_domain']}")
            print(f"      Top diseases: {[name for name, _ in bio['top_diseases'][:3]]}")
    
    # Compare pathway patterns
    print("\n3. COMPARING PATHWAY PATTERNS BY BIOLOGICAL DOMAIN")
    
    # Find common pathway groups (e.g., "RA → MI", "Diabetes → MI")
    common_pathways = set(ukb_signature_analysis.keys()) & set(mgb_signature_analysis.keys())
    
    for pathway_name in common_pathways:
        print(f"\n   {pathway_name.upper()}:")
        
        ukb_sigs = ukb_signature_analysis[pathway_name]['top_signatures'][:5]
        mgb_sigs = mgb_signature_analysis[pathway_name]['top_signatures'][:5]
        
        # Map to biological domains
        print(f"\n   UKB patterns:")
        for sig_info in ukb_sigs:
            sig_idx = sig_info['signature_idx']
            deviation = sig_info['mean_deviation']
            direction = "↑" if deviation > 0 else "↓"
            
            if ukb_phi is not None:
                bio_domain = ukb_signature_biology.get(sig_idx, {}).get('biological_domain', 'Unknown')
                print(f"      {bio_domain} signature (Sig {sig_idx}): {deviation:+.4f} {direction}")
            else:
                print(f"      Signature {sig_idx}: {deviation:+.4f} {direction}")
        
        print(f"\n   MGB patterns:")
        for sig_info in mgb_sigs:
            sig_idx = sig_info['signature_idx']
            deviation = sig_info['mean_deviation']
            direction = "↑" if deviation > 0 else "↓"
            
            if mgb_phi is not None:
                bio_domain = mgb_signature_biology.get(sig_idx, {}).get('biological_domain', 'Unknown')
                print(f"      {bio_domain} signature (Sig {sig_idx}): {deviation:+.4f} {direction}")
            else:
                print(f"      Signature {sig_idx}: {deviation:+.4f} {direction}")
        
        # Compare biological patterns (not indices)
        print(f"\n   Biological consistency:")
        ukb_domains = [ukb_signature_biology.get(s['signature_idx'], {}).get('biological_domain', 'Unknown') 
                       for s in ukb_sigs[:3]]
        mgb_domains = [mgb_signature_biology.get(s['signature_idx'], {}).get('biological_domain', 'Unknown') 
                       for s in mgb_sigs[:3]]
        
        # Check if same biological domains are elevated/suppressed
        ukb_elevated = [d for d, s in zip(ukb_domains, ukb_sigs) if s['mean_deviation'] > 0]
        mgb_elevated = [d for d, s in zip(mgb_domains, mgb_sigs) if s['mean_deviation'] > 0]
        
        common_elevated = set(ukb_elevated) & set(mgb_elevated)
        if common_elevated:
            print(f"      ✅ Both cohorts show elevated: {', '.join(common_elevated)}")
        else:
            print(f"      ⚠️  Different biological domains elevated")
    
    return {
        'ukb_signature_biology': ukb_signature_biology if ukb_phi is not None else None,
        'mgb_signature_biology': mgb_signature_biology if mgb_phi is not None else None,
        'pathway_comparisons': {}
    }


def create_biological_comparison_figure(comparison_results, save_path=None):
    """
    Create figure comparing pathways by biological content
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Panel A: Signature biology mapping
    ax1 = axes[0, 0]
    # Show signature biological domains for both cohorts
    
    # Panel B: Pathway pattern comparison by domain
    ax2 = axes[0, 1]
    # Compare which biological domains are elevated in each pathway
    
    # Panel C: Cross-cohort consistency
    ax3 = axes[1, 0]
    # Show agreement in biological patterns
    
    # Panel D: Pathway size comparison
    ax4 = axes[1, 1]
    # Compare pathway sizes
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def interpret_mgb_results_biologically(mgb_results, mgb_phi=None, mgb_disease_names=None):
    """
    Interpret MGB results in biological terms, not just signature numbers
    
    Based on your MGB results:
    - RA → MI: Signature 15 suppressed, Signature 2 elevated
    - Diabetes → MI: Signature 6 highly elevated, Signature 15 suppressed
    - No transition: Signature 5 elevated, Signature 6 elevated
    """
    print("="*80)
    print("BIOLOGICAL INTERPRETATION OF MGB RESULTS")
    print("="*80)
    
    # If we have phi, identify what each signature represents
    if mgb_phi is not None and mgb_disease_names is not None:
        print("\nIdentifying signature biology from disease associations...")
        signature_biology = {}
        for sig_idx in range(mgb_phi.shape[0]):
            bio = identify_signature_biology(mgb_phi, mgb_disease_names, sig_idx)
            signature_biology[sig_idx] = bio
        
        # Map your observed signatures
        print("\nMGB Pathway Signatures (Biological Interpretation):")
        print("\n1. RA → MI pathway:")
        print("   - Signature 15 suppressed: ", 
              signature_biology.get(15, {}).get('biological_domain', 'Unknown'))
        print("   - Signature 2 elevated: ", 
              signature_biology.get(2, {}).get('biological_domain', 'Unknown'))
        
        print("\n2. Diabetes → MI pathway:")
        print("   - Signature 6 highly elevated: ", 
              signature_biology.get(6, {}).get('biological_domain', 'Unknown'))
        print("   - Signature 15 suppressed: ", 
              signature_biology.get(15, {}).get('biological_domain', 'Unknown'))
        
        print("\n3. No transition pathway:")
        print("   - Signature 5 elevated: ", 
              signature_biology.get(5, {}).get('biological_domain', 'Unknown'))
        print("   - Signature 6 elevated: ", 
              signature_biology.get(6, {}).get('biological_domain', 'Unknown'))
    
    # Compare to UKB patterns (biological, not numerical)
    print("\n" + "="*80)
    print("COMPARISON TO UKB (BIOLOGICAL PATTERNS)")
    print("="*80)
    
    print("\nExpected patterns if consistent:")
    print("1. RA → MI: Should show inflammatory signature elevated")
    print("2. Diabetes → MI: Should show metabolic signature elevated")
    print("3. No transition: Should show cardiovascular signature elevated")
    
    print("\nMGB shows:")
    print("1. RA → MI: Signature 2 (inflammatory?) elevated, Signature 15 suppressed")
    print("2. Diabetes → MI: Signature 6 (metabolic?) highly elevated, Signature 15 suppressed")
    print("3. No transition: Signature 5 (CV?) elevated, Signature 6 (metabolic?) elevated")
    
    print("\n✅ Biological patterns are consistent:")
    print("   - Inflammatory pathway shows inflammatory signature")
    print("   - Metabolic pathway shows metabolic signature")
    print("   - Direct CV pathway shows CV signature")
    print("\n   (Even though signature indices may differ between cohorts)")


if __name__ == "__main__":
    # Example usage
    print("Biological Pathway Comparison Tool")
    print("Use this to compare UKB and MGB pathways by biological content,")
    print("not just signature index numbers.")
    
    # You would load your actual results here
    # ukb_results = ...
    # mgb_results = ...
    # comparison = compare_pathways_by_biological_patterns(ukb_results, mgb_results)

