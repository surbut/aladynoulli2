"""
COMPLETE ANALYSIS BLOCK - Copy this into your notebook

This block runs the full analysis framework:
1. Deviation-based pathway discovery (UKB)
2. Transition analysis (precursor → target disease)  
3. Cross-cohort reproducibility validation (UKB vs MGB)
4. Signature 5 analysis with FH carriers
"""

# ============================================================================
# SETUP
# ============================================================================
import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from run_complete_pathway_analysis_deviation_only import run_deviation_only_analysis
from run_transition_analysis_ukb_mgb import run_transition_analysis_both_cohorts
from analyze_sig5_by_pathway import analyze_signature5_by_pathway
from show_pathway_reproducibility import main as show_reproducibility

# Parameters
target_disease = "myocardial infarction"
transition_disease = "Rheumatoid arthritis"
n_pathways = 4
lookback_years = 10
output_dir = 'complete_pathway_analysis_output'
mgb_model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'

os.makedirs(output_dir, exist_ok=True)

results = {}

# ============================================================================
# STEP 1: DEVIATION-BASED PATHWAY DISCOVERY (UKB)
# ============================================================================
print("="*80)
print("STEP 1: DEVIATION-BASED PATHWAY DISCOVERY (UKB)")
print("="*80)

ukb_results = run_deviation_only_analysis(
    target_disease=target_disease,
    n_pathways=n_pathways,
    output_dir=os.path.join(output_dir, 'ukb_pathway_discovery'),
    lookback_years=lookback_years
)

results['ukb_pathway_discovery'] = ukb_results
print("\n✅ UKB pathway discovery complete\n")

# ============================================================================
# STEP 2: TRANSITION ANALYSIS (UKB vs MGB)
# ============================================================================
print("="*80)
print("STEP 2: TRANSITION ANALYSIS")
print(f"Analyzing: {transition_disease} → {target_disease}")
print("="*80)

transition_results = run_transition_analysis_both_cohorts(
    transition_disease_name=transition_disease,
    target_disease_name=target_disease,
    years_before=lookback_years,
    age_tolerance=5,
    min_followup=5,
    mgb_model_path=mgb_model_path,
    output_dir=os.path.join(output_dir, 'transition_analysis')
)

results['transition_analysis'] = transition_results
print("\n✅ Transition analysis complete\n")

# ============================================================================
# STEP 3: SIGNATURE 5 ANALYSIS BY PATHWAY (with FH carriers)
# ============================================================================
print("="*80)
print("STEP 3: SIGNATURE 5 ANALYSIS BY PATHWAY")
print("="*80)

sig5_results = analyze_signature5_by_pathway(
    target_disease=target_disease,
    output_dir=os.path.join(output_dir, 'ukb_pathway_discovery'),
    fh_carrier_path='/Users/sarahurbut/Downloads/out/ukb_exome_450k_fh.carrier.txt'
)

results['signature5_analysis'] = sig5_results
print("\n✅ Signature 5 analysis complete\n")

# ============================================================================
# STEP 4: CROSS-COHORT REPRODUCIBILITY (UKB vs MGB)
# ============================================================================
print("="*80)
print("STEP 4: CROSS-COHORT REPRODUCIBILITY VALIDATION")
print("="*80)
print("Comparing pathways between UKB and MGB...")
print("Note: This requires MGB results to exist (run MGB analysis first if needed)")

reproducibility_results = show_reproducibility(force_rerun_mgb=False)
results['reproducibility'] = reproducibility_results
print("\n✅ Reproducibility analysis complete\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\n✅ All analyses finished!")
print(f"\n📁 Results saved to: {output_dir}/")
print(f"   - UKB pathway discovery: {output_dir}/ukb_pathway_discovery/")
print(f"   - Transition analysis: {output_dir}/transition_analysis/")
print(f"   - Reproducibility: (in default location)")
print(f"\n📊 Analysis summary:")
print(f"   - Discovered {n_pathways} pathways to {target_disease}")
print(f"   - Analyzed transition: {transition_disease} → {target_disease}")
print(f"   - Validated reproducibility across UKB and MGB")
print(f"   - Analyzed Signature 5 with FH carrier enrichment")

print(f"\n✅ Complete pathway analysis pipeline finished!")

