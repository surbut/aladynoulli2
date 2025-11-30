#!/usr/bin/env python3
"""
CHIP analysis across multiple signatures and outcomes.

CHIP is associated with cardiovascular disease, but may work through different mechanisms than FH.
This script tests multiple signatures and outcomes to find the most relevant associations.

Key hypotheses:
- Signature 16 (critical care/acute illness) - CHIP causes inflammation and acute events
- Signature 0 (cardiac structure) - CHIP may cause structural heart disease
- Stroke outcomes - CHIP is strongly associated with stroke
- Heart failure outcomes - CHIP may cause heart failure through inflammation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
import sys
from pathlib import Path

sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')
from analyze_chip_carriers_signature import load_chip_carriers, analyze_signature_enrichment_chip, visualize_signature_trajectory_chip

# Load data
print("Loading data...")
import torch

chip_file_path = '/Users/sarahurbut/Dropbox-Personal/CH_UKB.txt'
theta_path = '/Users/sarahurbut/aladynoulli2/pyScripts/new_thetas_with_pcs_retrospective.pt'
Y_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt'
processed_ids_path = '/Users/sarahurbut/aladynoulli2/pyScripts/processed_patient_ids.npy'

theta = torch.load(theta_path, map_location='cpu')
if hasattr(theta, 'numpy'):
    theta = theta.numpy()

Y = torch.load(Y_path, map_location='cpu')
processed_ids = np.load(processed_ids_path)

# Load disease names to find relevant outcomes
disease_names = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv')['x']

# Find disease indices for different outcomes
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision/new_notebooks')
from find_disease_indices import get_major_disease_indices

disease_indices_dict = get_major_disease_indices(disease_names)

# Define different outcome sets
# CHIP is associated with multiple outcomes beyond CVD:
# - Cardiovascular (stroke, MI, HF)
# - Blood cancers (leukemia, MDS) - may not be in disease list
# - Inflammatory conditions
# - Infections (pneumonia)
# - Acute events (sepsis, acute renal failure)
outcome_sets = {
    'ASCVD': disease_indices_dict.get('ASCVD', [112, 113, 114, 115, 116]),  # Fallback to known indices
    'Stroke': disease_indices_dict.get('Stroke', []),
    'Heart_Failure': disease_indices_dict.get('Heart_Failure', []),
    'Atrial_Fib': disease_indices_dict.get('Atrial_Fib', []),
    'All_Cancers': disease_indices_dict.get('All_Cancers', []),
    'Leukemia_MDS': disease_indices_dict.get('Leukemia_MDS', []),  # Blood cancers - may be empty
    'Anemia': disease_indices_dict.get('Anemia', []),  # Blood disorders
    'Pneumonia': disease_indices_dict.get('Pneumonia', []),  # Infections
    'COPD': disease_indices_dict.get('COPD', []),  # Inflammatory lung disease
    'Sepsis': disease_indices_dict.get('Sepsis', []),  # Acute inflammatory event
    'Acute_Renal_Failure': disease_indices_dict.get('Acute_Renal_Failure', []),  # Acute event
}

# Define signatures to test
# CHIP causes inflammation and affects multiple systems, so test multiple signatures:
# Signature 5: Ischemic cardiovascular (classic atherosclerosis)
# Signature 16: Critical care/acute illness (inflammation, acute events) - CHIP causes inflammation!
# Signature 0: Cardiac structure (heart failure, arrhythmias) - CHIP associated with HF
# Signature 11: Cerebrovascular (stroke) - CHIP strongly associated with stroke
# Signature 2: GI disorders (inflammation-related)
# Signature 7: Pain/Inflammation (chronic inflammation)
signatures_to_test = {
    5: 'Ischemic_CV',
    16: 'Critical_Care',  # Inflammation - most relevant for CHIP!
    0: 'Cardiac_Structure',  # Heart failure
    11: 'Cerebrovascular',  # Stroke
    2: 'GI_Disorders',  # Inflammation-related
    7: 'Pain_Inflammation',  # Chronic inflammation
    # Add others if relevant
}

# Set up output directory for plots and results
# Use absolute path based on script location to ensure consistency
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
output_dir = script_dir / 'results' / 'chip_multiple_signatures'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n📁 Results and plots will be saved to: {output_dir}")
print(f"   Script directory: {script_dir}")
print(f"   Current working directory: {Path.cwd()}")

# Check if results already exist
summary_path = output_dir / 'chip_multiple_signatures_summary.csv'
print(f"\n🔍 Checking for existing results at: {summary_path}")
print(f"   Path exists: {summary_path.exists()}")
if summary_path.exists():
    print(f"\n✓✓✓ Results already exist at: {summary_path}")
    print(f"   File size: {summary_path.stat().st_size:,} bytes")
    print("   Skipping analysis. To re-run, delete the file first.")
    # Use SystemExit which works better in notebooks
    raise SystemExit(0)
else:
    print("   No existing results found - will run analysis")

# Load CHIP carriers
print("\nLoading CHIP carriers...")
is_chip, chip_df = load_chip_carriers(chip_file_path, processed_ids, 'hasCH')
is_dnmt3a, _ = load_chip_carriers(chip_file_path, processed_ids, 'hasDNMT3A')
is_tet2, _ = load_chip_carriers(chip_file_path, processed_ids, 'hasTET2')

# Run analysis for each combination
results_summary = []

for outcome_name, event_indices in outcome_sets.items():
    if len(event_indices) == 0:
        print(f"\n⚠️  Skipping {outcome_name}: No disease indices found")
        continue
    
    print(f"\n{'='*80}")
    print(f"OUTCOME: {outcome_name} (indices: {event_indices})")
    print(f"{'='*80}")
    
    for sig_idx, sig_name in signatures_to_test.items():
        print(f"\n--- Signature {sig_idx} ({sig_name}) ---")
        
        # Test CHIP
        try:
            # Create output subdirectory for this combination
            combo_output_dir = output_dir / f'{outcome_name}_sig{sig_idx}'
            
            res_chip = analyze_signature_enrichment_chip(
                chip_file_path, 'CHIP', 'hasCH', sig_idx,
                event_indices, theta, Y, processed_ids,
                pre_window=5, epsilon=0.0, output_dir=combo_output_dir
            )
            
            # Generate and save plot
            try:
                fig = visualize_signature_trajectory_chip(
                    res_chip, theta, Y, processed_ids, event_indices,
                    sig_idx, 'CHIP', pre_window=5
                )
                plot_path = combo_output_dir / 'CHIP_signature_trajectory.png'
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Saved plot: {plot_path}")
            except Exception as e:
                print(f"  ⚠️  Could not generate plot: {e}")
            results_summary.append({
                'mutation': 'CHIP',
                'outcome': outcome_name,
                'signature': sig_idx,
                'signature_name': sig_name,
                'n_carriers': res_chip['n_carriers'],
                'carrier_prop_rising': res_chip['carriers_rising'] / max(res_chip['n_carriers'], 1),
                'noncarrier_prop_rising': res_chip['noncarriers_rising'] / max(res_chip['n_noncarriers'], 1),
                'OR': res_chip['OR'],
                'p_value': res_chip['p_value']
            })
        except Exception as e:
            print(f"  Error analyzing CHIP: {e}")
        
        # Test DNMT3A
        try:
            combo_output_dir = output_dir / f'{outcome_name}_sig{sig_idx}'
            
            res_dnmt3a = analyze_signature_enrichment_chip(
                chip_file_path, 'DNMT3A', 'hasDNMT3A', sig_idx,
                event_indices, theta, Y, processed_ids,
                pre_window=5, epsilon=0.0, output_dir=combo_output_dir
            )
            
            # Generate and save plot
            try:
                fig = visualize_signature_trajectory_chip(
                    res_dnmt3a, theta, Y, processed_ids, event_indices,
                    sig_idx, 'DNMT3A', pre_window=5
                )
                plot_path = combo_output_dir / 'DNMT3A_signature_trajectory.png'
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Saved plot: {plot_path}")
            except Exception as e:
                print(f"  ⚠️  Could not generate plot: {e}")
            results_summary.append({
                'mutation': 'DNMT3A',
                'outcome': outcome_name,
                'signature': sig_idx,
                'signature_name': sig_name,
                'n_carriers': res_dnmt3a['n_carriers'],
                'carrier_prop_rising': res_dnmt3a['carriers_rising'] / max(res_dnmt3a['n_carriers'], 1),
                'noncarrier_prop_rising': res_dnmt3a['noncarriers_rising'] / max(res_dnmt3a['n_noncarriers'], 1),
                'OR': res_dnmt3a['OR'],
                'p_value': res_dnmt3a['p_value']
            })
        except Exception as e:
            print(f"  Error analyzing DNMT3A: {e}")
        
        # Test TET2
        try:
            combo_output_dir = output_dir / f'{outcome_name}_sig{sig_idx}'
            
            res_tet2 = analyze_signature_enrichment_chip(
                chip_file_path, 'TET2', 'hasTET2', sig_idx,
                event_indices, theta, Y, processed_ids,
                pre_window=5, epsilon=0.0, output_dir=combo_output_dir
            )
            
            # Generate and save plot
            try:
                fig = visualize_signature_trajectory_chip(
                    res_tet2, theta, Y, processed_ids, event_indices,
                    sig_idx, 'TET2', pre_window=5
                )
                plot_path = combo_output_dir / 'TET2_signature_trajectory.png'
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Saved plot: {plot_path}")
            except Exception as e:
                print(f"  ⚠️  Could not generate plot: {e}")
            results_summary.append({
                'mutation': 'TET2',
                'outcome': outcome_name,
                'signature': sig_idx,
                'signature_name': sig_name,
                'n_carriers': res_tet2['n_carriers'],
                'carrier_prop_rising': res_tet2['carriers_rising'] / max(res_tet2['n_carriers'], 1),
                'noncarrier_prop_rising': res_tet2['noncarriers_rising'] / max(res_tet2['n_noncarriers'], 1),
                'OR': res_tet2['OR'],
                'p_value': res_tet2['p_value']
            })
        except Exception as e:
            print(f"  Error analyzing TET2: {e}")

# Create summary table
summary_df = pd.DataFrame(results_summary)

print("\n" + "="*100)
print("SUMMARY: CHIP MUTATION CARRIER ANALYSIS ACROSS SIGNATURES")
print("="*100)

if len(summary_df) > 0:
    # Sort by OR (highest enrichment first)
    summary_df = summary_df.sort_values('OR', ascending=False)
    
    print(f"\n{'Mutation':<12} {'Outcome':<10} {'Signature':<20} {'N_Carriers':<12} {'Carrier%':<12} {'NonCarrier%':<12} {'OR':<10} {'p-value':<12}")
    print("-" * 100)
    
    for _, row in summary_df.iterrows():
        print(f"{row['mutation']:<12} {row['outcome']:<10} {row['signature_name']:<20} "
              f"{int(row['n_carriers']):<12} {row['carrier_prop_rising']*100:<11.2f}% "
              f"{row['noncarrier_prop_rising']*100:<11.2f}% {row['OR']:<10.3f} {row['p_value']:<12.3e}")
    
    # Find most significant associations
    print("\n" + "="*100)
    print("MOST SIGNIFICANT ASSOCIATIONS (p < 0.05, OR > 1.0):")
    print("="*100)
    
    significant = summary_df[(summary_df['p_value'] < 0.05) & (summary_df['OR'] > 1.0)]
    if len(significant) > 0:
        for _, row in significant.iterrows():
            print(f"{row['mutation']} + Signature {row['signature']} ({row['signature_name']}) "
                  f"for {row['outcome']}: OR={row['OR']:.3f}, p={row['p_value']:.3e}")
    else:
        print("No significant associations found (p < 0.05, OR > 1.0)")
    
    # Find strongest associations (highest OR)
    print("\n" + "="*100)
    print("STRONGEST ASSOCIATIONS (Top 5 by OR):")
    print("="*100)
    
    top5 = summary_df.head(5)
    for _, row in top5.iterrows():
        print(f"{row['mutation']} + Signature {row['signature']} ({row['signature_name']}) "
              f"for {row['outcome']}: OR={row['OR']:.3f}, p={row['p_value']:.3e}, "
              f"carriers={row['carrier_prop_rising']*100:.1f}% vs noncarriers={row['noncarrier_prop_rising']*100:.1f}%")
    
    # Save summary table
    summary_path = output_dir / 'chip_multiple_signatures_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary table to: {summary_path}")
else:
    print("No results to summarize")

print("\n" + "="*100)
print(f"📁 All results and plots saved to: {output_dir}")
print("="*100)

