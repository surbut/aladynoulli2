#!/usr/bin/env python3
"""
Full Washout Analysis for 0-400K patients
Processes all batches systematically and aggregates results
"""

import torch
import pandas as pd
import numpy as np
from evaluatetdccode import evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex
from load_model_essentials import load_model_essentials

def run_full_washout_analysis():
    """Run washout analysis on all batches from 0-400K"""
    
    print("="*80)
    print("FULL WASHOUT ANALYSIS: 0-400K PATIENTS")
    print("="*80)
    
    # Load the full data once
    print("\n1. LOADING FULL DATASET")
    Y, E, G, essentials = load_model_essentials()
    fh_processed = pd.read_csv('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/baselinagefamh.csv')
    
    print(f"Loaded Y: {Y.shape}")
    print(f"Loaded E: {E.shape}")
    print(f"Loaded fh_processed: {fh_processed.shape}")
    
    # Define all batches (0-400K in 10K increments)
    batches = [(i, i+10000) for i in range(0, 400000, 10000)]
    print(f"\n2. PROCESSING {len(batches)} BATCHES")
    print(f"Batches: {batches[:5]}...{batches[-5:]}")
    
    # Storage for results
    washout_results = {
        '0yr': {},  # No washout
        '1yr': {},  # 1-year washout  
        '2yr': {}   # 2-year washout
    }
    
    # Run washout analysis on each batch
    for batch_idx, (start, stop) in enumerate(batches):
        print(f"\n=== Processing batch {batch_idx+1}/{len(batches)}: {start}-{stop} ===")
        
        try:
            # Load batch predictions
            pi_filename = f"/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi/pi_enroll_fixedphi_sex_{start}_{stop}.pt"
            pi_batch = torch.load(pi_filename, map_location='cpu')
            
            # Check gamma shape to verify PCS inclusion
            gamma_shape = pi_batch['model_state_dict']['gamma'].shape
            print(f"  Gamma shape: {gamma_shape}")
            
            # Subset other data to match
            Y_batch = Y[start:stop]
            E_batch = E[start:stop] 
            pce_df_batch = fh_processed.iloc[start:stop].reset_index(drop=True)
            
            print(f"  Y_batch: {Y_batch.shape}")
            print(f"  E_batch: {E_batch.shape}")
            print(f"  pce_df_batch: {pce_df_batch.shape}")
            
            # Run washout analysis for this batch
            for washout_name, offset in [('0yr', 0), ('1yr', 1), ('2yr', 2)]:
                print(f"    Running {washout_name} washout...")
                
                try:
                    results = evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex(
                        pi=pi_batch,
                        Y_100k=Y_batch,
                        E_100k=E_batch,
                        disease_names=essentials['disease_names'],
                        pce_df=pce_df_batch,
                        n_bootstraps=50,  # Fewer bootstraps per batch
                        follow_up_duration_years=1,
                        start_offset=offset
                    )
                    
                    # Store results
                    for disease, metrics in results.items():
                        if disease not in washout_results[washout_name]:
                            washout_results[washout_name][disease] = {
                                'aucs': [], 'cis': [], 'events': [], 'rates': []
                            }
                        
                        washout_results[washout_name][disease]['aucs'].append(metrics['auc'])
                        washout_results[washout_name][disease]['cis'].append((metrics['ci_lower'], metrics['ci_upper']))
                        washout_results[washout_name][disease]['events'].append(metrics['n_events'])
                        washout_results[washout_name][disease]['rates'].append(metrics['event_rate'])
                    
                    print(f"      ✅ {washout_name} completed")
                    
                except Exception as e:
                    print(f"      ❌ Error in {washout_name}: {e}")
                    continue
            
            # Clean up memory
            del pi_batch, Y_batch, E_batch, pce_df_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ❌ Error loading batch {start}-{stop}: {e}")
            continue
    
    # Aggregate results across batches
    print("\n" + "="*80)
    print("AGGREGATED WASHOUT RESULTS")
    print("="*80)
    
    for washout_name, diseases in washout_results.items():
        print(f"\n{washout_name.upper()} WASHOUT:")
        print("-" * 50)
        
        for disease, metrics in diseases.items():
            aucs = [a for a in metrics['aucs'] if not pd.isna(a)]
            if aucs:
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                n_batches = len(aucs)
                
                # Calculate overall CI from individual CIs
                all_cis = [ci for ci in metrics['cis'] if ci is not None]
                if all_cis:
                    ci_lowers = [ci[0] for ci in all_cis]
                    ci_uppers = [ci[1] for ci in all_cis]
                    overall_ci_lower = np.mean(ci_lowers)
                    overall_ci_upper = np.mean(ci_uppers)
                    
                    print(f"  {disease:30} | AUC: {mean_auc:.3f}±{std_auc:.3f} | CI: [{overall_ci_lower:.3f}, {overall_ci_upper:.3f}] | Batches: {n_batches}")
                else:
                    print(f"  {disease:30} | AUC: {mean_auc:.3f}±{std_auc:.3f} | Batches: {n_batches}")
            else:
                print(f"  {disease:30} | No valid AUCs")
    
    # Save results
    print(f"\n3. SAVING RESULTS")
    results_filename = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/full_washout_results_0_400k.pt"
    torch.save(washout_results, results_filename)
    print(f"Results saved to: {results_filename}")
    
    return washout_results

if __name__ == "__main__":
    results = run_full_washout_analysis()
