#!/usr/bin/env python3
"""
Generate Supplementary Figure S11: 1-year AUCs vs PCE/PREVENT over time.

This script evaluates 1-year risk predictions at different time offsets (0-9 years)
for both Aladynoulli and PCE/PREVENT scores, showing how performance changes over time.

Usage:
    python create_S11_1year_aucs_over_time.py
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

# Add paths
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts')
from evaluatetdccode import evaluate_major_diseases_rolling_1year_roc_curves_highres


def create_s11_figure(start_idx=0, end_idx=10000, output_dir=None):
    """
    Generate S11 figure showing 1-year AUCs vs PCE/PREVENT over time.
    
    Args:
        start_idx: Start index for batch (default: 0)
        end_idx: End index for batch (default: 10000)
        output_dir: Output directory (default: results/paper_figs/supp/s11)
    """
    print("="*80)
    print("CREATING S11: 1-year AUCs vs PCE/PREVENT over time")
    print("="*80)
    
    # Set default output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'results' / 'paper_figs' / 'supp' / 's11'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Y, E, essentials
    base_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/')
    print(f"\nLoading data from: {base_path}")
    
    Y = torch.load(base_path / 'Y_tensor.pt', weights_only=False)[start_idx:end_idx]
    E = torch.load(base_path / 'E_matrix.pt', weights_only=False)[start_idx:end_idx]
    essentials = torch.load(base_path / 'model_essentials.pt', weights_only=False)
    disease_names = essentials['disease_names']
    
    print(f"Loaded Y: {Y.shape}, E: {E.shape}")
    print(f"Number of diseases: {len(disease_names)}")
    
    # Load pce_df from RDS file (has pce_goff_fuull and prevent_impute columns)
    print("\nLoading PCE/PREVENT data from RDS file...")
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    
    readRDS = robjects.r['readRDS']
    pce_data = readRDS('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/pce_df_prevent.rds')
    pce_df = pandas2ri.rpy2py(pce_data)
    pce_df = pce_df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    print(f"Loaded pce_df: {len(pce_df)} rows")
    print(f"PCE available: {'pce_goff_fuull' in pce_df.columns}")
    print(f"PREVENT available: {'prevent_impute' in pce_df.columns}")
    
    # Load pi batches for offsets 0-9
    print("\nLoading pi batches for offsets 0-9...")
    pi_base_dir = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/age_offset_local_vectorized_E_corrected/')
    pi_batches = []
    
    for k in range(10):  # Offsets 0-9
        pi_filename = f'pi_enroll_fixedphi_age_offset_{k}_sex_{start_idx}_{end_idx}_try2_withpcs_newrun_pooledall.pt'
        pi_path = pi_base_dir / pi_filename
        if pi_path.exists():
            pi_batch = torch.load(pi_path, weights_only=False)
            pi_batches.append(pi_batch)
            print(f"  Loaded offset {k}: {pi_batch.shape}")
        else:
            print(f"  Warning: {pi_filename} not found")
            raise FileNotFoundError(f"Pi file not found: {pi_path}")
    
    print(f"\nLoaded {len(pi_batches)} pi batches")
    
    # Run evaluation (this will create the plot automatically)
    print("\nEvaluating 1-year predictions at each offset...")
    print("(This computes AUCs for Aladynoulli, PCE, and PREVENT at each time offset)")
    
    results = evaluate_major_diseases_rolling_1year_roc_curves_highres(
        pi_batches, Y, E, disease_names, pce_df, 
        patient_indices=None, 
        plot_group='ASCVD'
    )
    
    # Move the generated plot to S11 output directory
    print("\nSaving output...")
    
    # The function saves to current directory, so move it
    plot_source = Path('roc_curves_ASCVD_all_years.pdf')
    plot_dest = output_dir / 'S11.pdf'
    
    if plot_source.exists():
        shutil.move(str(plot_source), str(plot_dest))
        print(f"✓ Saved S11 to: {plot_dest}")
    else:
        print(f"Warning: Plot file not found at {plot_source}")
        print("  The function may have saved it with a different name.")
        print("  Checking for alternative filenames...")
        
        # Check for alternative names
        alt_names = ['roc_curves_ASCVD.pdf', 'roc_curves_ASCVD_all_years.png']
        found = False
        for alt_name in alt_names:
            alt_path = Path(alt_name)
            if alt_path.exists():
                shutil.move(str(alt_path), str(plot_dest))
                print(f"✓ Found and moved {alt_name} to: {plot_dest}")
                found = True
                break
        
        if not found:
            print("  Could not find plot file. Please check the function output.")
    
    print("\n" + "="*80)
    print("S11 GENERATION COMPLETE")
    print("="*80)
    
    return results, plot_dest


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Supplementary Figure S11")
    parser.add_argument('--start_idx', type=int, default=0,
                        help="Start index for batch (default: 0)")
    parser.add_argument('--end_idx', type=int, default=10000,
                        help="End index for batch (default: 10000)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory (default: results/paper_figs/supp/s11)")
    
    args = parser.parse_args()
    
    create_s11_figure(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        output_dir=args.output_dir
    )

