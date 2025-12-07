#!/usr/bin/env python3
"""
Extract the list of patient indices that have max_censor > 70.
This identifies the 247,207 patients used in the age 70 filtered predictions.
"""

import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract filtered patient indices')
    parser.add_argument('--censor_info_path', type=str,
                       default='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/censor_info.csv',
                       help='Path to censor_info.csv')
    parser.add_argument('--min_censor_age', type=float, default=70.0,
                       help='Minimum max_censor age (default: 70.0)')
    parser.add_argument('--output_path', type=str,
                       default=None,
                       help='Output path for patient indices (default: print to stdout)')
    args = parser.parse_args()
    
    print(f"Loading censor info from: {args.censor_info_path}")
    censor_df = pd.read_csv(args.censor_info_path)
    
    print(f"Total patients in censor_info.csv: {len(censor_df)}")
    print(f"Max censor age range: {censor_df['max_censor'].min():.1f} - {censor_df['max_censor'].max():.1f}")
    
    # Filter to patients with max_censor > min_censor_age
    mask = censor_df['max_censor'].values > args.min_censor_age
    filtered_indices = np.where(mask)[0]
    
    print(f"\nPatients with max_censor > {args.min_censor_age}: {len(filtered_indices)}")
    print(f"  Percentage: {100*len(filtered_indices)/len(censor_df):.1f}%")
    print(f"  Index range: {filtered_indices[0]} to {filtered_indices[-1]}")
    
    if args.output_path:
        # Save as numpy array
        np.save(args.output_path, filtered_indices)
        print(f"\n✓ Saved patient indices to: {args.output_path}")
        
        # Also save as CSV with additional info
        csv_path = args.output_path.replace('.npy', '.csv')
        filtered_df = censor_df.iloc[filtered_indices].copy()
        filtered_df['original_index'] = filtered_indices
        filtered_df.to_csv(csv_path, index=False)
        print(f"✓ Saved patient info to: {csv_path}")
    else:
        print(f"\nFirst 10 indices: {filtered_indices[:10]}")
        print(f"Last 10 indices: {filtered_indices[-10:]}")
        print("\nTo save, use --output_path option")
    
    return filtered_indices

if __name__ == '__main__':
    main()

