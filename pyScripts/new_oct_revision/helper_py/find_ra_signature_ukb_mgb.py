#!/usr/bin/env python3
"""
Find which signature Rheumatoid Arthritis belongs to in both UKB and MGB

This script:
1. Loads phi (signature-disease association) values from both cohorts
2. Finds Rheumatoid Arthritis in disease lists
3. Shows which signatures have the strongest association with RA
"""

import torch
import numpy as np
import sys
import os
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')

from pathway_discovery import load_full_data
from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
from find_disease_in_cohort import find_disease_flexible


def find_ra_signature_ukb(ukb_model_path=None):
    """
    Find which signature Rheumatoid Arthritis belongs to in UKB using clusters
    
    Parameters:
    -----------
    ukb_model_path : str, optional
        Path to UKB model file. If None, tries to find it automatically.
    
    Returns:
    --------
    dict with RA signature information
    """
    print("="*80)
    print("FINDING RA SIGNATURE IN UKB")
    print("="*80)
    
    # Load UKB data to get disease names
    print("\n1. Loading UKB data...")
    Y_ukb, thetas_ukb, disease_names_ukb, _ = load_full_data()
    
    # Find RA in UKB disease list
    print("\n2. Finding Rheumatoid Arthritis in UKB...")
    ra_matches = find_disease_flexible("rheumatoid arthritis", disease_names_ukb, verbose=True)
    
    if not ra_matches:
        print("❌ Could not find Rheumatoid Arthritis in UKB!")
        return None
    
    ra_name_ukb = ra_matches[0][1]
    ra_idx_ukb = ra_matches[0][0]
    print(f"✅ Found: '{ra_name_ukb}' at index {ra_idx_ukb}")
    
    # Try to load UKB model to get clusters
    print("\n3. Loading UKB model to get clusters...")
    
    # Try common UKB model paths
    if ukb_model_path is None:
        possible_paths = [
            '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_10000.pt',
            '/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_0_50000.pt',
            '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/enrollment_model_W0.0001_fulldata_sexspecific.pt',
        ]
        
        # Try first available batch model
        for path in possible_paths:
            if os.path.exists(path):
                ukb_model_path = path
                break
        else:
            ukb_model_path = None
    
    if ukb_model_path and os.path.exists(ukb_model_path):
        try:
            print(f"   Loading from: {ukb_model_path}")
            ukb_model_data = torch.load(ukb_model_path, map_location=torch.device('cpu'))
            
            # Get clusters - direct disease-to-signature mapping
            if 'clusters' in ukb_model_data:
                clusters_ukb = ukb_model_data['clusters']
                print(f"   ✅ Loaded clusters: type {type(clusters_ukb)}")
                
                # Handle different cluster formats
                ra_signature = None
                
                if isinstance(clusters_ukb, (list, np.ndarray)):
                    if len(clusters_ukb) > ra_idx_ukb:
                        ra_signature = clusters_ukb[ra_idx_ukb]
                        if hasattr(ra_signature, 'item'):
                            ra_signature = ra_signature.item()
                elif hasattr(clusters_ukb, '__getitem__'):
                    # Try direct indexing
                    try:
                        ra_signature = clusters_ukb[ra_idx_ukb]
                        if hasattr(ra_signature, 'item'):
                            ra_signature = ra_signature.item()
                    except (KeyError, IndexError):
                        pass
                
                if ra_signature is not None:
                    print(f"\n4. RA Signature Assignment:")
                    print(f"   ✅ Rheumatoid Arthritis (index {ra_idx_ukb}) belongs to Signature {ra_signature}")
                    
                    return {
                        'disease_name': ra_name_ukb,
                        'disease_idx': ra_idx_ukb,
                        'signature': int(ra_signature)
                    }
                else:
                    print(f"   ⚠️  Could not extract signature for disease index {ra_idx_ukb}")
                    print(f"   Clusters type: {type(clusters_ukb)}")
                    if hasattr(clusters_ukb, 'shape'):
                        print(f"   Clusters shape: {clusters_ukb.shape}")
                    return None
            else:
                print("   ⚠️  'clusters' not found in model")
                print(f"   Available keys: {list(ukb_model_data.keys())}")
                return None
        except Exception as e:
            print(f"   ⚠️  Error loading UKB model: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"   ⚠️  UKB model file not found")
        print(f"   Please provide ukb_model_path parameter")
        return None


def find_ra_signature_mgb(mgb_model_path='/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'):
    """
    Find which signature Rheumatoid Arthritis belongs to in MGB using clusters
    
    Parameters:
    -----------
    mgb_model_path : str
        Path to MGB model file
    
    Returns:
    --------
    dict with RA signature information
    """
    print("\n" + "="*80)
    print("FINDING RA SIGNATURE IN MGB")
    print("="*80)
    
    # Load MGB data
    print("\n1. Loading MGB data...")
    Y_mgb, thetas_mgb, disease_names_mgb, _ = load_mgb_data_from_model(mgb_model_path)
    
    # Find RA in MGB disease list
    print("\n2. Finding Rheumatoid Arthritis in MGB...")
    ra_matches = find_disease_flexible("rheumatoid arthritis", disease_names_mgb, verbose=True)
    
    if not ra_matches:
        print("❌ Could not find Rheumatoid Arthritis in MGB!")
        return None
    
    ra_name_mgb = ra_matches[0][1]
    ra_idx_mgb = ra_matches[0][0]
    print(f"✅ Found: '{ra_name_mgb}' at index {ra_idx_mgb}")
    
    # Load clusters from MGB model
    print("\n3. Loading clusters from MGB model...")
    
    try:
        mgb_data = torch.load(mgb_model_path, map_location=torch.device('cpu'))
        
        # Get clusters - direct disease-to-signature mapping
        if 'clusters' in mgb_data:
            clusters_mgb = mgb_data['clusters']
            print(f"   ✅ Loaded clusters: type {type(clusters_mgb)}")
            
            # Handle different cluster formats
            ra_signature = None
            
            if isinstance(clusters_mgb, (list, np.ndarray)):
                if len(clusters_mgb) > ra_idx_mgb:
                    ra_signature = clusters_mgb[ra_idx_mgb]
                    if hasattr(ra_signature, 'item'):
                        ra_signature = ra_signature.item()
            elif hasattr(clusters_mgb, '__getitem__'):
                # Try direct indexing
                try:
                    ra_signature = clusters_mgb[ra_idx_mgb]
                    if hasattr(ra_signature, 'item'):
                        ra_signature = ra_signature.item()
                except (KeyError, IndexError):
                    pass
            
            if ra_signature is not None:
                print(f"\n4. RA Signature Assignment:")
                print(f"   ✅ Rheumatoid Arthritis (index {ra_idx_mgb}) belongs to Signature {ra_signature}")
                
                return {
                    'disease_name': ra_name_mgb,
                    'disease_idx': ra_idx_mgb,
                    'signature': int(ra_signature)
                }
            else:
                print(f"   ⚠️  Could not extract signature for disease index {ra_idx_mgb}")
                print(f"   Clusters type: {type(clusters_mgb)}")
                if hasattr(clusters_mgb, 'shape'):
                    print(f"   Clusters shape: {clusters_mgb.shape}")
                return None
        else:
            print("   ⚠️  'clusters' not found in model")
            print(f"   Available keys: {list(mgb_data.keys())}")
            return None
    except Exception as e:
        print(f"   ⚠️  Error loading MGB model: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_ra_signatures(ukb_model_path=None, mgb_model_path='/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'):
    """
    Compare RA signatures between UKB and MGB
    """
    print("="*80)
    print("COMPARING RA SIGNATURES: UKB vs MGB")
    print("="*80)
    
    # Find RA signatures in both cohorts
    ukb_results = find_ra_signature_ukb(ukb_model_path=ukb_model_path)
    mgb_results = find_ra_signature_mgb(mgb_model_path=mgb_model_path)
    
    if ukb_results is None or mgb_results is None:
        print("\n❌ Could not find RA signatures in one or both cohorts")
        return None
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nUKB:")
    print(f"  Disease: '{ukb_results['disease_name']}' (index {ukb_results['disease_idx']})")
    print(f"  Signature: Signature {ukb_results['signature']}")
    
    print(f"\nMGB:")
    print(f"  Disease: '{mgb_results['disease_name']}' (index {mgb_results['disease_idx']})")
    print(f"  Signature: Signature {mgb_results['signature']}")
    
    # Check if signatures match
    if ukb_results['signature'] == mgb_results['signature']:
        print(f"\n✅ Signatures MATCH: Both cohorts have Signature {ukb_results['signature']}")
    else:
        print(f"\n⚠️  Signatures DIFFER:")
        print(f"   UKB: Signature {ukb_results['signature']}")
        print(f"   MGB: Signature {mgb_results['signature']}")
        print(f"   (This is expected - signature indices are arbitrary across cohorts)")
    
    return {
        'ukb': ukb_results,
        'mgb': mgb_results
    }


if __name__ == "__main__":
    # Compare RA signatures
    results = compare_ra_signatures()
    
    if results:
        print("\n✅ Analysis complete!")

