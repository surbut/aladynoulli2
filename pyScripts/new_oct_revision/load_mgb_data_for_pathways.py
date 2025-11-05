#!/usr/bin/env python3
"""
Load MGB Data for Pathway Analysis

MGB data structure is different from UKB:
- Different disease names/coding
- Different model file format
- Different data organization

This script loads MGB data and transforms it to match pathway analysis requirements.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

def load_mgb_model_data(mgb_model_path):
    """
    Load MGB model data from .pt file
    
    MGB model structure:
    - model_state_dict: Contains lambda_ and other model parameters
    - Y: Binary disease matrix
    - disease_names: List of disease names (may be different from UKB)
    
    Returns:
    --------
    dict with keys: Y, thetas, disease_names, lambda_
    """
    print("="*80)
    print("LOADING MGB MODEL DATA")
    print("="*80)
    
    print(f"\n1. Loading MGB model from: {mgb_model_path}")
    mgb_data = torch.load(mgb_model_path, map_location=torch.device('cpu'))
    
    print(f"   Model keys: {list(mgb_data.keys())}")
    
    # Extract key components
    print(f"\n2. Extracting model components")
    
    # Lambda (signature loadings before softmax)
    if 'model_state_dict' in mgb_data:
        lambda_mgb = mgb_data['model_state_dict']['lambda_'].detach().numpy()
        print(f"   ✅ Loaded lambda: {lambda_mgb.shape}")
    else:
        raise ValueError("Could not find 'model_state_dict' in MGB model")
    
    # Y (binary disease matrix)
    if 'Y' in mgb_data:
        Y_mgb = mgb_data['Y']
        if isinstance(Y_mgb, torch.Tensor):
            Y_mgb = Y_mgb.numpy()
        print(f"   ✅ Loaded Y: {Y_mgb.shape}")
    else:
        raise ValueError("Could not find 'Y' in MGB model")
    
    # Disease names
    if 'disease_names' in mgb_data:
        disease_names_mgb = mgb_data['disease_names']
        # Convert to list if needed
        if hasattr(disease_names_mgb, 'values'):
            disease_names_mgb = disease_names_mgb.values.tolist()
        elif isinstance(disease_names_mgb, (list, tuple)):
            disease_names_mgb = list(disease_names_mgb)
        elif isinstance(disease_names_mgb, np.ndarray):
            disease_names_mgb = disease_names_mgb.tolist()
        print(f"   ✅ Loaded disease names: {len(disease_names_mgb)} diseases")
    else:
        raise ValueError("Could not find 'disease_names' in MGB model")
    
    # Compute thetas from lambda (softmax)
    print(f"\n3. Computing thetas from lambda (softmax normalization)")
    def softmax(x, axis=1):
        """Compute softmax along specified axis"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    # Lambda shape: (N, K, T) - normalize across signatures (axis=1)
    thetas_mgb = softmax(lambda_mgb, axis=1)
    print(f"   ✅ Computed thetas: {thetas_mgb.shape}")
    
    # Verify shapes match
    N_lambda, K, T = lambda_mgb.shape
    N_y, D, T_y = Y_mgb.shape
    
    if N_lambda != N_y:
        print(f"   ⚠️  WARNING: Lambda has {N_lambda} patients, Y has {N_y} patients")
    
    if T != T_y:
        print(f"   ⚠️  WARNING: Lambda has {T} timepoints, Y has {T_y} timepoints")
    
    print(f"\n4. DATA SUMMARY")
    print(f"   Patients: {N_lambda}")
    print(f"   Signatures: {K}")
    print(f"   Time points: {T}")
    print(f"   Diseases: {D}")
    
    # Show sample disease names
    print(f"\n   First 10 disease names:")
    for i, name in enumerate(disease_names_mgb[:10]):
        print(f"     {i}: {name}")
    
    return {
        'Y': Y_mgb,
        'thetas': thetas_mgb,
        'lambda': lambda_mgb,
        'disease_names': disease_names_mgb,
        'N': N_lambda,
        'K': K,
        'T': T,
        'D': D
    }


def find_disease_matches(disease_names_mgb, target_disease_name):
    """
    Find disease name matches in MGB data
    
    MGB may use different disease names than UKB, so we need flexible matching.
    
    Parameters:
    -----------
    disease_names_mgb : list
        List of disease names from MGB
    target_disease_name : str
        Disease name to search for (e.g., "myocardial infarction")
    
    Returns:
    --------
    list of tuples: (index, disease_name, match_score)
    """
    target_lower = target_disease_name.lower()
    target_words = set(target_lower.split())
    
    matches = []
    for i, disease_name in enumerate(disease_names_mgb):
        disease_lower = str(disease_name).lower()
        
        # Exact match
        if target_lower in disease_lower or disease_lower in target_lower:
            matches.append((i, disease_name, 'exact'))
        # Word overlap
        else:
            disease_words = set(disease_lower.split())
            overlap = len(target_words.intersection(disease_words))
            if overlap > 0:
                matches.append((i, disease_name, overlap))
    
    # Sort by match quality (exact first, then by overlap)
    matches.sort(key=lambda x: (x[2] != 'exact', -x[2] if isinstance(x[2], int) else 0))
    
    return matches


def create_mgb_pathway_data(mgb_data, target_disease_name, processed_ids=None):
    """
    Create pathway data structure from MGB data
    
    This transforms MGB data into the format expected by pathway analysis.
    
    Parameters:
    -----------
    mgb_data : dict
        Output from load_mgb_model_data()
    target_disease_name : str
        Disease name to analyze (will search for matches in MGB disease names)
    processed_ids : array-like, optional
        Patient IDs (EMPI) if available. If None, uses index 0..N-1
    
    Returns:
    --------
    dict with Y, thetas, disease_names, processed_ids ready for pathway analysis
    """
    print("="*80)
    print(f"PREPARING MGB DATA FOR PATHWAY ANALYSIS: {target_disease_name}")
    print("="*80)
    
    Y_mgb = mgb_data['Y']
    thetas_mgb = mgb_data['thetas']
    disease_names_mgb = mgb_data['disease_names']
    
    # Find disease matches
    print(f"\n1. Searching for '{target_disease_name}' in MGB disease names...")
    matches = find_disease_matches(disease_names_mgb, target_disease_name)
    
    if not matches:
        print(f"   ❌ No matches found for '{target_disease_name}'")
        print(f"   Available diseases (first 20):")
        for i, name in enumerate(disease_names_mgb[:20]):
            print(f"     {i}: {name}")
        return None
    
    print(f"   Found {len(matches)} potential matches:")
    for idx, name, score in matches[:5]:  # Show top 5
        print(f"     {idx}: {name} (match: {score})")
    
    # Use best match (first in sorted list)
    target_idx, target_name, match_score = matches[0]
    print(f"\n   Using: {target_name} (index {target_idx})")
    
    # Handle processed IDs
    if processed_ids is None:
        N = mgb_data['N']
        processed_ids = np.arange(N)
        print(f"\n2. No processed_ids provided - using index 0..{N-1}")
    else:
        processed_ids = np.array(processed_ids)
        print(f"\n2. Using provided processed_ids: {len(processed_ids)} patients")
        if len(processed_ids) != mgb_data['N']:
            print(f"   ⚠️  WARNING: processed_ids length ({len(processed_ids)}) != N ({mgb_data['N']})")
    
    return {
        'Y': Y_mgb,
        'thetas': thetas_mgb,
        'disease_names': disease_names_mgb,
        'processed_ids': processed_ids,
        'target_disease_name': target_name,  # Use MGB name
        'target_disease_idx': target_idx,
        'source': 'MGB'
    }


def map_mgb_to_ukb_disease_names(mgb_disease_names, ukb_disease_names=None):
    """
    Create a mapping between MGB and UKB disease names
    
    This is useful if you want to compare pathways across cohorts.
    Many diseases may not have direct matches.
    
    Parameters:
    -----------
    mgb_disease_names : list
        Disease names from MGB
    ukb_disease_names : list, optional
        Disease names from UKB (for comparison)
    
    Returns:
    --------
    dict: {mgb_name: ukb_name} or {mgb_name: None} if no match
    """
    if ukb_disease_names is None:
        # Load UKB disease names
        ukb_disease_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/disease_names.csv'
        if Path(ukb_disease_path).exists():
            ukb_df = pd.read_csv(ukb_disease_path)
            ukb_disease_names = ukb_df['x'].tolist()
        else:
            print(f"   ⚠️  UKB disease names file not found at {ukb_disease_path}")
            return {}
    
    mapping = {}
    for mgb_name in mgb_disease_names:
        matches = find_disease_matches(ukb_disease_names, mgb_name)
        if matches and matches[0][2] == 'exact':
            mapping[mgb_name] = ukb_disease_names[matches[0][0]]
        else:
            mapping[mgb_name] = None
    
    n_matched = sum(1 for v in mapping.values() if v is not None)
    print(f"   Matched {n_matched} of {len(mgb_disease_names)} diseases to UKB names")
    
    return mapping


if __name__ == "__main__":
    # Example usage
    mgb_model_path = '/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt'
    
    # Load MGB data
    mgb_data = load_mgb_model_data(mgb_model_path)
    
    # Prepare for pathway analysis
    pathway_data = create_mgb_pathway_data(
        mgb_data, 
        target_disease_name="myocardial infarction",
        processed_ids=None  # Will use indices if not provided
    )
    
    if pathway_data:
        print(f"\n✅ MGB data ready for pathway analysis!")
        print(f"   Target disease: {pathway_data['target_disease_name']}")
        print(f"   Target index: {pathway_data['target_disease_idx']}")
        print(f"   Data shapes: Y={pathway_data['Y'].shape}, thetas={pathway_data['thetas'].shape}")

