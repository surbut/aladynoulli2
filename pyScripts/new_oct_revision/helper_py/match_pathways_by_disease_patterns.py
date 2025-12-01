#!/usr/bin/env python3
"""
Match Pathways Between Cohorts Based on Disease Patterns

Since pathway labels (0, 1, 2, 3) are arbitrary and disease names differ between cohorts,
we match pathways by their biological content - which diseases are enriched in each pathway.

This allows us to compare:
- "Pathway with high coronary atherosclerosis" in UKB vs MGB
- "Pathway with high diabetes" in UKB vs MGB
- etc.

Rather than comparing "Pathway 0" in UKB vs "Pathway 0" in MGB (which may be different!)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
import warnings
warnings.filterwarnings('ignore')


def extract_disease_enrichment_by_pathway(pathway_data, Y, disease_names, top_n=20):
    """
    Extract disease enrichment patterns for each pathway
    
    For each pathway, identify which diseases are most enriched (highest prevalence)
    
    Returns:
    --------
    dict: {pathway_id: [(disease_name, prevalence, enrichment_ratio), ...]}
    """
    patients = pathway_data['patients']
    unique_pathways = sorted(set([p['pathway'] for p in patients]))
    
    # Get overall disease prevalences (across all MI patients)
    all_mi_patients = [p['patient_id'] for p in patients]
    overall_disease_counts = Y[all_mi_patients, :, :].sum(axis=(0, 2))  # Sum over patients and time
    overall_prevalences = overall_disease_counts / len(all_mi_patients)
    
    pathway_disease_patterns = {}
    
    for pathway_id in unique_pathways:
        pathway_patients = [p for p in patients if p['pathway'] == pathway_id]
        patient_ids = [p['patient_id'] for p in pathway_patients]
        n_pathway = len(patient_ids)
        
        if n_pathway == 0:
            continue
        
        # Calculate disease prevalences for this pathway
        pathway_disease_counts = Y[patient_ids, :, :].sum(axis=(0, 2))
        pathway_prevalences = pathway_disease_counts / n_pathway
        
        # Calculate enrichment ratios (pathway prevalence / overall prevalence)
        enrichment_ratios = []
        for disease_idx in range(len(disease_names)):
            overall_prev = overall_prevalences[disease_idx].item()
            pathway_prev = pathway_prevalences[disease_idx].item()
            
            if overall_prev > 0.001:  # Only consider diseases with >0.1% overall prevalence
                enrichment = pathway_prev / overall_prev if overall_prev > 0 else 0
            else:
                enrichment = 0
            
            enrichment_ratios.append({
                'disease_idx': disease_idx,
                'disease_name': disease_names[disease_idx],
                'pathway_prevalence': pathway_prev,
                'overall_prevalence': overall_prev,
                'enrichment_ratio': enrichment
            })
        
        # Sort by enrichment ratio (highest first)
        enrichment_ratios.sort(key=lambda x: x['enrichment_ratio'], reverse=True)
        
        # Get top N diseases
        pathway_disease_patterns[pathway_id] = enrichment_ratios[:top_n]
    
    return pathway_disease_patterns


def match_disease_names_by_keywords(ukb_disease_name, mgb_disease_names):
    """
    Try to match disease names between cohorts using keywords with fuzzy matching
    
    Returns:
    --------
    list: Indices of MGB diseases that might match UKB disease, sorted by similarity
    """
    ukb_lower = ukb_disease_name.lower()
    
    # Extract key terms (remove common words)
    common_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                   'disease', 'disorder', 'syndrome', 'condition', 'acute', 'chronic'}
    ukb_terms = set(ukb_lower.split()) - common_words
    ukb_terms = {t for t in ukb_terms if len(t) > 2}  # Only keep meaningful terms
    
    matches_with_scores = []
    
    for i, mgb_name in enumerate(mgb_disease_names):
        mgb_lower = mgb_name.lower()
        mgb_terms = set(mgb_lower.split()) - common_words
        mgb_terms = {t for t in mgb_terms if len(t) > 2}
        
        if len(ukb_terms) == 0 or len(mgb_terms) == 0:
            continue
        
        # Check for exact word matches
        overlap = ukb_terms & mgb_terms
        exact_match_score = len(overlap) / max(len(ukb_terms), len(mgb_terms))
        
        # Check for substring matches (e.g., "myocardial" in "myocardial_infarction")
        substring_match = False
        for ukb_term in ukb_terms:
            for mgb_term in mgb_terms:
                if ukb_term in mgb_term or mgb_term in ukb_term:
                    substring_match = True
                    break
            if substring_match:
                break
        
        # Check for character overlap (fuzzy matching)
        char_overlap = 0
        ukb_chars = set(ukb_lower.replace(' ', '').replace('_', '').replace('-', ''))
        mgb_chars = set(mgb_lower.replace(' ', '').replace('_', '').replace('-', ''))
        if len(ukb_chars) > 0 and len(mgb_chars) > 0:
            char_overlap = len(ukb_chars & mgb_chars) / max(len(ukb_chars), len(mgb_chars))
        
        # Combined score
        total_score = 0.0
        if exact_match_score > 0:
            total_score += exact_match_score * 0.7
        if substring_match:
            total_score += 0.2
        if char_overlap > 0.5:
            total_score += char_overlap * 0.1
        
        if total_score > 0.2:  # Lower threshold for matching (more lenient)
            matches_with_scores.append((i, total_score))
    
    # Sort by score (highest first)
    matches_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, score in matches_with_scores]


def calculate_pathway_similarity(ukb_pathway_pattern, mgb_pathway_pattern,
                                 ukb_disease_names, mgb_disease_names):
    """
    Calculate similarity between two pathways based on disease enrichment patterns
    
    Uses cosine similarity on enrichment vectors, matching diseases by keywords
    Improved to use more diseases and better matching
    """
    # Create a combined disease space
    # Use top 20 diseases for better matching
    disease_mapping = {}
    
    # Create reverse lookup for MGB diseases
    mgb_enrichment_dict = {info['disease_idx']: info['enrichment_ratio'] 
                          for info in mgb_pathway_pattern}
    
    # Try matching UKB diseases to MGB diseases (use top 30 for better coverage)
    for ukb_disease_info in ukb_pathway_pattern[:30]:  # Top 30 diseases
        ukb_disease_name = ukb_disease_info['disease_name']
        ukb_enrichment = ukb_disease_info['enrichment_ratio']
        
        # Skip if enrichment is too low (lower threshold to capture more diseases)
        if ukb_enrichment < 1.1:  # Only consider diseases with >10% enrichment
            continue
        
        # Find matching MGB diseases
        mgb_matches = match_disease_names_by_keywords(ukb_disease_name, mgb_disease_names)
        
        if mgb_matches:
            # Use the best matching MGB disease
            for mgb_idx in mgb_matches:
                mgb_disease_name = mgb_disease_names[mgb_idx]
                mgb_enrichment = mgb_enrichment_dict.get(mgb_idx, 0)
                
                # Only add if MGB also has meaningful enrichment (lower threshold)
                if mgb_enrichment > 1.1:
                    disease_mapping[ukb_disease_name] = {
                        'ukb_enrichment': ukb_enrichment,
                        'mgb_enrichment': mgb_enrichment,
                        'mgb_disease_name': mgb_disease_name,
                        'mgb_disease_idx': mgb_idx
                    }
                    break  # Use first match
    
    # Also try reverse matching (MGB to UKB) for diseases we might have missed
    ukb_enrichment_dict = {info['disease_idx']: info['enrichment_ratio'] 
                          for info in ukb_pathway_pattern}
    
    for mgb_disease_info in mgb_pathway_pattern[:30]:
        mgb_disease_name = mgb_disease_names[mgb_disease_info['disease_idx']]
        mgb_enrichment = mgb_disease_info['enrichment_ratio']
        
        if mgb_enrichment < 1.1:
            continue
        
        # Skip if already matched
        already_matched = any(m['mgb_disease_name'] == mgb_disease_name 
                            for m in disease_mapping.values())
        if already_matched:
            continue
        
        # Find matching UKB diseases
        ukb_matches = match_disease_names_by_keywords(mgb_disease_name, ukb_disease_names)
        
        if ukb_matches:
            for ukb_idx in ukb_matches:
                ukb_disease_name = ukb_disease_names[ukb_idx]
                ukb_enrichment = ukb_enrichment_dict.get(ukb_idx, 0)
                
                if ukb_enrichment > 1.1:
                    disease_mapping[ukb_disease_name] = {
                        'ukb_enrichment': ukb_enrichment,
                        'mgb_enrichment': mgb_enrichment,
                        'mgb_disease_name': mgb_disease_name,
                        'mgb_disease_idx': mgb_disease_info['disease_idx']
                    }
                    break
    
    if len(disease_mapping) == 0:
        return 0.0, {}
    
    # Calculate multiple similarity metrics
    ukb_enrichments = np.array([d['ukb_enrichment'] for d in disease_mapping.values()])
    mgb_enrichments = np.array([d['mgb_enrichment'] for d in disease_mapping.values()])
    
    # 1. Cosine similarity (direction)
    ukb_norm = np.linalg.norm(ukb_enrichments)
    mgb_norm = np.linalg.norm(mgb_enrichments)
    
    if ukb_norm == 0 or mgb_norm == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = np.dot(ukb_enrichments, mgb_enrichments) / (ukb_norm * mgb_norm)
    
    # 2. Pearson correlation (linear relationship)
    if len(ukb_enrichments) > 1:
        correlation = np.corrcoef(ukb_enrichments, mgb_enrichments)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # 3. Rank correlation (order preservation)
    if len(ukb_enrichments) > 1:
        rank_corr, _ = spearmanr(ukb_enrichments, mgb_enrichments)
        if np.isnan(rank_corr):
            rank_corr = 0.0
    else:
        rank_corr = 0.0
    
    # Combined similarity (weighted average)
    # Weight cosine similarity more heavily, but also consider correlation
    similarity = 0.5 * cosine_sim + 0.3 * max(0, correlation) + 0.2 * max(0, rank_corr)
    
    return similarity, disease_mapping


def match_pathways_between_cohorts(ukb_pathway_data, ukb_Y, ukb_disease_names,
                                   mgb_pathway_data, mgb_Y, mgb_disease_names,
                                   top_n_diseases=20):
    """
    Match pathways between UKB and MGB based on disease enrichment patterns
    
    Returns:
    --------
    dict: {
        'pathway_matching': {(ukb_pathway_id, mgb_pathway_id): similarity_score, ...},
        'best_matches': {ukb_pathway_id: mgb_pathway_id, ...},
        'ukb_patterns': {...},
        'mgb_patterns': {...}
    }
    """
    print("="*80)
    print("MATCHING PATHWAYS BY DISEASE PATTERNS")
    print("="*80)
    print("\nExtracting disease enrichment patterns for each pathway...")
    
    # Extract disease patterns for each pathway in both cohorts
    ukb_patterns = extract_disease_enrichment_by_pathway(
        ukb_pathway_data, ukb_Y, ukb_disease_names, top_n=top_n_diseases
    )
    mgb_patterns = extract_disease_enrichment_by_pathway(
        mgb_pathway_data, mgb_Y, mgb_disease_names, top_n=top_n_diseases
    )
    
    print(f"\nUKB pathways: {sorted(ukb_patterns.keys())}")
    print(f"MGB pathways: {sorted(mgb_patterns.keys())}")
    
    # Calculate similarity between all pathway pairs
    print("\nCalculating pathway similarities...")
    pathway_similarities = {}
    pathway_mappings = {}
    
    for ukb_pathway_id in sorted(ukb_patterns.keys()):
        ukb_pattern = ukb_patterns[ukb_pathway_id]
        
        for mgb_pathway_id in sorted(mgb_patterns.keys()):
            mgb_pattern = mgb_patterns[mgb_pathway_id]
            
            similarity, disease_mapping = calculate_pathway_similarity(
                ukb_pattern, mgb_pattern,
                ukb_disease_names, mgb_disease_names
            )
            
            pathway_similarities[(ukb_pathway_id, mgb_pathway_id)] = similarity
            pathway_mappings[(ukb_pathway_id, mgb_pathway_id)] = disease_mapping
    
    # Print similarity matrix for debugging
    print("\nSimilarity Matrix (all pathway pairs):")
    header = "UKB\\MGB"
    print(f"{header:<10}", end="")
    for mgb_id in sorted(mgb_patterns.keys()):
        mgb_label = f"MGB {mgb_id}"
        print(f"{mgb_label:<12}", end="")
    print()
    print("-" * (10 + 12 * len(mgb_patterns)))
    
    for ukb_id in sorted(ukb_patterns.keys()):
        ukb_label = f"UKB {ukb_id}"
        print(f"{ukb_label:<10}", end="")
        for mgb_id in sorted(mgb_patterns.keys()):
            similarity = pathway_similarities.get((ukb_id, mgb_id), 0.0)
            print(f"{similarity:<12.3f}", end="")
        print()
    
    # Find best matches using Hungarian algorithm (optimal assignment)
    print("\nFinding best pathway matches (using optimal assignment)...")
    
    try:
        from scipy.optimize import linear_sum_assignment
        
        # Create similarity matrix
        ukb_pathways = sorted(ukb_patterns.keys())
        mgb_pathways = sorted(mgb_patterns.keys())
        
        # Create cost matrix (1 - similarity, since Hungarian minimizes cost)
        n_ukb = len(ukb_pathways)
        n_mgb = len(mgb_pathways)
        
        cost_matrix = np.ones((max(n_ukb, n_mgb), max(n_ukb, n_mgb)))
        
        for i, ukb_id in enumerate(ukb_pathways):
            for j, mgb_id in enumerate(mgb_pathways):
                similarity = pathway_similarities.get((ukb_id, mgb_id), 0.0)
                cost_matrix[i, j] = 1.0 - similarity  # Convert to cost
        
        # Find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        best_matches = {}
        used_mgb_pathways = set()
        
        for i, j in zip(row_indices, col_indices):
            if i < n_ukb and j < n_mgb:
                ukb_id = ukb_pathways[i]
                mgb_id = mgb_pathways[j]
                similarity = pathway_similarities.get((ukb_id, mgb_id), 0.0)
                
                if ukb_id not in best_matches and mgb_id not in used_mgb_pathways:
                    best_matches[ukb_id] = mgb_id
                    used_mgb_pathways.add(mgb_id)
                    print(f"  UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id} (similarity: {similarity:.3f})")
        
    except ImportError:
        # Fallback to greedy if scipy not available
        print("  (Using greedy matching - install scipy for optimal assignment)")
        best_matches = {}
        used_mgb_pathways = set()
        
        sorted_pairs = sorted(pathway_similarities.items(), key=lambda x: x[1], reverse=True)
        
        for (ukb_id, mgb_id), similarity in sorted_pairs:
            if ukb_id not in best_matches and mgb_id not in used_mgb_pathways:
                best_matches[ukb_id] = mgb_id
                used_mgb_pathways.add(mgb_id)
                print(f"  UKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id} (similarity: {similarity:.3f})")
    
    # Print disease mappings for matched pathways
    print("\nDisease pattern matches:")
    print("-" * 80)
    for ukb_id, mgb_id in best_matches.items():
        similarity = pathway_similarities[(ukb_id, mgb_id)]
        disease_mapping = pathway_mappings[(ukb_id, mgb_id)]
        
        print(f"\nUKB Pathway {ukb_id} ↔ MGB Pathway {mgb_id} (similarity: {similarity:.3f})")
        
        if len(disease_mapping) == 0:
            print(f"  ⚠️  No diseases matched! This pathway may not have clear matches.")
            print(f"  Top UKB diseases in Pathway {ukb_id}:")
            for i, disease_info in enumerate(ukb_patterns[ukb_id][:5]):
                print(f"    {i+1}. {disease_info['disease_name']} (enrichment: {disease_info['enrichment_ratio']:.2f})")
            print(f"  Top MGB diseases in Pathway {mgb_id}:")
            for i, disease_info in enumerate(mgb_patterns[mgb_id][:5]):
                print(f"    {i+1}. {disease_info['disease_name']} (enrichment: {disease_info['enrichment_ratio']:.2f})")
        else:
            # Show top matching diseases
            sorted_diseases = sorted(disease_mapping.items(), 
                                   key=lambda x: x[1]['ukb_enrichment'], reverse=True)
            
            print(f"  Matched {len(disease_mapping)} diseases. Top 5 matching diseases:")
            for ukb_disease, mapping_info in sorted_diseases[:5]:
                print(f"    UKB: {ukb_disease} (enrichment: {mapping_info['ukb_enrichment']:.2f}x)")
                print(f"    MGB: {mapping_info['mgb_disease_name']} (enrichment: {mapping_info['mgb_enrichment']:.2f}x)")
    
    return {
        'pathway_similarities': pathway_similarities,
        'best_matches': best_matches,
        'ukb_patterns': ukb_patterns,
        'mgb_patterns': mgb_patterns,
        'disease_mappings': pathway_mappings
    }


def compare_matched_pathways(ukb_pathway_data, mgb_pathway_data, pathway_matching,
                             ukb_results, mgb_results):
    """
    Compare matched pathways (e.g., UKB Pathway 0 vs MGB Pathway 2, if they're matched)
    
    Now we can compare pathway sizes, signature patterns, etc. for the matched pairs
    """
    print("\n" + "="*80)
    print("COMPARING MATCHED PATHWAYS")
    print("="*80)
    
    ukb_patients = ukb_pathway_data['patients']
    mgb_patients = mgb_pathway_data['patients']
    
    best_matches = pathway_matching['best_matches']
    similarities = pathway_matching['pathway_similarities']
    
    print("\nPathway Size Comparison (for matched pathways):")
    print(f"{'UKB Pathway':<15} {'MGB Pathway':<15} {'UKB Size':<15} {'MGB Size':<15} {'UKB %':<12} {'MGB %':<12} {'Similarity':<12}")
    print("-" * 100)
    
    ukb_labels = np.array([p['pathway'] for p in ukb_patients])
    mgb_labels = np.array([p['pathway'] for p in mgb_patients])
    
    ukb_total = len(ukb_labels)
    mgb_total = len(mgb_labels)
    
    for ukb_id, mgb_id in sorted(best_matches.items()):
        ukb_size = np.sum(ukb_labels == ukb_id)
        mgb_size = np.sum(mgb_labels == mgb_id)
        
        ukb_pct = (ukb_size / ukb_total * 100) if ukb_total > 0 else 0
        mgb_pct = (mgb_size / mgb_total * 100) if mgb_total > 0 else 0
        
        similarity = similarities[(ukb_id, mgb_id)]
        
        print(f"Pathway {ukb_id:<12} Pathway {mgb_id:<12} {ukb_size:<15,} {mgb_size:<15,} {ukb_pct:<12.1f} {mgb_pct:<12.1f} {similarity:<12.3f}")
    
    return {
        'matched_pathway_sizes': {
            (ukb_id, mgb_id): {
                'ukb_size': np.sum(ukb_labels == ukb_id),
                'mgb_size': np.sum(mgb_labels == mgb_id),
                'similarity': similarities[(ukb_id, mgb_id)]
            }
            for ukb_id, mgb_id in best_matches.items()
        }
    }


if __name__ == "__main__":
    print("Pathway Matching by Disease Patterns")
    print("This module matches pathways between cohorts based on biological content,")
    print("not arbitrary pathway index numbers.")

