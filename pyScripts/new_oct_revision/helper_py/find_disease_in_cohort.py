#!/usr/bin/env python3
"""
Helper function to find diseases in cohorts using flexible matching
"""

def find_disease_flexible(disease_name, disease_names, verbose=True):
    """
    Find disease in disease_names list using flexible matching
    
    Parameters:
    -----------
    disease_name : str
        Disease name to search for (e.g., "Rheumatoid arthritis")
    disease_names : list
        List of all disease names in the cohort
    verbose : bool
        Whether to print matching results
        
    Returns:
    --------
    list: List of (index, matched_name) tuples, sorted by match quality
    """
    
    disease_name_lower = disease_name.lower()
    
    # Extract key terms (remove common words)
    common_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
                   'disease', 'disorder', 'syndrome', 'condition', 'acute', 'chronic'}
    
    search_terms = set(disease_name_lower.split()) - common_words
    search_terms = {t for t in search_terms if len(t) > 2}
    
    matches = []
    
    for i, name in enumerate(disease_names):
        name_lower = name.lower()
        
        # Score 1: Exact substring match (highest priority)
        if disease_name_lower in name_lower or name_lower in disease_name_lower:
            score = 100
            matches.append((i, name, score, 'exact_substring'))
            continue
        
        # Score 2: All search terms present
        name_terms = set(name_lower.split()) - common_words
        name_terms = {t for t in name_terms if len(t) > 2}
        
        if len(search_terms) > 0:
            overlap = search_terms & name_terms
            if len(overlap) == len(search_terms):  # All terms match
                score = 90
                matches.append((i, name, score, 'all_terms'))
                continue
            elif len(overlap) > 0:  # Some terms match
                score = 50 + (len(overlap) / len(search_terms)) * 40
                matches.append((i, name, score, 'partial_terms'))
                continue
        
        # Score 3: Individual term matches
        for term in search_terms:
            if term in name_lower:
                score = 30
                matches.append((i, name, score, 'single_term'))
                break
    
    # Sort by score (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Remove duplicates (keep highest score)
    seen_indices = set()
    unique_matches = []
    for idx, name, score, match_type in matches:
        if idx not in seen_indices:
            seen_indices.add(idx)
            unique_matches.append((idx, name, score, match_type))
    
    if verbose:
        print(f"\nSearching for: '{disease_name}'")
        print(f"Found {len(unique_matches)} potential matches:")
        for idx, name, score, match_type in unique_matches[:10]:  # Show top 10
            print(f"  [{idx:3d}] {name:<60} (score: {score:.1f}, type: {match_type})")
        if len(unique_matches) == 0:
            print("  ❌ No matches found!")
            print(f"\n  Searching for terms: {search_terms}")
            print(f"  Sample disease names in cohort:")
            for i, name in enumerate(disease_names[:20]):
                print(f"    [{i:3d}] {name}")
    
    return [(idx, name) for idx, name, score, match_type in unique_matches]


def find_disease_in_both_cohorts(disease_name, disease_names_ukb, disease_names_mgb, verbose=True):
    """
    Find disease in both UKB and MGB cohorts
    
    Returns:
    --------
    dict with 'ukb_matches' and 'mgb_matches' lists
    """
    
    ukb_matches = find_disease_flexible(disease_name, disease_names_ukb, verbose=verbose)
    mgb_matches = find_disease_flexible(disease_name, disease_names_mgb, verbose=verbose)
    
    return {
        'ukb_matches': ukb_matches,
        'mgb_matches': mgb_matches
    }


if __name__ == "__main__":
    # Test function
    import sys
    sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/new_oct_revision')
    
    from pathway_discovery import load_full_data
    from run_mgb_deviation_analysis_and_compare import load_mgb_data_from_model
    
    print("Loading UKB data...")
    _, _, disease_names_ukb, _ = load_full_data()
    
    print("\nLoading MGB data...")
    _, _, disease_names_mgb, _ = load_mgb_data_from_model()
    
    # Test finding diseases
    test_diseases = [
        'Rheumatoid arthritis',
        'myocardial infarction',
        'Breast cancer',
        'Type 2 diabetes'
    ]
    
    for disease in test_diseases:
        print("\n" + "="*80)
        results = find_disease_in_both_cohorts(disease, disease_names_ukb, disease_names_mgb)
        
        if results['ukb_matches']:
            print(f"\n✅ UKB: Best match: [{results['ukb_matches'][0][0]}] {results['ukb_matches'][0][1]}")
        else:
            print(f"\n❌ UKB: No matches found")
        
        if results['mgb_matches']:
            print(f"✅ MGB: Best match: [{results['mgb_matches'][0][0]}] {results['mgb_matches'][0][1]}")
        else:
            print(f"❌ MGB: No matches found")

