"""
Disease Sequence Analysis

This script analyzes detailed disease sequences before target disease onset
using granular ICD-10 codes from the UK Biobank.

It identifies:
1. Most common disease sequences before target disease
2. Signature patterns associated with different sequences
3. Temporal ordering and enrichment of pre-disease conditions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
import torch
import os

def load_icd10_data(icd_file_path='/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode.rds'):
    """
    Load ICD-10 diagnosis data from RDS or CSV file
    
    Expected format:
    - eid: patient identifier
    - diag_icd10: ICD-10 diagnosis code
    - age_diag: age at diagnosis (for RDS)
    - age: age at diagnosis (for CSV, if different)
    """
    import os
    
    # Check if file is CSV or RDS
    if icd_file_path.endswith('.csv'):
        try:
            icd_data = pd.read_csv(icd_file_path, on_bad_lines='skip', engine='python')
            print(f"✅ Loaded ICD-10 data from CSV: {len(icd_data)} diagnoses")
            
            # Check for required columns
            required_cols = ['eid', 'diag_icd10']
            if not all(col in icd_data.columns for col in required_cols):
                print(f"❌ CSV missing required columns. Found: {icd_data.columns.tolist()}")
                print(f"   Required: {required_cols}")
                return None
            
            # Handle age column (may be 'age' or 'age_diag')
            if 'age' in icd_data.columns:
                icd_data['age_diag'] = icd_data['age']
            elif 'age_diag' not in icd_data.columns:
                print("⚠️  Warning: No age column found. Creating dummy ages.")
                icd_data['age_diag'] = 50.0  # Default age
            
            print(f"   {len(icd_data['eid'].unique())} unique patients")
            print(f"   {len(icd_data['diag_icd10'].unique())} unique ICD-10 codes")
            return icd_data
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            return None
    else:
        # Try loading as RDS file
        try:
            import pyreadr
            result = pyreadr.read_r(icd_file_path)
            icd_data = result[None]  # Get the dataframe
            print(f"✅ Loaded ICD-10 data from RDS: {len(icd_data)} diagnoses")
            print(f"   {len(icd_data['eid'].unique())} unique patients")
            print(f"   {len(icd_data['diag_icd10'].unique())} unique ICD-10 codes")
            return icd_data
        except Exception as e:
            print(f"❌ Error loading RDS data: {e}")
            print("   Install pyreadr: pip install pyreadr")
            return None


def load_icd10_mapping_from_csv(mapping_file_path):
    """
    Load ICD-10 to phecode mapping from detailed CSV or RDS file
    
    Returns a dictionary mapping ICD-10 codes to phecodes
    """
    try:
        # Check if RDS or CSV
        if mapping_file_path.endswith('.rds'):
            import pyreadr
            result = pyreadr.read_r(mapping_file_path)
            mapping_df = result[None]  # Get the dataframe
            print(f"✅ Loaded ICD-10 mapping from RDS: {len(mapping_df)} codes")
        else:
            mapping_df = pd.read_csv(mapping_file_path, on_bad_lines='skip', engine='python')
            print(f"✅ Loaded ICD-10 mapping from CSV: {len(mapping_df)} codes")
        
        # Create mapping dictionary
        icd_to_phecode = {}
        for _, row in mapping_df.iterrows():
            # Try different possible column names
            icd_code = None
            if 'ICD10' in row:
                icd_code = str(row['ICD10']).strip().upper()
            elif 'diag_icd10' in row:
                icd_code = str(row['diag_icd10']).strip().upper()
            elif 'icd_code' in row:
                icd_code = str(row['icd_code']).strip().upper()
            
            if not icd_code or icd_code == 'NAN' or icd_code == 'NONE':
                continue
            
            # Get phecode and phenotype
            phecode = row.get('phecode', 'Unknown')
            phenotype = row.get('phenotype', 'Unknown')
            
            # Add mappings for both the full code and shortened versions
            icd_to_phecode[icd_code] = {
                'phecode': phecode,
                'phenotype': phenotype
            }
            
            # Also add mappings for codes without the last digit (e.g., I23 -> I2 or I236 -> I23)
            # This helps match codes that might be stored differently
            if len(icd_code) > 3:
                # Map I236 -> I23 in addition to I236
                short_code = icd_code[:3]
                if short_code not in icd_to_phecode:
                    icd_to_phecode[short_code] = {
                        'phecode': phecode,
                        'phenotype': phenotype
                    }
        
        print(f"   Created mapping for {len(icd_to_phecode)} ICD-10 codes")
        
        # Debug: print a few sample mappings
        sample_keys = list(icd_to_phecode.keys())[:5]
        print(f"   Sample mappings:")
        for key in sample_keys:
            print(f"     {key} -> {icd_to_phecode[key]['phenotype']}")
        
        return icd_to_phecode
        
    except Exception as e:
        print(f"❌ Error loading mapping file: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_icd10_mapping(mapping_file_path=None):
    """
    Create a mapping from ICD-10 codes to disease categories
    If mapping_file_path is provided, uses detailed CSV mapping.
    Otherwise uses basic chapter-based mapping.
    """
    # If detailed mapping is provided, use it
    if mapping_file_path:
        detailed_mapping = load_icd10_mapping_from_csv(mapping_file_path)
        if detailed_mapping:
            def get_category(icd_code):
                """Get category from detailed mapping"""
                if pd.isna(icd_code) or not isinstance(icd_code, str):
                    return 'Unknown'
                
                # Clean the code (remove spaces, make uppercase)
                icd_code = icd_code.strip().upper()
                
                # Try exact match first
                if icd_code in detailed_mapping:
                    return detailed_mapping[icd_code]['phenotype']
                
                # Try without suffix (e.g., 'I211' -> 'I21')
                if len(icd_code) > 3:
                    base_code = icd_code[:3]
                    if base_code in detailed_mapping:
                        return detailed_mapping[base_code]['phenotype']
                
                # Debug: print unknown codes (only first 10)
                if not hasattr(get_category, 'unknown_count'):
                    get_category.unknown_count = 0
                if get_category.unknown_count < 10:
                    print(f"DEBUG: Unknown ICD code: {icd_code}")
                    get_category.unknown_count += 1
                
                return 'Unknown'
            
            return get_category
    
    # Otherwise, use basic chapter-based mapping
    # Major ICD-10 chapters
    icd10_chapters = {
        'A': 'Infectious diseases',
        'B': 'Infectious diseases',
        'C': 'Neoplasms',
        'D0': 'Neoplasms',
        'D1': 'Neoplasms',
        'D2': 'Neoplasms',
        'D3': 'Neoplasms',
        'D4': 'Neoplasms',
        'D5': 'Blood/immune',
        'D6': 'Blood/immune',
        'D7': 'Blood/immune',
        'D8': 'Blood/immune',
        'E': 'Endocrine/metabolic',
        'F': 'Mental/behavioral',
        'G': 'Nervous system',
        'H0': 'Eye',
        'H1': 'Eye',
        'H2': 'Eye',
        'H3': 'Eye',
        'H4': 'Eye',
        'H5': 'Eye',
        'H6': 'Ear',
        'H7': 'Ear',
        'H8': 'Ear',
        'H9': 'Ear',
        'I': 'Cardiovascular',
        'J': 'Respiratory',
        'K': 'Digestive',
        'L': 'Skin',
        'M': 'Musculoskeletal',
        'N': 'Genitourinary',
        'O': 'Pregnancy',
        'P': 'Perinatal',
        'Q': 'Congenital',
        'R': 'Symptoms/signs',
        'S': 'Injury',
        'T': 'Injury',
        'V': 'External causes',
        'W': 'External causes',
        'X': 'External causes',
        'Y': 'External causes',
        'Z': 'Health status'
    }
    
    def get_category(icd_code):
        """Get category for an ICD-10 code"""
        if pd.isna(icd_code) or not isinstance(icd_code, str):
            return 'Unknown'
        
        # Try first character
        if icd_code[0] in icd10_chapters:
            return icd10_chapters[icd_code[0]]
        
        # Try first two characters
        if len(icd_code) >= 2 and icd_code[:2] in icd10_chapters:
            return icd10_chapters[icd_code[:2]]
        
        return 'Unknown'
    
    return get_category


def find_disease_sequences_before_target(icd_data, target_icd_codes, processed_ids, 
                                         min_sequence_length=2, max_sequence_length=5,
                                         time_window_years=10, get_category=None, use_phenotype_from_data=True):
    """
    Find common disease sequences before target disease
    
    Parameters:
    -----------
    icd_data : DataFrame
        ICD-10 diagnosis data with columns: eid, diag_icd10, age_diag
    target_icd_codes : list
        List of ICD-10 codes for target disease (e.g., ['I21', 'I22'] for MI)
    processed_ids : array
        Array of patient IDs from main analysis (first 400K)
    min_sequence_length : int
        Minimum number of diagnoses in a sequence
    max_sequence_length : int
        Maximum number of diagnoses in a sequence
    time_window_years : int
        Look back this many years before target disease
    """
    print(f"\n=== FINDING DISEASE SEQUENCES BEFORE TARGET ===")
    print(f"Target ICD-10 codes: {target_icd_codes}")
    print(f"Time window: {time_window_years} years before target")
    print(f"Sequence length: {min_sequence_length}-{max_sequence_length} diagnoses")
    
    # Filter to patients in our analysis
    icd_data_filtered = icd_data[icd_data['eid'].isin(processed_ids)].copy()
    print(f"Filtered to {len(icd_data_filtered)} diagnoses in analysis cohort")
    
    # Convert age to numeric (handle different formats)
    # age_diag might be "81.51088 days" or just a number
    def parse_age(age_str):
        """Parse age from various formats"""
        if pd.isna(age_str):
            return np.nan
        
        age_str = str(age_str).strip()
        
        # Try to extract numeric part
        import re
        match = re.search(r'([\d.]+)', age_str)
        if match:
            age_val = float(match.group(1))
            
            # If it says "days", convert to years
            if 'day' in age_str.lower():
                age_val = age_val / 365.25
            
            return age_val
        
        return np.nan
    
    icd_data_filtered['age_numeric'] = icd_data_filtered['age_diag'].apply(parse_age)
    
    # Drop rows with missing age or diagnosis code
    icd_data_filtered = icd_data_filtered.dropna(subset=['age_numeric', 'diag_icd10'])
    print(f"After removing missing data: {len(icd_data_filtered)} diagnoses")
    
    # Ensure diag_icd10 is string type and strip whitespace
    icd_data_filtered['diag_icd10'] = icd_data_filtered['diag_icd10'].astype(str).str.strip()
    
    # Find patients with target disease
    # Use startswith to match both "I21" and "I21.9" for "I21"
    target_mask = icd_data_filtered['diag_icd10'].str.startswith(tuple(target_icd_codes), na=False)
    target_patients = icd_data_filtered[target_mask].copy()
    
    # Debug: print some sample ICD codes to verify
    print(f"Sample ICD codes in data: {icd_data_filtered['diag_icd10'].unique()[:10].tolist()}")
    
    # Debug: check if target codes exist
    all_icd_codes = icd_data_filtered['diag_icd10'].unique()
    print(f"Looking for codes starting with: {target_icd_codes}")
    matching_codes = [code for code in all_icd_codes if any(code.startswith(target) for target in target_icd_codes)]
    print(f"Found {len(matching_codes)} matching codes: {matching_codes[:10]}")
    
    print(f"Found {len(target_patients)} target disease diagnoses")
    print(f"In {len(target_patients['eid'].unique())} unique patients")
    
    # Get first occurrence of target disease for each patient
    target_first = target_patients.groupby('eid')['age_numeric'].min().reset_index()
    target_first.columns = ['eid', 'target_age']
    
    print(f"Analyzing {len(target_first)} patients with target disease")
    
    # For each patient, find diagnoses before target
    sequences = []
    sequence_categories = []
    
    # Check if phenotype column exists in the data (from merged RDS)
    has_phenotype_column = 'phenotype' in icd_data_filtered.columns
    
    for idx, row in target_first.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing patient {idx}/{len(target_first)}...")
        
        eid = row['eid']
        target_age = row['target_age']
        
        # Get all diagnoses for this patient before target
        patient_diagnoses = icd_data_filtered[
            (icd_data_filtered['eid'] == eid) & 
            (icd_data_filtered['age_numeric'] < target_age) &
            (icd_data_filtered['age_numeric'] >= target_age - time_window_years)
        ].copy()
        
        # Sort by age
        patient_diagnoses = patient_diagnoses.sort_values('age_numeric')
        
        # Get sequence of ICD-10 codes
        icd_sequence = patient_diagnoses['diag_icd10'].tolist()
        
        # Get sequence of categories
        if has_phenotype_column:
            # Use phenotype column directly from the RDS data
            category_sequence = patient_diagnoses['phenotype'].tolist()
        else:
            # Fall back to mapping function
            if get_category is None:
                get_category = create_icd10_mapping()
            category_sequence = [get_category(code) for code in icd_sequence]
        
        if len(icd_sequence) >= min_sequence_length:
            # Store patient info and sequence
            sequences.append({
                'eid': eid,
                'target_age': target_age,
                'icd_sequence': tuple(icd_sequence[:max_sequence_length]),
                'category_sequence': tuple(category_sequence[:max_sequence_length]),
                'n_diagnoses': len(icd_sequence),
                'time_to_target': target_age - patient_diagnoses['age_numeric'].min()
            })
    
    sequences_df = pd.DataFrame(sequences)
    print(f"\nFound {len(sequences_df)} patients with sufficient pre-target disease history")
    
    # Find most common sequences
    print(f"\n=== MOST COMMON DISEASE CATEGORY SEQUENCES ===")
    category_sequence_counts = Counter(sequences_df['category_sequence'])
    
    print(f"Top 20 most common category sequences:")
    for i, (seq, count) in enumerate(category_sequence_counts.most_common(20)):
        pct = count / len(sequences_df) * 100
        seq_str = ' → '.join(seq)
        print(f"  {i+1}. {seq_str}")
        print(f"     {count} patients ({pct:.1f}%)")
    
    # Analyze 2-disease sequences (bigrams)
    print(f"\n=== MOST COMMON 2-DISEASE SEQUENCES (BIGRAMS) ===")
    bigrams = []
    for seq in sequences_df['category_sequence']:
        if len(seq) >= 2:
            for i in range(len(seq) - 1):
                bigrams.append((seq[i], seq[i+1]))
    
    bigram_counts = Counter(bigrams)
    print(f"Top 20 most common bigrams:")
    for i, (bigram, count) in enumerate(bigram_counts.most_common(20)):
        print(f"  {i+1}. {bigram[0]} → {bigram[1]}: {count} occurrences")
    
    # Analyze disease category enrichment
    print(f"\n=== DISEASE CATEGORY ENRICHMENT BEFORE TARGET ===")
    all_categories = []
    for seq in sequences_df['category_sequence']:
        all_categories.extend(seq)
    
    category_counts = Counter(all_categories)
    total_diagnoses = len(all_categories)
    
    print(f"Disease categories ranked by frequency:")
    for i, (category, count) in enumerate(category_counts.most_common()):
        pct = count / total_diagnoses * 100
        print(f"  {i+1}. {category}: {count} ({pct:.1f}%)")
    
    return sequences_df, category_sequence_counts, bigram_counts


def link_sequences_to_signatures(sequences_df, thetas, processed_ids, 
                                 signature_threshold=0.1, years_before=10):
    """
    Link disease sequences to signature patterns using DEVIATIONS from population average
    
    For each sequence, identify which signatures were elevated relative to age-matched population
    
    Parameters:
    - years_before: Number of years before target disease to analyze (default: 10 for 10-year lookback)
    """
    print(f"\n=== LINKING SEQUENCES TO SIGNATURE PATTERNS (DEVIATION METHOD) ===")
    print(f"Using {years_before}-year lookback window")
    
    # Create mapping from eid to index in thetas
    eid_to_idx = {eid: idx for idx, eid in enumerate(processed_ids)}
    
    # For each patient, get their signature trajectory before target
    sequences_with_sigs = []
    
    # Calculate population reference (average across all patients at each timepoint)
    population_reference = np.mean(thetas, axis=0)  # Shape: (K, T)
    
    for idx, row in sequences_df.iterrows():
        eid = row['eid']
        target_age = row['target_age']
        
        if eid not in eid_to_idx:
            continue
        
        patient_idx = eid_to_idx[eid]
        
        # Get signature trajectory (K signatures x T timepoints)
        theta_patient = thetas[patient_idx, :, :]  # Shape: (K, T)
        
        # Get signatures at target age (or just before)
        target_time_idx = min(int(target_age - 30), theta_patient.shape[1] - 1)
        
        if target_time_idx < years_before:
            continue
        
        # Get signatures in years_before years before target
        start_idx = max(0, target_time_idx - years_before)
        pre_target_sigs = theta_patient[:, start_idx:target_time_idx]  # Shape: (K, years_before)
        
        # Calculate population reference for same time window
        ref_sigs = population_reference[:, start_idx:target_time_idx]  # Shape: (K, years_before)
        
        # Calculate deviations from population reference at each timepoint
        sig_deviations_temporal = pre_target_sigs - ref_sigs  # Shape: (K, years_before)
        
        # Flatten for clustering consistency (matching pathway discovery method)
        sig_deviations_flattened = sig_deviations_temporal.flatten()  # Shape: (K*years_before,)
        
        # Also calculate average over time for summary/display purposes
        avg_pre_target_sigs = np.mean(pre_target_sigs, axis=1)
        avg_ref_sigs = np.mean(ref_sigs, axis=1)
        avg_sig_deviations = avg_pre_target_sigs - avg_ref_sigs
        
        # Identify elevated signatures (above threshold deviation)
        elevated_sigs = np.where(avg_sig_deviations > signature_threshold)[0]
        
        # Get top 3 signatures by deviation (not raw loading)
        top_sigs = np.argsort(avg_sig_deviations)[::-1][:3]
        
        sequences_with_sigs.append({
            'eid': eid,
            'category_sequence': row['category_sequence'],
            'icd_sequence': row['icd_sequence'],
            'elevated_sigs': tuple(elevated_sigs),
            'top_3_sigs': tuple(top_sigs),
            'sig_loadings': avg_pre_target_sigs,  # Keep raw for reference
            'sig_deviations': avg_sig_deviations,  # Averaged deviations (for display)
            'sig_deviations_temporal': sig_deviations_temporal,  # Full temporal deviations (K x years_before)
            'sig_deviations_flattened': sig_deviations_flattened,  # Flattened for clustering consistency
            'target_age': target_age,
            'start_age': 30 + start_idx,
            'dominant_sig': top_sigs[0]
        })
    
    sequences_with_sigs_df = pd.DataFrame(sequences_with_sigs)
    
    print(f"Linked {len(sequences_with_sigs_df)} sequences to signature patterns")
    print(f"\nAge Matching:")
    print(f"  - All patients developed {target_disease} at their respective ages (stored in target_age)")
    print(f"  - Signature deviations calculated relative to age-matched population reference")
    print(f"  - Deviations computed for {years_before}-year window before diagnosis")
    print(f"\nDeviation Storage:")
    print(f"  - Flattened: K×{years_before} = {sequences_with_sigs_df['sig_deviations_flattened'].iloc[0].shape[0]} features (for clustering consistency)")
    print(f"  - Averaged: K = 21 averaged deviations (for summary stats)")
    
    # Analyze signature patterns by sequence type
    print(f"\n=== SIGNATURE PATTERNS BY DISEASE SEQUENCE ===")
    print(f"(Displaying averaged deviations for interpretability)")
    
    # Group by category sequence
    for seq, group in sequences_with_sigs_df.groupby('category_sequence'):
        if len(group) >= 50:  # Only show sequences with sufficient sample size
            seq_str = ' → '.join(seq)
            print(f"\nSequence: {seq_str} ({len(group)} patients)")
            
            # Find most common signature patterns
            dominant_sig_counts = Counter(group['dominant_sig'])
            print(f"  Most common dominant signatures:")
            for sig_idx, count in dominant_sig_counts.most_common(5):
                pct = count / len(group) * 100
                print(f"    Signature {sig_idx}: {count} patients ({pct:.1f}%)")
            
            # Average signature deviations for this sequence
            avg_sig_deviations = np.mean(np.vstack(group['sig_deviations'].values), axis=0)
            top_sigs_for_seq = np.argsort(avg_sig_deviations)[::-1][:5]
            print(f"  Average top 5 signature deviations:")
            for rank, sig_idx in enumerate(top_sigs_for_seq):
                print(f"    {rank+1}. Signature {sig_idx}: {avg_sig_deviations[sig_idx]:.3f}")
    
    return sequences_with_sigs_df


def visualize_sequence_signature_relationships(sequences_with_sigs_df, top_n_sequences=10, 
                                               output_dir='pathway_analysis_output', 
                                               target_disease="myocardial infarction"):
    """
    Create visualizations of relationships between disease sequences and signatures
    Now includes temporal signature deviation plots with colors
    """
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    # Get most common sequences
    sequence_counts = Counter(sequences_with_sigs_df['category_sequence'])
    top_sequences = [seq for seq, _ in sequence_counts.most_common(top_n_sequences)]
    
    # Filter to top sequences
    df_filtered = sequences_with_sigs_df[
        sequences_with_sigs_df['category_sequence'].isin(top_sequences)
    ]
    
    # NEW: Create temporal deviation plots for top sequences
    n_top_sequences = min(6, len(top_sequences))  # Show top 6 sequences
    fig_temporal = plt.figure(figsize=(20, 12))
    
    for seq_idx, seq in enumerate(top_sequences[:n_top_sequences]):
        seq_data = df_filtered[df_filtered['category_sequence'] == seq]
        
        if len(seq_data) == 0:
            continue
        
        # Calculate average temporal deviations for this sequence
        temporal_deviations_list = []
        for _, row in seq_data.iterrows():
            temporal_deviations_list.append(row['sig_deviations_temporal'])
        
        avg_temporal_deviations = np.mean(np.array(temporal_deviations_list), axis=0)  # Shape: (K, years_before)
        
        ax = fig_temporal.add_subplot(2, 3, seq_idx + 1)
        
        K, T = avg_temporal_deviations.shape
        years_before = T
        
        # Create time axis (years before target disease)
        time_axis = np.arange(-years_before, 0, 1)
        
        # Create stacked area plot
        cumulative = np.zeros(T)
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, K)))
        
        # Plot top 10 signatures (most variable or highest average deviation)
        sig_variance = np.var(avg_temporal_deviations, axis=1)
        top_sigs_to_plot = np.argsort(sig_variance)[::-1][:min(10, K)]
        
        for i, sig_idx in enumerate(top_sigs_to_plot):
            sig_values = avg_temporal_deviations[sig_idx, :]
            ax.fill_between(time_axis, cumulative, cumulative + sig_values, 
                           color=colors[sig_idx % len(colors)], 
                           alpha=0.7, label=f'Sig {sig_idx}')
            cumulative += sig_values
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        seq_str = ' → '.join(seq[:2]) + (' ...' if len(seq) > 2 else '')
        ax.set_title(f'{seq_str}\n(n={len(seq_data)} patients)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Years Before Target Disease')
        ax.set_ylabel('Signature Deviation from Population')
        ax.grid(True, alpha=0.3)
        
        if seq_idx == 0:  # Only show legend on first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
    
    fig_temporal.suptitle(f'Temporal Signature Deviations by Disease Sequence\n{target_disease} (10-Year Lookback)', 
                          fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save temporal plot
    os.makedirs(output_dir, exist_ok=True)
    filename_temporal = f'{output_dir}/signature_deviations_temporal_by_sequence_{target_disease.replace(" ", "_")}.pdf'
    plt.savefig(filename_temporal, dpi=300, bbox_inches='tight')
    print(f"   Saved temporal signature deviation plot: {filename_temporal}")
    plt.close()
    
    # Original 4-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Heatmap: Signature loadings by sequence
    ax1 = axes[0, 0]
    
    # Create matrix of average signature deviations per sequence
    K = len(sequences_with_sigs_df['sig_deviations'].iloc[0])
    sig_matrix = []
    seq_labels = []
    
    for seq in top_sequences:
        seq_data = df_filtered[df_filtered['category_sequence'] == seq]
        avg_deviations = np.mean(np.vstack(seq_data['sig_deviations'].values), axis=0)
        sig_matrix.append(avg_deviations)
        seq_labels.append(' → '.join(seq[:3]))  # Truncate long sequences
    
    sig_matrix = np.array(sig_matrix)
    
    im1 = ax1.imshow(sig_matrix, cmap='RdYlBu_r', aspect='auto')
    ax1.set_yticks(range(len(seq_labels)))
    ax1.set_yticklabels(seq_labels, fontsize=8)
    ax1.set_xlabel('Signature Index')
    ax1.set_ylabel('Disease Sequence')
    ax1.set_title('Average Signature Deviations by Disease Sequence', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Signature Deviation')
    
    # 2. Dominant signature distribution by sequence
    ax2 = axes[0, 1]
    
    # For each sequence, get distribution of dominant signatures
    dominant_sig_data = []
    for seq in top_sequences[:5]:  # Top 5 sequences
        seq_data = df_filtered[df_filtered['category_sequence'] == seq]
        dominant_sigs = Counter(seq_data['dominant_sig'])
        dominant_sig_data.append(dominant_sigs)
    
    # Create stacked bar chart
    all_sigs = set()
    for d in dominant_sig_data:
        all_sigs.update(d.keys())
    all_sigs = sorted(all_sigs)
    
    x_pos = np.arange(len(top_sequences[:5]))
    bottom = np.zeros(len(top_sequences[:5]))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_sigs)))
    
    for sig_idx, color in zip(all_sigs, colors):
        heights = []
        for dominant_sig_dict in dominant_sig_data:
            heights.append(dominant_sig_dict.get(sig_idx, 0))
        
        ax2.bar(x_pos, heights, bottom=bottom, label=f'Sig {sig_idx}', 
                color=color, alpha=0.8)
        bottom += heights
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([' → '.join(seq[:2]) for seq in top_sequences[:5]], 
                         rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Number of Patients')
    ax2.set_title('Dominant Signature Distribution by Sequence', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 3. Sequence length distribution
    ax3 = axes[1, 0]
    
    sequence_lengths = df_filtered['category_sequence'].apply(len)
    ax3.hist(sequence_lengths, bins=range(2, 8), alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Number of Diagnoses in Sequence')
    ax3.set_ylabel('Number of Patients')
    ax3.set_title('Distribution of Sequence Lengths', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Most discriminating signatures across sequences
    ax4 = axes[1, 1]
    
    # Calculate variance in signature deviation across sequences
    sig_variance = np.var(sig_matrix, axis=0)
    top_var_sigs = np.argsort(sig_variance)[::-1][:10]
    
    ax4.bar(range(len(top_var_sigs)), sig_variance[top_var_sigs], 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(top_var_sigs)))
    ax4.set_xticklabels([f'Sig {i}' for i in top_var_sigs], rotation=45)
    ax4.set_ylabel('Variance in Deviations Across Sequences')
    ax4.set_title('Signatures that Most Differentiate Sequences (by Deviation Variance)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{output_dir}/sequence_signature_relationships_{target_disease.replace(" ", "_")}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved sequence-signature relationship plot: {filename}")
    plt.close()


def analyze_disease_sequences_for_target(target_disease_name, target_icd_codes, 
                                        thetas, processed_ids, 
                                        icd_file_path=None,
                                        mapping_file_path=None,
                                        output_dir='pathway_analysis_output'):
    """
    Complete analysis of disease sequences before target disease
    
    Parameters:
    - icd_file_path: Path to RDS file with patient ICD-10 diagnosis data
    - mapping_file_path: Path to CSV file with ICD-10 to phecode mapping (optional, uses detailed mapping if provided)
    """
    print(f"="*80)
    print(f"DISEASE SEQUENCE ANALYSIS: {target_disease_name.upper()}")
    print(f"="*80)
    
    # Load ICD-10 patient diagnosis data (still from RDS)
    if icd_file_path is None:
        icd_file_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode.rds'
    
    icd_data = load_icd10_data(icd_file_path)
    
    if icd_data is None:
        return None
    
    # Load mapping if provided
    get_category = create_icd10_mapping(mapping_file_path)
    
    # Find disease sequences
    sequences_df, category_counts, bigram_counts = find_disease_sequences_before_target(
        icd_data, target_icd_codes, processed_ids, get_category=get_category
    )
    
    # Link sequences to signatures
    sequences_with_sigs_df = link_sequences_to_signatures(
        sequences_df, thetas, processed_ids
    )
    
    # Visualize
    visualize_sequence_signature_relationships(
        sequences_with_sigs_df, 
        output_dir=output_dir,
        target_disease=target_disease_name
    )
    
    return {
        'sequences_df': sequences_df,
        'sequences_with_sigs_df': sequences_with_sigs_df,
        'category_counts': category_counts,
        'bigram_counts': bigram_counts
    }


if __name__ == "__main__":
    print("Disease Sequence Analysis Script")
    print("Analyzes granular disease sequences before target disease onset")


