"""
Compare ICD-10 code aggregation: Aladynoulli (via PheCodes) vs Delphi (direct ICD-10)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Major diseases and their phenotype names (from fig5utils.py

# Load major_diseases mapping from evaluate_major_disease_wsex_rolling_tte.py
major_diseases = {
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
              'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
    'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'],
    'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
    'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
    'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
    'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
    'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
    'Osteoporosis': ['Osteoporosis NOS'],
    'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
    'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
    'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],
    'Prostate_Cancer': ['Cancer of prostate'],
    'Lung_Cancer': ['Cancer of bronchus; lung'],
    'Bladder_Cancer': ['Malignant neoplasm of bladder'],
    'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
    'Depression': ['Major depressive disorder'],
    'Anxiety': ['Anxiety disorder'],
    'Bipolar_Disorder': ['Bipolar'],
    'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
    'Psoriasis': ['Psoriasis vulgaris'],
    'Ulcerative_Colitis': ['Ulcerative colitis'],
    'Crohns_Disease': ['Regional enteritis'],
    'Asthma': ['Asthma'],
    'Parkinsons': ["Parkinson's disease"],
    'Multiple_Sclerosis': ['Multiple sclerosis'],
    'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
}

def load_phecode_mapping(file_path):
    """Load the PheCode mapping file (CSV or RDS)"""
    file_path = Path(file_path)
    print(f"Loading PheCode mapping from: {file_path}")
    
    if file_path.suffix == '.rds':
        try:
            import pyreadr
            result = pyreadr.read_r(str(file_path))
            df = result[None]
            print(f"Loaded {len(df)} rows from RDS")
        except ImportError:
            raise ImportError("pyreadr is required to read .rds files. Install with: pip install pyreadr")
        except Exception as e:
            raise Exception(f"Error loading RDS file: {e}")
    else:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from CSV")
    
    print(f"Columns: {df.columns.tolist()}")
    return df

def map_diseases_to_phecodes(subset_df, full_df, major_diseases):
    """
    Map major diseases to their Phecodes using subset file, then count aggregated top-level ICD-10 codes 
    from the full file. Delphi uses top-level ICD-10 codes, so we extract the first 3 characters for fair comparison.
    
    Args:
        subset_df: Subset file (348 rows) - used to find which Phecodes match our diseases
        full_df: Full file (88K rows) - used to count all top-level ICD-10 codes per PheCode
    """
    results = []
    
    for disease, phenotype_names in major_diseases.items():
        # Step 1: Find matching Phecodes using the SUBSET file
        phecodes_found = set()
        
        for phenotype_name in phenotype_names:
            # Use the same matching approach as R2 notebook: str.contains with regex=False
            matches = subset_df[
                subset_df['phenotype'].str.contains(phenotype_name, case=False, na=False, regex=False)
            ]
            
            if len(matches) > 0:
                # Get unique Phecodes from matches
                matched_phecodes = matches['phecode'].unique()
                if disease == 'ASCVD':  # Debug for ASCVD
                    print(f"    DEBUG: '{phenotype_name}' matched {len(matches)} rows, Phecodes: {sorted(matched_phecodes)}")
                    # Show sample matching phenotype names
                    sample_phenotypes = matches['phenotype'].unique()[:3]
                    print(f"      Sample matching phenotypes: {list(sample_phenotypes)}")
                for phecode in matched_phecodes:
                    phecodes_found.add(phecode)
            else:
                if disease == 'ASCVD':  # Debug for ASCVD
                    print(f"    DEBUG: '{phenotype_name}' - NO MATCH FOUND")
                    # Show similar phenotype names for debugging
                    if 'angina' in phenotype_name.lower():
                        similar = subset_df[subset_df['phenotype'].str.contains('angina', case=False, na=False, regex=False)]
                        if len(similar) > 0:
                            print(f"      Similar phenotypes with 'angina': {similar['phenotype'].unique().tolist()[:5]}")
                    # Also check what phenotypes exist for phecode 411.1
                    phecode_411_1_rows = subset_df[subset_df['phecode'] == 411.1]
                    if len(phecode_411_1_rows) > 0:
                        print(f"      Phenotypes for PheCode 411.1: {phecode_411_1_rows['phenotype'].unique().tolist()}")
        
        # Debug: Show what was found before collapsing
        if disease == 'ASCVD':
            print(f"    DEBUG: Total phecodes found before collapsing: {len(phecodes_found)}")
            print(f"    DEBUG: Phecodes: {sorted(phecodes_found)}")
        
        # Collapse sub-Phecodes to parent Phecodes
        # e.g., 250.2, 250.21, 250.22, 250.24 -> just use 250.2
        parent_phecodes = set()
        phecodes_list = sorted(list(phecodes_found), key=lambda x: (len(str(x)), str(x)))
        
        for pc in phecodes_list:
            pc_str = str(pc)
            # Check if this PheCode is a child of any other PheCode in our set
            is_child = False
            for other_pc in phecodes_list:
                if other_pc != pc:
                    other_pc_str = str(other_pc)
                    # Check if pc is a child of other_pc (pc starts with other_pc + ".")
                    if pc_str.startswith(other_pc_str + "."):
                        is_child = True
                        break
            # If it's not a child, it's a parent (or standalone)
            if not is_child:
                parent_phecodes.add(pc)
        
        # If we have sub-Phecodes, use only parent Phecodes
        # Otherwise, use all found Phecodes
        if len(parent_phecodes) > 0 and len(parent_phecodes) < len(phecodes_found):
            phecodes_to_use = parent_phecodes
            print(f"  Note: Collapsing sub-Phecodes to parent Phecodes: {sorted(phecodes_found)} -> {sorted(phecodes_to_use)}")
        else:
            phecodes_to_use = phecodes_found
        
        # Step 2: For each parent PheCode, get ALL ICD-10 codes from the FULL file
        phecode_details = []
        phecode_top_level_codes = {}  # Store which top-level codes map to each PheCode
        phecode_all_codes = {}  # Store which ALL codes map to each PheCode
        phecode_info_top = {}  # Store phecode -> count of unique top-level ICD-10 codes
        phecode_info_all = {}  # Store phecode -> count of ALL unique ICD-10 codes
        
        for parent_pc in sorted(phecodes_to_use):
            # Get all rows for this parent PheCode AND any sub-Phecodes from the FULL file
            parent_pc_str = str(parent_pc)
            # Match exact PheCode or sub-Phecodes that start with parent + "."
            all_related_rows = full_df[
                (full_df['phecode'] == parent_pc) | 
                (full_df['phecode'].astype(str).str.startswith(parent_pc_str + ".", na=False))
            ]
            
            # Handle both ICD10.x and ICD10.y if present, otherwise use ICD10
            icd10_cols = [col for col in all_related_rows.columns if 'ICD10' in col]
            if not icd10_cols:
                continue
            
            # Extract ALL ICD-10 codes (full codes) and top-level codes (first 3 characters)
            all_codes = set()
            top_level_codes = set()
            
            for col in icd10_cols:
                icd10_codes = all_related_rows[col].dropna().astype(str)
                for code in icd10_codes:
                    # Clean the code (remove dots, strip whitespace)
                    clean_code = code.replace('.', '').strip()
                    if len(clean_code) >= 3:
                        # Add full code
                        all_codes.add(clean_code)
                        # Add top-level code (first 3 characters)
                        top_level_codes.add(clean_code[:3])
            
            # Count unique codes
            icd_count_all = len(all_codes)
            icd_count_top = len(top_level_codes)
            top_level_codes_list = sorted(list(top_level_codes))
            all_codes_list = sorted(list(all_codes))
            
            if icd_count_all > 0:
                # Store the counts and actual codes for this parent PheCode
                phecode_info_top[parent_pc] = icd_count_top
                phecode_info_all[parent_pc] = icd_count_all
                phecode_top_level_codes[parent_pc] = top_level_codes_list
                phecode_all_codes[parent_pc] = all_codes_list
                phecode_details.append(f"{parent_pc}(top:{icd_count_top}, all:{icd_count_all})")
        
        # Count UNIQUE ICD-10 codes across all Phecodes for this disease (not sum, to avoid double-counting)
        # Collect all unique codes from all Phecodes
        all_disease_top_level_codes = set()
        all_disease_all_codes = set()
        for pc in phecodes_to_use:
            all_disease_top_level_codes.update(phecode_top_level_codes.get(pc, []))
            all_disease_all_codes.update(phecode_all_codes.get(pc, []))
        
        total_top_level_icd10 = len(all_disease_top_level_codes)
        total_all_icd10 = len(all_disease_all_codes)
        n_phecodes = len(phecodes_to_use)
        
        results.append({
            'Disease': disease,
            'N_Phecodes': n_phecodes,
            'N_top_level_ICD10': int(total_top_level_icd10),  # Top-level (3-character) ICD-10 codes
            'N_all_ICD10': int(total_all_icd10),  # ALL ICD-10 codes (full codes)
            'ICD10_per_PheCode_avg': total_all_icd10 / n_phecodes if n_phecodes > 0 else 0,
            'Phecodes_with_counts': ', '.join(phecode_details) if phecode_details else 'Not found',
            'Top_level_ICD10_codes': '; '.join([f"{pc}: {','.join(codes[:10])}{'...' if len(codes) > 10 else ''}" for pc, codes in phecode_top_level_codes.items()]) if phecode_top_level_codes else 'Not found',
            'Reduction_factor': total_all_icd10 / n_phecodes if n_phecodes > 0 else np.nan
        })
        
        print(f"\n{disease}:")
        print(f"  Phenotypes: {phenotype_names}")
        print(f"  Phecodes found in subset: {sorted(phecodes_found)}")
        print(f"  Parent Phecodes used: {sorted(phecodes_to_use)}")
        for pc in sorted(phecodes_to_use):
            top_codes = phecode_top_level_codes.get(pc, [])
            all_codes_list = phecode_all_codes.get(pc, [])
            count_top = phecode_info_top.get(pc, 0)
            count_all = phecode_info_all.get(pc, 0)
            print(f"    PheCode {pc}:")
            print(f"      Top-level ICD-10 codes: {count_top} → {', '.join(top_codes[:10])}{'...' if len(top_codes) > 10 else ''}")
            print(f"      ALL ICD-10 codes: {count_all} → {', '.join(all_codes_list[:10])}{'...' if len(all_codes_list) > 10 else ''}")
        print(f"  Total top-level ICD-10 codes (Delphi top-level): {int(total_top_level_icd10)}")
        print(f"  Total ALL ICD-10 codes (Delphi all codes): {int(total_all_icd10)}")
        print(f"  Aladynoulli uses: {n_phecodes} Phecodes")
        print(f"  Reduction (all codes): {int(total_all_icd10)} ICD-10 codes → {n_phecodes} Phecodes")
        if n_phecodes > 0:
            print(f"  Average ICD-10 codes per PheCode: {total_all_icd10 / n_phecodes:.1f}")
    
    return pd.DataFrame(results)

def create_plots(results_df, output_dir):
    """Create visualization plots comparing Aladynoulli vs Delphi approaches"""
    
    # Filter out diseases with no matches
    plot_df = results_df[results_df['N_Phecodes'] > 0].copy()
    plot_df = plot_df.sort_values('N_all_ICD10', ascending=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Side-by-side bar chart comparing Phecodes vs ICD-10 codes
    ax1 = plt.subplot(2, 2, 1)
    y_pos = np.arange(len(plot_df))
    bar_width = 0.35
    
    bars1 = ax1.barh(y_pos - bar_width/2, plot_df['N_Phecodes'], 
                     bar_width, label='Aladynoulli (Phecodes)', 
                     color='#2c7fb8', alpha=0.8)
    bars2 = ax1.barh(y_pos + bar_width/2, plot_df['N_all_ICD10'], 
                     bar_width, label='Delphi (ALL ICD-10 codes)', 
                     color='#f03b20', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(plot_df['Disease'], fontsize=9)
    ax1.set_xlabel('Number of Predictions Needed', fontsize=11, fontweight='bold')
    ax1.set_title('Aladynoulli vs Delphi: Number of Predictions per Disease', 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, plot_df['N_all_ICD10'].max() * 1.1)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        if row['N_Phecodes'] > 0:
            ax1.text(row['N_Phecodes'], i - bar_width/2, f"{int(row['N_Phecodes'])}", 
                    va='center', ha='right', fontsize=8, fontweight='bold')
        if row['N_all_ICD10'] > 0:
            ax1.text(row['N_all_ICD10'], i + bar_width/2, f"{int(row['N_all_ICD10'])}", 
                    va='center', ha='right', fontsize=8, fontweight='bold')
    
    # Plot 2: Reduction factor (ICD-10 codes per PheCode)
    ax2 = plt.subplot(2, 2, 2)
    reduction_df = plot_df.sort_values('Reduction_factor', ascending=True)
    colors = plt.cm.viridis(reduction_df['Reduction_factor'] / reduction_df['Reduction_factor'].max())
    
    bars = ax2.barh(range(len(reduction_df)), reduction_df['Reduction_factor'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(reduction_df)))
    ax2.set_yticklabels(reduction_df['Disease'], fontsize=9)
    ax2.set_xlabel('Top-Level ICD-10 Codes per PheCode', fontsize=11, fontweight='bold')
    ax2.set_title('Aggregation Efficiency: Top-Level ICD-10 Codes per PheCode', 
                  fontsize=12, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(reduction_df.iterrows()):
        ax2.text(row['Reduction_factor'], i, f"{row['Reduction_factor']:.1f}", 
                va='center', ha='left', fontsize=8, fontweight='bold')
    
    # Plot 3: Scatter plot showing relationship
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(plot_df['N_Phecodes'], plot_df['N_all_ICD10'], 
                         s=100, alpha=0.6, c=plot_df['Reduction_factor'], 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line showing 1:1 ratio
    max_val = max(plot_df['N_Phecodes'].max(), plot_df['N_all_ICD10'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='1:1 ratio')
    
    ax3.set_xlabel('Number of Phecodes (Aladynoulli)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of ALL ICD-10 Codes (Delphi)', fontsize=11, fontweight='bold')
    ax3.set_title('Phecodes vs ALL ICD-10 Codes', fontsize=12, fontweight='bold', pad=15)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('ALL ICD-10 Codes per PheCode', fontsize=9, fontweight='bold')
    
    # Annotate a few key diseases
    for idx, row in plot_df.nlargest(5, 'N_all_ICD10').iterrows():
        ax3.annotate(row['Disease'].replace('_', ' '), 
                    (row['N_Phecodes'], row['N_all_ICD10']),
                    fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')
    
    # Plot 4: Summary comparison
    ax4 = plt.subplot(2, 2, 4)
    total_phecodes = plot_df['N_Phecodes'].sum()
    total_icd10 = plot_df['N_all_ICD10'].sum()
    reduction_ratio = total_icd10 / total_phecodes if total_phecodes > 0 else 0
    
    categories = ['Aladynoulli\n(Phecodes)', 'Delphi\n(ALL ICD-10)']
    values = [total_phecodes, total_icd10]
    colors_bar = ['#2c7fb8', '#f03b20']
    
    bars = ax4.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Total Number of Predictions', fontsize=11, fontweight='bold')
    ax4.set_title(f'Overall Comparison\n{reduction_ratio:.1f}x Reduction with PheCode Aggregation', 
                  fontsize=12, fontweight='bold', pad=15)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add reduction annotation
    ax4.text(0.5, max(values) * 0.7, f'{reduction_ratio:.1f}x fewer\npredictions', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / 'icd10_aggregation_comparison.png'
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    return fig

def main():
    # Use the same RDS file that the R2 notebook uses (which correctly finds all 6 ASCVD phenotypes)
    # This file contains all ICD-10 codes mapped to Phecodes with phenotype names
    rds_path = Path("/Users/sarahurbut/aladynoulli2/noulli_mapped_phecode_icd10cm.rds")
    
    # Fallback to CSV files if RDS not available
    subset_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode_subset.csv")
    full_path = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/icd2phecode_mergedwithdetailedphecode_info.csv")
    
    if rds_path.exists():
        print(f"Using RDS file (same as R2 notebook): {rds_path}")
        full_df = load_phecode_mapping(rds_path)
        # For RDS file, we use the same dataframe for both finding phecodes and counting ICD-10 codes
        subset_df = full_df
    elif subset_path.exists() and full_path.exists():
        print(f"RDS file not found, using CSV files instead")
        print(f"Loading subset file (to find Phecodes): {subset_path}")
        subset_df = load_phecode_mapping(subset_path)
        print(f"\nLoading full file (to count ICD-10 codes): {full_path}")
        full_df = load_phecode_mapping(full_path)
    else:
        print(f"Error: No mapping file found!")
        print(f"  Expected RDS: {rds_path}")
        print(f"  Or CSV files: {subset_path} and {full_path}")
        return
    
    # Map diseases to Phecodes using subset file, then count ICD-10 codes from full file
    print("\n" + "="*80)
    print("MAPPING DISEASES TO PHECODES")
    print("="*80)
    results_df = map_diseases_to_phecodes(subset_df, full_df, major_diseases)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total diseases analyzed: {len(results_df)}")
    print(f"Total Phecodes used: {results_df['N_Phecodes'].sum()}")
    print(f"Total top-level ICD-10 codes aggregated: {results_df['N_top_level_ICD10'].sum()}")
    print(f"Total ALL ICD-10 codes aggregated: {results_df['N_all_ICD10'].sum()}")
    print(f"Average ALL ICD-10 codes per PheCode: {results_df['ICD10_per_PheCode_avg'].mean():.1f}")
    print(f"Median ALL ICD-10 codes per PheCode: {results_df['ICD10_per_PheCode_avg'].median():.1f}")
    
    # Comparison: Aladynoulli vs Delphi
    print("\n" + "="*80)
    print("ALADYNOULLI vs DELPHI COMPARISON")
    print("="*80)
    print(f"Aladynoulli approach: Uses {results_df['N_Phecodes'].sum()} Phecodes")
    print(f"Delphi approach (top-level): Uses {results_df['N_top_level_ICD10'].sum()} top-level ICD-10 codes (3-character codes)")
    print(f"Delphi approach (all codes): Uses {results_df['N_all_ICD10'].sum()} ALL ICD-10 codes")
    print(f"Reduction (top-level): {results_df['N_top_level_ICD10'].sum() / results_df['N_Phecodes'].sum():.1f}x fewer predictions needed")
    print(f"Reduction (all codes): {results_df['N_all_ICD10'].sum() / results_df['N_Phecodes'].sum():.1f}x fewer predictions needed")
    
    # Sort by number of ALL ICD-10 codes aggregated
    results_df = results_df.sort_values('N_all_ICD10', ascending=False)
    
    # Save results
    output_dir = Path("/Users/sarahurbut/aladynoulli2/claudefile/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'icd10_aggregation_comparison.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Display top diseases by aggregation
    print("\n" + "="*80)
    print("TOP DISEASES BY ICD-10 AGGREGATION")
    print("="*80)
    display_cols = ['Disease', 'N_Phecodes', 'N_top_level_ICD10', 'N_all_ICD10', 'ICD10_per_PheCode_avg', 'Reduction_factor']
    if 'Phecodes_with_counts' in results_df.columns:
        display_cols.insert(-1, 'Phecodes_with_counts')
    print(results_df[display_cols].to_string(index=False))
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    create_plots(results_df, output_dir)
    
    return results_df

if __name__ == "__main__":
    results = main()

