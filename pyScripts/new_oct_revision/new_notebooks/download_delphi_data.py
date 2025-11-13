"""
Download and process Delphi-2M supplementary data from Nature paper
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path

def download_delphi_csv():
    """Download Delphi-2M supplementary CSV file"""
    url = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09529-3/MediaObjects/41586_2025_9529_MOESM3_ESM.csv"
    output_path = Path("/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_supplementary.csv")
    
    print(f"Downloading Delphi-2M data from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded to: {output_path}")
        return output_path
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return None

def load_delphi_data(csv_path):
    """Load and parse Delphi-2M CSV data"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from Delphi-2M data")
    return df

def create_disease_mapping():
    """
    Map Delphi-2M disease names (ICD-10 based) to Aladynoulli disease names
    This is a manual mapping based on common disease categories
    """
    mapping = {
        # Cardiovascular
        "I21": "ASCVD",  # Acute myocardial infarction
        "I22": "ASCVD",  # Subsequent myocardial infarction  
        "I25": "ASCVD",  # Chronic ischaemic heart disease
        "I48": "Atrial_Fib",  # Atrial fibrillation
        "I50": "Heart_Failure",  # Heart failure
        "I63": "Stroke",  # Cerebral infarction
        "I64": "Stroke",  # Stroke, not specified as haemorrhage or infarction
        
        # Diabetes
        "E10": "Diabetes",  # Type 1 diabetes
        "E11": "Diabetes",  # Type 2 diabetes
        
        # Cancer
        "C18": "Colorectal_Cancer",  # Malignant neoplasm of colon
        "C19": "Colorectal_Cancer",  # Malignant neoplasm of rectosigmoid junction
        "C20": "Colorectal_Cancer",  # Malignant neoplasm of rectum
        "C34": "Lung_Cancer",  # Malignant neoplasm of bronchus and lung
        "C50": "Breast_Cancer",  # Malignant neoplasm of breast
        "C61": "Prostate_Cancer",  # Malignant neoplasm of prostate
        "C67": "Bladder_Cancer",  # Malignant neoplasm of bladder
        
        # Other
        "N18": "CKD",  # Chronic kidney disease
        "J44": "COPD",  # Chronic obstructive pulmonary disease
        "J45": "Asthma",  # Asthma
        "J46": "Asthma",  # Status asthmaticus
        "J12": "Pneumonia",  # Viral pneumonia
        "J13": "Pneumonia",  # Pneumonia due to Streptococcus pneumoniae
        "J14": "Pneumonia",  # Pneumonia due to Haemophilus influenzae
        "J15": "Pneumonia",  # Bacterial pneumonia
        "J16": "Pneumonia",  # Pneumonia due to other infectious organisms
        "J17": "Pneumonia",  # Pneumonia in diseases classified elsewhere
        "J18": "Pneumonia",  # Pneumonia, organism unspecified
        
        # Mental health
        "F32": "Depression",  # Depressive episode
        "F33": "Depression",  # Recurrent depressive disorder
        "F41": "Anxiety",  # Other anxiety disorders
        "F31": "Bipolar_Disorder",  # Bipolar affective disorder
        
        # Other conditions
        "M80": "Osteoporosis",  # Osteoporosis with pathological fracture
        "M81": "Osteoporosis",  # Osteoporosis without pathological fracture
        "M05": "Rheumatoid_Arthritis",  # Seropositive rheumatoid arthritis
        "M06": "Rheumatoid_Arthritis",  # Other rheumatoid arthritis
        "G20": "Parkinsons",  # Parkinson disease
        "G35": "Multiple_Sclerosis",  # Multiple sclerosis
        "E00": "Thyroid_Disorders",  # Congenital iodine-deficiency syndrome
        "E01": "Thyroid_Disorders",  # Iodine-deficiency-related thyroid disorders
        "E02": "Thyroid_Disorders",  # Subclinical iodine-deficiency hypothyroidism
        "E03": "Thyroid_Disorders",  # Other hypothyroidism
        "E04": "Thyroid_Disorders",  # Other non-toxic goitre
        "E05": "Thyroid_Disorders",  # Thyrotoxicosis
        "E06": "Thyroid_Disorders",  # Thyroiditis
        "E07": "Thyroid_Disorders",  # Other disorders of thyroid
        "L40": "Psoriasis",  # Psoriasis
        "K50": "Crohns_Disease",  # Crohn disease
        "K51": "Ulcerative_Colitis",  # Ulcerative colitis
        "D50": "Anemia",  # Iron deficiency anaemia
        "D51": "Anemia",  # Vitamin B12 deficiency anaemia
        "D52": "Anemia",  # Folate deficiency anaemia
        "D53": "Anemia",  # Other nutritional anaemias
    }
    
    return mapping

def process_delphi_data(df):
    """
    Process Delphi-2M data to extract AUC values for matching diseases
    Returns a dictionary mapping Aladynoulli disease names to Delphi AUCs
    """
    disease_mapping = create_disease_mapping()
    
    # Reverse mapping: Aladynoulli disease -> list of ICD-10 codes
    aladyn_to_icd = {}
    for icd, aladyn in disease_mapping.items():
        if aladyn not in aladyn_to_icd:
            aladyn_to_icd[aladyn] = []
        aladyn_to_icd[aladyn].append(icd)
    
    # Process Delphi data
    delphi_results = {}
    
    for _, row in df.iterrows():
        name = str(row.get('Name', '')).strip()
        
        # Extract ICD-10 code from name (e.g., "I21 Acute myocardial infarction" -> "I21")
        if len(name) >= 3 and name[0].isalpha() and name[1:3].isdigit():
            icd_code = name[:3]
            
            # Check if this ICD code maps to one of our diseases
            for aladyn_disease, icd_codes in aladyn_to_icd.items():
                if icd_code in icd_codes:
                    # Get AUC values (1-year gap = prospective prediction)
                    auc_female_1yr = row.get('AUC Female, (1 year gap)', np.nan)
                    auc_male_1yr = row.get('AUC Male, (1 year gap)', np.nan)
                    
                    # Average across sexes (or use available)
                    if pd.notna(auc_female_1yr) and pd.notna(auc_male_1yr):
                        auc_1yr = (auc_female_1yr + auc_male_1yr) / 2
                    elif pd.notna(auc_female_1yr):
                        auc_1yr = auc_female_1yr
                    elif pd.notna(auc_male_1yr):
                        auc_1yr = auc_male_1yr
                    else:
                        auc_1yr = np.nan
                    
                    # Store (keep best match if multiple ICD codes map to same disease)
                    if aladyn_disease not in delphi_results or pd.notna(auc_1yr):
                        delphi_results[aladyn_disease] = {
                            '1yr': auc_1yr,
                            'icd_code': icd_code,
                            'name': name
                        }
    
    return delphi_results

if __name__ == "__main__":
    # Download data
    csv_path = download_delphi_csv()
    
    if csv_path and csv_path.exists():
        # Load and process
        df = load_delphi_data(csv_path)
        delphi_results = process_delphi_data(df)
        
        print(f"\n✓ Processed {len(delphi_results)} diseases from Delphi-2M")
        print("\nDelphi-2M results (1-year gap = prospective prediction):")
        for disease, data in sorted(delphi_results.items()):
            print(f"  {disease:25s} AUC_1yr: {data['1yr']:.3f} (from {data['icd_code']}: {data['name'][:50]})")
        
        # Save processed results
        import pickle
        output_path = Path("/Users/sarahurbut/aladynoulli2/claudefile/output/delphi_results.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(delphi_results, f)
        print(f"\n✓ Saved processed results to: {output_path}")

