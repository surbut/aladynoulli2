"""
Annotate genetic loci using Ensembl REST API.
Queries Ensembl VEP for gene annotations and compares with existing annotations.
"""

import pandas as pd
import requests
from pathlib import Path
import time
import json
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Disable SSL warnings (if needed)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
INPUT_FILE = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/unique_loci_per_signature.csv")
OUTPUT_FILE = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/unique_loci_annotated.csv")

# Ensembl REST API base URL
ENSEMBL_REST = "https://rest.ensembl.org"

# Create a session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def query_vep_by_id(variant_id, species="human"):
    """Query Ensembl VEP by variant ID (rsID)"""
    endpoint = f"{ENSEMBL_REST}/vep/{species}/id/{variant_id}"
    headers = {"Content-Type": "application/json"}
    params = {"content-type": "application/json"}
    
    try:
        # Try with SSL verification first
        response = session.get(endpoint, headers=headers, params=params, timeout=15, verify=True)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.SSLError:
        # If SSL error, try without verification (less secure but might work)
        try:
            response = session.get(endpoint, headers=headers, params=params, timeout=15, verify=False)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    except Exception as e:
        return None

def query_vep_by_region(chr_num, pos, ref=None, alt=None, species="human"):
    """Query Ensembl VEP by region"""
    if ref and alt:
        variant_str = f"{chr_num}:{pos}:{ref}:{alt}"
    else:
        variant_str = f"{chr_num} {pos} {pos}"
    
    endpoint = f"{ENSEMBL_REST}/vep/{species}/region/{variant_str}"
    headers = {"Content-Type": "application/json"}
    params = {"content-type": "application/json"}
    
    try:
        response = session.get(endpoint, headers=headers, params=params, timeout=15, verify=True)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.SSLError:
        try:
            response = session.get(endpoint, headers=headers, params=params, timeout=15, verify=False)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    except Exception as e:
        return None

def query_overlap_genes(chr_num, start, end, species="human"):
    """Query Ensembl for overlapping genes (fallback method)"""
    endpoint = f"{ENSEMBL_REST}/overlap/id/{species}"
    headers = {"Content-Type": "application/json"}
    params = {
        "feature": "gene",
        "region": f"{chr_num}:{start}-{end}",
        "content-type": "application/json"
    }
    
    try:
        response = session.get(endpoint, headers=headers, params=params, timeout=15, verify=True)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.SSLError:
        try:
            response = session.get(endpoint, headers=headers, params=params, timeout=15, verify=False)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    except Exception as e:
        return None

def safe_str(value):
    """Safely convert value to string, handling NaN"""
    if pd.isna(value):
        return 'N/A'
    return str(value)

def main():
    # Load loci table
    print(f"Loading loci from: {INPUT_FILE}")
    loci_df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(loci_df)} loci to annotate\n")
    
    # Test API connection with first variant
    print("Testing API connection with first variant...")
    test_row = loci_df.iloc[0]
    test_rsid = test_row.get('rsid', 'N/A')
    print(f"Test variant: {test_rsid}")
    
    if test_rsid != 'N/A' and str(test_rsid).startswith('rs'):
        test_result = query_vep_by_id(test_rsid)
        print(f"VEP by ID result: {test_result is not None}")
        if test_result:
            print(f"  Result keys: {list(test_result[0].keys()) if test_result else 'None'}")
    
    print("\nStarting annotation...\n")
    
    annotated_results = []
    errors = []
    
    for idx, row in loci_df.iterrows():
        chr_num = int(row['CHR'])
        pos = int(row['POS'])
        rsid = row.get('rsid', 'N/A')
        uid = row.get('UID', 'N/A')
        
        print(f"[{idx+1}/{len(loci_df)}] {rsid} at {chr_num}:{pos}...", end=' ', flush=True)
        
        gene_symbol = 'N/A'
        gene_id = 'N/A'
        consequence = 'N/A'
        biotype = 'N/A'
        method_used = 'None'
        
        # Strategy 1: Try by rsID first (most reliable)
        if rsid != 'N/A' and pd.notna(rsid) and str(rsid).startswith('rs'):
            result = query_vep_by_id(rsid)
            if result and len(result) > 0:
                method_used = 'rsID'
                vep_result = result[0]
                
                # Extract gene info
                if 'transcript_consequences' in vep_result:
                    for tc in vep_result['transcript_consequences']:
                        if tc.get('canonical') == 1:
                            gene_symbol = tc.get('gene_symbol', 'N/A')
                            gene_id = tc.get('gene_id', 'N/A')
                            biotype = tc.get('biotype', 'N/A')
                            if 'consequence_terms' in tc and len(tc['consequence_terms']) > 0:
                                consequence = tc['consequence_terms'][0]
                            break
                    
                    # If no canonical, use first
                    if gene_symbol == 'N/A' and len(vep_result['transcript_consequences']) > 0:
                        tc = vep_result['transcript_consequences'][0]
                        gene_symbol = tc.get('gene_symbol', 'N/A')
                        gene_id = tc.get('gene_id', 'N/A')
        
        # Strategy 2: Try by region with ref/alt from UID
        if gene_symbol == 'N/A' and pd.notna(uid) and ':' in str(uid):
            parts = str(uid).split(':')
            if len(parts) >= 4:
                ref = parts[2]
                alt = parts[3]
                result = query_vep_by_region(chr_num, pos, ref, alt)
                if result and len(result) > 0:
                    method_used = 'region_with_alleles'
                    vep_result = result[0]
                    
                    if 'transcript_consequences' in vep_result:
                        for tc in vep_result['transcript_consequences']:
                            if tc.get('canonical') == 1:
                                gene_symbol = tc.get('gene_symbol', 'N/A')
                                gene_id = tc.get('gene_id', 'N/A')
                                break
        
        # Strategy 3: Fallback to overlap query (find nearest gene)
        if gene_symbol == 'N/A':
            result = query_overlap_genes(chr_num, max(1, pos - 50000), pos + 50000)
            if result:
                method_used = 'overlap'
                # Find closest gene
                genes = []
                for item in result:
                    if item.get('feature_type') == 'gene' and 'external_name' in item:
                        gene_start = item.get('start', 0)
                        gene_end = item.get('end', 0)
                        distance = min(abs(gene_start - pos), abs(gene_end - pos))
                        genes.append({
                            'name': item.get('external_name', 'N/A'),
                            'id': item.get('id', 'N/A'),
                            'distance': distance
                        })
                
                if genes:
                    genes.sort(key=lambda x: x['distance'])
                    closest = genes[0]
                    gene_symbol = closest['name']
                    gene_id = closest['id']
        
        # Handle NaN values and convert to strings
        original_gene = safe_str(row.get('nearestgene', 'N/A'))
        gene_symbol = safe_str(gene_symbol)
        gene_id = safe_str(gene_id)
        consequence = safe_str(consequence)
        biotype = safe_str(biotype)
        
        # Compare with original annotation
        match = 'N/A'
        if original_gene != 'N/A' and gene_symbol != 'N/A':
            match = 'Yes' if original_gene.upper() == gene_symbol.upper() else 'No'
        elif original_gene == 'N/A' and gene_symbol != 'N/A':
            match = 'New'
        
        result = {
            'Signature': row['Signature'],
            'CHR': chr_num,
            'POS': pos,
            'UID': safe_str(uid),
            'rsid': safe_str(rsid),
            'LOG10P': row['LOG10P'],
            'original_nearestgene': original_gene,
            'ensembl_gene_symbol': gene_symbol,
            'ensembl_gene_id': gene_id,
            'consequence': consequence,
            'biotype': biotype,
            'annotation_method': method_used,
            'annotation_match': match
        }
        
        annotated_results.append(result)
        print(f"→ {gene_symbol} ({method_used})")
        
        # Be respectful to API
        time.sleep(0.1)
    
    # Create annotated dataframe
    annotated_df = pd.DataFrame(annotated_results)
    
    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    annotated_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*80)
    print("ANNOTATION COMPLETE")
    print("="*80)
    print(f"✓ Saved annotated results to: {OUTPUT_FILE}")
    print(f"\nTotal loci: {len(annotated_df)}")
    print(f"Loci with Ensembl gene annotation: {(annotated_df['ensembl_gene_symbol'] != 'N/A').sum()}")
    print(f"Matches with original: {(annotated_df['annotation_match'] == 'Yes').sum()}")
    print(f"Mismatches: {(annotated_df['annotation_match'] == 'No').sum()}")
    print(f"New annotations (original was N/A): {(annotated_df['annotation_match'] == 'New').sum()}")
    
    # Show annotation methods used
    print(f"\nAnnotation methods used:")
    method_counts = annotated_df['annotation_method'].value_counts()
    print(method_counts.to_string())
    
    # Show first few results
    print(f"\nFirst 5 results:")
    print(annotated_df[['rsid', 'original_nearestgene', 'ensembl_gene_symbol', 'annotation_method']].head().to_string())
    
    print("\nDone!")

if __name__ == "__main__":
    main()

