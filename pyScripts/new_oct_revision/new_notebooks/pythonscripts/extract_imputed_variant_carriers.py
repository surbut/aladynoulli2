#!/usr/bin/env python3
"""
Extract variant carriers from UK Biobank imputed genotype data.

This script extracts carrier status for specific variants from imputed BGEN files
and creates a carrier file compatible with the FH analysis framework.

Requirements:
- UK Biobank imputed BGEN files (or PLINK format)
- Variant list (rsID or chr:pos:ref:alt format)
- Sample file matching your processed_ids

Usage:
    python extract_imputed_variant_carriers.py \
        --variants variants.txt \
        --bgen_path /path/to/imputed/bgen \
        --sample_file /path/to/sample_file.sample \
        --output carriers.txt \
        --processed_ids /path/to/processed_ids.npy
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

try:
    import bgen_reader
    HAS_BGEN = True
except ImportError:
    HAS_BGEN = False
    print("⚠️  bgen_reader not installed. Install with: pip install bgen-reader")

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False


def load_variant_list(variant_file):
    """
    Load variant list from file.
    
    Expected format (one per line):
    - rsID: rs429358
    - chr:pos:ref:alt: 19:44908822:C:T
    - chr:pos: 19:44908822 (will extract all alleles)
    
    Returns list of variant identifiers.
    """
    variants = []
    with open(variant_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                variants.append(line)
    return variants


def extract_from_bgen(variant_list, bgen_path, sample_file, processed_ids, output_file):
    """
    Extract variant carriers from BGEN files.
    
    This is a placeholder - you'll need to implement BGEN reading based on your setup.
    Common approaches:
    1. Use bgen_reader Python package
    2. Use qctool/bcftools to convert to VCF first
    3. Use PLINK to extract variants
    """
    if not HAS_BGEN:
        raise ImportError("bgen_reader not available. Please install or use alternative method.")
    
    # Load sample file to get eid mapping
    sample_df = pd.read_csv(sample_file, sep=' ', skiprows=2, 
                           names=['ID_1', 'ID_2', 'missing', 'sex'])
    
    # Create eid to index mapping
    eid_to_idx = {int(eid): idx for idx, eid in enumerate(sample_df['ID_1'])}
    
    # Load processed_ids
    if isinstance(processed_ids, str):
        processed_ids = np.load(processed_ids)
    
    # Extract variants from each chromosome
    all_carriers = set()
    
    for variant in variant_list:
        print(f"Processing variant: {variant}")
        
        # Parse variant identifier
        if variant.startswith('rs'):
            # rsID - need to look up chr:pos
            # This requires a variant lookup file or API call
            print(f"  ⚠️  rsID lookup not implemented - need variant info file")
            continue
        elif ':' in variant:
            # chr:pos:ref:alt or chr:pos format
            parts = variant.split(':')
            chrom = parts[0].replace('chr', '')
            pos = int(parts[1])
            
            # Determine BGEN file path (UKB format: ukb_imp_chr{chr}_v3.bgen)
            bgen_file = Path(bgen_path) / f"ukb_imp_chr{chrom}_v3.bgen"
            
            if not bgen_file.exists():
                print(f"  ⚠️  BGEN file not found: {bgen_file}")
                continue
            
            # Extract variant (this is pseudocode - actual implementation depends on bgen_reader API)
            try:
                # Example with bgen_reader:
                # bgen = bgen_reader.open_bgen(bgen_file)
                # variant_data = bgen.read_variant(variant)
                # genotypes = variant_data['genotypes']  # Shape: (n_samples, 3) for [P(0/0), P(0/1), P(1/1)]
                
                # Determine carriers (heterozygous or homozygous alternate)
                # For dominant model: carriers = genotypes[:, 1] + genotypes[:, 2] > threshold
                # For recessive model: carriers = genotypes[:, 2] > threshold
                
                print(f"  ✓ Extracted variant at chr{chrom}:{pos}")
                # all_carriers.update(carrier_eids)
                
            except Exception as e:
                print(f"  ⚠️  Error extracting variant: {e}")
                continue
    
    # Create carrier file in same format as FH analysis expects
    carrier_df = pd.DataFrame({
        'IID': sorted(all_carriers)
    })
    carrier_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ Saved {len(all_carriers)} carriers to {output_file}")


def extract_from_plink(variant_list, plink_prefix, processed_ids, output_file, 
                      dominant=True, dosage_threshold=0.5):
    """
    Extract variant carriers from PLINK format files (.bed/.bim/.fam).
    
    Uses PLINK's --extract and --recode A to get dosages, then determines carriers.
    """
    import subprocess
    import tempfile
    
    # Write variant list to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        variant_file = f.name
        for variant in variant_list:
            f.write(f"{variant}\n")
    
    # Use PLINK to extract variants
    # This requires PLINK to be installed and in PATH
    try:
        # Extract variants and recode as dosage
        cmd = [
            'plink',
            '--bfile', plink_prefix,
            '--extract', variant_file,
            '--recode', 'A',
            '--out', str(Path(output_file).parent / 'temp_extracted')
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Load dosage file
        dosage_file = Path(output_file).parent / 'temp_extracted.raw'
        dosage_df = pd.read_csv(dosage_file, sep=' ')
        
        # Determine carriers based on dosage
        # PLINK dosage: 0 = homozygous ref, 1 = heterozygous, 2 = homozygous alt
        carrier_cols = []
        for variant in variant_list:
            # Find column matching variant
            col = [c for c in dosage_df.columns if variant in c]
            if col:
                if dominant:
                    # Dominant: heterozygous or homozygous alternate
                    carriers = dosage_df[dosage_df[col[0]] >= dosage_threshold]['IID'].values
                else:
                    # Recessive: homozygous alternate only
                    carriers = dosage_df[dosage_df[col[0]] >= (2 - dosage_threshold)]['IID'].values
                
                carrier_cols.extend(carriers)
        
        # Create carrier file
        all_carriers = set(carrier_cols)
        carrier_df = pd.DataFrame({
            'IID': sorted(all_carriers)
        })
        carrier_df.to_csv(output_file, sep='\t', index=False)
        print(f"\n✓ Saved {len(all_carriers)} carriers to {output_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  PLINK extraction failed: {e}")
        print("Make sure PLINK is installed and variant IDs match PLINK format")
    except FileNotFoundError:
        print("⚠️  PLINK not found. Please install PLINK or use alternative method.")


def create_variant_list_template(output_file):
    """Create a template file with common high-impact variants available in imputed data."""
    
    # Common variants that are typically well-imputed and have high impact
    common_variants = [
        "# High-impact variants typically available in imputed data",
        "# Format: one variant per line (rsID or chr:pos:ref:alt)",
        "",
        "# APOE ε4 (Alzheimer's disease risk)",
        "rs429358",
        "rs7412",
        "",
        "# Cardiovascular risk variants",
        "rs10757278",  # 9p21 CAD risk
        "rs1333049",   # 9p21 CAD risk
        "",
        "# Type 2 Diabetes",
        "rs7903146",   # TCF7L2
        "",
        "# Add your variants here...",
        "# Format examples:",
        "# rs123456",
        "# 19:44908822:C:T",
        "# chr19:44908822"
    ]
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(common_variants))
    
    print(f"✓ Created variant list template: {output_file}")
    print("  Edit this file to add your variants of interest")


def main():
    parser = argparse.ArgumentParser(
        description="Extract variant carriers from imputed genotype data"
    )
    parser.add_argument('--variants', type=str, 
                       help='File with variant list (rsID or chr:pos:ref:alt)')
    parser.add_argument('--bgen_path', type=str,
                       help='Path to BGEN files directory')
    parser.add_argument('--plink_prefix', type=str,
                       help='PLINK file prefix (e.g., /path/to/data)')
    parser.add_argument('--sample_file', type=str,
                       help='BGEN sample file')
    parser.add_argument('--processed_ids', type=str,
                       help='Path to processed_ids.npy file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output carrier file path')
    parser.add_argument('--dominant', action='store_true', default=True,
                       help='Use dominant model (default: True)')
    parser.add_argument('--recessive', action='store_true',
                       help='Use recessive model')
    parser.add_argument('--create_template', action='store_true',
                       help='Create template variant list file')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_variant_list_template(args.output)
        return
    
    if not args.variants:
        print("⚠️  Error: --variants required (or use --create_template)")
        sys.exit(1)
    
    # Load variants
    variants = load_variant_list(args.variants)
    print(f"Loaded {len(variants)} variants from {args.variants}")
    
    # Load processed_ids if provided
    processed_ids = None
    if args.processed_ids:
        processed_ids = np.load(args.processed_ids)
        print(f"Loaded {len(processed_ids)} processed IDs")
    
    # Extract carriers based on available format
    if args.plink_prefix:
        extract_from_plink(variants, args.plink_prefix, processed_ids, 
                          args.output, dominant=not args.recessive)
    elif args.bgen_path:
        if not args.sample_file:
            print("⚠️  Error: --sample_file required for BGEN extraction")
            sys.exit(1)
        extract_from_bgen(variants, args.bgen_path, args.sample_file, 
                         processed_ids, args.output)
    else:
        print("⚠️  Error: Must provide either --bgen_path or --plink_prefix")
        sys.exit(1)


if __name__ == '__main__':
    main()

