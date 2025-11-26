# Working with Imputed Variants for Mutation Carrier Analysis

## Overview

This guide explains how to extend the FH mutation carrier analysis to work with **imputed genetic variants** instead of WES (whole exome sequencing) data.

## Key Differences: Imputed vs WES Data

### WES Data (on RAP)
- **Direct sequencing**: Observes actual DNA sequence
- **Rare variants**: Can detect very rare mutations (MAF < 0.1%)
- **Pathogenic mutations**: Can identify specific disease-causing mutations
- **Example**: FH mutations (LDLR, APOB, PCSK9) - very rare, require WES

### Imputed Data (locally available)
- **Statistical inference**: Predicts genotypes from reference panels
- **Common variants**: Best for variants with MAF > 1%
- **Well-imputed rare variants**: Some rare variants (MAF 0.1-1%) can be imputed with high confidence
- **Format**: Usually BGEN or PLINK format

## What Variants Are Available in Imputed Data?

### High-Impact Variants Typically Well-Imputed:

1. **Common Pathogenic Variants**
   - APOE ε4 (rs429358, rs7412) - Alzheimer's disease
   - BRCA1/BRCA2 common variants (if MAF > 1%)
   - Common variants in Mendelian disease genes

2. **High-Effect-Size GWAS Variants**
   - 9p21 CAD risk variants (rs10757278, rs1333049)
   - TCF7L2 diabetes variant (rs7903146)
   - Many other GWAS hits with large effect sizes

3. **Common Variants in Disease-Associated Genes**
   - Variants in genes like PCSK9, LDLR (if common enough)
   - Variants in metabolic pathway genes

### What's NOT Available in Imputed Data:

- Very rare mutations (MAF < 0.1%) - these require WES
- Private mutations (family-specific)
- Structural variants (large deletions/duplications)
- Most pathogenic mutations in Mendelian disease genes

## Workflow: From Imputed Data to Carrier Analysis

### Step 1: Identify Variants of Interest

Create a variant list file (`variants.txt`) with one variant per line:

```
# Format: rsID or chr:pos:ref:alt
rs429358
rs7412
rs10757278
19:44908822:C:T
```

### Step 2: Extract Carriers from Imputed Data

Use `extract_imputed_variant_carriers.py`:

```bash
# For PLINK format:
python extract_imputed_variant_carriers.py \
    --variants variants.txt \
    --plink_prefix /path/to/ukb_imputed \
    --processed_ids /path/to/processed_ids.npy \
    --output apoe_carriers.txt \
    --dominant

# For BGEN format (requires bgen_reader):
python extract_imputed_variant_carriers.py \
    --variants variants.txt \
    --bgen_path /path/to/bgen_files \
    --sample_file /path/to/sample_file.sample \
    --processed_ids /path/to/processed_ids.npy \
    --output apoe_carriers.txt
```

### Step 3: Run Signature Enrichment Analysis

Use `analyze_mutation_carriers_signature.py`:

```python
# In notebook:
%run analyze_mutation_carriers_signature.py \
    --carrier_file apoe_carriers.txt \
    --mutation_name "APOE_epsilon4" \
    --signature_idx 5 \
    --event_indices 112,113,114,115,116 \
    --plot \
    --output_dir results/
```

## Example: APOE ε4 Analysis

APOE ε4 is a common variant (MAF ~15% in European populations) that:
- Is well-imputed (high imputation quality)
- Has large effect on Alzheimer's disease risk
- Could show signature enrichment before dementia events

### Variant List (`apoe_variants.txt`):
```
rs429358
rs7412
```

### Extract Carriers:
```bash
python extract_imputed_variant_carriers.py \
    --variants apoe_variants.txt \
    --plink_prefix /path/to/ukb_imputed \
    --output apoe_carriers.txt
```

### Analyze:
```python
%run analyze_mutation_carriers_signature.py \
    --carrier_file apoe_carriers.txt \
    --mutation_name "APOE_epsilon4" \
    --signature_idx 0 \  # Or whichever signature is relevant for dementia
    --event_indices 200,201,202 \  # Dementia-related disease indices
    --plot
```

## Alternative: Using Pre-Extracted Variant Files

If you already have variant carrier files (e.g., from previous analyses), you can use them directly:

```python
# Carrier file format: tab-separated with 'IID' column
# apoe_carriers.txt:
# IID
# 1000015
# 1000023
# ...

%run analyze_mutation_carriers_signature.py \
    --carrier_file apoe_carriers.txt \
    --mutation_name "APOE_epsilon4" \
    --signature_idx 5
```

## Tips for Finding Variants in Imputed Data

1. **Check MAF**: Variants with MAF > 1% are usually well-imputed
2. **Check Imputation Quality**: Look for INFO score > 0.8
3. **Use GWAS Catalog**: Search for high-effect-size variants in your disease of interest
4. **Check ClinVar**: Some pathogenic variants are common enough to be imputed

## Common Variants Worth Analyzing

### Cardiovascular:
- 9p21 variants (rs10757278, rs1333049) - CAD risk
- APOE variants (rs429358, rs7412) - Also cardiovascular
- Common PCSK9 variants

### Metabolic:
- TCF7L2 (rs7903146) - Type 2 diabetes
- FTO (rs1421085) - Obesity

### Neurological:
- APOE ε4 (rs429358, rs7412) - Alzheimer's
- Common variants in neurodegenerative disease genes

### Cancer:
- Common BRCA1/BRCA2 variants (if MAF > 1%)
- Other common cancer risk variants

## Notes

- **Dominant vs Recessive**: Most analyses use dominant model (heterozygous OR homozygous alternate), but you can specify `--recessive` for recessive variants
- **Multiple Variants**: You can analyze multiple variants by creating separate carrier files or combining them
- **Signature Selection**: Choose the signature most relevant to your disease (e.g., Signature 5 for cardiovascular, Signature 0 for neurological)

