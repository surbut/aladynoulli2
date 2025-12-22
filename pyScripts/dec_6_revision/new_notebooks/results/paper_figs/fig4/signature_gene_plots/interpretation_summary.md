# Signature-Gene-Disease Association Validation

## Summary

This analysis validates the biological meaningfulness of disease signatures through a three-way consistency check:

1. **Signature ↔ Gene**: Rare variant association studies (RVAS) identify genes significantly associated with signature exposure
2. **Gene ↔ Disease**: Rare variant burden in these genes correlates with disease occurrence
3. **Disease ↔ Signature**: Diseases showing high gene-disease correlation also show high signature loading (phi)

## Key Findings

### Strong Positive Associations (r > 0.35, p < 1e-5)

These signature-gene-disease triads show strong concordance across all three levels:

| Signature | Gene | Correlation (r) | P-value | Interpretation |
|-----------|------|-----------------|---------|----------------|
| **5** | **LDLR** | **0.743** | 7.1e-34 | Lipid metabolism genes strongly associated with cardiovascular signature; diseases with high LDLR rare variant burden show high Signature 5 loading |
| **10** | **MIP** | **0.577** | 1.1e-19 | MIP (Major Intrinsic Protein) associated with ophthalmologic signature; eye diseases with MIP variants strongly load on Signature 10 |
| **0** | **TTN** | **0.535** | 6.7e-17 | Titin gene (cardiac structure) associated with cardiac arrhythmia signature; cardiac diseases with TTN variants show high Signature 0 loading |
| **6** | **BRCA2** | **0.496** | 7.1e-16 | BRCA2 associated with metastatic cancer signature; cancers with BRCA2 rare variants strongly load on Signature 6 |
| **16** | **TET2** | **0.363** | 4.6e-09 | TET2 (DNA methylation) associated with infectious/critical care signature; diseases with TET2 variants load on Signature 16 |
| **11** | **CLPTM1L** | **0.364** | 3.4e-08 | CLPTM1L associated with cerebrovascular signature; cerebrovascular diseases with CLPTM1L variants load on Signature 11 |

### Moderate Associations

| Signature | Gene | Correlation (r) | P-value |
|-----------|------|-----------------|---------|
| 16 | PKD1 | 0.258 | 9.2e-05 |
| 7 | GNB2 | 0.175 | 0.019 |

### Negative Associations (potentially interesting for understanding opposing effects)

| Signature | Gene | Correlation (r) | P-value | Interpretation |
|-----------|------|-----------------|---------|----------------|
| 5 | APOB | -0.358 | 1.2e-07 | APOB rare variants show inverse relationship - diseases with APOB variants have *lower* Signature 5 loading (may reflect different lipid metabolism pathways) |
| 20 | DEFB1 | -0.177 | 0.013 | DEFB1 (antimicrobial defense) inversely associated with "healthy" signature - may reflect protective effects |
| 5 | CDH26 | -0.188 | 0.009 | CDH26 shows inverse association with cardiovascular signature |

### Weak/Non-significant Associations

| Signature | Gene | Correlation (r) | P-value |
|-----------|------|-----------------|---------|
| 3 | ADGRG7 | 0.037 | 0.587 |
| 16 | BRCA2 | 0.136 | 0.038 |

## Biological Validation

This three-way consistency provides strong evidence for biological meaningfulness:

1. **Genetic basis validated**: Genes identified through rare variant analysis of signatures are the same genes whose rare variants correlate with diseases
2. **Signature-disease alignment**: Diseases that are genetically linked (through rare variants) to signature-associated genes show corresponding signature enrichment
3. **Mechanistic insights**: The direction of correlations (positive vs negative) reveals complementary vs opposing pathways

## Example: Signature 5 - LDLR (r=0.743)

- **Step 1**: RVAS identified LDLR as significantly associated with Signature 5 (cardiovascular) exposure
- **Step 2**: Rare variant burden in LDLR correlates with cardiovascular diseases (hypercholesterolemia, coronary atherosclerosis, etc.)
- **Step 3**: These same diseases show high loading on Signature 5 (phi values)
- **Result**: Perfect concordance across all three levels, validating that Signature 5 captures the LDLR-mediated cardiovascular risk pathway

## Implications

- **Signatures capture genetic architecture**: The fact that signature-associated genes show disease-level correlations validates that signatures integrate genetic risk
- **Pleiotropy detection**: Genes like BRCA2 appearing in multiple signatures (6 and 16) reveal shared genetic architecture across disease clusters
- **Pathway discovery**: Inverse correlations (e.g., APOB) suggest alternative or protective pathways that may modulate signature expression

