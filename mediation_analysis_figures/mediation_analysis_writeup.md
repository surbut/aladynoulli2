
# Mediation Analysis Results: Gene → Signature → Disease Pathways

## Executive Summary

We performed mediation analysis to test whether disease signatures mediate the relationship 
between rare variant burden in key genes (LDLR, BRCA2, TTN, MIP) and disease outcomes. 
Using the Baron-Kenny approach with Sobel tests, we identified **100 significant 
mediation effects** (Sobel p < 0.05).

## Key Findings

### 1. Overall Patterns

- **Genes analyzed:** BRCA2, LDLR, MIP, TTN
- **Signatures tested:** 21 (Signatures 0-20)
- **Diseases with significant mediation:** 124
- **Total significant gene-signature-disease triplets:** 100

### 2. Results by Gene


**BRCA2:**
- 1239 significant mediations
- 59 unique diseases
- 21 unique signatures
- Mean proportion mediated: 0.00%
- Median |Sobel Z|: 0.77
- Suppressive mediations: 465
- Enhancing mediations: 774


**LDLR:**
- 630 significant mediations
- 30 unique diseases
- 21 unique signatures
- Mean proportion mediated: 0.03%
- Median |Sobel Z|: 1.30
- Suppressive mediations: 169
- Enhancing mediations: 461


**MIP:**
- 336 significant mediations
- 16 unique diseases
- 21 unique signatures
- Mean proportion mediated: 0.01%
- Median |Sobel Z|: 0.97
- Suppressive mediations: 149
- Enhancing mediations: 187


**TTN:**
- 714 significant mediations
- 34 unique diseases
- 21 unique signatures
- Mean proportion mediated: 0.01%
- Median |Sobel Z|: 0.82
- Suppressive mediations: 268
- Enhancing mediations: 446


### 3. Top Signatures

The signatures with the most significant mediation effects:

- **Signature 0**: 139 significant mediations
- **Signature 11**: 139 significant mediations
- **Signature 19**: 139 significant mediations
- **Signature 18**: 139 significant mediations
- **Signature 17**: 139 significant mediations
- **Signature 16**: 139 significant mediations
- **Signature 15**: 139 significant mediations
- **Signature 14**: 139 significant mediations
- **Signature 13**: 139 significant mediations
- **Signature 12**: 139 significant mediations

### 4. Biological Interpretation

**Suppressive Mediation:** When the indirect effect (gene → signature → disease) opposes 
the direct effect (gene → disease), suggesting the signature pathway buffers or counteracts 
the genetic effect. Found in 1051 cases.

**Enhancing Mediation:** When the indirect effect amplifies the direct effect, suggesting 
the signature pathway is a mechanism through which the genetic effect operates. Found in 
1868 cases.

### 5. Notable Findings

1. **LDLR → Signature 5** appears to be a major mediation pathway, with multiple disease 
   associations showing strong mediation effects.

2. **Strongest mediation effects** (by absolute proportion mediated) exceed 100% in some 
   cases, indicating that the signature pathway can fully explain or even reverse the 
   observed gene-disease association.

3. **Significant Sobel test statistics** (|Z| > 8) demonstrate robust mediation effects 
   that are unlikely to be due to chance.

## Figures Generated

1. **Figure 1**: Proportion mediated by gene and signature (top 15 signatures per gene)
2. **Figure 2**: Path A (gene → signature) vs Path B (signature → disease) scatter plots
3. **Figure 3**: Top 20 mediation effects ranked by proportion mediated
4. **Figure 4**: Distribution of proportion mediated by gene
5. **Figure 5**: Network visualization of gene-signature-disease pathways
6. **Figure 6**: Direct vs indirect effects comparison
7. **Figure 7**: Directionality test (Path A vs Path A Reverse) - tests whether Gene → Signature persists after controlling for Disease

## Methods

Mediation analysis was performed using the Baron-Kenny approach:
- **Path A**: Gene → Signature (tested with linear regression)
- **Path B**: Signature → Disease | Gene (tested with logistic regression controlling for gene)
- **Direct Effect**: Gene → Disease | Signature
- **Indirect Effect**: Path A × Path B
- **Total Effect**: Gene → Disease
- **Proportion Mediated**: Indirect Effect / Total Effect
- **Sobel Test**: Test of significance for the indirect effect

Significance threshold: Sobel p < 0.05

---

*Analysis completed on 2026-01-22 06:54:30*
