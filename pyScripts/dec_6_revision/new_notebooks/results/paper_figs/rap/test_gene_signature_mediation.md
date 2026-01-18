# Testing Gene → Signature → Disease Mediation

## Research Question
Is the gene-disease association **mediated through signatures**, or are gene and signature **independent** contributors to disease?

## Two Hypotheses

### H1: Mediation (Gene → Signature → Disease)
- Gene affects signature loading
- Signature loading affects disease risk
- Gene effect on disease is **indirect** (through signature)

### H2: Independence (Gene → Disease, Signature → Disease separately)
- Gene affects disease directly
- Signature affects disease directly  
- Gene and signature are **independent** predictors
- Signature just "summarizes" disease patterns but doesn't mediate gene effects

## Mediation Analysis Approaches

### Approach 1: Baron-Kenny 4-Step Test

Test these 4 relationships:

1. **Gene → Disease** (Total effect)
   - Model: `Disease ~ Gene_burden`
   - Measure: Total effect of gene on disease

2. **Gene → Signature** (Path a)  
   - Model: `Signature_loading ~ Gene_burden`
   - Measure: Does gene burden predict signature loading?

3. **Signature → Disease** (Path b, controlling for gene)
   - Model: `Disease ~ Signature_loading + Gene_burden`
   - Measure: Does signature predict disease when controlling for gene?

4. **Gene → Disease** (Direct effect, controlling for signature)
   - Model: `Disease ~ Gene_burden + Signature_loading`
   - Measure: Direct effect of gene after controlling for signature

**Interpretation:**
- **Mediation**: Significant paths 1, 2, 3, AND path 4 is smaller than path 1
- **Independence**: Significant paths 1 and 3, but path 4 ≈ path 1 (gene effect unchanged)

### Approach 2: Structural Equation Modeling (SEM)

Fit a path model:
```
Gene_burden → Signature_loading → Disease
      ↓                              ↑
      └──────────────────────────────┘
            (direct path)
```

**Test:**
- Indirect effect: `a × b` (Gene→Sig × Sig→Disease)
- Direct effect: `c'` (Gene→Disease controlling for Sig)
- Total effect: `c = a×b + c'`

**Mediation if:** Indirect effect significant AND direct effect < total effect

### Approach 3: Sobel Test / Bootstrap Mediation

Test if indirect effect `a × b` is significantly different from zero.

**Sobel test:**
```
z = (a × b) / sqrt(b²×SE_a² + a²×SE_b²)
```

**Bootstrap:** Resample and compute indirect effects, test if confidence interval excludes zero.

## Data Requirements

For patient-level analysis, you need:

1. **Gene burden** (from rare variant analysis)
   - Per patient: `burden_i = sum(rare_variants)` for gene G
   - Source: Already computed in `rare_variant_burden_associations.csv`

2. **Signature loadings** (patient-specific)
   - Per patient: `theta[i, k, t]` for signature k at time t
   - Or average: `theta[i, k]` = mean over time
   - Source: From lambda/theta in model data

3. **Disease status**
   - Per patient: `Y[i, d]` binary or time-to-event
   - Source: Y matrix

## Practical Implementation

### Option A: Patient-Level Regression (if you have matched patient data)

```R
# Load patient data with:
# - gene_burden (from genotype files)
# - signature_loading (from model theta)
# - disease_status (from Y matrix)

# Step 1: Total effect
fit1 <- glm(disease ~ gene_burden, family=binomial)
total_effect <- coef(fit1)[2]

# Step 2: Gene → Signature
fit2 <- lm(signature_loading ~ gene_burden)
path_a <- coef(fit2)[2]

# Step 3: Signature → Disease (controlling for gene)
fit3 <- glm(disease ~ signature_loading + gene_burden, family=binomial)
path_b <- coef(fit3)[2]  # Signature effect
direct_effect <- coef(fit3)[3]  # Gene direct effect

# Step 4: Test mediation
indirect_effect <- path_a * path_b
mediation_ratio <- indirect_effect / total_effect
```

### Option B: Disease-Level Meta-Analysis (using existing correlation data)

If you don't have patient-level matched data, you can use disease-level correlations:

```R
# From your existing analysis:
# - gene_disease_corr: correlation between gene burden and disease
# - sig_disease_phi: signature-disease association (phi)
# - Need: gene_sig_corr (correlation between gene burden and signature loading)

# Partial correlation: Gene-Disease controlling for Signature
# r_gd.s = (r_gd - r_gs × r_sd) / sqrt((1-r_gs²)(1-r_sd²))

# If partial correlation is much smaller than total correlation → mediation
```

## Key Tests to Distinguish Models

### Test 1: Does gene burden predict signature loading?
- **H1 (Mediation)**: YES - Gene → Signature path should be significant
- **H2 (Independence)**: NO - Gene and Signature should be independent

**How to test:** 
- Patient-level: Regress `signature_loading ~ gene_burden`
- Or: Test if gene burden correlates with signature loadings across patients

### Test 2: Does controlling for signature reduce gene effect?
- **H1 (Mediation)**: YES - Gene effect should drop when controlling for signature
- **H2 (Independence)**: NO - Gene effect should remain similar

**How to test:**
- Compare: `Disease ~ Gene` vs `Disease ~ Gene + Signature`
- If gene coefficient drops substantially → mediation
- If gene coefficient stays similar → independence

### Test 3: Is indirect effect significant?
- **H1 (Mediation)**: YES - `(Gene→Sig) × (Sig→Disease)` should be significant
- **H2 (Independence)**: NO - Indirect effect should be near zero

## What You'd Need to Compute

1. **Gene → Signature association**
   - Per patient: Correlate gene burden with their signature loadings
   - Across patients: `cor(gene_burden[i], theta[i, signature_idx])`
   - This tests if patients with higher gene burden have higher signature loading

2. **Signature → Disease (conditional on gene)**
   - Use existing phi/psi (signature-disease associations)
   - But test if these associations remain when controlling for gene

3. **Gene → Disease (conditional on signature)**
   - Partial correlation or logistic regression with both predictors
   - Test if gene effect remains after controlling for signature loading

## Recommendation

Start with **Option B** (disease-level) since you already have:
- Gene-disease correlations (from R script)
- Signature-disease associations (phi/psi)

You just need to compute:
- **Gene-signature correlations**: Correlate gene burden with signature loadings across patients for the relevant signature-gene pairs

Then use partial correlation to test if gene-disease correlation drops when controlling for signature-disease association.

