# Mediation Analysis: Gene → Signature → Disease

## Mathematical Framework

We test if genetic effects on disease are **mediated through signatures**.

### Pathway Model

```
Gene (G) ──[Path a]──> Signature (S) ──[Path b]──> Disease (D)
         \                                          /
          \                                        /
           └──────────[Direct Effect c']──────────┘
```

Where:
- **G**: Rare variant burden (gene-level)
- **S**: Signature loading (θ, individual-level)
- **D**: Disease phenotype (binary, 0/1)

---

## Step 1: Total Effect (Gene → Disease)

**Model:**
```
logit(P(D=1 | G)) = β₀ + c·G
```

**Interpretation:**
- **c** = Total effect of Gene on Disease (log odds ratio)
- If c > 0: Higher burden → Higher disease risk
- If c = 0: No association

**In R:**
```r
model_total <- glm(disease ~ gene_burden, family = binomial(link = "logit"))
c <- coef(model_total)[2]  # Total effect
```

---

## Step 2: Path a (Gene → Signature)

**Model:**
```
S = α₀ + a·G + ε
```

**Interpretation:**
- **a** = Effect of Gene on Signature loading
- If a > 0: Higher burden → Higher signature loading
- If a = 0: No association

**In R:**
```r
model_a <- lm(signature ~ gene_burden)
a <- coef(model_a)[2]  # Path a
```

---

## Step 3: Path b (Signature → Disease | Gene)

**Model:**
```
logit(P(D=1 | S, G)) = β₀ + b·S + c'·G
```

**Interpretation:**
- **b** = Effect of Signature on Disease, controlling for Gene (log odds ratio)
- **c'** = Direct effect of Gene on Disease, controlling for Signature
- If b > 0: Higher signature loading → Higher disease risk (controlling for gene)
- If b = 0: No mediation

**In R:**
```r
model_b <- glm(disease ~ signature + gene_burden, family = binomial(link = "logit"))
b <- coef(model_b)[2]   # Path b (Signature coefficient)
c_prime <- coef(model_b)[3]  # Direct effect (Gene coefficient controlling for Signature)
```

---

## Step 4: Mediation Statistics

### Indirect Effect (Mediated through Signature)

**Formula:**
```
Indirect Effect = a × b
```

**Interpretation:**
- Product of Path a and Path b
- Change in log-odds of disease per unit increase in gene burden, **via its effect on signature**

**Standard Error (Delta method):**
```
SE(indirect) = √(b²·SE(a)² + a²·SE(b)²)
```

**Sobel Test:**
```
Z_sobel = (a × b) / SE(indirect)
P_sobel = 2 × (1 - Φ(|Z_sobel|))
```

### Proportion Mediated

**Formula:**
```
Proportion Mediated = (Indirect Effect) / (Total Effect)
                   = (a × b) / c
```

**Interpretation:**
- Fraction of total gene effect that is mediated through signature
- If = 1.0: Complete mediation (all effect through signature)
- If = 0.0: No mediation (all effect direct)
- If > 1.0 or < 0: Inconsistent mediation (sign suppression or inconsistent effects)

---

## Step 5: Baron-Kenny Criteria

Mediation is supported if **all four conditions** are met:

1. **Gene → Disease** (Total Effect):
   - `c ≠ 0` and `P < 0.05`
   - Gene is associated with disease

2. **Gene → Signature** (Path a):
   - `a ≠ 0` and `P < 0.05`
   - Gene is associated with signature

3. **Signature → Disease | Gene** (Path b):
   - `b ≠ 0` and `P < 0.05`
   - Signature is associated with disease controlling for gene

4. **Direct Effect Reduction**:
   - `|c'| < |c|` (direct effect smaller than total effect)
   - Direct effect `c'` should be smaller (or non-significant) compared to total effect `c`

**Complete Mediation:** `c' = 0` and `P > 0.05` (direct effect eliminated)

**Partial Mediation:** `c' ≠ 0` but `|c'| < |c|` (some direct effect remains)

---

## Example Calculation

### Hypothetical Results:

- **Total Effect (c):** 0.5, SE = 0.1, P < 0.001
- **Path a:** 0.3, SE = 0.05, P < 0.001
- **Path b:** 0.4, SE = 0.08, P < 0.001
- **Direct Effect (c'):** 0.38, SE = 0.11, P < 0.001

### Mediation Statistics:

**Indirect Effect:**
```
Indirect = a × b = 0.3 × 0.4 = 0.12
```

**SE(Indirect):**
```
SE(indirect) = √(0.4² × 0.05² + 0.3² × 0.08²)
             = √(0.16 × 0.0025 + 0.09 × 0.0064)
             = √(0.0004 + 0.000576)
             = √0.000976
             = 0.0312
```

**Sobel Test:**
```
Z_sobel = 0.12 / 0.0312 = 3.85
P_sobel = 2 × (1 - Φ(3.85)) ≈ 0.0001
```

**Proportion Mediated:**
```
Proportion = 0.12 / 0.5 = 0.24 = 24%
```

**Interpretation:**
- 24% of the gene effect on disease is mediated through signature
- 76% is direct (not through signature)
- Mediation is statistically significant (P < 0.001)

---

## Notes

1. **Causal Interpretation:** This is associational, not necessarily causal. Confounding could affect all paths.

2. **Scale Dependency:** On the log-odds scale, mediation statistics are approximations. For binary outcomes with logistic regression, the product method (a × b) is standard but not exact.

3. **Bootstrap Confidence Intervals:** For more robust inference, consider bootstrap confidence intervals for the indirect effect rather than the Sobel test.

4. **Multiple Mediators:** If multiple signatures mediate the gene-disease relationship, this framework extends to parallel or serial mediation models.

