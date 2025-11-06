# How Pathway Similarity is Calculated

## Overview

Pathway similarity is calculated by comparing **disease enrichment patterns** between pathways. This allows us to match pathways across cohorts (UKB vs MGB) based on their **biological content**, not arbitrary index numbers.

---

## Step-by-Step Process

### Step 1: Extract Disease Enrichment Patterns

For each pathway, we calculate which diseases are most enriched (have highest prevalence).

**Formula**:
```
For each pathway:
  1. Get all patients in this pathway
  2. Calculate disease prevalence in pathway: 
     pathway_prevalence[d] = (number of patients with disease d) / (total patients in pathway)
  
  3. Calculate overall disease prevalence (across all MI patients):
     overall_prevalence[d] = (number of all MI patients with disease d) / (total MI patients)
  
  4. Calculate enrichment ratio:
     enrichment_ratio[d] = pathway_prevalence[d] / overall_prevalence[d]
  
  5. Sort diseases by enrichment ratio (highest first)
  6. Keep top N diseases (default: top 30)
```

**Example**:
- Pathway 0 has 1,836 patients
- 500 patients in Pathway 0 have "coronary atherosclerosis"
- Overall, 5,000 out of 24,803 MI patients have "coronary atherosclerosis"
- Pathway 0 prevalence: 500/1,836 = 0.272 (27.2%)
- Overall prevalence: 5,000/24,803 = 0.202 (20.2%)
- Enrichment ratio: 0.272 / 0.202 = **1.35x** (35% more common in Pathway 0)

---

### Step 2: Match Disease Names Between Cohorts

Since UKB and MGB may use different disease names (e.g., "myocardial_infarction" vs "MI"), we use **fuzzy keyword matching**.

**Algorithm**:
```python
For each UKB disease:
  1. Extract key terms (remove common words like "the", "disease", etc.)
  2. For each MGB disease:
     a. Check for exact word matches
     b. Check for substring matches (e.g., "myocardial" in "myocardial_infarction")
     c. Check for character overlap (fuzzy matching)
  3. Score each potential match
  4. Return best matches (score > 0.2)
```

**Example**:
- UKB: "coronary_artery_disease"
- MGB: "coronary_artery_disease" → **Exact match!**
- MGB: "coronary_heart_disease" → **High similarity** (shared words: "coronary", "disease")
- MGB: "diabetes" → **Low similarity** (no shared terms)

---

### Step 3: Create Disease Mapping

For each pathway pair (UKB Pathway X vs MGB Pathway Y), we create a mapping of matched diseases.

**Process**:
1. **Forward matching**: For each UKB disease in Pathway X, find matching MGB diseases in Pathway Y
2. **Reverse matching**: For each MGB disease in Pathway Y, find matching UKB diseases in Pathway X
3. **Filter**: Only keep diseases with enrichment ratio > 1.1x in BOTH pathways

**Result**: A dictionary mapping UKB diseases to MGB diseases with their enrichment ratios:
```python
{
  "coronary_artery_disease": {
    "ukb_enrichment": 1.35,  # 35% more common in UKB Pathway X
    "mgb_enrichment": 1.42,  # 42% more common in MGB Pathway Y
    "mgb_disease_name": "coronary_artery_disease"
  },
  "diabetes": {
    "ukb_enrichment": 1.28,
    "mgb_enrichment": 1.31,
    "mgb_disease_name": "diabetes_mellitus"
  },
  ...
}
```

---

### Step 4: Calculate Similarity Metrics

We compute **three different similarity metrics** and combine them:

#### Metric 1: Cosine Similarity
**Formula**:
```
cosine_similarity = (ukb_enrichments · mgb_enrichments) / (||ukb_enrichments|| × ||mgb_enrichments||)
```

**What it measures**: Directional similarity (are the enrichment patterns pointing in the same direction?)

**Range**: [-1, 1]
- +1: Perfect alignment (same direction)
- 0: Orthogonal (no relationship)
- -1: Opposite directions

**Example**:
- UKB Pathway: [1.5, 1.3, 1.2] (coronary, diabetes, hypertension)
- MGB Pathway: [1.4, 1.3, 1.25] (coronary, diabetes, hypertension)
- Cosine similarity: **~0.99** (very similar!)

#### Metric 2: Pearson Correlation
**Formula**:
```
correlation = Corr(ukb_enrichments, mgb_enrichments)
```

**What it measures**: Linear relationship (do enrichment ratios scale together?)

**Range**: [-1, 1]
- +1: Perfect positive linear relationship
- 0: No linear relationship
- -1: Perfect negative linear relationship

**Example**:
- UKB Pathway: [1.5, 1.3, 1.2]
- MGB Pathway: [1.4, 1.3, 1.25]
- Pearson correlation: **~0.98** (strong linear relationship)

#### Metric 3: Spearman Rank Correlation
**Formula**:
```
rank_correlation = SpearmanCorr(rank(ukb_enrichments), rank(mgb_enrichments))
```

**What it measures**: Rank order preservation (are diseases ranked similarly?)

**Range**: [-1, 1]
- +1: Perfect rank order match
- 0: No rank order relationship
- -1: Perfect reverse rank order

**Example**:
- UKB Pathway: coronary (1.5) > diabetes (1.3) > hypertension (1.2)
- MGB Pathway: coronary (1.4) > diabetes (1.3) > hypertension (1.25)
- Spearman correlation: **1.0** (perfect rank order match!)

---

### Step 5: Combine Metrics

**Formula**:
```python
similarity = 0.5 × cosine_similarity + 0.3 × max(0, pearson_correlation) + 0.2 × max(0, spearman_correlation)
```

**Weights**:
- **50%** cosine similarity (directional alignment)
- **30%** Pearson correlation (linear scaling)
- **20%** Spearman correlation (rank order)

**Why these weights?**
- Cosine similarity is most important (captures overall pattern similarity)
- Pearson correlation captures magnitude relationships
- Spearman correlation captures ordering (less important but still useful)

**Range**: [0, 1] (we take max(0, ...) for correlations to avoid negative contributions)

---

### Step 6: Find Optimal Pathway Matching

We use the **Hungarian algorithm** (optimal assignment) to find the best one-to-one matching between UKB and MGB pathways.

**Process**:
1. Create a **similarity matrix**: All pairwise similarities between UKB and MGB pathways
2. Convert to **cost matrix**: `cost = 1 - similarity` (Hungarian minimizes cost)
3. Find **optimal assignment**: One-to-one matching that maximizes total similarity
4. Return **best matches**: `{ukb_pathway_id: mgb_pathway_id, ...}`

**Example**:
```
Similarity Matrix:
        MGB 0  MGB 1  MGB 2  MGB 3
UKB 0   0.12   0.85   0.23   0.45
UKB 1   0.34   0.15   0.78   0.22
UKB 2   0.67   0.31   0.19   0.56
UKB 3   0.23   0.42   0.11   0.89

Optimal Assignment:
UKB 0 ↔ MGB 1 (similarity: 0.85)
UKB 1 ↔ MGB 2 (similarity: 0.78)
UKB 2 ↔ MGB 0 (similarity: 0.67)
UKB 3 ↔ MGB 3 (similarity: 0.89)
```

---

## Key Parameters

### `top_n_diseases` (default: 30)
- Number of top diseases to consider for matching
- Higher = more diseases considered, but may include noise
- Lower = fewer diseases, but may miss important matches

### `enrichment_threshold` (default: 1.1x)
- Minimum enrichment ratio to consider a disease
- 1.1x = disease must be at least 10% more common in pathway than overall
- Higher = more stringent (only very enriched diseases)
- Lower = more lenient (includes more diseases)

### `fuzzy_match_threshold` (default: 0.2)
- Minimum similarity score for disease name matching
- Higher = more stringent (only very similar names)
- Lower = more lenient (allows more fuzzy matches)

---

## Example: Full Calculation

### UKB Pathway 0:
- Top diseases:
  1. Coronary artery disease (enrichment: 1.45x)
  2. Diabetes (enrichment: 1.32x)
  3. Hypertension (enrichment: 1.28x)
  4. Heart failure (enrichment: 1.15x)

### MGB Pathway 2:
- Top diseases:
  1. Coronary artery disease (enrichment: 1.52x)
  2. Diabetes mellitus (enrichment: 1.29x)
  3. Hypertension (enrichment: 1.31x)
  4. Heart failure (enrichment: 1.18x)

### Matching:
1. **Coronary artery disease** ↔ **Coronary artery disease** (exact match)
2. **Diabetes** ↔ **Diabetes mellitus** (fuzzy match)
3. **Hypertension** ↔ **Hypertension** (exact match)
4. **Heart failure** ↔ **Heart failure** (exact match)

### Enrichment Vectors:
- UKB: [1.45, 1.32, 1.28, 1.15]
- MGB: [1.52, 1.29, 1.31, 1.18]

### Similarity Metrics:
- Cosine similarity: **0.998** (almost perfect alignment)
- Pearson correlation: **0.995** (strong linear relationship)
- Spearman correlation: **1.0** (perfect rank order)

### Combined Similarity:
```
similarity = 0.5 × 0.998 + 0.3 × 0.995 + 0.2 × 1.0
           = 0.499 + 0.299 + 0.2
           = 0.998
```

**Result**: UKB Pathway 0 and MGB Pathway 2 are **highly similar** (0.998)!

---

## Why This Works

1. **Biological Content**: Matches pathways by what diseases they're associated with, not arbitrary labels
2. **Robust to Naming Differences**: Fuzzy matching handles different disease naming conventions
3. **Multiple Metrics**: Combines directional, linear, and rank-order similarities for robustness
4. **Optimal Matching**: Hungarian algorithm ensures best possible one-to-one matching

---

## Limitations

1. **Disease Name Matching**: May miss matches if disease names are very different
2. **Enrichment Threshold**: May miss pathways with subtle disease patterns
3. **Top N Diseases**: May miss important diseases beyond top N
4. **Cohort Differences**: Different disease prevalences between cohorts can affect enrichment ratios

---

## Code Reference

See `match_pathways_by_disease_patterns.py`:
- `extract_disease_enrichment_by_pathway()`: Step 1
- `match_disease_names_by_keywords()`: Step 2
- `calculate_pathway_similarity()`: Steps 3-5
- `match_pathways_between_cohorts()`: Step 6

