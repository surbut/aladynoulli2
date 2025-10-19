# Deviation-Based Clustering: Why It Matters

## The Problem You Identified

You correctly pointed out that clustering by **raw signature loadings at age of onset** will capture **age-related patterns** common to the entire population, not the **deviations that make pathways truly distinct**.

### Example:
- **Patient A** (age 55 at MI): Signature 3 = 0.8
- **Patient B** (age 55 at MI): Signature 3 = 0.8
- **Population average** (age 55): Signature 3 = 0.5

**Problem**: Patient A and B look identical (both 0.8), but they're BOTH elevated compared to population!

**Solution**: Cluster by **deviation from population reference**:
- **Patient A deviation**: 0.8 - 0.5 = +0.3
- **Patient B deviation**: 0.8 - 0.5 = +0.3

Now we can distinguish:
- **Patient C** (age 55 at MI): Signature 3 = 0.8, Signature 5 = 0.9
- **Population average** (age 55): Signature 3 = 0.5, Signature 5 = 0.2
- **Patient C deviations**: Sig 3 = +0.3, **Sig 5 = +0.7** ← inflammatory pathway!

## Three Clustering Methods Compared

### Method 1: Average Loading (Raw Values)
**What it does**: Clusters by average signature loading 5 years before disease

**Features**: `[Sig1, Sig2, ..., Sig21]` (21 features)

**Pros**:
- Simple
- Interpretable
- Matches your R script approach

**Cons**:
- **Confounded by age** - older patients have higher loadings
- May cluster by "age at onset" rather than "pathway to disease"
- Misses patients who deviate in opposite directions from population

**When to use**: When you want to find "high-risk" vs. "low-risk" groups

---

### Method 2: Trajectory Similarity (Dynamics)
**What it does**: Clusters by trajectory dynamics (average + slope + variance + peaks + age)

**Features**: `[Sig1_avg, ..., Sig21_avg, Sig1_slope, ..., Sig21_slope, Sig1_var, ..., Sig21_var, Sig1_peak, ..., Sig21_peak, age/100]` (85 features)

**Pros**:
- Captures temporal dynamics
- Includes instability/variance
- More sophisticated

**Cons**:
- **Still confounded by age** - slope and peaks are age-related
- Complex (85 features)
- Harder to interpret

**When to use**: When you care about "accelerating" vs. "stable" pathways

---

### Method 3: Deviation from Reference (Age-Independent) ← **NEW**
**What it does**: Clusters by **deviation from population average** in the same time window

**Features**: `[Sig1_dev, ..., Sig21_dev, Sig1_var_dev, ..., Sig21_var_dev]` (42 features)

**Calculation**:
```python
# For each patient:
# 1. Get their pre-disease trajectory (5 years before MI)
pre_disease_traj = theta_patient[:, age-5:age]  # Shape: (21 sigs, 5 years)

# 2. Get population reference for SAME ages
ref_traj = population_average[:, age-5:age]  # Shape: (21 sigs, 5 years)

# 3. Calculate deviation
deviation = pre_disease_traj - ref_traj  # What makes THIS patient different?

# 4. Average deviation over 5 years
avg_deviation = mean(deviation, axis=time)  # (21,)

# 5. Also include variance in deviation (instability)
dev_variance = var(deviation, axis=time)  # (21,)

# 6. Cluster on [avg_deviation, dev_variance]
```

**Pros**:
- **Removes age confounding** - deviations are relative to age-matched population
- Identifies truly distinct biological pathways
- Interpretable: "This patient had +0.7 elevated Sig 5 compared to typical"
- Matches your R script philosophy (deviation plots)

**Cons**:
- Requires computing population reference (but we have 400K patients!)
- May miss absolute magnitude effects

**When to use**: When you want **biologically meaningful heterogeneity**, not just age/risk stratification

---

## Why Deviation Method is Most Interesting

### 1. **Removes Age Confounding**
Two 40-year-olds with MI are compared to typical 40-year-olds.
Two 70-year-olds with MI are compared to typical 70-year-olds.

### 2. **Identifies Biological Pathways**
- **Pathway 1**: High deviation in Sig 5 (inflammatory) → "inflammatory pathway to MI"
- **Pathway 2**: High deviation in Sig 12 (metabolic) → "metabolic pathway to MI"
- **Pathway 3**: Low deviation in Sig 5 and 12, high deviation in Sig 3 (CV) → "traditional risk factor pathway"

### 3. **Matches Your Visualization Approach**
Your R script plots **deviations from reference** - now clustering does the same!

### 4. **Clinically Actionable**
"You're 2 standard deviations above population on inflammatory signature" is more actionable than "Your inflammatory signature is 0.8"

### 5. **Handles Bi-Directional Deviations**
- Patient A: +0.7 on Sig 5 (more inflammatory than typical)
- Patient B: -0.3 on Sig 5 (less inflammatory than typical)
- Both are "different" but Method 1 would just see absolute values

---

## Expected Results

### Method 1 (Raw Loading) Might Show:
- **Pathway 0**: Young onset (age 40-50), low loadings
- **Pathway 1**: Mid-age onset (age 50-60), medium loadings
- **Pathway 2**: Late onset (age 60-70), high loadings
- **Pathway 3**: Very late onset (age 70-80), very high loadings

**Interpretation**: Mostly age-stratified

---

### Method 3 (Deviation) Should Show:
- **Pathway 0**: High inflammatory deviation (Sig 5 +0.5), rheumatologic diseases
- **Pathway 1**: High metabolic deviation (Sig 12 +0.6), diabetes, obesity
- **Pathway 2**: High traditional CV deviation (Sig 3 +0.4), hypertension, hyperlipidemia
- **Pathway 3**: Mixed/balanced deviations, less clear pre-disease signal

**Interpretation**: Biologically distinct pathways, age-independent

---

## How to Interpret Results

### Look for:

1. **Discriminating Signatures** (from interrogation output)
   - Which signatures most differentiate pathways?
   - Do these align with known biology?

2. **Pre-Disease Conditions** (from disease analysis)
   - Pathway 0: Rheumatoid arthritis, lupus, psoriasis? → inflammatory
   - Pathway 1: Type 2 diabetes, obesity, metabolic syndrome? → metabolic
   - Pathway 2: Hypertension, hypercholesterolemia, angina? → traditional

3. **Medication Patterns** (from medication integration)
   - Pathway 0: Immunosuppressants, biologics?
   - Pathway 1: Metformin, insulin, GLP-1 agonists?
   - Pathway 2: Statins, ACE inhibitors, beta blockers?

4. **PRS Patterns** (from PRS analysis)
   - Pathway 0: Higher inflammatory disease PRS?
   - Pathway 1: Higher diabetes PRS?
   - Pathway 2: Higher CAD PRS?

---

## For the Paper

### Key Message:
> "To identify pathways that reflect biological heterogeneity rather than age-stratified risk, we clustered patients by their **deviation from age-matched population reference** in the 5 years before disease onset. This approach revealed [N] distinct pathways characterized by differential elevation in specific signatures."

### Figure:
**Panel A**: Heatmap of deviations by pathway
- Rows: Pathways
- Columns: Signatures
- Values: Average deviation from reference
- Color: Red = elevated, Blue = suppressed, White = typical

**Panel B**: Signature trajectory plots
- X-axis: Years before MI
- Y-axis: Deviation from reference
- Lines: Each pathway
- Shows how pathways diverge over time

**Panel C**: Disease characterization
- Rows: Pathways
- Columns: Top differentiating diseases
- Values: % prevalence (corrected for pathway size)

**Panel D**: PRS enrichment
- Bars: Pathways
- Y-axis: PRS z-score (relative to population)
- Different bars for different PRS (CAD, diabetes, inflammatory)

---

## Running the Analysis

```python
from run_pathway_analysis import run_complete_pathway_analysis

# This now runs ALL THREE methods
results = run_complete_pathway_analysis("myocardial infarction", n_pathways=4)

# Compare the three methods:
print("Method 1 (Average Loading):")
print(results['pathway_data_avg'])

print("\nMethod 2 (Trajectory Similarity):")
print(results['pathway_data_traj'])

print("\nMethod 3 (Deviation from Reference): ← RECOMMENDED")
print(results['pathway_data_dev'])

# Access deviation-specific results:
dev_results = results['results_dev']
dev_medications = results['medication_results_dev']
dev_prs = results['prs_results_dev']
dev_diseases = results['granular_results_dev']
```

---

## Troubleshooting

### "All pathways look the same"
→ Try different `n_pathways` (3, 4, 5, 6)
→ Check if target disease has sufficient heterogeneity
→ Look at discriminating signature scores

### "Deviations are too small"
→ This is expected - most patients are close to population average
→ Look at **relative** differences between pathways
→ Focus on signatures with high variance across pathways

### "Results don't make biological sense"
→ Check pre-disease conditions for each pathway
→ Validate with medication patterns
→ Compare with PRS results
→ May need more/fewer pathways

---

## Next Steps

1. **Run all three methods** on MI
2. **Compare pathway characterizations**
   - Which method gives most interpretable results?
   - Which aligns best with known biology?
3. **Focus on deviation method** for main paper figures
4. **Show all three** in supplementary material for robustness
5. **Validate** with medications and PRS

---

## Summary

The deviation-based method addresses your concern about age confounding by:
1. Computing population-level signature reference trajectories
2. Calculating patient-specific deviations from this reference
3. Clustering on deviations (not raw values)
4. Identifying pathways that differ biologically, not just by age

This gives you **interpretable, biologically meaningful heterogeneity** rather than just risk stratification.

