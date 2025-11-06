# Analysis Framework: Pathway Discovery and Validation

## Overview

This document summarizes the comprehensive analysis framework for identifying distinct pathways to myocardial infarction (MI) and validating them across cohorts.

## Three-Level Analysis Structure

### 1. **Deviation-Based Pathway Discovery** 🎯
**What it does**: Identifies distinct patient trajectories by clustering signature deviations from population reference

**Key features**:
- Calculates deviation from population average for each patient
- Clusters patients by their signature deviation patterns
- Identifies distinct pathways (e.g., k=4 chosen for interpretability)
- Uses 10-year lookback before disease onset

**Output**: 
- Pathway assignments for each patient
- Distinct signature trajectories per pathway
- Pathway-specific disease patterns

**Why it works**:
- Deviation captures individual risk patterns relative to population
- Age-independent (compares to same-age population)
- Identifies subclinical patterns not captured by diagnosis codes

---

### 2. **Transition Analysis** 🔄
**What it does**: Compares patients with precursor diseases who develop MI vs. those who don't

**Key features**:
- Identifies patients with precursor disease (e.g., Rheumatoid Arthritis)
- Matches patients who develop MI (progressors) vs. those who don't (non-progressors)
- Compares signature trajectories between groups
- Age-matched to control for age effects

**Output**:
- Signature trajectory differences between progressors and non-progressors
- Identification of signatures that predict transition
- Temporal patterns before disease onset

**Why it works**:
- Directly tests which signatures predict disease progression
- Controls for precursor disease (all have it)
- Identifies signatures that distinguish who progresses vs. doesn't

**Example**: RA → MI transition shows Signature 3 is higher in RA patients who DON'T develop MI (counterintuitive finding)

---

### 3. **Cross-Cohort Reproducibility** 🔁
**What it does**: Validates findings across UKB and MGB cohorts

**Validation approaches**:

#### A. Pathway Reproducibility
- Matches pathways by disease enrichment patterns (not arbitrary indices)
- Compares signature trajectories for matched pathways
- Validates pathway sizes and proportions

#### B. Signature Correspondence
- Maps signatures between cohorts using cross-tabulation
- Compares signature-disease associations (phi values)
- Validates signature trajectory patterns

#### C. Genetic Validation (PRS)
- Compares polygenic risk scores across matched pathways
- Validates that same genetic patterns → same pathways
- Provides biological validation beyond just patterns

#### D. Transition Analysis Reproducibility
- Runs same transition analysis on both cohorts
- Compares signature patterns (e.g., RA → MI)
- Validates that transition signatures are consistent

**Why it works**:
- Reproducibility across cohorts = real biology, not cohort artifacts
- Different healthcare systems, coding practices, populations
- If findings replicate, they're robust

---

## Integrated Analysis Flow

```
1. PATHWAY DISCOVERY (UKB)
   ↓
   Identify distinct pathways by signature deviations
   ↓
2. PATHWAY CHARACTERIZATION
   ↓
   - Signature trajectories
   - Disease patterns
   - Precursor prevalence
   - FH carrier enrichment
   ↓
3. TRANSITION ANALYSIS (UKB)
   ↓
   Compare progressors vs. non-progressors
   ↓
4. REPRODUCIBILITY VALIDATION (MGB)
   ↓
   - Match pathways
   - Compare signatures
   - Validate transitions
   - Genetic validation (PRS)
   ↓
5. INTEGRATED INSIGHTS
   ↓
   Distinct, reproducible pathways to MI
```

---

## Key Strengths of This Framework

### 1. **Multi-Level Validation**
- Pathway-level: Distinct trajectories identified
- Signature-level: Biological mechanisms validated
- Genetic-level: PRS provides biological grounding
- Cohort-level: Reproducibility across systems

### 2. **Complements Clinical Data**
- Deviation analysis: Captures subclinical patterns
- Transition analysis: Tests predictive signatures
- FH carriers: Validates genetic risk component

### 3. **Robust Methodology**
- Deviation-based: Age-independent, population-referenced
- Matched comparisons: Controls for confounders
- Cross-cohort validation: Reduces false discoveries

### 4. **Clinically Relevant**
- Identifies distinct patient types
- Predicts disease progression
- Reproducible across healthcare systems

---

## Example: Complete Analysis for MI

### Step 1: Pathway Discovery
- Input: 24,803 MI patients in UKB
- Method: Deviation-based clustering (k=4)
- Output: 4 distinct pathways

### Step 2: Pathway Characterization
- Pathway 0: High precursor prevalence, high FH carriers (classical atherosclerosis)
- Pathway 1: Low precursors, low FH (hidden risk)
- Pathway 2: Mixed patterns
- Pathway 3: High precursors, high FH (genetic risk)

### Step 3: Transition Analysis
- RA → MI: Signature 3 higher in non-progressors (protective?)
- Validates that signatures predict progression

### Step 4: MGB Reproducibility
- Same 4 pathways identified
- Same signature patterns
- Same transition findings
- PRS validation confirms genetic basis

---

## What This Framework Achieves

✅ **Identifies distinct patient types** with different pathways to MI
✅ **Validates biological mechanisms** through signature patterns
✅ **Predicts disease progression** via transition analysis
✅ **Demonstrates reproducibility** across cohorts
✅ **Provides genetic validation** through PRS
✅ **Clinically interpretable** pathways for personalized medicine

---

## Next Steps / Future Directions

1. **Validate pathways in additional cohorts**
2. **Test pathway-specific interventions**
3. **Develop pathway-based risk prediction models**
4. **Integrate with additional omics data**
5. **Longitudinal validation in prospective cohorts**

