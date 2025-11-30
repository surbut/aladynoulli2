# Analysis Notebook Mapping: Where Original Analyses Go

This document maps the original analysis notebooks (`fh_analysis_summary.ipynb`, `pc_analysis_clean.ipynb`, `performancen_notebook_clean.ipynb`) to their corresponding reviewer response notebooks.

---

## 📊 Original Analysis Notebooks → Reviewer Response Mapping

### 1. **`fh_analysis_summary.ipynb`** → **`R1_Q3_Clinical_Meaning.ipynb`**

**Location**: `reviewer_responses/notebooks/R1_Q3_Clinical_Meaning.ipynb`

**What it addresses**: 
- **R1 Q3**: Clinical/Biological Meaningfulness
- **R2**: Interpretability of Signatures

**Key findings used**:
- FH carriers show **2.3× enrichment** of Signature 5 rise before ASCVD events (p<0.001)
- Validates LDL/cholesterol → cardiovascular disease pathway
- Demonstrates biological pathway (LDL → CVD)

**Status**: ✅ **Already integrated** - The `R1_Q3_Clinical_Meaning.ipynb` notebook contains the FH analysis code and results.

---

### 2. **`pc_analysis_clean.ipynb`** → **`R3_Population_Stratification_Ancestry.ipynb`**

**Location**: `reviewer_responses/notebooks/R3_Population_Stratification_Ancestry.ipynb`

**What it addresses**:
- **R3**: Population Stratification (continuous ancestry effects)

**Key findings used**:
- PRS distribution across ancestries
- Phi stability (signature ↔ disease associations) with/without PC adjustment
- Theta changes (patient signature loadings) with PC adjustment
- Clinical validation (patients most affected by PC adjustment)

**Status**: ✅ **Already integrated** - The `R3_Population_Stratification_Ancestry.ipynb` notebook references PC analysis findings and uses the same reference theta approach.

**Note**: The PC analysis notebook (`pc_analysis_clean.ipynb`) is in the parent `new_notebooks/` directory and is used as a reference. The reviewer response notebook (`R3_Population_Stratification_Ancestry.ipynb`) contains the specific analysis addressing the reviewer's question about continuous vs. binary ancestry effects.

---

### 3. **`performancen_notebook_clean.ipynb`** → **Multiple Reviewer Response Notebooks**

**Current Location**: `new_notebooks/performancen_notebook_clean.ipynb` (parent directory)

**Where it should be referenced**:

#### **R1 Q2: Lifetime Risk Comparisons**
- **Notebook**: Should create `R1_Q2_Lifetime_Risk.ipynb` or reference performance notebook
- **Section**: Age Offset section
- **What it shows**: Age-offset predictions (ages 40-80) vs. clinical models

#### **R1 Q9: AUC vs Clinical Risk Scores**
- **Notebook**: Should create `R1_Q9_AUC_Comparisons.ipynb` or reference performance notebook  
- **Section**: External Scores section
- **What it shows**: PCE, PREVENT, QRISK3 comparisons

#### **R1 Q10: Age-Specific Discrimination**
- **Notebook**: Should create `R1_Q10_Age_Specific.ipynb` or reference performance notebook
- **Section**: Age Offset section
- **What it shows**: AUC by age group (40-49, 50-59, 60-69, 70+)

#### **R2: Temporal Leakage**
- **Notebook**: `R2_Temporal_Leakage.ipynb` (already exists)
- **Section**: Washout section
- **What it shows**: Washout window analyses (0yr, 1yr, 2yr)

#### **R3 Q3: Washout Windows**
- **Notebook**: `R2_Temporal_Leakage.ipynb` (shared with R2)
- **Section**: Washout section
- **What it shows**: Reverse causation prevention

**Status**: ⚠️ **Needs organization** - The performance notebook is currently in the parent directory and is referenced in multiple places, but there's no dedicated reviewer response notebook that directly links to it.

---

## 🗂️ Recommended Repository Structure

```
reviewer_responses/
├── notebooks/
│   ├── R1_Q3_Clinical_Meaning.ipynb          ← Uses fh_analysis_summary.ipynb
│   ├── R3_Population_Stratification_Ancestry.ipynb  ← Uses pc_analysis_clean.ipynb
│   ├── R1_Q2_Lifetime_Risk.ipynb             ← Should reference performancen_notebook_clean.ipynb
│   ├── R1_Q9_AUC_Comparisons.ipynb            ← Should reference performancen_notebook_clean.ipynb
│   ├── R1_Q10_Age_Specific.ipynb              ← Should reference performancen_notebook_clean.ipynb
│   ├── R2_Temporal_Leakage.ipynb              ← Should reference performancen_notebook_clean.ipynb
│   └── Discovery_Prediction_Framework_Overview.ipynb  ← New framework overview
│
├── docs/
│   ├── README_FOR_REVIEWERS.md
│   ├── REVIEWER_RESPONSE_ORGANIZATION.md
│   └── ANALYSIS_NOTEBOOK_MAPPING.md           ← This file
│
└── (parent directory)/
    ├── fh_analysis_summary.ipynb             ← Original analysis (referenced by R1_Q3)
    ├── pc_analysis_clean.ipynb                ← Original analysis (referenced by R3_Population)
    └── performancen_notebook_clean.ipynb      ← Original analysis (referenced by multiple R notebooks)
```

---

## 📝 Recommendations

### Option 1: Keep Performance Notebook in Parent Directory (Current Approach)
**Pros**: 
- Single source of truth for all performance analyses
- Avoids duplication
- Easy to maintain

**Cons**:
- Reviewers need to navigate between directories
- Less self-contained reviewer response notebooks

**Action**: Add clear references in each reviewer response notebook pointing to the performance notebook with specific section numbers.

### Option 2: Create Dedicated Reviewer Response Notebooks
**Pros**:
- Each reviewer question has its own self-contained notebook
- Easier for reviewers to navigate
- More organized

**Cons**:
- May duplicate code from performance notebook
- Need to maintain consistency

**Action**: Create notebooks like `R1_Q2_Lifetime_Risk.ipynb`, `R1_Q9_AUC_Comparisons.ipynb`, etc., that either:
- Copy relevant sections from `performancen_notebook_clean.ipynb`, OR
- Use `%run` magic to execute specific cells from the performance notebook

### Option 3: Hybrid Approach (Recommended)
**Pros**:
- Balance between organization and maintainability
- Reviewer response notebooks are self-contained but reference the main analysis

**Action**:
1. Keep `performancen_notebook_clean.ipynb` in parent directory as the main analysis
2. Create reviewer response notebooks that:
   - Explain the reviewer question
   - Reference the performance notebook with specific cell numbers
   - Include key results/plots inline
   - Link to the full analysis for details

---

## ✅ Current Status

| Original Notebook | Reviewer Response Notebook | Status |
|------------------|---------------------------|--------|
| `fh_analysis_summary.ipynb` | `R1_Q3_Clinical_Meaning.ipynb` | ✅ Integrated |
| `pc_analysis_clean.ipynb` | `R3_Population_Stratification_Ancestry.ipynb` | ✅ Referenced |
| `performancen_notebook_clean.ipynb` | Multiple (R1 Q2, Q9, Q10, R2, R3 Q3) | ⚠️ Needs organization |

---

## 🎯 Next Steps

1. **For `performancen_notebook_clean.ipynb`**:
   - Decide on approach (Option 1, 2, or 3 above)
   - Create dedicated reviewer response notebooks OR add clear references
   - Update `README_FOR_REVIEWERS.md` with links

2. **Verify all analyses are accessible**:
   - Ensure all paths work from reviewer response notebooks
   - Test that reviewers can run notebooks independently

3. **Create index/table of contents**:
   - Update `REVIEWER_QUESTIONS_INDEX.ipynb` with links to performance notebook sections
   - Add clear navigation between notebooks

