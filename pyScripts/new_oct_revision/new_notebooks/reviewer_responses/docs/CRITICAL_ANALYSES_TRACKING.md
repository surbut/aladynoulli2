# Critical Analyses Tracking - Don't Lose These!

This document tracks all critical analyses that MUST be preserved in the reviewer response structure.

---

## ✅ **1. PC Analysis (With/Without PCs Comparison)**

**Original Notebook**: `pc_analysis_clean.ipynb` (parent directory)

**Key Findings**:
- Phi stability (signature ↔ disease associations) with/without PC adjustment
- Theta changes (patient signature loadings) with/without PC adjustment  
- Clinical validation (patients most affected by PC adjustment)
- PRS distribution across ancestries

**Current Status**: 
- ⚠️ **PARTIALLY INTEGRATED**
- `R3_Population_Stratification_Ancestry.ipynb` uses WITH PCs only
- **MISSING**: WITH vs WITHOUT PCs comparison

**Action Needed**: 
- Add section to `R3_Population_Stratification_Ancestry.ipynb` showing:
  - Phi comparison (with PC vs without PC)
  - Theta shift analysis (how much patients change)
  - Clinical impact (CAD event rates for most affected patients)

**Why Critical**: Shows that PC adjustment doesn't destroy biological interpretations while controlling for population stratification.

---

## ✅ **2. FH Analysis (Clinical Meaningfulness)**

**Original Notebook**: `fh_analysis_summary.ipynb` (parent directory)

**Key Findings**:
- FH carriers show **2.3× enrichment** of Signature 5 rise before ASCVD events (p<0.001)
- Validates LDL/cholesterol → cardiovascular disease pathway
- Demonstrates biological pathway (LDL → CVD)

**Current Status**: 
- ✅ **FULLY INTEGRATED**
- `R1_Q3_Clinical_Meaning.ipynb` contains the full FH analysis

**Action Needed**: None - already preserved!

---

## ✅ **3. IPW Analysis (Selection Bias)**

**Original Notebook**: `ipw_analysis_summary.ipynb` (parent directory)

**Key Findings**:
- IPW weighting impact on model
- Population representativeness comparisons
- Weight distributions by subgroup
- Shows minimal impact on signature structure (mean difference <0.002)

**Current Status**: 
- ⚠️ **NEEDS VERIFICATION**
- `R1_Q1_Selection_Bias.ipynb` exists and mentions IPW
- Need to verify it actually includes the IPW analysis code/results

**Action Needed**: 
- Check if `R1_Q1_Selection_Bias.ipynb` includes IPW analysis or just references it
- If missing, add IPW analysis section

**Why Critical**: Addresses R1 Q1 and R3 Q1 (Selection bias) - core reviewer concern.

---

## ✅ **4. Delphi Comparison (ICD Codes per PheCode)**

**Original Notebook**: `delphicomp.ipynb` (parent directory)

**Key Findings**:
- **21.4× reduction** in predictions needed (53 Phecodes vs 1133 ICD-10 codes)
- Shows aggregation efficiency: top-level ICD-10 codes per PheCode
- Comparison plots showing Aladynoulli vs Delphi efficiency
- Demonstrates parsimony of PheCode approach

**Current Status**: 
- ⚠️ **NOT INTEGRATED INTO REVIEWER RESPONSE**
- Mentioned in `performancen_notebook_clean.ipynb` (around line 988-1039)
- Has dedicated notebook `delphicomp.ipynb` in parent directory
- **MISSING**: No dedicated reviewer response notebook

**Action Needed**: 
- Decide which reviewer question this addresses (likely R1 Q9 or general efficiency question)
- Either:
  - Create dedicated notebook `R1_Q9_Efficiency_PheCode_Aggregation.ipynb`, OR
  - Add section to existing `R1_Q9_AUC_Comparisons.ipynb` (if it exists), OR
  - Ensure `delphicomp.ipynb` is clearly referenced in the reviewer response index

**Why Critical**: Shows major advantage of PheCode aggregation - 21.4× fewer predictions needed!

---

## ✅ **5. Performance Notebook Sections**

**Original Notebook**: `performancen_notebook_clean.ipynb` (parent directory)

**Key Sections to Preserve**:

### 5a. **Delphi Comparison Section** (Lines ~988-1039)
- Aladynoulli vs Delphi AUC comparisons
- Shows which diseases Aladynoulli wins on
- **Status**: In performance notebook, needs clear reference

### 5b. **Age Offset Analyses** (R1 Q2, R1 Q10)
- Lifetime risk predictions
- Age-specific discrimination
- **Status**: Referenced but notebooks `R1_Q2_Lifetime_Risk.ipynb` and `R1_Q10_Age_Specific.ipynb` may not exist yet

### 5c. **Washout Analyses** (R2, R3 Q3)
- 0-year, 1-year, 2-year washout windows
- Temporal leakage prevention
- **Status**: `R2_Temporal_Leakage.ipynb` exists, should reference performance notebook

### 5d. **External Score Comparisons** (R1 Q9)
- PCE, PREVENT, QRISK3 comparisons
- **Status**: Should be in `R1_Q9_AUC_Comparisons.ipynb` (may not exist)

**Action Needed**: 
- Verify which reviewer response notebooks exist
- Ensure each references the correct section of `performancen_notebook_clean.ipynb`
- Add clear cell number references for easy navigation

---

## 📋 Summary Checklist

| Analysis | Original Location | Reviewer Response Location | Status | Action Needed |
|----------|------------------|---------------------------|--------|---------------|
| **PC Analysis (with/without)** | `pc_analysis_clean.ipynb` | `R3_Population_Stratification_Ancestry.ipynb` | ⚠️ Partial | Add WITH/WITHOUT comparison |
| **FH Analysis** | `fh_analysis_summary.ipynb` | `R1_Q3_Clinical_Meaning.ipynb` | ✅ Complete | None |
| **IPW Analysis** | `ipw_analysis_summary.ipynb` | `R1_Q1_Selection_Bias.ipynb` | ⚠️ Verify | Check if included, add if missing |
| **Delphi Comparison** | `delphicomp.ipynb` | ❌ Not integrated | ⚠️ Missing | Create notebook or add to R1_Q9 |
| **Performance Sections** | `performancen_notebook_clean.ipynb` | Multiple notebooks | ⚠️ Verify | Ensure all sections referenced |

---

## 🎯 Immediate Actions

1. **PC Analysis**: Add WITH/WITHOUT PC comparison to `R3_Population_Stratification_Ancestry.ipynb`
2. **IPW Analysis**: Verify `R1_Q1_Selection_Bias.ipynb` includes full IPW analysis
3. **Delphi Comparison**: Decide where to integrate `delphicomp.ipynb` results
4. **Performance Notebook**: Create missing reviewer response notebooks OR add clear references with cell numbers

---

## 📝 Notes

- All original notebooks should remain in parent directory as "source of truth"
- Reviewer response notebooks should either:
  - Include the analysis code directly, OR
  - Clearly reference the original notebook with specific cell/section numbers
- Use this document to verify nothing gets lost during reorganization

---

**Last Updated**: [Current Date]
**Status**: Tracking document - update as analyses are integrated

