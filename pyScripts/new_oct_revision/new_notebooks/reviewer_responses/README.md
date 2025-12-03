# Reviewer Response Analyses

This directory contains all interactive analyses addressing reviewer questions and concerns.

## 📁 Structure

```
reviewer_responses/
├── README.md                      # This file - navigation hub
├── notebooks/
│   ├── R1/                        # Referee #1 analyses (13 notebooks)
│   ├── R2/                        # Referee #2 analyses (3 notebooks)
│   ├── R3/                        # Referee #3 analyses (7 notebooks)
│   ├── framework/                 # Framework overview (1 notebook)
│   └── results/                   # Reviewer-specific results
├── preprocessing/                 # Data preprocessing utilities
│   ├── preprocessing_utils.py     # Standalone preprocessing functions
│   ├── create_preprocessing_files.ipynb  # Interactive preprocessing notebook
│   └── WORKFLOW.md                # Complete workflow documentation
└── SIMPLE_EXAMPLE.py              # Simple example of preprocessing functions
```

## 🎯 How to Use

1. Click on any question below to navigate to its dedicated analysis notebook
2. Each notebook is self-contained and can be run independently
3. All notebooks use the same data paths and setup

## 🔧 Technical Notes

- **Results**: Stored in `notebooks/results/` (within notebooks directory)
- **Source Code**: Shared code is in `pyScripts_forPublish/` (not duplicated here)
- **Paths**: Notebooks use absolute paths for reliability
- **Data**: All notebooks are self-contained and can be run independently

---

## Referee #1: Human Genetics, Disease Risk

| Question | Notebook | Status |
|----------|----------|--------|
| **Q1**: Selection bias / socioeconomic bias | [`notebooks/R1/R1_Q1_Selection_Bias.ipynb`](notebooks/R1/R1_Q1_Selection_Bias.ipynb) | ✅ Complete |
| **Q2**: Lifetime risk comparisons | [`notebooks/R1/R1_Q2_Lifetime_Risk.ipynb`](notebooks/R1/R1_Q2_Lifetime_Risk.ipynb) | ✅ Complete |
| **Q3**: Clinical/biological meaningfulness | [`notebooks/R1/R1_Q3_Clinical_Meaning.ipynb`](notebooks/R1/R1_Q3_Clinical_Meaning.ipynb) | ✅ Complete |
| **Q3**: ICD vs PheCode aggregation | [`notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.ipynb`](notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.ipynb) | ✅ Complete |
| **Q7**: Heritability estimates | [`notebooks/R1/R1_Q7_Heritability.ipynb`](notebooks/R1/R1_Q7_Heritability.ipynb) | ✅ Complete |
| **Q9**: AUC vs clinical risk scores | [`notebooks/R1/R1_Q9_AUC_Comparisons.ipynb`](notebooks/R1/R1_Q9_AUC_Comparisons.ipynb) | ✅ Complete |
| **Q10**: Age-specific discrimination | [`notebooks/R1/R1_Q10_Age_Specific.ipynb`](notebooks/R1/R1_Q10_Age_Specific.ipynb) | ✅ Complete |
| **Additional**: Biological plausibility (CHIP) | [`notebooks/R1/R1_Biological_Plausibility_CHIP.ipynb`](notebooks/R1/R1_Biological_Plausibility_CHIP.ipynb) | ✅ Complete |
| **Additional**: Clinical utility (dynamic risk) | [`notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.ipynb`](notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.ipynb) | ✅ Complete |
| **Additional**: Genetic validation (GWAS) | [`notebooks/R1/R1_Genetic_Validation_GWAS.ipynb`](notebooks/R1/R1_Genetic_Validation_GWAS.ipynb) | ✅ Complete |
| **Additional**: Multi-disease patterns | [`notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.ipynb`](notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.ipynb) | ✅ Complete |
| **Additional**: Robustness (LOO validation) | [`notebooks/R1/R1_Robustness_LOO_Validation.ipynb`](notebooks/R1/R1_Robustness_LOO_Validation.ipynb) | ✅ Complete |

## Referee #2: EHRs

| Concern | Notebook | Status |
|---------|----------|--------|
| Temporal accuracy / leakage | [`notebooks/R2/R2_Temporal_Leakage.ipynb`](notebooks/R2/R2_Temporal_Leakage.ipynb) | ✅ Complete |
| Model validity / learning | [`notebooks/R2/R2_R3_Model_Validity_Learning.ipynb`](notebooks/R2/R2_R3_Model_Validity_Learning.ipynb) | ✅ Complete |
| Washout analysis (continued) | [`notebooks/R2/R2_Washout_Continued.ipynb`](notebooks/R2/R2_Washout_Continued.ipynb) | ✅ Complete |

## Referee #3: Statistical Genetics, PRS

| Question | Notebook | Status |
|----------|----------|--------|
| **Q4**: Competing risks | [`notebooks/R3/R3_Competing_Risks.ipynb`](notebooks/R3/R3_Competing_Risks.ipynb) | ✅ Complete |
| **Q8**: Heterogeneity definition | [`notebooks/R3/R3_Q8_Heterogeneity.ipynb`](notebooks/R3/R3_Q8_Heterogeneity.ipynb) | ✅ Complete |
| **Q8**: Heterogeneity analysis (main paper method) | [`notebooks/R3/R3_Q8_Heterogeneity_MainPaper_Method.ipynb`](notebooks/R3/R3_Q8_Heterogeneity_MainPaper_Method.ipynb) | ✅ Complete |
| **Q8**: Heterogeneity analysis (continued) | [`notebooks/R3/R3_Q8_Heterogeneity_Continued.ipynb`](notebooks/R3/R3_Q8_Heterogeneity_Continued.ipynb) | ✅ Complete |
| **Population Stratification**: Continuous ancestry effects | [`notebooks/R3/R3_Population_Stratification_Ancestry.ipynb`](notebooks/R3/R3_Population_Stratification_Ancestry.ipynb) | ✅ Complete |
| **Additional**: FullE vs ReducedE comparison | [`notebooks/R3/R3_FullE_vs_ReducedE_Comparison.ipynb`](notebooks/R3/R3_FullE_vs_ReducedE_Comparison.ipynb) | ✅ Complete |
| **Additional**: Linear vs Nonlinear mixing | [`notebooks/R3/R3_Linear_vs_NonLinear_Mixing.ipynb`](notebooks/R3/R3_Linear_vs_NonLinear_Mixing.ipynb) | ✅ Complete |

---

## Framework Overview

| Notebook | Description |
|----------|-------------|
| [`notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb`](notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb) | Overview of the discovery and prediction framework |

---

## Preprocessing & Workflow

**Addresses reviewer questions about data preprocessing and the complete analysis workflow.**

| Resource | Description |
|----------|-------------|
| [`preprocessing/WORKFLOW.md`](preprocessing/WORKFLOW.md) | **Complete end-to-end workflow documentation** - Step-by-step guide from preprocessing → batch training → master checkpoint → prediction |
| [`preprocessing/create_preprocessing_files.ipynb`](preprocessing/create_preprocessing_files.ipynb) | Interactive notebook for data preprocessing with visualizations (smoothed prevalence, clustering, signature references) |
| [`preprocessing/preprocessing_utils.py`](preprocessing/preprocessing_utils.py) | Standalone preprocessing functions (`compute_smoothed_prevalence`, `create_initial_clusters_and_psi`, `create_reference_trajectories`) |
| [`preprocessing/SIMPLE_EXAMPLE.py`](preprocessing/SIMPLE_EXAMPLE.py) | Minimal copy-paste example demonstrating how to use the preprocessing functions |

**Workflow Overview:**
1. **Preprocessing**: Create smoothed prevalence, initial clusters, and reference trajectories
2. **Batch Training**: Run `run_aladyn_batch` with FULL E matrix
3. **Master Checkpoint**: Generate master checkpoint (phi and psi)
4. **Prediction**: Run `run_aladyn_predict_with_master` (automatically loads `E_enrollment_full.pt`)

See [`preprocessing/WORKFLOW.md`](preprocessing/WORKFLOW.md) for detailed instructions.

---

## Supporting Analyses

These notebooks provide detailed analyses that support the reviewer responses. They are located in the parent `new_notebooks/` directory:

| Notebook | Description | Used By |
|----------|-------------|---------|
| [`../performancen_notebook_clean.ipynb`](../additional_notebooks/performancen_notebook_clean.ipynb) | Performance evaluation (AUC, comparisons, washout, age offsets) | R1 Q9, R1 Q2, R1 Q10, R2 |
| [`../fh_analysis_summary.ipynb`](../additional_notebooks/fh_analysis_summary.ipynb) | Familial Hypercholesterolemia carrier analysis | R1 Q3 |
| [`../ipw_analysis_summary.ipynb`](../additional_notebooks/ipw_analysis_summary.ipynb) | Inverse Probability Weighting analysis | R1 Q1 |
| [`../pc_analysis_clean.ipynb`](../additional_notebooks/pc_analysis_clean.ipynb) | Principal component adjustment analysis | R3 Population Stratification |
| [`../heritability_analysis_summary.ipynb`](../additional_notebooks/heritability_analysis_summary.ipynb) | LDSC heritability estimates | R1 Q7 |
| [`../heterogeneity_analysis_summary.ipynb`](../additional_notebooks/heterogeneity_analysis_summary.ipynb) | Disease pathway heterogeneity | R3 Q8 |
| [`../washout_analysis_summary.ipynb`](../additional_notebooks/washout_analysis_summary.ipynb) | Washout window analysis | R2, R3 Q3 |
| [`/delphicomp.ipynb`](../additional_notebooks/delphicomp.ipynb) | Delphi comparison analysis | R1 Q9 |

---

## Quick Navigation

### ✅ All Completed Analyses

**Referee #1 (13 notebooks):**
- Selection bias (IPW): [`notebooks/R1/R1_Q1_Selection_Bias.ipynb`](notebooks/R1/R1_Q1_Selection_Bias.ipynb)
- Lifetime risk: [`notebooks/R1/R1_Q2_Lifetime_Risk.ipynb`](notebooks/R1/R1_Q2_Lifetime_Risk.ipynb)
- Clinical meaning (FH): [`notebooks/R1/R1_Q3_Clinical_Meaning.ipynb`](notebooks/R1/R1_Q3_Clinical_Meaning.ipynb)
- ICD vs PheCode aggregation: [`notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.ipynb`](notebooks/R1/R1_Q3_ICD_vs_PheCode_Comparison.ipynb)
- Heritability: [`notebooks/R1/R1_Q7_Heritability.ipynb`](notebooks/R1/R1_Q7_Heritability.ipynb)
- AUC comparisons: [`notebooks/R1/R1_Q9_AUC_Comparisons.ipynb`](notebooks/R1/R1_Q9_AUC_Comparisons.ipynb)
- Age-specific discrimination: [`notebooks/R1/R1_Q10_Age_Specific.ipynb`](notebooks/R1/R1_Q10_Age_Specific.ipynb)
- Biological plausibility (CHIP): [`notebooks/R1/R1_Biological_Plausibility_CHIP.ipynb`](notebooks/R1/R1_Biological_Plausibility_CHIP.ipynb)
- Clinical utility (dynamic risk): [`notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.ipynb`](notebooks/R1/R1_Clinical_Utility_Dynamic_Risk_Updating.ipynb)
- Genetic validation (GWAS): [`notebooks/R1/R1_Genetic_Validation_GWAS.ipynb`](notebooks/R1/R1_Genetic_Validation_GWAS.ipynb)
- Genetic validation (Gene-based RVAS): [`notebooks/R1/R1_Genetic_Validation_Gene_Based_RVAS.ipynb`](notebooks/R1/R1_Genetic_Validation_Gene_Based_RVAS.ipynb)
- Multi-disease patterns: [`notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.ipynb`](notebooks/R1/R1_Multi_Disease_Patterns_Competing_Risks.ipynb)
- Robustness (LOO validation): [`notebooks/R1/R1_Robustness_LOO_Validation.ipynb`](notebooks/R1/R1_Robustness_LOO_Validation.ipynb)

**Referee #2 (3 notebooks):**
- Temporal leakage: [`notebooks/R2/R2_Temporal_Leakage.ipynb`](notebooks/R2/R2_Temporal_Leakage.ipynb)
- Model validity / learning: [`notebooks/R2/R2_R3_Model_Validity_Learning.ipynb`](notebooks/R2/R2_R3_Model_Validity_Learning.ipynb)
- Washout analysis (continued): [`notebooks/R2/R2_Washout_Continued.ipynb`](notebooks/R2/R2_Washout_Continued.ipynb)

**Referee #3 (7 notebooks):**
- Competing risks: [`notebooks/R3/R3_Competing_Risks.ipynb`](notebooks/R3/R3_Competing_Risks.ipynb)
- Heterogeneity: [`notebooks/R3/R3_Q8_Heterogeneity.ipynb`](notebooks/R3/R3_Q8_Heterogeneity.ipynb)
- Heterogeneity analysis (main paper method): [`notebooks/R3/R3_Q8_Heterogeneity_MainPaper_Method.ipynb`](notebooks/R3/R3_Q8_Heterogeneity_MainPaper_Method.ipynb)
- Heterogeneity analysis (continued): [`notebooks/R3/R3_Q8_Heterogeneity_Continued.ipynb`](notebooks/R3/R3_Q8_Heterogeneity_Continued.ipynb)
- Population stratification: [`notebooks/R3/R3_Population_Stratification_Ancestry.ipynb`](notebooks/R3/R3_Population_Stratification_Ancestry.ipynb)
- FullE vs ReducedE comparison: [`notebooks/R3/R3_FullE_vs_ReducedE_Comparison.ipynb`](notebooks/R3/R3_FullE_vs_ReducedE_Comparison.ipynb)
- Linear vs Nonlinear mixing: [`notebooks/R3/R3_Linear_vs_NonLinear_Mixing.ipynb`](notebooks/R3/R3_Linear_vs_NonLinear_Mixing.ipynb)

**Framework (1 notebook):**
- Framework overview: [`notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb`](notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb)

**Total: 24 notebooks** ✅ All complete

