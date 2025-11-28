# Analysis Notebooks for Reviewers

This directory contains all analyses addressing reviewer concerns. Each notebook is self-contained and can be run independently.

## 🎯 Quick Start: Find Analysis by Reviewer Question

### **Referee #1** (Human Genetics, Disease Risk)

| Question | Notebook | What It Shows |
|----------|----------|--------------|
| **Q1**: Selection bias / socioeconomic bias | `ipw_analysis_summary.ipynb` | IPW weighting impact, population representativeness |
| **Q2**: Lifetime risk comparisons | `performancen_notebook_clean.ipynb` (Age Offset section) | Age-offset predictions vs. clinical models |
| **Q3**: Clinical/biological meaningfulness | `fh_analysis_summary.ipynb` | FH carriers show Signature 5 enrichment before events |
| **Q7**: Heritability estimates | `heritability_analysis_summary.ipynb` | LDSC heritability for all signatures |
| **Q9**: AUC vs clinical risk scores | `performancen_notebook_clean.ipynb` (External Scores) | PCE, PREVENT, QRISK3 comparisons |
| **Q10**: Age-specific discrimination | `analyze_age_offset_signatures.py` | AUC by age group |

### **Referee #2** (EHRs)

| Concern | Notebook | What It Shows |
|---------|----------|--------------|
| Temporal accuracy / leakage | `performancen_notebook_clean.ipynb` (Washout section) | Washout window analyses (0yr, 1yr, 2yr) |
| Interpretability | `fh_analysis_summary.ipynb` | Biological pathway validation (FH → Signature 5) |

### **Referee #3** (Statistical Genetics, PRS)

| Question | Notebook | What It Shows |
|----------|----------|--------------|
| **Q1**: Selection/participation bias | `ipw_analysis_summary.ipynb` | IPW weighting, population comparisons |
| **Q3**: Washout windows | `performancen_notebook_clean.ipynb` (Washout section) | Reverse causation prevention |
| **Q4**: Competing risks | ⚠️ **Pending** | Fine-Gray model, cumulative incidence |
| **Q8**: Heterogeneity definition | `heterogeneity_analysis_summary.ipynb` | Pathway analysis, patient heterogeneity |
| **Q12**: Heritability methods | `heritability_analysis_summary.ipynb` | LDSC estimates (SBayesS pending) |

---

## 📚 Summary Notebooks (Start Here)

These notebooks provide concise summaries of key analyses:

1. **`ipw_analysis_summary.ipynb`**
   - Inverse Probability Weighting analysis
   - Population representativeness
   - Weight distributions by subgroup
   - **Addresses**: R1 Q1, R3 Q1 (Selection bias)

2. **`heritability_analysis_summary.ipynb`**
   - LDSC heritability estimates for all 21 signatures
   - Signature 5 (cardiovascular) detailed results
   - Comparison with trait-level heritabilities
   - **Addresses**: R1 Q7, R3 Q12 (Heritability)

3. **`fh_analysis_summary.ipynb`**
   - Familial Hypercholesterolemia (FH) carrier analysis
   - Signature 5 enrichment before ASCVD events
   - Demonstrates biological pathway (LDL → CVD)
   - **Addresses**: R1 Q3 (Clinical meaningfulness)

4. **`heterogeneity_analysis_summary.ipynb`**
   - Disease pathway heterogeneity analysis
   - Identifies 4 distinct pathways to myocardial infarction
   - Quantifies patient heterogeneity (different pathways to same disease)
   - Shows biological heterogeneity (metabolic vs. inflammatory pathways)
   - **Addresses**: R3 Q8 (Heterogeneity definition)

---

## 📊 Main Analysis Notebook

**`performancen_notebook_clean.ipynb`** contains all performance evaluation analyses:

### Sections:
1. **Setup & Data Preparation** - Data loading, paths
2. **Generate Predictions** - Time horizons, washout, age offsets
3. **Load Results** - CSV loading utilities
4. **Comparisons**:
   - External scores (PCE, PREVENT, QRISK3) → **R1 Q9**
   - Delphi comparisons
   - Cox baseline
   - Prediction drops
5. **Visualizations** - All plots and figures
6. **Washout Analyses** → **R2, R3 Q3** (Temporal leakage)
7. **Age Offset Analyses** → **R1 Q2, R1 Q10** (Lifetime risk, age-specific)

---

## 🔧 Supporting Scripts

| Script | Purpose | Reviewer Question |
|--------|---------|-------------------|
| `generate_washout_predictions.py` | Generate washout predictions | R2, R3 Q3 |
| `generate_age_offset_predictions.py` | Generate age-offset predictions | R1 Q2, R1 Q10 |
| `compare_with_external_scores.py` | Compare with PCE/PREVENT | R1 Q9 |
| `compare_age_offset_washout.py` | Compare age offset vs washout | R1 Q2, R3 Q3 |
| `analyze_fh_carriers_signature.py` | FH carrier enrichment | R1 Q3 |
| `analyze_age_offset_signatures.py` | Age-stratified analysis | R1 Q10 |

---

## 📋 Analysis Status

### ✅ **Completed**
- [x] IPW weighting analysis
- [x] Heritability estimates (LDSC)
- [x] FH carrier enrichment
- [x] Washout window analyses
- [x] Age offset predictions
- [x] External score comparisons

### 🔄 **In Progress**
- [ ] Lifetime risk comparisons (needs completion)
- [ ] Age-specific discrimination breakdown
- [ ] Competing risk analysis (Fine-Gray)
- [ ] Cumulative incidence vs. population data

---

## 🗂️ File Organization

```
new_notebooks/
├── README_FOR_REVIEWERS.md          ← You are here
├── REVIEWER_RESPONSE_ORGANIZATION.md ← Detailed mapping
│
├── Summary Notebooks
│   ├── ipw_analysis_summary.ipynb
│   ├── heritability_analysis_summary.ipynb
│   ├── fh_analysis_summary.ipynb
│   └── heterogeneity_analysis_summary.ipynb
│
├── Main Notebook
│   └── performancen_notebook_clean.ipynb
│
└── Supporting Scripts
    ├── generate_washout_predictions.py
    ├── generate_age_offset_predictions.py
    ├── compare_with_external_scores.py
    ├── compare_age_offset_washout.py
    ├── analyze_fh_carriers_signature.py
    └── analyze_age_offset_signatures.py
```

---

## 💡 How to Use This Directory

1. **For specific reviewer questions**: Use the Quick Start table above
2. **For overview**: Start with the three Summary Notebooks
3. **For detailed analysis**: See `performancen_notebook_clean.ipynb`
4. **For reproducibility**: All scripts are self-contained with paths

---

## 📞 Questions?

See `REVIEWER_RESPONSE_ORGANIZATION.md` for detailed mapping of all reviewer concerns to analyses.

---

**Last Updated**: November 27, 2024

