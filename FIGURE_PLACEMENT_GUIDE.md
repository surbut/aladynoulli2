# Figure Placement Guide for LaTeX Files

## Overview
This guide shows where to place figures based on references in `reviewer_response.tex` and `current.tex`.

## Directory Structure Needed

### For `reviewer_response.tex`:
The file references figures with paths like `figures/...` or just filenames. You'll need a `figures/` directory relative to where the LaTeX file is compiled.

**Location:** `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/figures/`

### For `current.tex`:
The file references Extended Data figures with paths like `alffigures/S*.pdf`.

**Location:** `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/alffigures/`

---

## Figures Referenced in `reviewer_response.tex`

### IPW and Selection Bias Figures:
1. **`ukb_age_distribution.png`** (line 267)
   - **Current location:** `supp/s29/ukb_age_distribution.png`
   - **Should be:** `figures/ukb_age_distribution.png` OR copy to root if referenced without path

2. **`figures/ipw_phi_comparison.png`** (line 286)
   - **Current location:** Not found in supp (may need to generate)
   - **Should be:** `figures/ipw_phi_comparison.png`

### RVAS and Genetic Validation:
3. **`figures/rvas.png`** (line 384)
   - **Current location:** `supp/rvas.png`
   - **Should be:** `figures/rvas.png` (copy from supp root)

4. **`FH_signature5_trajectory.png`** (line 394)
   - **Current location:** `supp/FH_signature5_trajectory.png`
   - **Should be:** `figures/FH_signature5_trajectory.png` (copy from supp root)

5. **`figures/Leukemia_MDS_sig16_CHIP_signature_trajectory.png`** (line 407)
   - **Current location:** Not found in supp (may need to generate)
   - **Should be:** `figures/Leukemia_MDS_sig16_CHIP_signature_trajectory.png`

### Multi-Disease Patterns:
6. **`figures/multi_disease_patterns_visualization.png`** (line 464)
   - **Current location:** `supp/multidiseases/multi_disease_patterns_visualization.png`
   - **Should be:** `figures/multi_disease_patterns_visualization.png` (copy from supp/multidiseases/)

### Clinical Score Comparisons:
7. **`figures/external_scores_comparison.png`** (line 711)
   - **Current location:** `supp/external.png`
   - **Should be:** `figures/external_scores_comparison.png` (copy and rename from supp/external.png)

### Delphi Comparisons:
8. **`figures/delphi_comparison_simple_mapping_1yr.png`** (line 1451)
   - **Current location:** `supp/delphi/delphi_comparison_simple_mapping_1yr.png`
   - **Should be:** `figures/delphi_comparison_simple_mapping_1yr.png` (copy from supp/delphi/)

9. **`figures/delphi_comparison_phecode_mapping_1yr_1tomany.png`** (line 1458)
   - **Current location:** `supp/delphi/delphi_comparison_phecode_mapping_1yr_1tomany.png`
   - **Should be:** `figures/delphi_comparison_phecode_mapping_1yr_1tomany.png` (copy from supp/delphi/)

### ICD vs PheCode:
10. **`figures/icd_v_phe.png`** (line 1120)
    - **Current location:** `supp/phecodespng.png`
    - **Should be:** `figures/icd_v_phe.png` (copy and rename from supp/phecodespng.png)

### Population Stratification (PC Analysis):
11. **`figures/pc_phi_comp.png`** (line 1236)
    - **Current location:** Not found in supp (may be in s31 folder)
    - **Should be:** `figures/pc_phi_comp.png`
    - **Note:** Check if this matches `supp/s31/S31_PC_phi_comparison.pdf` (may need to convert)

12. **`figures/pc_lambda_shift.png`** (line 1243)
    - **Current location:** `supp/s31/pc_lambda_shift.png`
    - **Should be:** `figures/pc_lambda_shift.png` (copy from supp/s31/)

13. **`figures/pc_line_plot.png`** (line 1250)
    - **Current location:** `supp/s31/pc_line_plot.png`
    - **Should be:** `figures/pc_line_plot.png` (copy from supp/s31/)

### Washout Analysis:
14. **`figures/fixed_timepoint_horizons.png`** (line 1000)
    - **Current location:** Not found in supp (may be in washoutstuff/)
    - **Should be:** `figures/fixed_timepoint_horizons.png`

15. **`figures/true_washout_comparison_10yr_30yr.png`** (line 1006)
    - **Current location:** `supp/washoutstuff/true_washout_comparison_10yr_30yr.png` ✅
    - **Should be:** `figures/true_washout_comparison_10yr_30yr.png` (copy from supp/washoutstuff/)

16. **`figures/paper_figs/fig5/washout_performance_plot.pdf`** (line 1016)
    - **Current location:** Not found in supp
    - **Should be:** `figures/paper_figs/fig5/washout_performance_plot.pdf`

17. **`figures/washout_performance_plot.png`** (line 1277)
    - **Current location:** Not found in supp (may be in washoutstuff/)
    - **Should be:** `figures/washout_performance_plot.png`

### Competing Risks:
18. **`figures/competing_risk.png`** (line 1357)
    - **Current location:** `supp/s30/competing.png`
    - **Should be:** `figures/competing_risk.png` (copy and rename from supp/s30/competing.png)

19. **`figures/competing_2.png`** (line 1359)
    - **Current location:** Not found in supp
    - **Should be:** `figures/competing_2.png`

20. **`figures/subsequent_disease_temporal_patterns.png`** (line 1366)
    - **Current location:** Check multidiseases/ folder or may be named differently
    - **Should be:** `figures/subsequent_disease_temporal_patterns.png` (may need to locate or generate)

### Calibration:
21. **`figures/calibration_plots_full_400k.pdf`** (line 1391)
    - **Current location:** In main paper Figure 5 (not needed for reviewer response)
    - **Should be:** Skip - already in main paper

### Cohort Signature Probabilities:
22. **`ukb_all_signatures_probabilities.pdf`** (line 1071)
    - **Current location:** `supp/s14-S16/ukb_all_signatures_probabilities.pdf`
    - **Should be:** `figures/ukb_all_signatures_probabilities.pdf` OR root level if no path

23. **`mgb_all_signatures_probabilities.pdf`** (line 1078)
    - **Current location:** `supp/s14-S16/mgb_all_signatures_probabilities.pdf`
    - **Should be:** `figures/mgb_all_signatures_probabilities.pdf` OR root level if no path

24. **`figures/aou_all_signatures_probabilities.pdf`** (line 1086)
    - **Current location:** `supp/s14-S16/aou_all_signatures_probabilities.pdf`
    - **Should be:** `figures/aou_all_signatures_probabilities.pdf` (copy from supp/s14-S16/)

### Other:
25. **`signature_ancestry_lineplot.png`** (line 1184)
    - **Current location:** Not found in supp
    - **Should be:** `figures/signature_ancestry_lineplot.png` OR root level

---

## Figures Referenced in `current.tex` (Extended Data)

All Extended Data figures use path: `alffigures/S*.pdf`

**Location needed:** `/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/Apps/Overleaf/Aladynoulli_Nature/alffigures/`

### Extended Data Figures (S1-S28):

1. **`alffigures/S1.pdf`** (line 912)
   - **Current location:** `supp/s1/S1.pdf` ✅
   - **Should be:** Copy to `alffigures/S1.pdf`

2. **`alffigures/S2.pdf`** (line 1048)
   - **Current location:** `supp/s2/S2.pdf` ✅
   - **Should be:** Copy to `alffigures/S2.pdf`

3. **`alffigures/S3.pdf`** (line 874)
   - **Current location:** `supp/S3.pdf` ✅
   - **Should be:** Copy to `alffigures/S3.pdf`

4. **`alffigures/S4.pdf`** (line 976)
   - **Current location:** `supp/S4.pdf` ✅
   - **Should be:** Copy to `alffigures/S4.pdf`

5. **`alffigures/S5.pdf`** (line 903)
   - **Current location:** `supp/s5/S5.pdf` ✅
   - **Should be:** Copy to `alffigures/S5.pdf`

6. **`alffigures/S6.pdf`** (line 953)
   - **Current location:** `supp/s6/S6.pdf` ✅
   - **Should be:** Copy to `alffigures/S6.pdf`

7. **`alffigures/S7.pdf`** (line 987)
   - **Current location:** `supp/S7.pdf` ✅
   - **Should be:** Copy to `alffigures/S7.pdf`

8. **`alffigures/S8.pdf`** (line 1006)
   - **Current location:** `supp/s8/S8.pdf` ✅
   - **Should be:** Copy to `alffigures/S8.pdf`

9. **`alffigures/S9.pdf`** (line 885)
   - **Current location:** `supp/S9.pdf` ✅
   - **Should be:** Copy to `alffigures/S9.pdf`

10. **`alffigures/S10.pdf`** (line 1017)
    - **Current location:** Not found in supp
    - **Should be:** Copy to `alffigures/S10.pdf` (may need to generate)

11. **`alffigures/S11.pdf`** (line 1038)
    - **Current location:** `supp/s11/S11.pdf` ✅
    - **Should be:** Copy to `alffigures/S11.pdf`

12. **`alffigures/S12.pdf`** (line 944)
    - **Current location:** `supp/S12.pdf` ✅
    - **Should be:** Copy to `alffigures/S12.pdf`

13. **`alffigures/S13.pdf`** (line 894)
    - **Current location:** `supp/S13.pdf` ✅
    - **Should be:** Copy to `alffigures/S13.pdf`

14. **`alffigures/S14.pdf`** (line 920)
    - **Current location:** Not found as S14.pdf (but see s14-S16 folder)
    - **Should be:** Copy to `alffigures/S14.pdf` (may be one of the files in s14-S16/)

15. **`alffigures/S15.pdf`** (line 928)
    - **Current location:** Not found as S15.pdf (but see s14-S16 folder)
    - **Should be:** Copy to `alffigures/S15.pdf` (may be one of the files in s14-S16/)

16. **`alffigures/S16.pdf`** (line 936)
    - **Current location:** Not found as S16.pdf (but see s14-S16 folder)
    - **Should be:** Copy to `alffigures/S16.pdf` (may be one of the files in s14-S16/)

17. **`alffigures/S17.pdf`** (line 967)
    - **Current location:** `supp/s17/` has individual PDFs but not S17.pdf
    - **Should be:** Copy to `alffigures/S17.pdf` (may need to combine or use one of the files)

18. **`alffigures/S28.pdf`** (referenced in text, line 240)
    - **Current location:** Not found in supp
    - **Should be:** Copy to `alffigures/S28.pdf` (may be pathway heterogeneity figure)

### Other Extended Data Files:
- **`alffigures/gail.png`** (line 1027)
  - **Current location:** Not found in supp
  - **Should be:** Copy to `alffigures/gail.png`

- **`alffigures/auc_trends_by_age.pdf`** (line 996)
  - **Current location:** Not found in supp
  - **Should be:** Copy to `alffigures/auc_trends_by_age.pdf`

---

## Quick Action Items

### For `reviewer_response.tex`:
1. Create directory: `figures/` in Overleaf project root
2. Copy/rename these files from supp:
   - `supp/rvas.png` → `figures/rvas.png`
   - `supp/FH_signature5_trajectory.png` → `figures/FH_signature5_trajectory.png`
   - `supp/external.png` → `figures/external_scores_comparison.png`
   - `supp/phecodespng.png` → `figures/icd_v_phe.png`
   - `supp/s29/ukb_age_distribution.png` → `figures/ukb_age_distribution.png`
   - `supp/s31/pc_lambda_shift.png` → `figures/pc_lambda_shift.png`
   - `supp/s31/pc_line_plot.png` → `figures/pc_line_plot.png`
   - `supp/s30/competing.png` → `figures/competing_risk.png`
   - `supp/multidiseases/multi_disease_patterns_visualization.png` → `figures/multi_disease_patterns_visualization.png`
   - `supp/delphi/delphi_comparison_simple_mapping_1yr.png` → `figures/delphi_comparison_simple_mapping_1yr.png`
   - `supp/delphi/delphi_comparison_phecode_mapping_1yr_1tomany.png` → `figures/delphi_comparison_phecode_mapping_1yr_1tomany.png`
   - `supp/s14-S16/ukb_all_signatures_probabilities.pdf` → `figures/ukb_all_signatures_probabilities.pdf`
   - `supp/s14-S16/mgb_all_signatures_probabilities.pdf` → `figures/mgb_all_signatures_probabilities.pdf`
   - `supp/s14-S16/aou_all_signatures_probabilities.pdf` → `figures/aou_all_signatures_probabilities.pdf`

### For `current.tex`:
1. Create directory: `alffigures/` in Overleaf project root
2. Copy all S*.pdf files from supp to alffigures/:
   - `supp/s1/S1.pdf` → `alffigures/S1.pdf`
   - `supp/s2/S2.pdf` → `alffigures/S2.pdf`
   - `supp/S3.pdf` → `alffigures/S3.pdf`
   - `supp/S4.pdf` → `alffigures/S4.pdf`
   - `supp/s5/S5.pdf` → `alffigures/S5.pdf`
   - `supp/s6/S6.pdf` → `alffigures/S6.pdf`
   - `supp/S7.pdf` → `alffigures/S7.pdf`
   - `supp/s8/S8.pdf` → `alffigures/S8.pdf`
   - `supp/S9.pdf` → `alffigures/S9.pdf`
   - `supp/s11/S11.pdf` → `alffigures/S11.pdf`
   - `supp/S12.pdf` → `alffigures/S12.pdf`
   - `supp/S13.pdf` → `alffigures/S13.pdf`

---

## Missing Figures (Need to Generate or Locate)

### For reviewer_response.tex:
- `figures/ipw_phi_comparison.png` (may need to generate from IPW analysis)
- `figures/Leukemia_MDS_sig16_CHIP_signature_trajectory.png` (may need to generate from CHIP analysis)
- `figures/fixed_timepoint_horizons.png` (check washoutstuff/ folder)
- `figures/paper_figs/fig5/washout_performance_plot.pdf` (may be in main paper figures)
- `figures/washout_performance_plot.png` (check washoutstuff/ folder)
- `figures/competing_2.png` (may need to generate or check s30/)
- `figures/subsequent_disease_temporal_patterns.png` (check multidiseases/ folder)
- `figures/signature_ancestry_lineplot.png` (may need to generate)
- `figures/pc_phi_comp.png` (may be `supp/s31/S31_PC_phi_comparison.pdf` - check if needs conversion)
- **SKIP:** `figures/calibration_plots_full_400k.pdf` (already in main paper Figure 5)

### For current.tex:
- `alffigures/S10.pdf`
- `alffigures/S14.pdf`, `S15.pdf`, `S16.pdf` (check s14-S16 folder)
- `alffigures/S17.pdf` (check s17 folder - may need to combine)
- `alffigures/S18-S27.pdf` (GWAS lead SNPs - check supp_lead_snps/)
- `alffigures/S28.pdf` (pathway analysis - check pathway_heterogeneity/)
- `alffigures/gail.png`
- `alffigures/auc_trends_by_age.pdf`

---

## Notes:
- The `reviewer_response.tex` file uses relative paths starting with `figures/`, so all those files should be in a `figures/` subdirectory.
- The `current.tex` file uses `alffigures/` for Extended Data figures.
- Some figures may need format conversion (PDF ↔ PNG).
- Check the `washoutstuff/` folder for washout-related figures.
- Check the `pathway_heterogeneity/` folder for S28.
- Check the `supp_lead_snps/` folder for S18-S27 GWAS figures.

