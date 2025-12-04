# Numbers Verified and Updated in reviewer_response.tex

## Age-Stratified Analysis (from age_stratified_auc_results.csv)

### ASCVD 10-year predictions:
- Ages 39-50: AUC = 0.650 ✓
- Ages 50-60: AUC = 0.694 ✓
- Ages 60-72: AUC = 0.705 ✓

### ASCVD 1-year predictions:
- Ages 39-50: AUC = 0.802 ✓
- Ages 50-60: AUC = 0.911 ✓
- Ages 60-72: AUC = 0.934 ✓

### Diabetes 1-year predictions:
- Ages 39-50: AUC = 0.621 ✓
- Ages 50-60: AUC = 0.707 ✓
- Ages 60-72: AUC = 0.925 ✓

## Clinical Score Comparisons (from external_scores_comparison.csv)

### ASCVD 10-year:
- Aladynoulli: 0.737 ✓
- PCE: 0.683 ✓
- QRISK3: 0.702 ✓
- Improvement vs PCE: +7.9% ✓
- Improvement vs QRISK3: +5.0% ✓

### ASCVD 30-year:
- Aladynoulli: 0.708 ✓
- PREVENT: 0.650 ✓
- Improvement: +9.0% ✓

### Breast Cancer 10-year (females):
- Aladynoulli: 0.556 ✓
- GAIL: 0.539 ✓
- Improvement: +3.2% ✓

## Cox Baseline Comparisons (from cox_baseline_comparison_static10yr_wins.csv)

All improvements verified:
- Parkinson's: +35.2% (AUC 0.722 vs 0.534) ✓
- CKD: +33.2% (AUC 0.705 vs 0.529) ✓
- Prostate Cancer: +31.9% (AUC 0.684 vs 0.519) ✓
- Stroke: +31.6% (AUC 0.681 vs 0.518) ✓
- COPD: +25.8% (AUC 0.659 vs 0.524) ✓
- All Cancers: +24.0% (AUC 0.671 vs 0.541) ✓
- Colorectal Cancer: +24.0% (AUC 0.646 vs 0.521) ✓
- Atrial Fibrillation: +20.2% (AUC 0.707 vs 0.588) ✓
- Lung Cancer: +20.7% (AUC 0.668 vs 0.554) ✓
- Heart Failure: +18.6% (AUC 0.702 vs 0.592) ✓
- ASCVD: +16.3% (AUC 0.737 vs 0.634) ✓

## Delphi-2M Comparisons (from delphi_comparison_1yr_wins_0gap.csv and washout_vs_delphi_key_diseases.csv)

### 1-year predictions with 1-year washout:
- Secondary Cancer: +26.8% (AUC 0.629 vs 0.361) ✓
- Breast Cancer: +14.9% (AUC 0.679 vs 0.530) ✓
- All Cancers: +13.9% (AUC 0.691 vs 0.551) ✓
- Lung Cancer: +12.7% (AUC 0.799 vs 0.672) ✓
- Prostate Cancer: +11.8% (AUC 0.737 vs 0.619) ✓
- Colorectal Cancer: +10.7% (AUC 0.694 vs 0.587) ✓
- ASCVD: +8.1% (AUC 0.742 vs 0.661) ✓

### 1-year predictions with 0-year washout:
- Parkinson's: +21.6% (AUC 0.827 vs 0.611) ✓
- Multiple Sclerosis: +18.2% (AUC 0.836 vs 0.655) ✓
- ASCVD: +15.9% (AUC 0.896 vs 0.737) ✓

## Time Horizon Analysis (from comparison_all_horizons.csv)

### ASCVD:
- 5-year: AUC = 0.761 ✓
- 10-year: AUC = 0.718 ✓
- 30-year: AUC = 0.708 ✓
- Static 10-year: AUC = 0.737 ✓

## Washout Analysis (from comprehensive_washout_results.csv)

### ASCVD at timepoint 1:
- 0-year washout: AUC = 0.862 ✓
- 1-year washout: AUC = 0.742 ✓
- 2-year washout: AUC = 0.747 ✓

## CHIP Analysis (from chip_multiple_signatures_summary.csv)

### DNMT3A carriers:
- Leukemia/MDS, Signature 16: OR = 1.97, p = 0.0007 ✓
- Carrier prop rising: 81.1% vs non-carrier 68.5% ✓

### TET2 carriers:
- Heart Failure, Signature 16: OR = 1.61, p = 8.9×10⁻⁵ ✓

### CHIP (combined):
- Leukemia/MDS, Signature 16: OR = 1.60, p = 0.0004 ✓
- Heart Failure, Signature 16: OR = 1.17, p = 0.003 ✓
- COPD, Signature 16: OR = 1.25, p = 2.9×10⁻⁶ ✓
- Anemia, Signature 16: OR = 1.16, p = 0.0003 ✓

## Familial Hypercholesterolemia (from FH_signature5_enrichment.csv)

- n_carriers: 464 ✓
- n_noncarriers: 55,844 ✓
- prop_carriers_rising: 95.9% ✓
- prop_noncarriers_rising: 93.5% ✓
- OR: 1.63 ✓
- p_value: 0.017 ✓

## Age Offset Analysis (from age_offset_aucs_summary_batch_0_10000.csv)

### ASCVD across 10 age offsets:
- Mean AUC: 0.894 ✓
- Median AUC: 0.898 ✓
- Range: 0.862-0.920 ✓

## Numbers Still Requiring Verification

The following numbers may need to be checked against source data:

1. **Cross-cohort validation**: 79.2% signature correspondence (median modified Jaccard index = 0.792, IQR = 0.65-0.89)
2. **Genetic validation**: 
   - 150 genome-wide significant loci
   - Cardiovascular h² = 0.041 (SE = 0.003)
   - Musculoskeletal h² = 0.035 (SE = 0.002)
   - Pain/inflammation h² = 0.027 (SE = 0.002)
   - 75 significant PRS associations (9.9% of tests)
3. **RVAS results**:
   - 7 genome-wide significant genes
   - LDLR: p = 1.05×10⁻³³
   - APOB: p = 7.16×10⁻¹⁰
   - BRCA2: p = 3.91×10⁻¹¹
   - PKD1: p = 3.98×10⁻¹⁵
4. **LOO robustness**: mean SE = 0.0010, median SE = 0.0002, 95% of SE values ≤ 0.004
5. **IPW analysis**: mean difference <0.002
6. **Genomic control**: λgc = 1.02-1.22
7. **Effect sizes**: Cohen's d = 2.46-3.87
8. **Population sizes**: UKB n=427,239, MGB n=48,069, AoU n=208,263
9. **Ancestry percentages**: UKB 90.6% EUR, MGB 69.2% EUR, AoU 40.4% EUR

## Summary

All age-stratified AUC values, clinical score comparisons, Cox baseline improvements, Delphi comparisons, time horizon results, washout results, CHIP analysis, and FH analysis numbers have been verified and updated in the document using the actual data files.

