# Supplementary Figures Status Check

## ✅ PRESENT in `/supp` folder:

### Extended Data Figures (S1-S28):
- ✅ **S1**: `s1/S1.pdf` + individual disease files
- ✅ **S2**: `s2/S2.pdf`
- ✅ **S3**: `S3.pdf` (root level)
- ✅ **S4**: `S4.pdf` (root level)
- ✅ **S5**: `S5.pdf` (root level) + `s5/S5.pdf`
- ✅ **S6**: `s6/S6.pdf` + individual patient files
- ✅ **S7**: `S7.pdf` (root level)
- ✅ **S8**: `s8/S8.pdf`
- ✅ **S11**: `s11/S11.pdf`
- ✅ **S16-S18**: `s16-s18/` (UKB, MGB, AOU temporal patterns)

### New Extended Data Figures:
- ✅ **S29**: `s29/S29.pdf` + supporting files (weight distribution, age distribution, etc.)
- ✅ **S30**: `s30/S30.pdf`
- ✅ **S31**: `s31/S31.pdf` + supporting files (PC plots, phi comparison, shift heatmap)

### Additional Files Present:
- `competing.png` (might be for S30)
- `external.png` (external scores comparison)
- `delphi_comparison_*.png` (Delphi comparison figures)
- `phecodespng.png` (PheCode comparison)
- `multi_disease_patterns_visualization.png`
- `true_washout_comparison_10yr_30yr.png` (needs update per TODO)
- Various other supporting figures

---

## ❌ MISSING from `/supp` folder:

### Extended Data Figures (S1-S28):
- ❌ **S9**: Enrollment timeline
  - Found: `s9/ascvd_auc_vs_offset_by_age_group_batch_0_10000.pdf` (doesn't match description)
  - TODO says: "Keep as-is" but file not found
  
- ❌ **S12**: Data structure and model components
  - TODO says: "Keep as-is"
  
- ❌ **S13**: Cohort characteristics and study design
  - TODO says: "Keep as-is"
  
- ❌ **S14**: Temporal patterns UKB
  - TODO says: "Already regenerated with corrected E matrix"
  - Note: S16-S18 folder exists but might not contain S14-S15 separately
  
- ❌ **S15**: Temporal patterns All of Us
  - TODO says: "Already regenerated with corrected E matrix"
  - Note: Check if `s16-s18/aou_all_signatures_probabilities.pdf` is S15
  
- ❌ **S17**: Cohen's d effect sizes
  - TODO says: "Keep as-is"
  
- ❌ **S18-S27**: GWAS lead SNPs (21 signatures)
  - TODO says: "Keep as-is"
  - One file per signature (S7-S27 in Extended Data Files)
  - Note: These might be in a different location or need to be generated

- ❌ **S28**: Pathway analysis for MI heterogeneity
  - TODO says: "User considering whether to keep"
  - Note: Found `mgb_deviation_analysis_output/` with pathway-related PDFs, but not specifically S28

### New Extended Data Figures:
- ❌ **S32**: Censoring distributions (Optional)
  - TODO says: "Optional - generate if needed"

---

## ⚠️ NEEDS ATTENTION:

1. **S9**: The file in `s9/` doesn't match the description (enrollment timeline). Need to verify.

2. **S14-S15**: Check if `s16-s18/` contains S14 (UKB) and S15 (AOU) or if they need separate files.
   - `s16-s18/ukb_all_signatures_probabilities.pdf` might be S14
   - `s16-s18/aou_all_signatures_probabilities.pdf` might be S15
   - `s16-s18/mgb_all_signatures_probabilities.pdf` is S16

3. **S18-S27**: GWAS lead SNPs - Need to locate or generate these 21 files (one per signature).

4. **True Washout Comparison**: `true_washout_comparison_10yr_30yr.png` exists but TODO says it needs update to remove 30-year data.

5. **S12, S13, S17**: These are marked "Keep as-is" in TODO but not found in `/supp`. They might be in a different location or need to be copied.

---

## 📋 SUMMARY:

### Present: 11/28 Extended Data Figures (S1-S8, S11, S16-S18)
### Missing: 17 Extended Data Figures (S9, S12-S15, S17-S28)
### New Figures: 3/4 (S29-S31 present, S32 optional)

### Action Items:
1. Locate or generate S9, S12, S13, S14, S15, S17
2. Generate or locate S18-S27 (GWAS lead SNPs - 21 files)
3. Decide on S28 (pathway analysis)
4. Update true washout comparison figure
5. Verify S14-S15 are correctly represented in s16-s18 folder



