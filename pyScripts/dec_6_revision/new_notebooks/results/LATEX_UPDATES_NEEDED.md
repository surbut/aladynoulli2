# LaTeX File Updates Needed - AUC Value Corrections

Based on actual results in `pyScripts/dec_6_revision/new_notebooks/results/`, here are all the numbers that need to be updated in the reviewer response LaTeX file.

## 1. External Scores Comparison (PCE, PREVENT, GAIL, QRISK3)

**File**: `comparisons/pooled_retrospective/external_scores_comparison.csv`

### ASCVD 10-year:
- **LaTeX says**: Aladynoulli 0.737 vs PCE 0.683 (+7.9%) vs QRISK3 0.702 (+5.0%)
- **Actual**: Aladynoulli **0.7327** vs PCE 0.6830 (+**4.97%**) vs QRISK3 0.7021 (+**3.06%**)
- **CI**: Aladynoulli (0.7298-0.7354), PCE (0.6808-0.6853), QRISK3 (0.6991-0.7051)

### ASCVD 30-year:
- **LaTeX says**: Aladynoulli 0.708 vs PREVENT 0.650 (+9.0%)
- **Actual**: Aladynoulli **0.7030** vs PREVENT 0.6501 (+**5.29%**)
- **CI**: Aladynoulli (0.6967-0.7093), PREVENT (0.6440-0.6563)

### Breast Cancer 10-year (Female):
- **LaTeX says**: Aladynoulli 0.556 vs GAIL 0.539 (+3.2%)
- **Actual**: Aladynoulli **0.5504** vs GAIL 0.5394 (+**1.10%**)
- **CI**: Aladynoulli (0.5452-0.5560), GAIL (0.5339-0.5448)

## 2. Cox Baseline Comparison (10-year static)

**File**: `comparisons/pooled_retrospective/cox_baseline_comparison_static10yr_full.csv`

### Top Diseases (in order of improvement):

1. **Parkinsons**:
   - **LaTeX says**: 0.722 vs 0.534 (+35.2%)
   - **Actual**: **0.7231** vs **0.5339** (+**35.4%**)

2. **CKD**:
   - **LaTeX says**: 0.705 vs 0.529 (+33.2%)
   - **Actual**: **0.7057** vs **0.5292** (+**33.3%**)

3. **Prostate Cancer**:
   - **LaTeX says**: 0.684 vs 0.519 (+31.9%)
   - **Actual**: **0.6828** vs **0.5189** (+**31.6%**)

4. **Stroke**:
   - **LaTeX says**: 0.681 vs 0.518 (+31.6%)
   - **Actual**: **0.6811** vs **0.5175** (+**31.6%**) ✓ (matches)

5. **COPD**:
   - **LaTeX says**: 0.659 vs 0.524 (+25.8%)
   - **Actual**: **0.6581** vs **0.5236** (+**25.7%**)

6. **All Cancers**:
   - **LaTeX says**: 0.671 vs 0.541 (+24.0%)
   - **Actual**: **0.6693** vs **0.5411** (+**23.7%**)

7. **Colorectal Cancer**:
   - **LaTeX says**: 0.646 vs 0.521 (+24.0%)
   - **Actual**: **0.6456** vs **0.5212** (+**23.9%**)

8. **Atrial Fibrillation**:
   - **LaTeX says**: 0.707 vs 0.588 (+20.2%)
   - **Actual**: **0.7067** vs **0.5883** (+**20.1%**)

9. **Lung Cancer**:
   - **LaTeX says**: 0.668 vs 0.554 (+20.7%)
   - **Actual**: **0.6683** vs **0.5538** (+**20.7%**) ✓ (matches)

10. **Heart Failure**:
    - **LaTeX says**: 0.702 vs 0.592 (+18.6%)
    - **Actual**: **0.7013** vs **0.5919** (+**18.5%**)

11. **ASCVD**:
    - **LaTeX says**: 0.737 vs 0.634 (+16.3%)
    - **Actual**: **0.7329** vs **0.6338** (+**15.6%**)

## 3. Delphi Comparison (1-year predictions with 1-year washout)

**File**: `washout_fixed_timepoint/pooled_retrospective/washout_vs_delphi_all_diseases.csv`

### Key Diseases Where Aladynoulli Wins (16/27 = 59.3%):

1. **Parkinsons**:
   - **LaTeX says**: 0.907 vs 0.617 (+29.0%)
   - **Actual**: **0.9513** vs 0.6166 (+**33.5%**)

2. **Secondary Cancer**:
   - **LaTeX says**: 0.629 vs 0.361 (+26.8%)
   - **Actual**: **0.6287** vs 0.3607 (+**26.8%**) ✓ (matches)

3. **Breast Cancer**:
   - **LaTeX says**: 0.679 vs 0.530 (+14.9%)
   - **Actual**: **0.6951** vs 0.5300 (+**16.5%**)

4. **Ulcerative Colitis**:
   - **LaTeX says**: 0.811 vs 0.671 (+14.0%)
   - **Actual**: **0.8234** vs 0.6706 (+**15.3%**)

5. **All Cancers**:
   - **LaTeX says**: 0.691 vs 0.551 (+13.9%)
   - **Actual**: **0.6921** vs 0.5514 (+**14.1%**)

6. **Lung Cancer**:
   - **LaTeX says**: 0.799 vs 0.672 (+12.7%)
   - **Actual**: **0.8099** vs 0.6716 (+**13.8%**)

7. **Multiple Sclerosis**:
   - **LaTeX says**: 0.673 vs 0.615 (+5.9%)
   - **Actual**: **0.7338** vs 0.6146 (+**11.9%**)

8. **Stroke**:
   - **LaTeX says**: 0.776 vs 0.656 (+12.0%)
   - **Actual**: **0.7748** vs 0.6557 (+**11.9%**)

9. **Rheumatoid Arthritis**:
   - **LaTeX says**: 0.897 vs 0.763 (+13.4%)
   - **Actual**: **0.8757** vs 0.7627 (+**11.3%**)

10. **Prostate Cancer**:
    - **LaTeX says**: 0.737 vs 0.619 (+11.8%)
    - **Actual**: **0.7316** vs 0.6190 (+**11.3%**)

11. **Colorectal Cancer**:
    - **LaTeX says**: 0.694 vs 0.587 (+10.7%)
    - **Actual**: **0.6775** vs 0.5866 (+**9.1%**)

12. **ASCVD**:
    - **LaTeX says**: 0.742 vs 0.661 (+8.1%)
    - **Actual**: **0.7349** vs 0.6611 (+**7.4%**)

13. **Osteoporosis**:
    - **LaTeX says**: 0.715 vs 0.672 (+4.3%)
    - **Actual**: **0.7139** vs 0.6716 (+**4.2%**)

14. **Atrial Fibrillation**:
    - **LaTeX says**: 0.669 vs 0.631 (+3.8%)
    - **Actual**: **0.6731** vs 0.6312 (+**4.2%**)

15. **Thyroid Disorders**:
    - **LaTeX says**: 0.626 vs 0.600 (+2.6%)
    - **Actual**: **0.6188** vs 0.6004 (+**1.8%**)

16. **Bladder Cancer**:
    - **LaTeX says**: (not mentioned)
    - **Actual**: **0.6308** vs 0.6112 (+**2.0%**)

### Win Rate:
- **LaTeX says**: 15/28 diseases (53.6%)
- **Actual**: **16/27 diseases (59.3%)** (Note: Only 27 diseases in washout comparison, not 28)

## 4. Delphi Comparison (1-year predictions, 0-year gap)

**File**: `comparisons/pooled_retrospective/delphi_comparison_1yr_full.csv`

### Key Numbers:
- **ASCVD**: Aladynoulli 0.8809 vs Delphi 0.7370 (+14.4%)
- **Parkinsons**: Aladynoulli 0.8091 vs Delphi 0.6108 (+19.8%)
- **Multiple Sclerosis**: Aladynoulli 0.8395 vs Delphi 0.6545 (+18.5%)

## Summary of Key Changes Needed:

1. **ASCVD 10-year**: 0.737 → **0.7327** (and recalculate % improvements)
2. **ASCVD 30-year**: 0.708 → **0.7030** (and recalculate % improvement)
3. **Breast Cancer**: 0.556 → **0.5504** (and recalculate % improvement)
4. **Cox baseline numbers**: Many need slight adjustments (see above)
5. **Delphi washout win rate**: 15/28 (53.6%) → **16/27 (59.3%)**
6. **Delphi washout numbers**: Several need updates (see above)

## Notes:
- All percentages need to be recalculated based on the corrected AUC values
- CI values should also be updated where provided
- The win rate calculation should use 27 diseases (not 28) for the washout comparison






