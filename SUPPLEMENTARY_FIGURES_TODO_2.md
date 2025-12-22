# Supplementary Figures To-Do List

## Status Legend
- ⬜ Not Started
- 🟡 In Progress  
- ✅ Complete
- ⏭️ Skip/Not Needed

---

## Extended Data Figures (S1-S28) - Main Paper

### Existing Figures to Update/Regenerate

#### ⬜ S1: Robustness of φ estimation across subsets
- **Status**: Needs remake
- **Method**: Borrow phi values from `censor_e_batchrun_vectorized`
- **Output**: `alffigures/S1.pdf`
- **Description**: Show φ trajectory stability across subsets with standard errors
- **Code Location**: Main paper figures directory

#### ⬜ S2: Simulation study
- **Status**: Needs regeneration
- **Method**: Use `enhanced_simulation_showcase_v2` notebook
- **Output**: `alffigures/S2.pdf`
- **Description**: Simulation demonstrating accurate recovery of latent clusters and temporal dynamics
- **Code Location**: Simulation notebooks directory

#### ✅ S3: Training architecture (retrospective/prospective)
- **Status**: Keep as-is
- **Output**: `alffigures/S3.pdf`

#### ✅ S4: AEX calculation for genetic discovery
- **Status**: Keep as-is
- **Output**: `alffigures/S4.pdf`

#### ✅ S5: Cross-cohort validation heatmaps
- **Status**: Keep as-is
- **Output**: `alffigures/S5.pdf`

#### ⬜ S6: Individual patient trajectories
- **Status**: Needs update
- **Method**: Borrow approach from Figure 3
- **Output**: `alffigures/S6.pdf`
- **Description**: Show individual trajectories with signature loadings, timelines, and contributions
- **Code Location**: `Figure3_Individual_Trajectories.ipynb` or similar

#### ✅ S7: Signature-specific SNP associations (pleiotropy)
- **Status**: Keep as-is
- **Output**: `alffigures/S7.pdf`

#### ⬜ S8: ROC curves vs PREVENT
- **Status**: Needs regeneration
- **Method**: Generate sex-specific ROC curves for PREVENT with new Aladynoulli static 10-year predictions
- **Output**: `alffigures/S8.pdf`
- **Description**: Sex-specific ROC curves comparing Aladynoulli to PREVENT for 10-year predictions
- **Code Location**: Performance comparison notebooks/scripts

#### ✅ S9: Enrollment timeline
- **Status**: Keep as-is
- **Output**: `alffigures/S9.pdf`

#### ⬜ S10: Distribution of 1-year AUCs across offsets
- **Status**: REMOVE
- **Action**: Delete reference from Extended Data section

#### ⬜ S11: 1-year AUCs vs PCE/PREVENT over time
- **Status**: Needs regeneration
- **Output**: `alffigures/S11.pdf`
- **Description**: Distribution of 1-year AUCs across offsets for ASCVD vs clinical risk scores
- **Note**: User mentioned "I'm sure we have code" - check existing notebooks

#### ✅ S12: Data structure and model components
- **Status**: Keep as-is
- **Output**: `alffigures/S12.pdf`

#### ✅ S13: Cohort characteristics and study design
- **Status**: Keep as-is
- **Output**: `alffigures/S13.pdf`

#### ✅ S14: Temporal patterns UKB
- **Status**: Already regenerated with corrected E matrix
- **Output**: `alffigures/S14.pdf`
- **Note**: This already shows the corrected solution (realistic increasing hazards)

#### ✅ S15: Temporal patterns All of Us
- **Status**: Already regenerated with corrected E matrix
- **Output**: `alffigures/S15.pdf`

#### ✅ S16: Temporal patterns MGB
- **Status**: Already regenerated with corrected E matrix
- **Output**: `alffigures/S16.pdf`

#### ✅ S17: Cohen's d effect sizes
- **Status**: Keep as-is
- **Output**: `alffigures/S17.pdf`

#### ✅ S18-S27: GWAS lead SNPs
- **Status**: Keep as-is
- **Output**: One file per signature (S7-S27 in Extended Data Files)
- **Description**: Lead SNPs for each disease signature (21 signatures total)

#### ⬜ S28: Pathway analysis for MI heterogeneity
- **Status**: User considering whether to keep
- **Output**: Extended Data Files (PDF)
- **Description**: Four distinct biological pathways to MI
- **Code Location**: `R3_Q8_Heterogeneity_Continued.ipynb` or pathway analysis notebooks
- **Note**: User mentioned "not sure the pathway analysis adds much"

---

## New Extended Data Figures to Create

### ⬜ S29: IPW Analysis (Multi-panel Figure)
- **Status**: Needs creation
- **Output**: `alffigures/S29.pdf` or `figures/ipw_analysis_multi_panel.pdf`
- **Panels**:
  - **Panel A**: Weight distribution histogram (mean=0.93, median=0.59, range 0.17-6.63)
    - Show median line at 0.59
    - Show reference line at 1.0 (representative)
  - **Panel B**: Average weights by subgroup (bar chart)
    - Age <60 (mean weight ~1.89, under-represented)
    - Other Ethnicity (mean weight ~1.39, under-represented)
    - Male (mean weight ~1.00, at threshold)
    - Fair/Poor Health (mean weight ~1.00, at threshold)
    - No University Degree (mean weight ~0.99, over-represented)
    - Good Health (mean weight ~0.91, over-represented)
    - White British (mean weight ~0.87, over-represented)
    - Female (mean weight ~0.87, over-represented)
    - Age 60+ (mean weight ~0.85, over-represented)
    - University Degree (mean weight ~0.81, most over-represented)
  - **Panel C**: Population characteristics before/after weighting (bar chart)
    - Good/Excellent Health: 74.8% → 72.9%
    - Age 60+: 92.1% → 84.0%
    - White British: 89.3% → 84.0%
    - University Degree: 33.0% → 28.7%
  - **Panel D**: Age distribution unweighted vs weighted
    - 50-59: 10.3% → 19.8%
    - 60-69: 28.4% → 35.4%
    - 70+: 61.3% → 44.8%
  - **Panel E**: IPW phi comparison (correlation plot)
    - Weighted vs unweighted phi correlation
    - Show high correlation (>0.99)
    - **Key point**: Phi values are essentially identical, showing model structure (signature-disease associations) is preserved
  - **Panel F**: Weighted vs Unweighted Prevalence Trajectories (5 diseases)
    - From `prevalence_weighted_vs_unweighted_comparison.pdf`
    - Shows: MI, Depression, Breast Cancer, Atrial Fibrillation, Type 2 Diabetes
    - **Key point**: Weighted and unweighted prevalences are very similar (max diff 0.0011-0.0019, mean diff 0.0005-0.0010)
    - Demonstrates that IPW doesn't materially change prevalence estimates
  - **Panel G**: Weighted vs Unweighted Pi Trajectories (5 diseases)
    - Pi trajectories from weighted vs unweighted models
    - Shows: MI, Depression, Breast Cancer, Atrial Fibrillation, Type 2 Diabetes
    - **Key point**: Pi (disease hazard) trajectories show small differences between weighted/unweighted models (max diff 0.0001-0.0021, mean diff 0.0000-0.0009)
    - Even though phi is the same, pi can differ slightly because it's computed from the model
  - **Panel H**: Weighted Prevalence vs Weighted Pi Averages (validation)
    - From the "Weighted Prevalence vs Weighted Pi Averages" figure
    - Shows: Weighted pi averages (from model) can recreate weighted prevalence (from data)
    - Overall correlation: 0.9928
    - **Key point**: The weighted model's pi values can accurately recreate the weighted prevalence, validating that the model works correctly even with IPW
- **Code Location**: `R1_Q1_Selection_Bias.ipynb` or related IPW notebooks
- **Existing Files**: 
  - Weight distribution plots (from IPW analysis results)
  - `prevalence_weighted_vs_unweighted_comparison.pdf` (Panel F)
  - Pi trajectories comparison (Panel G - check if exists, or generate from model outputs)
  - Weighted prevalence vs weighted pi averages figure (Panel H)
- **Note**: Remove the old comparison script that uses only 100K patients (the one referenced in the notebook snippet)
- **Key Message**: 
  1. IPW weights successfully rebalance the sample (Panels A-D)
  2. Model structure (phi) is preserved (Panel E: correlation >0.99)
  3. Prevalence estimates are robust (Panel F: very similar weighted vs unweighted)
  4. Pi trajectories differ slightly but are still small (Panel G)
  5. Weighted model can recreate weighted prevalence (Panel H: validation)
- **Caption**: Inverse probability weighting (IPW) analysis for selection bias correction. (A) Distribution of IPW weights (median=0.59). (B) Average weights by demographic subgroup. (C) Population characteristics before and after weighting. (D) Age distribution before and after weighting. (E) Correlation between weighted and unweighted signature-disease associations (φ), demonstrating preservation of model structure. (F) Prevalence trajectories comparing weighted vs unweighted (max diff: 0.0011-0.0019), showing robustness of prevalence estimates. (G) Pi (disease hazard) trajectories from weighted vs unweighted models (max diff: 0.0001-0.0021), showing small differences. (H) Validation: weighted pi averages (from model) accurately recreate weighted prevalence (from data), correlation=0.9928.

### ⬜ S30: Competing Risks Analysis (Multi-panel Figure)
- **Status**: Needs creation (combine existing figures)
- **Output**: `alffigures/S30.pdf` or `figures/competing_risks_analysis.pdf`
- **Panels**:
  - **Panel A**: `figures/competing_risk.png` (patient examples showing multiple outcomes)
  - **Panel B**: `figures/competing_2.png` (additional competing risks examples)
  - **Panel C**: `figures/subsequent_disease_temporal_patterns.png` (temporal patterns of subsequent diseases)
- **Code Location**: `R3_Competing_Risks.ipynb`
- **Action**: Combine three existing figures into one multi-panel Extended Data figure
- **Caption**: Competing risks and subsequent disease analysis. (A-B) Patient trajectory examples showing multiple competing outcomes over time. (C) Temporal patterns of subsequent disease development, illustrating how diseases develop sequentially and interact over the life course.

### ⬜ S31: Population Stratification (PC Adjustments) (Multi-panel Figure)
- **Status**: Needs creation (combine existing figures)
- **Output**: `alffigures/S31.pdf` or `figures/population_stratification_pc.pdf`
- **Panels**:
  - **Panel A**: `figures/pc_line_plot.png` (PC effects over time/age)
  - **Panel B**: `figures/pc_lambda_shift.png` (lambda shifts by ancestry with/without PC adjustment)
  - **Panel C**: `figures/pc_phi_comp.png` (phi correlations with/without PC adjustment)
- **Code Location**: `R3_Population_Stratification_Ancestry.ipynb`
- **Action**: Combine three existing figures into one multi-panel Extended Data figure
- **Caption**: Population stratification and principal component adjustment analysis. (A) Principal component effects on signature loadings over time. (B) Ancestry-specific signature loading shifts with (blue) and without (red) PC adjustment, showing amplification of ancestry effects when PCs are included. (C) Correlation of signature-disease associations (φ) with and without PC adjustment, demonstrating minimal structural changes.

### ⬜ S32: Censoring Distributions (Optional)
- **Status**: Optional - generate if needed
- **Output**: `alffigures/S32.pdf` or `figures/censoring_distributions.pdf`
- **Description**: Entry age, exit age, and follow-up duration distributions for UKB, MGB, and AOU
- **Code Location**: Similar to snippet shown in `R3_Q4_Decreasing_Hazards_Censoring_Bias.ipynb`
- **Panels**:
  - Entry age distributions (three cohorts)
  - Exit age distributions (three cohorts)
  - Follow-up duration distributions (three cohorts)
- **Note**: User can generate similar to the UKB example shown (lines 1-32 of notebook snippet)

---

## Reviewer Response Figures (May Move to Extended Data)

### Washout Analyses
These are currently in reviewer response but could be consolidated into Extended Data if desired:

#### ⬜ Washout Performance Plot (Short-term: 1, 3, 6 months)
- **Status**: Exists, may move to Extended Data
- **Existing File**: `figures/washout_performance_plot.png` and `figures/paper_figs/fig5/washout_performance_plot.pdf`
- **Code Location**: `R3_AvoidingReverseCausation.ipynb`
- **Description**: AUC for 1-year and 10-year predictions with 0, 1, 3, 6 month washout periods

#### ⬜ Fixed Timepoint Washout
- **Status**: Exists, may move to Extended Data
- **Existing File**: `figures/fixed_timepoint_horizons.png`
- **Code Location**: `R2_Washout_Comparisons.ipynb`
- **Description**: Model performance with fixed timepoint washout periods (0, 1, 2 years) for 1-year predictions

#### ⬜ True Washout Comparison (10-year only)
- **Status**: **NEEDS UPDATE** - Remove 30-year data
- **Existing File**: `figures/true_washout_comparison_10yr_30yr.png`
- **Code Location**: `R2_Temporal_Leakage.ipynb` (around lines 579-581)
- **Action**: Update notebook to only show 10-year predictions (remove 30-year)
- **New File Name**: `figures/true_washout_comparison_10yr.png` (or keep name, just update content)
- **Description**: 10-year static predictions with 0-year vs 1-year washout comparison

---

## Additional Figures (May Stay in Reviewer Response or Move to Extended Data)

### ⬜ External Scores Comparison
- **Status**: Exists
- **File**: `figures/external_scores_comparison.png`
- **Code Location**: `R1_Q9_AUC_Comparisons.ipynb`
- **Decision**: Keep in reviewer response or move to Extended Data?

### ⬜ RVAS Visualization
- **Status**: Exists
- **File**: `figures/rvas.png`
- **Code Location**: `R1_Genetic_Validation_Gene_Based_RVAS2.ipynb`
- **Decision**: Keep in reviewer response or move to Extended Data?

### ⬜ Multi-disease Patterns
- **Status**: Exists
- **File**: `figures/multi_disease_patterns_visualization.png`
- **Decision**: Keep in reviewer response or move to Extended Data?

### ⬜ CHIP-Leukemia Trajectory
- **Status**: Exists
- **File**: `figures/Leukemia_MDS_sig16_CHIP_signature_trajectory.png`
- **Decision**: Keep in reviewer response or move to Extended Data?

### ⬜ ICD vs PheCode Comparison
- **Status**: Exists
- **File**: `figures/icd_v_phe.png`
- **Code Location**: `R1_ICD_Phecode_Comparison.ipynb`
- **Decision**: Keep in reviewer response or move to Extended Data?

### ⬜ Delphi Comparison Figures
- **Status**: Exist
- **Files**: 
  - `figures/delphi_comparison_simple_mapping_1yr.png`
  - `figures/delphi_comparison_phecode_mapping_1yr_1tomany.png`
- **Code Location**: `R2_Delphi_Phecode_Mapping.ipynb`
- **Decision**: Keep in reviewer response or move to Extended Data?

---

## Reference Information

### Calibration Figure
- **Location**: Already in main paper **Figure 5, Panel B** (line 774)
- **Action**: Just reference it there, no new figure needed
- **Caption**: "Calibration plot across all follow-up periods for all at-risk individuals, showing observed versus predicted event rates on a log-log scale."

### Censoring Bias Correction - Important Note
- **Key Point**: S14-S16 (temporal patterns) ARE the corrected solution showing realistic increasing hazards
- **No separate "censoring bias correction" figure needed**: The corrected temporal patterns (S14-S16) already demonstrate the fix works - they show biologically realistic, monotonically increasing disease risk with age, which is the solution to the censoring bias problem
- **What was done**: E matrix was corrected to use realistic censoring times (from last ICD-10 diagnosis date) instead of uniform follow-up to age 81
- **Result**: The regenerated S14-S16 already show the corrected, realistic patterns
- **We don't need**: A separate before/after comparison figure showing the problem (decreasing hazards) vs the solution (increasing hazards) - the corrected figures ARE the solution

---

## Priority Order

### High Priority (Must Do):
1. ⬜ S8: Regenerate sex-specific ROC curves vs PREVENT
2. ⬜ S11: Regenerate 1-year AUCs vs PCE/PREVENT
3. ⬜ S29: Create IPW analysis multi-panel figure
4. ⬜ S30: Create competing risks multi-panel figure
5. ⬜ S31: Create population stratification multi-panel figure
6. ⬜ Update true washout comparison (remove 30-year)

### Medium Priority:
7. ⬜ S1: Remake robustness of φ estimation
8. ⬜ S2: Regenerate simulation study
9. ⬜ S6: Update individual patient trajectories

### Low Priority / Optional:
10. ⬜ S32: Censoring distributions (if needed)
11. ⬜ Decide on S28 (pathway analysis)
12. ⬜ Decide which reviewer response figures to move to Extended Data

---

## Code Locations Summary

### Main Notebook Directories:
- Main paper figures: `/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/main_paper_figures/`
- Reviewer response notebooks: `/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/reviewer_responses/notebooks/`
- Python scripts: `/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/pythonscripts/`

### Key Notebooks:
- `R1_Q1_Selection_Bias.ipynb` → IPW analysis (S29)
- `R3_Competing_Risks.ipynb` → Competing risks (S30)
- `R3_Population_Stratification_Ancestry.ipynb` → PC adjustments (S31)
- `R2_Temporal_Leakage.ipynb` → True washout comparison (needs update)
- `R3_AvoidingReverseCausation.ipynb` → Short-term washout
- `R2_Washout_Comparisons.ipynb` → Fixed timepoint washout
- `Figure3_Individual_Trajectories.ipynb` → S6 approach
- `R1_Q9_AUC_Comparisons.ipynb` → S8, S11, external scores

---

## Notes
- Update status emoji as you progress: ⬜ → 🟡 → ✅
- Check off items as they're completed
- Add notes or blockers as needed

