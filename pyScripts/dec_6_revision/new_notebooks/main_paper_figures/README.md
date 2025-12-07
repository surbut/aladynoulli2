# Main Paper Figures

This directory contains notebooks to recreate all main paper figures for publication.

## Figure Structure

Based on the main paper outline, we need to create:

### Figure 1: Model Overview
- Panel A: Model architecture (lambda, theta, phi relationships)
- Panel B: Theta distributions across population
- Panel C: Disease-signature association heatmap (psi values)
- Panel D: Model applications (prediction, subtypes, dynamic updates)

### Figure 2: Population-Level Patterns
- Panel A: Heatmap of discovered signatures with top diseases
  - **Note:** Can use existing `plot_disease_blocks()` function from `utils.py` (see `visualizing_transitions.ipynb` for example usage)
  - Shows disease-signature assignments (clusters) and cross-cohort correspondence
- Panel B: Patient clustering based on signature weights
- Panel C: Signature prevalence across age cohorts
- Panel D: Temporal evolution of signature prevalence

### Figure 3: Individual Trajectories
- Panel A: Case studies of 2-3 individuals showing signature evolution
- Panel B: Multimorbidity patterns (theta changes before/after diagnosis)
- Panel C: Signature response to new diagnoses (real-time updating)
- Panel D: Comparison of trajectories between disease subtypes

### Figure 4: Predictive Performance
- Panel A: ROC curves for several diseases
- Panel B: Calibration plots
- Panel C: Performance comparison with baseline models
- Panel D: Lead time analysis (early detection capability)

### Figure 5: Genetic Validation
- Panel A: PRS differences between disease subtypes
- Panel B: Manhattan plot of signature-specific genetic associations
- Panel C: Signature-modifying variants (genetic × signature interaction)
- Panel D: Genetic correlation network of signatures

## Data Sources

- **Thetas/Signatures:** `load_full_data()` from `helper_py.pathway_discovery`
- **Model Checkpoints:** 
  - UKB: `/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam.pt`
  - MGB: `/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt`
  - AoU: `/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_AOU.pt`
- **Existing Functions:** 
  - `plot_disease_blocks()` in `utils.py` (see `visualizing_transitions.ipynb` for usage)
- **Predictions:** `pooled_retrospective/` directories
- **GWAS Results:** `R1_Genetic_Validation_GWAS.ipynb`
- **Heterogeneity Analysis:** `R3_Q8_Heterogeneity_MainPaper_Method.ipynb`
- **Clinical Comparisons:** `pooled_retrospective/external_scores_comparison.csv`

## Output

All figures should be saved as high-resolution PDFs suitable for publication:
- Resolution: 300 DPI minimum
- Format: PDF (vector) preferred, PNG acceptable for complex plots
- Location: Save to `figures/` directory in main project root

## Notes

- All quantitative results should come from `pooled_retrospective/` directories for consistency
- Figures should match the style and format requirements of the target journal
- Each panel should be clearly labeled and self-contained
- Color schemes should be accessible (consider colorblind-friendly palettes)

