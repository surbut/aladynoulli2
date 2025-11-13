# Redo Analysis Notebooks

This directory collects a set of refreshed summary notebooks that document the major re-runs completed in fall 2025. Each notebook is designed to be self-contained: the first few cells explain the scope, the middle cells load precomputed exports, and the closing cells highlight key takeaways while linking back to the heavy training notebooks.

## Notebook Guide

### [`fh_analysis_summary.ipynb`](fh_analysis_summary.ipynb)
- **Goal**: Quantify how familial hypercholesterolemia (FH) carriers behave in the retrospective enrollment model, focusing on Signature 5 escalation prior to CAD events.
- **Inputs**: Batched φ/θ checkpoints with and without PCs, UKB exome carrier list (`ukb_exome_450k_fh.carrier.txt`), processed patient IDs, PRS table, ancestry projections.
- **Highlights**:
  - Loads all 41 batches, averages φ, and compares θ trajectories across ancestry groups.
  - Calculates enrichment of Signature 5 rises for FH carriers vs. non-carriers, with Fisher exact tests and Wilson CIs.
  - Produces prevalence tables for precursor diseases and exports plots under `complete_pathway_analysis_output/ukb_pathway_discovery/`.

### [`heterogeneity_analysis_summary.ipynb`](heterogeneity_analysis_summary.ipynb)
- **Goal**: End-to-end rerun of the MI heterogeneity pipeline, from UKB pathway discovery through MGB validation.
- **Pipeline**:
  1. `run_deviation_only_analysis` → saves UKB pathway outputs to `complete_pathway_analysis_output/ukb_pathway_discovery/`.
  2. `run_transition_analysis_both_cohorts` → builds RA→MI progression comparisons across cohorts.
  3. `analyze_signature5_by_pathway` → evaluates Signature 5 deviations and FH carrier enrichment per pathway.
  4. `show_pathway_reproducibility` → matches UKB/MGB pathways and produces the reproducibility figures in this folder.
- **Outputs**: Consolidated pickle (`complete_analysis_results.pkl`), PDF/PNG figures, and MGB deviation artifacts (see `mgb_deviation_analysis_output/`).

### [`ipw_analysis_summary.ipynb`](ipw_analysis_summary.ipynb)
- **Goal**: Provide a concise recap of the inverse-probability-weighted training run.
- **Inputs**: R exports from `runningviasulizingweights.R` (`population_weighting_summary.csv`, `weights_by_subgroup.csv`, PNG plots) and the combined φ comparison checkpoint (`fair_phi_comparison_results.pt`).
- **Highlights**:
  - Loads the summary CSVs into tidy tables and surfaces the largest shifts between unweighted and weighted cohorts.
  - Embeds the weighting visualization PNGs for slide-ready review.
  - Displays stored phi-difference statistics to confirm weighted vs. legacy model alignment.

### [`pc_analysis_clean.ipynb`](pc_analysis_clean.ipynb)
- **Goal**: Document the effect of principal-component (PC) adjustment on the retrospective model.
- **Focus Areas**:
  - Ancestry-specific PRS distributions (violin plots for key scores).
  - φ stability scatter/difference plots for sentinel diseases (MI, CAD, diabetes, hypertension, RA).
  - θ shift diagnostics at the patient level and downstream CAD validation checks.
- **Assets**: Uses the same φ/θ batches as the FH notebook and writes comparison figures back into this directory when saved manually.

### [`washout_analysis_summary.ipynb`](washout_analysis_summary.ipynb)
- **Goal**: Summarize the batched washout experiment where predictions are re-evaluated under 0-, 1-, and 2-year exclusion windows.
- **Inputs**: Aggregated metrics in `pyScripts/new_oct_revision/washout_summary_table.csv` and PNGs exported by `runningviasulizingweights.R` (for context).
- **Highlights**:
  - Formats the summary table with mean±SD AUC, washout deltas, retention percentages, and batch coverage.
  - Generates a two-panel figure (trajectories + retention bar chart) to visualize sensitivity across diseases.
  - Provides bullet takeaways that differentiate cardiometabolic, psychiatric, and oncologic responses to washout.

### [`heritability_analysis_summary.ipynb`](heritability_analysis_summary.ipynb)
- **Goal**: Capture the LDSC read-outs for both signature-level and trait-level GWAS that inform the cardiovascular heterogeneity work.
- **Inputs**: `ldsc_summary.tsv` (per-signature SNP-heritability) and `ldsc_summary_bytrait.tsv` (component GWAS per lesion/trait).
- **Highlights**:
  - Preloads the two TSVs, parses `h2`, `Intercept`, `LambdaGC`, and `Ratio` with their standard errors.
  - Flags Signature 5 as outlier low-heritability (`h² ≈ 0.0033 ± 0.0013`) with a high attenuation ratio (~0.43), suggesting most signal is baseline confounding; contrasts with other signatures at ~0.03–0.05 heritability and ratios ~0.10–0.13.
  - Crosswalks signature-level h² with component GWAS (angina, coronary atherosclerosis, hypercholesterolemia, MI, etc.) so downstream notes can reference which endpoint dominates the genetic architecture.
  - Provides a short interpretation cell summarizing takeaways (e.g., signatures downstream of acute MI maintain moderate heritability; unstable angina shows lower h² but higher intercept, indicating potential noise).

### [`fig5new.ipynb`](fig5new.ipynb)
- **Goal**: Evaluate enrollment model predictions for major diseases over a 10-year follow-up window using enrollment checkpoints with joint φ and sex/PC covariates.
- **Inputs**: Enrollment model checkpoints from `enrollment_prediction_jointphi_sex_pcs`, full Y tensor, E matrix, PCE/Prevent covariates, disease names.
- **Highlights**:
  - Processes all 40 batches of enrollment model checkpoints, evaluating 10-year prospective predictions for 30+ major diseases (ASCVD, diabetes, cancers, psychiatric conditions, etc.).
  - Computes AUC with bootstrap confidence intervals for each disease, comparing model predictions against PCE/Prevent scores where available.
  - Aggregates batch-wise results to produce median AUC across batches and generates summary tables/plots for publication-ready figures.
  - Uses `evaluate_major_diseases_wsex_with_bootstrap` from `fig5utils` with `follow_up_duration_years=10` to ensure prospective evaluation.

### [`fig5new_year2.ipynb`](fig5new_year2.ipynb)
- **Goal**: Evaluate enrollment model predictions for major diseases over a 5-year follow-up window (shorter-term horizon analysis).
- **Inputs**: Same as `fig5new.ipynb` (enrollment checkpoints, Y/E tensors, covariates).
- **Highlights**:
  - Identical structure to `fig5new.ipynb` but uses `follow_up_duration_years=5` for shorter-term prediction evaluation.
  - Enables comparison of model performance at different prediction horizons (5-year vs 10-year).
  - Useful for understanding how prediction accuracy changes with time horizon, particularly for diseases with different temporal dynamics.
  - Saves results to `batch_results_joint_5yr.pkl` for downstream aggregation and comparison.

### [`delphicomp.ipynb`](delphicomp.ipynb)
- **Goal**: Compare Aladynoulli (PheCode-based) disease predictions with Delphi-2M (ICD-10 based) predictions across multiple time horizons.
- **Inputs**: 
  - Aladynoulli results: `washout_summary_table.csv` (1-year), `median_auc_results_5_year.csv` (5-year), `median_auc_results_10yearjointphi.csv` (10-year).
  - Delphi-2M supplementary table: `41586_2025_9529_MOESM3_ESM.csv` (ICD-10 level results from Shmatko et al., Nature 2025).
- **Highlights**:
  - Maps Aladynoulli PheCode diseases to Delphi ICD-10 codes for direct comparison across 30+ disease categories.
  - Creates comprehensive comparison tables showing Aladynoulli vs Delphi performance at 1-year, 5-year, and 10-year horizons.
  - Identifies diseases where Aladynoulli outperforms Delphi and vice versa, with context on the different coding systems (PheCode aggregation vs raw ICD-10).
  - Generates publication-ready comparison plots and summary statistics for benchmarking against state-of-the-art transformer-based disease prediction models.

## Usage Notes
- The notebooks expect the heavy preprocessing to be complete: weights, φ/θ checkpoints, and pathway outputs should already be present in the paths referenced above.
- Re-running entire pipelines (especially `heterogeneity_analysis_summary.ipynb`) can take hours and requires full data access. For most collaborators, opening the notebooks in read-only mode and reviewing the rendered tables/plots is sufficient.
- Generated figures are saved alongside the notebooks unless an explicit path is provided inside the code cell; check `complete_pathway_analysis_output/` and `mgb_deviation_analysis_output/` for artifacts.

Need to regenerate an asset? Consult the corresponding notebook’s first markdown cell for the authoritative reference scripts and data locations.


