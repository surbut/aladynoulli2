# Redo Analysis Notebooks

This directory collects a set of refreshed summary notebooks that document the major re-runs completed in fall 2025. Each notebook is designed to be self-contained: the first few cells explain the scope, the middle cells load precomputed exports, and the closing cells highlight key takeaways while linking back to the heavy training notebooks.

## Notebook Guide

### `fh_analysis_summary.ipynb`
- **Goal**: Quantify how familial hypercholesterolemia (FH) carriers behave in the retrospective enrollment model, focusing on Signature 5 escalation prior to CAD events.
- **Inputs**: Batched φ/θ checkpoints with and without PCs, UKB exome carrier list (`ukb_exome_450k_fh.carrier.txt`), processed patient IDs, PRS table, ancestry projections.
- **Highlights**:
  - Loads all 41 batches, averages φ, and compares θ trajectories across ancestry groups.
  - Calculates enrichment of Signature 5 rises for FH carriers vs. non-carriers, with Fisher exact tests and Wilson CIs.
  - Produces prevalence tables for precursor diseases and exports plots under `complete_pathway_analysis_output/ukb_pathway_discovery/`.

### `heterogeneity_analysis_summary.ipynb`
- **Goal**: End-to-end rerun of the MI heterogeneity pipeline, from UKB pathway discovery through MGB validation.
- **Pipeline**:
  1. `run_deviation_only_analysis` → saves UKB pathway outputs to `complete_pathway_analysis_output/ukb_pathway_discovery/`.
  2. `run_transition_analysis_both_cohorts` → builds RA→MI progression comparisons across cohorts.
  3. `analyze_signature5_by_pathway` → evaluates Signature 5 deviations and FH carrier enrichment per pathway.
  4. `show_pathway_reproducibility` → matches UKB/MGB pathways and produces the reproducibility figures in this folder.
- **Outputs**: Consolidated pickle (`complete_analysis_results.pkl`), PDF/PNG figures, and MGB deviation artifacts (see `mgb_deviation_analysis_output/`).

### `ipw_analysis_summary.ipynb`
- **Goal**: Provide a concise recap of the inverse-probability-weighted training run.
- **Inputs**: R exports from `runningviasulizingweights.R` (`population_weighting_summary.csv`, `weights_by_subgroup.csv`, PNG plots) and the combined φ comparison checkpoint (`fair_phi_comparison_results.pt`).
- **Highlights**:
  - Loads the summary CSVs into tidy tables and surfaces the largest shifts between unweighted and weighted cohorts.
  - Embeds the weighting visualization PNGs for slide-ready review.
  - Displays stored phi-difference statistics to confirm weighted vs. legacy model alignment.

### `pc_analysis_clean.ipynb`
- **Goal**: Document the effect of principal-component (PC) adjustment on the retrospective model.
- **Focus Areas**:
  - Ancestry-specific PRS distributions (violin plots for key scores).
  - φ stability scatter/difference plots for sentinel diseases (MI, CAD, diabetes, hypertension, RA).
  - θ shift diagnostics at the patient level and downstream CAD validation checks.
- **Assets**: Uses the same φ/θ batches as the FH notebook and writes comparison figures back into this directory when saved manually.

### `washout_analysis_summary.ipynb`
- **Goal**: Summarize the batched washout experiment where predictions are re-evaluated under 0-, 1-, and 2-year exclusion windows.
- **Inputs**: Aggregated metrics in `pyScripts/new_oct_revision/washout_summary_table.csv` and PNGs exported by `runningviasulizingweights.R` (for context).
- **Highlights**:
  - Formats the summary table with mean±SD AUC, washout deltas, retention percentages, and batch coverage.
  - Generates a two-panel figure (trajectories + retention bar chart) to visualize sensitivity across diseases.
  - Provides bullet takeaways that differentiate cardiometabolic, psychiatric, and oncologic responses to washout.

## Usage Notes
- The notebooks expect the heavy preprocessing to be complete: weights, φ/θ checkpoints, and pathway outputs should already be present in the paths referenced above.
- Re-running entire pipelines (especially `heterogeneity_analysis_summary.ipynb`) can take hours and requires full data access. For most collaborators, opening the notebooks in read-only mode and reviewing the rendered tables/plots is sufficient.
- Generated figures are saved alongside the notebooks unless an explicit path is provided inside the code cell; check `complete_pathway_analysis_output/` and `mgb_deviation_analysis_output/` for artifacts.

Need to regenerate an asset? Consult the corresponding notebook’s first markdown cell for the authoritative reference scripts and data locations.


