# Generated Outputs Log

Tracks all CSVs, PDFs, plots, checkpoints, and other outputs generated during analysis.

---

## Training Checkpoints

| Config | Directory | Files | Notes |
|--------|-----------|-------|-------|
| Nokappa v3 Constant (W=1e-4) | `~/Dropbox/nokappa_v3_W1e-4/` | 10 `.pt` files (`enrollment_model_REPARAM_NOKAPPA_W0.0001_batch_{start}_{end}.pt`) | 300ep, constant LR=0.1, kappa=1 |
| Nokappa v3 Cos300 (W=1e-4) | `~/Dropbox/nokappa_v3_cos300/` | 10 `.pt` files (`enrollment_model_REPARAM_NOKAPPA_COS300_W0.0001_batch_{start}_{end}.pt`) | 300ep, cosine LR 0.1->0.001, kappa=1 |
| Nokappa v2 (cosine+clip, 500ep) | `~/Dropbox/censor_e_batchrun_vectorized_REPARAM_v2_nokappa/` | 40 `.pt` files | First nokappa run |
| Reparam v1 | `~/Dropbox/censor_e_batchrun_vectorized_REPARAM/` | 40 `.pt` files | Original reparam |
| Nolr (centered) | `~/Dropbox/censor_e_batchrun_vectorized_nolr/` | 40 `.pt` files | Original centered |
| Grid search | `claudefile/grid_results/` | 8 config subdirs with `metrics.csv` + `checkpoint.pt` | Phase A: LR/scheduler sweep |
| W sweep grid | `claudefile/grid_results/nok_lr01_300_w*/` | 6 W-value subdirs | Phase B: W sweep |

## Pooled Parameters

| File | Location | Contents |
|------|----------|----------|
| `pooled_phi_kappa_gamma_nolr.pt` | `~/Dropbox-Personal/data_for_running/` | Pooled from nolr batches |
| `pooled_phi_kappa_gamma_reparam.pt` | `~/Dropbox-Personal/data_for_running/` | Pooled from reparam v1 batches |
| `pooled_phi_kappa_gamma_nokappa.pt` | `~/Dropbox-Personal/data_for_running/` | Pooled from nokappa v2 batches |

## LOO Prediction Outputs

| Config | Directory | Files | Status |
|--------|-----------|-------|--------|
| Nolr LOO | `~/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_loo/` | `pi_enroll_fixedphi_sex_{start}_{end}.pt` | Complete |
| Reparam LOO | `~/Dropbox/enrollment_predictions_fixedphi_fixedgk_reparam_loo/` | `pi_enroll_fixedphi_sex_{start}_{end}.pt` | Complete |
| Nokappa v3 Constant LOO | `~/Dropbox/enrollment_predictions_nokappa_v3_constant_loo/` | `pi_enroll_fixedphi_sex_{start}_{end}.pt` | Pending |
| Nokappa v3 Cos300 LOO | `~/Dropbox/enrollment_predictions_nokappa_v3_cos300_loo/` | `pi_enroll_fixedphi_sex_{start}_{end}.pt` | Pending |

## CSVs

| File | Location | Contents |
|------|----------|----------|
| `nolr_vs_reparam_5batches_auc.csv` | `claudefile/` | AUC comparison: nolr vs reparam (5 batches, 3 horizons) |
| `nolr_vs_reparam_5batches_auc_LOO.csv` | `claudefile/` | LOO AUC comparison: nolr vs reparam |
| `three_way_psi_stability.csv` | `claudefile/` | Psi stability: nolr vs reparam v1 vs nokappa v2 |
| `nokappa_v3_const_vs_cos_auc.csv` | `claudefile/` | AUC comparison: constant vs cos300 (pending LOO) |

## PDFs / Plots

| File | Location | Contents |
|------|----------|----------|
| `parameter_talk.pdf` | `claudefile/` | Presentation for Parameter company (24 slides) |
| `map_decoupling_slides.pdf` | `claudefile/` | MAP gradient decoupling explanation slides |
| `calibration_loo_nolr_vs_reparam.pdf` | `claudefile/` | Calibration: LOO nolr vs reparam (side by side) |
| `calibration_loo_overlay.pdf` | `claudefile/` | Calibration: LOO nolr vs reparam (overlay) |
| `calibration_loo_three_way.pdf` | `claudefile/` | Calibration: three-way comparison |
| `calibration_loo_three_way_overlay.pdf` | `claudefile/` | Calibration: three-way overlay |
| `lambda_curves_nolr_vs_reparam.pdf` | `claudefile/` | Lambda trajectory comparison |
| `three_way_param_scatter.png` | `claudefile/` | Parameter scatter: nolr vs reparam vs nokappa |
| `three_way_prs_signatures.png` | `claudefile/` | PRS-signature heatmaps (three-way) |
| `three_way_psi_stability.png` | `claudefile/` | Psi stability distributions |
| `calibration_nokappa_v3_const_vs_cos.pdf` | `claudefile/` | Calibration: constant vs cos300 (pending LOO) |

## Notebooks

| File | Location | Purpose |
|------|----------|---------|
| `nolr_vs_reparam_pipeline.ipynb` | `claudefile/` | Full nolr vs reparam comparison pipeline |
| `nolr_vs_reparam_summary.ipynb` | `claudefile/` | Summary of nolr vs reparam results |
| `nokappa_pipeline.ipynb` | `claudefile/` | First nokappa (v2) pipeline |
| `nokappa_grid_search.ipynb` | `claudefile/` | Grid search analysis notebook |
| `reparam_v2_pipeline.ipynb` | `claudefile/` | Reparam v2 pipeline |
| `nokappa_v3_pipeline.ipynb` | `claudefile/` | **Current**: Constant vs Cos300 pipeline |
| `parameter_recovery_simulation.ipynb` | `claudefile/` | Simulation: centered vs reparam vs nokappa gamma recovery |

## Scripts

| File | Location | Purpose |
|------|----------|---------|
| `train_nokappa_v3.py` | `claudefile/` | Constant LR training (v3) |
| `train_nokappa_v3_cos.py` | `claudefile/` | Cosine LR training (cos300) |
| `run_nokappa_v3_three_W.py` | `claudefile/` | Three-W runner (1e-5, 1e-4, 5e-4) |
| `run_nokappa_v3_cos.py` | `claudefile/` | Cos300 batch runner |
| `compare_lambda_three_W.py` | `claudefile/` | Lambda comparison across W values |
| `optim_grid_search.py` | `claudefile/` | Optimization grid search (Phase A + B) |
| `pool_phi_kappa_gamma_from_batches.py` | `claudefile/` | Pool params from training batches |
| `run_loo_predict_both.py` | `claudefile/` | LOO prediction (nolr + reparam) |
| `compare_loo_auc.py` | `claudefile/` | LOO AUC comparison |
| `compare_nolr_vs_reparam_5batches_auc.py` | `claudefile/` | 5-batch AUC comparison |
| `three_way_param_comparison.py` | `claudefile/` | Three-way param comparison |
| `calibration_loo_nolr_vs_reparam.py` | `claudefile/` | Calibration plots |
