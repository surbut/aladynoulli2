# How results_holdout_auc files were generated (non-LOO pooling)

Quick reference for the different **holdout** (non-LOO) pooling setups.

**Bottom line:**  
- **10yr holdout:** Slope vs no-slope was similar to slightly improved (no large gain).  
- **1yr holdout:** Slope showed clearer improvement over no-slope; the progression slides (`slide_auc_1yr`: "1-year AUC — where slope model really shines") and narrative treat that as **not unreasonable**, since the slope model has time-varying genetic effects (γ_slope · t), so near-term (1yr) prediction is where the time term can help.  
- The **big** jump appears only in **LOO** (slope ~0.85 on 100k vs holdout ~0.76), which remains unexplained.

**Where AUC is summarized:** Holdout/LOO AUC tables and figures come from the CSVs listed below and from `progression_slides.py` (slides for 10yr, 1yr, wide comparison). `slope_training.ipynb` does **not** summarize AUC; it focuses on training (NLL, |γ_slope|) and holdout **NLL** comparison (slope vs no-slope, per-disease NLL improvement). All use **test batches 0–9 (first 100k)** unless noted. E = E_enrollment_full.pt for prediction.

---

## 1. holdout_auc_slope_1phase_pool30_vs_noslope.csv

- **Script:** `python slope_holdout_auc.py --single_phase --single_phase_wide`
- **Pool:** Train batches **10–39** (30 batches). One shared gamma (slope: γ_level, γ_slope from 1-phase checkpoints; no-slope: baked φ, γ from base nokappa).
- **Slope checkpoints:** 1-phase (`slope_model_nokappa_v3_single_phase`).
- **Test:** Batches 0–9 (100k). Fit delta on each test batch with that shared pool.
- **Output:** Long table: model, horizon, disease, auc, ci_lower, ci_upper. Horizons: static_10yr, dynamic_10yr.

---

## 2. holdout_auc_slope_1phase_vs_noslope.csv

- **Script:** `python slope_holdout_auc.py --single_phase` (no `--single_phase_wide`)
- **Pool:** Train batches **10–14** (5 batches). One shared gamma.
- **Slope checkpoints:** 1-phase.
- **Test:** Batches 0–9 (100k).
- **Output:** Same format as above. So this is **pool-5** (1-phase) vs **pool-30** (1-phase) in file (1).

---

## 3. holdout_auc_slope_vs_noslope.csv

- **Script:** `python slope_holdout_auc.py` (no --single_phase)
- **Pool:** Train batches **10–39** (30 batches). One shared gamma.
- **Slope checkpoints:** **2-phase** (`slope_model_nokappa_v3`).
- **Test:** Batches 0–9 (100k).
- **Output:** Same long format. So this is **pool-30** with **2-phase** slope (vs (1) which is pool-30 with 1-phase).

---

## 4. holdout_auc_1yr_slope_1phase_vs_noslope.csv

- **Script:** `python slope_holdout_auc_1yr.py --single_phase`
- **Input:** Uses **saved pi** from the run that produced `holdout_auc_slope_1phase_vs_noslope.csv`: `pi_slope_holdout_1phase.pt`, `pi_noslope_holdout_1phase_run.pt` (pool-5, test 0–9).
- **Evaluation:** Static 1yr and dynamic 1yr AUC on that same 100k (follow_up_duration_years=1).
- **Output:** Long table with horizons static_1yr, dynamic_1yr. So **1-year** AUC for the **pool-5, 1-phase** holdout run.

---

## 4b. holdout_auc_1yr_slope_1phase_pool30_vs_noslope.csv

- **Script:** `python slope_holdout_auc_1yr.py --single_phase_wide`
- **Input:** Saved pi from **pool-30, 1-phase** run: `pi_slope_holdout_1phase_pool30.pt`, `pi_noslope_holdout_1phase_pool30.pt` (produced by `slope_holdout_auc.py --single_phase --single_phase_wide`).
- **Evaluation:** Static 1yr and dynamic 1yr AUC on test 0–9 (100k).
- **Output:** Long table with horizons static_1yr, dynamic_1yr. **1-year** AUC for **pool-30, 1-phase** holdout.

---

## 5. holdout_auc_1yr_slope_vs_noslope.csv

- **Script:** `python slope_holdout_auc_1yr.py` (no --single_phase)
- **Input:** Saved pi from **2-phase pool-30** run: `pi_slope_holdout.pt`, `pi_noslope_holdout.pt`.
- **Evaluation:** Static 1yr and dynamic 1yr on same 100k.
- **Output:** 1-year AUC for **pool-30, 2-phase** holdout.

---

## 6. gamma_slope_prs_by_signature.csv

- **Script:** `python slope_value_analyses.py` (holdout mode)
- **Input:** Loads slope checkpoint(s) from pool **10–39** (2-phase by default), pools γ_slope, and maps to PRS/signature names.
- **Output:** CSV of gamma_slope (PRS × signature). Optional `--loo` uses LOO checkpoints and writes `gamma_slope_prs_by_signature_loo.csv`.

---

## 7. age_stratified_1yr_auc.csv / age_stratified_1yr_auc.pdf

- **Script:** `python slope_value_analyses.py` (holdout mode)
- **Input:** Same saved pi as holdout (default: `pi_slope_holdout.pt`, `pi_noslope_holdout.pt` → pool-30, 2-phase, 100k).
- **Analysis:** 1-year AUC by age tertile (Young / Middle / Old) for slope vs no-slope.
- **Output:** CSV of disease × age_group × auc_slope, auc_noslope, diff; PDF bar charts. With `--loo`, uses LOO pi and writes `age_stratified_1yr_auc_loo.*`.

---

## 8. wide_comparison.csv

- **Source:** Not written by the scripts checked in claudefile. Likely built by pivoting the long-format holdout CSVs (e.g. `holdout_auc_slope_vs_noslope.csv` and 1yr versions) to one row per disease with columns: static_1yr_slope, static_1yr_noslope, static_1yr_diff, static_10yr_*, dynamic_1yr_*, dynamic_10yr_*.
- **Content:** One row per disease; slope vs no-slope AUC and diff for static/dynamic 1yr and 10yr. Used by `progression_slides.py`.

---

## Summary: pooling variants (all test on 0–9, 100k)

| Result file / run | Train batches | Pool size | Slope type |
|-------------------|---------------|-----------|------------|
| holdout_auc_slope_1phase_pool30_vs_noslope | 10–39 | 30 | 1-phase |
| holdout_auc_slope_1phase_vs_noslope | 10–14 | 5 | 1-phase |
| holdout_auc_slope_vs_noslope | 10–39 | 30 | 2-phase |
| holdout_auc_1yr_* (1phase) | (from pool-5 run) | 5 | 1-phase |
| holdout_auc_1yr_* (1phase_pool30) | (from pool-30 run) | 30 | 1-phase |
| holdout_auc_1yr_* (no flag) | (from pool-30 run) | 30 | 2-phase |

**LOO** is separate: for each batch i we pool from the other 39 batches and fit delta on batch i; no single “train batch” set.
