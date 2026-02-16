# Validity of 5-Batches AUC Comparison (nolr vs reparam)

## What was run

| Model | Prediction script | Fixed params source |
|-------|-------------------|---------------------|
| **Nolr** | `run_aladyn_predict_with_master_vector_cenosrE_fixedgk.py` (with `--output_dir ..._nolr_vectorized/`) | φ, ψ from master (nolr); γ, κ from `pooled_kappa_gamma_nolr.pt` |
| **Reparam** | `run_aladyn_predict_reparam_fixedgk.py` | φ, ψ, γ, κ all from `pooled_phi_kappa_gamma_reparam.pt` |

Both fit λ (nolr) or δ (reparam) per 10K batch to pre-enrollment data; extract π; same AUC evaluation.

**Verify reparam used pooled reparam γ/κ:** Yes. `run_aladyn_predict_reparam_fixedgk.py` line 84: `pooled = torch.load(args.pooled_reparam_path)` and uses `phi`, `gamma`, `kappa`, `psi` from that file.

---

## Is the comparison valid?

**Structurally yes:** same patients (first 50K), same Y/E/pce_df, same evaluation functions. Each model uses its own trained pooled params; that’s the intended comparison.

**Main caveat: evaluation set = training set.** The first 50K were in batches 0–4 during training. Both models saw them. So this is **in-sample** performance, not holdout.

---

## Why might reparam look “too good”?

### 1. Genetic signal via γ

- Reparam: γ ~14× larger than nolr. λ = mean(γ) + δ, so the prior mean is genetically informative.
- At prediction: δ is fit, but the prior mean G@γ already encodes PRS–signature associations. Better genetic signal can raise AUC.
- Nolr: γ ≈ 0; prior mean ≈ ref for everyone; λ has to learn genetic effects from scratch.

### 2. Same patients in training

- Pooled γ was estimated from training batches that included these 50K.
- Reparam γ gets NLL gradient during training, so it was partly tuned on these patients.
- That creates favorable conditions for reparam on this 50K.

### 3. Dynamic 1-year AUCs and small n_events

- Dynamic 1yr uses very few events (e.g. Bipolar 4, Bladder_Cancer 9, Crohn’s 7).
- AUC is unstable with few events; bootstrap CIs are wide.
- Values like Asthma 0.96, Depression 0.94, Anemia 0.87 are plausible but volatile — treat with caution.

### 4. Different φ, ψ, κ

- φ: r=0.94 (almost the same)
- ψ: r=0.76, 13 reparam flips
- κ: reparam 4.52 vs nolr 2.93

Different ψ can change which signature drives each disease, which can help or hurt discrimination. κ affects scaling but not ranks for AUC.

---

## Recommendation

1. **Treat as in-sample:** interpret as “how well each parameterization fits the same 50K used in training,” not as external validation.
2. **Run holdout evaluation:** use `holdout_auc_nolr_vs_reparam.py` with `--holdout_start 399000 --holdout_end 400000` (or similar) for true holdout AUC.
3. **Check reparam params:** confirm `pooled_phi_kappa_gamma_reparam.pt` is the one used by `run_aladyn_predict_reparam_fixedgk.py`.
4. **Dynamic 1yr:** report CIs; treat point estimates as noisy when n_events < 50.
