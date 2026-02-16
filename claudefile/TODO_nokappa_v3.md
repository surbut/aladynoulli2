# Nokappa v3 Holdout Evaluation — TODO

## Status: Training in progress (cos300 running)

### Notebook: `nokappa_v3_holdout_evaluation.ipynb`
All steps below are implemented in this notebook. Run once training finishes.

- [x] **Setup**: imports, paths, config setup for 3 configs (constant, cos300, clip)
- [x] **Section 1**: Load & pool parameters from b20_30 checkpoints (10 per config)
- [x] **Section 2**: Diagnostics — gamma magnitude per batch, top features per signature, gamma heatmap, psi shifts from init, cross-config correlations
- [x] **Section 3**: Holdout prediction (fit delta on 390k–400k, no overlap with training 200k–300k)
- [x] **Section 4**: Holdout loss + dynamic 10-year AUC comparison table with bootstrap CIs → saves `holdout_auc_nokappa_v3_b20_30.csv`

### Training status (batches 20–29, samples 200k–300k)
- [x] **Constant** (LR=0.1, 300ep) — 10/10 checkpoints done
- [x] **Clip** (LR=0.1, grad_clip=5, 300ep) — 10/10 checkpoints done
- [ ] **Cos300** (LR=0.1→0.001 cosine, 300ep) — running (PID 18000 serial runner)

### When cos300 finishes
1. Run the notebook: `nokappa_v3_holdout_evaluation.ipynb`
2. Or run the script: `/opt/miniconda3/envs/new_env_pyro2/bin/python claudefile/holdout_predict_and_auc_nokappa_v3_b20_30.py`
3. Compare holdout loss + AUC across configs

### Key files
| File | Purpose |
|------|---------|
| `nokappa_v3_holdout_evaluation.ipynb` | Clean notebook: diagnostics + holdout eval |
| `holdout_predict_and_auc_nokappa_v3_b20_30.py` | Script version of holdout eval |
| `nokappa_prediction_utils.py` | Shared `fit_and_extract_pi` (no NaN) |
| `run_nokappa_v3_batches20_30.py` | Training runner (constant → cos300 → clip) |
| `train_nokappa_v3_clip.py` | Clip training script |
| `parameter_recovery_simulation.ipynb` | Simulation: LR choice doesn't matter |
| `nokappa_v3_pipeline.ipynb` | Old pipeline (LOO, in-sample — superseded) |

### Checkpoint directories (Dropbox)
- `nokappa_v3_W1e-4_b20_30/` — constant
- `nokappa_v3_cos300_b20_30/` — cos300
- `nokappa_v3_clip_b20_30/` — clip
