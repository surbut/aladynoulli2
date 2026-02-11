# Centered vs Non-Centered Parameterization: Summary for Giovanni

## The two parameterizations

| | Centered ("nolr") | Non-centered ("reparam") |
|---|---|---|
| **Free params** | λ, γ | δ, γ (where λ = μ(γ) + δ) |
| **γ gets NLL gradient?** | No — only prior gradient | Yes — via chain rule |
| **Same objective?** | Yes — same stationary points, different optimization paths | |
| **Standard in** | Stan, lme4, INLA, PyMC (default) | Stan (non-centered), VAE-style models |

---

## 1. Pooled parameter comparison (39 training batches)

| Parameter | Correlation | Interpretation |
|---|---|---|
| **φ** (disease trajectories) | **0.94** | Nearly identical — both fit the data well |
| **ψ** (disease→signature map) | **0.76** | Moderately different — reparam less stable |
| **γ** (PRS→signature effects) | **0.37** | Very different — nolr γ ≈ 0 |
| **κ** (scaling) | nolr = 2.93, reparam = 4.52 | Reparam compensates with larger κ |

**γ magnitude:**
- Centered (nolr): mean |γ| = **0.006** (effectively zero)
- Non-centered (reparam): mean |γ| = **0.081** (14× larger)

---

## 2. PRS–Signature associations (top hits per signature)

### Key clinically interpretable signatures

**Sig 5 — Ischemic Cardiovascular:**

| Rank | Centered (nolr) | Non-centered (reparam) |
|---|---|---|
| 1 | **Sex = 0.299** | **Sex = 0.618** |
| 2 | **CAD = 0.225** | **CAD = 0.314** |
| 3 | LDL = 0.079 | PC10 = 0.138 |

Both agree: Sex and CAD PRS are the top drivers. Reparam amplifies the effect sizes.

**Sig 15 — Metabolic/Diabetes:**

| Rank | Centered (nolr) | Non-centered (reparam) |
|---|---|---|
| 1 | **T2D = 0.136** | **T2D = 1.240** |
| 2 | PC1 = 0.072 | Sex = 0.579 |
| 3 | PC3 = 0.067 | PC1 = 0.479 |

Both agree: T2D PRS is the top driver. Reparam amplifies by ~9×.

**Sig 13 — Male Urogenital:**

| Rank | Centered (nolr) | Non-centered (reparam) |
|---|---|---|
| 1 | **Sex = 0.118** | **Sex = 3.629** |
| 2 | PC = 0.045 | PC = 0.524 |
| 3 | OP = 0.026 | PC1 = 0.293 |

Both agree on rank order; reparam has 30× larger Sex effect.

### Signatures where reparam diverges (PCs dominate)

| Signature | Nolr top PRS | Reparam top PRS |
|---|---|---|
| Sig 0 (Cardiac Arrhythmias) | Sex, CVD | **PC2, Sex, PC1** |
| Sig 4 (Upper Respiratory) | AST, Sex | **PC1, Sex, PC2** |
| Sig 6 (Metastatic Cancer) | HT, ISS | **PC1, Sex, PC2** |
| Sig 11 (Cerebrovascular) | Sex, ISS | **Sex, PC4, PC2** |
| Sig 12 (Renal/Urologic) | Sex, BMI | **Sex, PC1, PC2** |
| Sig 14 (Pulmonary/Smoking) | Sex, POAG | **PC2, PC1, Sex** |

In reparam, many signatures are dominated by **PCs and Sex** with very large coefficients — this is concerning. PCs are population structure controls, not biological PRS. Having PC1 = −4.7 (Sig 6) or PC2 = 2.5 (Sig 14) suggests the reparam may be overfitting to population structure.

---

## 3. ψ stability (disease→signature assignments)

| Metric | Centered (nolr) | Non-centered (reparam) |
|---|---|---|
| Mean \|Δψ\| | 0.08 | 0.81 (10× larger) |
| Correlation (initial → final) | 0.977 | 0.738 |
| Diseases keeping same primary signature | **348/348 (100%)** | 335/348 (96%) |
| Diseases that flip | **0** | **13** |

### The 13 flipped diseases (reparam only)

| Disease | Initial signature | Reparam flips to | Batches flipped |
|---|---|---|---|
| Peritoneal adhesions | Hepatobiliary | Infectious/Critical Care | **39/39 (100%)** |
| Pyelonephritis | Renal/Urologic | Infectious/Critical Care | **39/39 (100%)** |
| Hydronephrosis | Renal/Urologic | Infectious/Critical Care | **39/39 (100%)** |
| Cholecystitis w/o stones | Hepatobiliary | Infectious/Critical Care | 38/39 (97%) |
| Other biliary tract | Hepatobiliary | Infectious/Critical Care | 38/39 (97%) |
| Acute pancreatitis | Hepatobiliary | Infectious/Critical Care | 37/39 (95%) |
| Other kidney/ureter disorders | Renal/Urologic | Infectious/Critical Care | 37/39 (95%) |
| Stricture/obstruction ureter | Renal/Urologic | Infectious/Critical Care | 35/39 (90%) |
| Other gallbladder disorders | Hepatobiliary | Infectious/Critical Care | 34/39 (87%) |
| Cholelithiasis | Hepatobiliary | Infectious/Critical Care | 33/39 (85%) |
| Cholelithiasis w/ cholecystitis | Hepatobiliary | Infectious/Critical Care | 33/39 (85%) |
| Infertility, female | Mixed/General Medical | Gynecologic | 29/39 (74%) |
| Cerebral ischemia | Cerebrovascular | Infectious/Critical Care | 28/39 (72%) |

**Pattern:** 12/13 flip into Infectious/Critical Care. Flips are consistent across batches (not averaging artifacts). Only the Infertility→Gynecologic flip is biologically plausible.

---

## 4. Bottom line

- **Centered (nolr) is the right choice** for the current model and prediction task.
- **γ ≈ 0** is not a bug — it reflects the fact that in joint MAP, hyperparameters only receive prior gradient. This is standard (same behavior as Stan `optimizing`, lme4 variance components, etc.).
- **Prediction works** because λ (K×T free parameters per individual) compensates. We fit λ to pre-enrollment data and predict 10-year post-enrollment events. The GP kernel carries the individual pattern forward.
- **γ would matter** for long-horizon extrapolation beyond the kernel's reach (future work), but for current 10-year prediction the centered version performs well.
- **Reparam** gives γ data signal, but at the cost of ψ instability, numerical NaN (around epoch 46), and PC-dominated γ that may reflect overfitting to population structure rather than biological signal.
