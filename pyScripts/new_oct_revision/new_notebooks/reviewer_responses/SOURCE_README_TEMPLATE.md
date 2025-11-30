# Model Source Code

This directory contains the core Aladynoulli model implementations used in all analyses.

## Files

### 1. `clust_huge_amp.py`
**Purpose**: Main model with joint phi estimation (Discovery Mode)

**When to use**: 
- Learning disease signatures (phi) from data
- Discovery analyses (e.g., signature interpretation, pathway analysis)
- Initial model fitting

**Key class**: `AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest`

**Usage**:
```python
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=Y.shape[0], D=Y.shape[1], T=Y.shape[2], K=20,
    P=G.shape[1], G=G, Y=Y, W=0.0001, R=0,
    prevalence_t=prevalence_t,
    signature_references=signature_refs,
    healthy_reference=True,
    disease_names=disease_names
)
history = model.fit(E, num_epochs=200)
pi, theta, phi = model.forward()
```

---

### 2. `clust_huge_amp_fixedPhi.py`
**Purpose**: Fixed phi model (Prediction Mode)

**When to use**:
- Clinical prediction scenarios
- Age-offset predictions
- Any analysis where phi is pre-learned and held constant

**Key class**: `AladynSurvivalFixedPhi`

**Usage**:
```python
from clust_huge_amp_fixedPhi import AladynSurvivalFixedPhi

# Load pre-trained phi from master checkpoint
pretrained_phi = torch.load('master_checkpoint.pt')['model_state_dict']['phi']
pretrained_psi = torch.load('master_checkpoint.pt')['model_state_dict']['psi']

model = AladynSurvivalFixedPhi(
    N=Y.shape[0], D=Y.shape[1], T=Y.shape[2], K=20,
    P=G.shape[1], G=G, Y=Y, W=0.0001, R=0,
    prevalence_t=prevalence_t,
    pretrained_phi=pretrained_phi,
    pretrained_psi=pretrained_psi,
    signature_references=signature_refs,
    healthy_reference=True,
    disease_names=disease_names
)
history = model.fit(E, num_epochs=200)
pi, theta = model.forward()  # phi is fixed, not returned
```

---

### 3. `weighted_aladyn.py`
**Purpose**: IPW-weighted model (Selection Bias Correction)

**When to use**:
- Correcting for UK Biobank selection bias
- IPW weighting analyses
- Population representativeness studies

**Key class**: `AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest_weighted`

**Usage**:
```python
from weighted_aladyn import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest_weighted

# Load IPW weights
weights = torch.load('ipw_weights.pt')

model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest_weighted(
    N=Y.shape[0], D=Y.shape[1], T=Y.shape[2], K=20,
    P=G.shape[1], G=G, Y=Y, W=0.0001, R=0,
    prevalence_t=prevalence_t,
    signature_references=signature_refs,
    healthy_reference=True,
    disease_names=disease_names,
    weights=weights  # IPW weights
)
history = model.fit(E, num_epochs=200)
pi, theta, phi = model.forward()
```

---

## Model Modes: Discovery vs Prediction

### Discovery Mode (`clust_huge_amp.py`)
- **Phi**: Learned from data (joint estimation)
- **Use case**: Understanding disease biology, signature interpretation
- **Output**: phi, theta, pi

### Prediction Mode (`clust_huge_amp_fixedPhi.py`)
- **Phi**: Pre-learned and fixed (from master checkpoint)
- **Use case**: Clinical prediction, age-offset analyses
- **Output**: theta, pi (phi is fixed)

See `notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb` for detailed explanation.

---

## Dependencies

All models require:
- `torch` (PyTorch)
- `numpy`
- `scipy`
- `sklearn` (for clustering)

---

## Data Requirements

All models need:
- `Y`: Disease outcome tensor (N × D × T)
- `E`: Event time matrix (N × D)
- `G`: Genetic data matrix (N × P)
- `prevalence_t`: Disease prevalence over time
- `signature_references`: Reference trajectories (optional)
- `disease_names`: List of disease names (optional)

---

## References

- Discovery vs Prediction framework: `notebooks/framework/Discovery_Prediction_Framework_Overview.ipynb`
- IPW weighting analysis: `notebooks/supporting/ipw_analysis_summary.ipynb`
- Fixed vs Joint Phi comparison: `notebooks/R3/R3_Fixed_vs_Joint_Phi_Comparison.ipynb`

