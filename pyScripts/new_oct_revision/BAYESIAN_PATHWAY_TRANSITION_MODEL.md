# Bayesian Pathway Transition Model (BPTM)

## Overview

A Bayesian model for predicting disease transitions (e.g., Rheumatoid arthritis → Myocardial Infarction) based on signature deviation trajectories, extending the aladynoulli framework to pathway-level analysis.

## Model Motivation

Current transition analysis:
- Computes signature deviations: `δ_i,k,t = θ_i,k,t - μ_k,t`
- Identifies pathways via clustering on deviation patterns
- Compares progressors vs non-progressors descriptively

**Limitation**: No probabilistic model for transition risk as a function of signature deviations.

## Model Formulation

### Core Idea

Model the **transition probability** `π_i(t|d_precursor)` for patient `i` to develop target disease at time `t`, given they have precursor disease `d_precursor`, as a function of their signature deviation trajectory.

### Mathematical Framework

For patient `i` with precursor disease `d_precursor` at time `t_precursor`:

**1. Transition Probability Model:**

```
π_i(t | d_precursor, t_precursor) = κ · sigmoid(η_i(t))
```

where `η_i(t)` is a logit combining:
- Baseline transition risk (disease-specific)
- Signature deviation contributions
- Temporal dynamics
- Individual genetic factors

**2. Logit Function:**

```
η_i(t) = α_{d_precursor} + β_{d_precursor} · (t - t_precursor) + 
         Σ_k γ_k · δ_i,k,t + 
         Σ_k Σ_τ ω_k,τ · δ_i,k,t-τ · I(τ ≤ lookback) +
         G_i^T · Γ_{d_precursor}
```

where:
- `α_{d_precursor}`: Baseline log-odds for transition from `d_precursor`
- `β_{d_precursor}`: Time-dependent baseline (linear trend)
- `γ_k`: Signature-specific transition coefficients (how much signature `k` deviations affect transition)
- `ω_k,τ`: Lagged effects (signature deviations `τ` timepoints ago)
- `δ_i,k,t = θ_i,k,t - μ_k,t`: Signature deviation from population reference
- `G_i`: Genetic/demographic factors
- `Γ_{d_precursor}`: Genetic effects on transition from `d_precursor`

**3. Signature Deviation Dynamics:**

Model deviations using Gaussian Processes (like aladynoulli):

```
δ_i,k,t ~ GP(μ_δ,k(t), K_δ,k(t, t'))
```

where:
- `μ_δ,k(t)`: Mean deviation trajectory for signature `k` (can be pathway-specific)
- `K_δ,k`: Temporal covariance kernel (squared exponential)

**4. Pathway-Specific Effects:**

For pathway `p` (discovered via clustering):

```
γ_k,p = γ_k + γ_k^pathway · I(patient i in pathway p)
```

This allows different signature contributions for different pathways.

### Hierarchical Structure

```
Level 1: Population-level
  - Baseline transition rates: α_d, β_d
  - Signature effects: γ_k, ω_k,τ
  - Genetic effects: Γ_d

Level 2: Pathway-level
  - Pathway-specific deviations: μ_δ,k,p(t)
  - Pathway-specific signature effects: γ_k,p

Level 3: Individual-level
  - Individual deviations: δ_i,k,t
  - Individual transition risk: π_i(t)
```

## Key Differences from Aladynoulli

| Aspect | Aladynoulli | BPTM |
|--------|-------------|------|
| **Outcome** | Disease occurrence `Y_i,d,t` | Transition occurrence `T_i,d_precursor→d_target` |
| **Features** | Signature loadings `θ_i,k,t` | Signature deviations `δ_i,k,t` |
| **Conditioning** | Unconditional disease risk | Conditional on precursor disease |
| **Temporal focus** | All timepoints | Pre-transition window |
| **Pathway integration** | Implicit (via signatures) | Explicit (pathway-specific effects) |

## Advantages

1. **Probabilistic predictions**: Quantify transition risk, not just descriptive patterns
2. **Uncertainty quantification**: Bayesian framework provides credible intervals
3. **Pathway-specific effects**: Different pathways can have different signature contributions
4. **Temporal dynamics**: Captures how deviation trajectories predict transitions
5. **Cross-cohort generalization**: Can compare pathway effects across UKB/MGB

## Implementation Strategy

### Phase 1: Basic Model
- Fixed signature effects (no pathway-specific)
- Simple temporal kernel
- MCMC inference

### Phase 2: Pathway Integration
- Pathway-specific effects
- Hierarchical priors

### Phase 3: Cross-Cohort
- Shared pathway structure
- Cohort-specific baseline rates

## Data Requirements

- `Y`: Disease outcomes (N × D × T)
- `θ`: Signature loadings (N × K × T)
- `G`: Genetic/demographic factors (N × P)
- `E`: Event times for precursor diseases
- Pathway labels (from clustering)

## Output

- Transition probability trajectories: `π_i(t)` for each patient
- Signature importance: `γ_k` coefficients
- Pathway-specific effects: `γ_k,p`
- Predictive performance: AUC, calibration

