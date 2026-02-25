# Full Joint Distribution of the Aladynoulli Model

## Parameters

| Symbol | Shape | Description |
|--------|-------|-------------|
| $\Gamma$ | $P \times (K+1)$ | Genetic effect matrix (P features, K+1 signatures) |
| $\delta$ | $N \times (K+1) \times T$ | Individual residual trajectories |
| $\psi$ | $(K+1) \times D$ | Signature-to-disease mapping |
| $\epsilon$ | $(K+1) \times D \times T$ | Residual disease loadings |
| $r$ | $K+1$ | Signature intercepts (fixed from training) |
| $G$ | $N \times P$ | Observed genetic covariates (PRS, Sex, PCs) |

## Prior

$$p(\Gamma) \propto 1 \quad \text{(flat — no explicit prior on gamma)}$$

$$p(\delta_i) = \prod_{k=0}^{K} \mathcal{N}(\delta_{ik} \mid 0,\; K_\theta) \quad \text{for each patient } i$$

where $K_\theta$ is the GP covariance kernel over time.

## Deterministic Transforms

**Signature loadings:**

$$\lambda_{ikt} = r_k + \mathbf{g}_i^\top \Gamma_k + \delta_{ikt}$$

**Signature weights (softmax over signatures):**

$$\theta_{ikt} = \frac{\exp(\lambda_{ikt})}{\sum_{k'} \exp(\lambda_{ik't})}$$

**Disease probabilities within each signature:**

$$\phi_{kdt} = \sigma\!\left(\text{logit\_prev}_{dt} + \psi_{kd} + \epsilon_{kdt}\right)$$

where $\sigma$ is the sigmoid function.

**Predicted probability of disease $d$ for patient $i$ at time $t$:**

$$\pi_{idt} = \sum_{k=0}^{K} \theta_{ikt} \cdot \phi_{kdt}$$

## Likelihood

$$p(y_{idt} \mid \pi_{idt}) = \text{Bernoulli}(\pi_{idt})$$

## Full Joint Distribution

$$p(\Gamma,\; \delta,\; \psi,\; \epsilon,\; y \mid G) = p(\Gamma) \cdot \prod_{i=1}^{N}\left[p(\delta_i) \cdot \prod_{d,t} p(y_{idt} \mid \pi_{idt})\right]$$

Expanding:

$$= \underbrace{1}_{\text{flat prior on } \Gamma} \;\cdot\; \prod_{i=1}^{N}\prod_{k=0}^{K} \underbrace{\mathcal{N}(\delta_{ik} \mid 0,\; K_\theta)}_{\text{GP prior on residuals}} \;\cdot\; \prod_{i,d,t} \underbrace{p(y_{idt} \mid \pi_{idt}(\Gamma, \delta, \psi, \epsilon))}_{\text{data likelihood}}$$

## MAP Loss (Negative Log Joint)

$$\mathcal{L} = \underbrace{-\sum_{i,d,t}\log p(y_{idt} \mid \pi_{idt})}_{\text{NLL (data fit)}} \;+\; \underbrace{W\sum_{i,k}\delta_{ik}^\top K_\theta^{-1}\,\delta_{ik}}_{\text{GP penalty on } \delta}$$

## The Prior on Gamma is Flat

Factorize the hierarchy:

$$p(Y, \Gamma, \delta, \phi, \psi) = p(Y \mid \lambda, \phi, \psi) \cdot p(\delta) \cdot p(\Gamma)$$

where:
- $p(\Gamma) \propto 1$ — **flat, no prior on gamma**
- $p(\delta_i) = \prod_k \mathcal{N}(\delta_{ik} \mid 0, K_\theta)$ — GP prior on the residuals
- $\lambda = r + G\Gamma + \delta$ — deterministic

The GP penalty in the MAP loss is $W \cdot \delta^\top K_\theta^{-1} \delta$. There is **no term involving $\Gamma$ alone**. The prior on $\Gamma$ is flat.

## Why the Centered Parameterization Starves Gamma

### Centered: GP penalty has a direct gradient on $\Gamma$

In the **centered** parameterization, $\lambda$ is the free parameter and $\delta$ is derived:

$$\delta_{ik} = \lambda_{ik} - r_k \mathbf{1} - G_i \Gamma_k$$

Substituting into the GP penalty and differentiating:

$$\frac{\partial\, \text{GP penalty}}{\partial \Gamma_k} = -2W \sum_{i=1}^{N} G_i^\top \, K_\theta^{-1} \underbrace{\left(\lambda_{ik} - r_k\mathbf{1} - G_i\Gamma_k\right)}_{\delta_{ik}}$$

This is **nonzero** — it directly regularizes $\Gamma$, acting like L2 shrinkage. Meanwhile, the NLL gradient on $\Gamma$ is **zero** because $\Gamma$ doesn't appear in the forward pass ($\lambda$ is read directly from memory). So $\Gamma$ only sees the weak prior gradient ($\propto W = 10^{-4}$), which actively shrinks it.

### Non-centered: GP penalty gradient on $\Gamma$ is zero

In the **non-centered** parameterization, $\delta$ is an independent free parameter. The GP penalty is $W \|\delta\|^2_{K^{-1}}$, and:

$$\frac{\partial\, \text{GP penalty}}{\partial \Gamma_k} = 0$$

**Zero.** The GP penalty doesn't touch $\Gamma$ at all.

Instead, $\Gamma$ gets a **strong NLL gradient** via the chain rule: $Y \to \pi \to \theta \to \lambda \to \Gamma$, because $\lambda = G\Gamma + \delta$ puts $\Gamma$ in the forward pass.

### The gradient comparison

| | Centered | Non-centered |
|---|---|---|
| GP penalty gradient on $\Gamma$ | **Nonzero**: $-2W \sum_i G_i^\top K^{-1} \delta_{ik}$ (shrinks $\Gamma$) | **Zero** |
| NLL gradient on $\Gamma$ | **Zero** ($\Gamma$ not in forward pass) | **Strong** (chain rule through $\lambda$) |
| Net effect | $\Gamma$ starved — only weak shrinkage signal | $\Gamma$ well-estimated — data-driven |
| Gamma recovery | $r = 0.796$ (over-shrunk) | $r = 0.954$ (mild shrinkage) |

## Residual Shrinkage in the Non-Centered Model

Even in the non-centered model, simulations show mild shrinkage of $\hat\Gamma$ toward zero. This is **not** from a prior — it's an **optimization phenomenon**.

$\Gamma$ and $\delta$ jointly determine $\lambda = G\Gamma + \delta$. The NLL wants $\lambda$ at specific values. If $\Gamma$ increases, $\delta$ must decrease to maintain the same $\lambda$ — but that costs GP penalty. So the optimizer faces a tradeoff:

- **Increase $\Gamma_k$** → genetic signal better captured → but $\delta$ must shift, increasing GP cost
- **Keep $\Gamma_k$ small** → $\delta$ stays GP-friendly → but genetic signal underused

This is coupled optimization dynamics, not a prior. The GP shapes the $\delta$ landscape, and $\Gamma$ lives in whatever space $\delta$ leaves for it.

**Pooling** across batches further stabilizes $\Gamma$ by averaging out the remaining sampling variability.
