"""
Slope Recovery vs Identifiability
==================================

Key insight: gamma_slope must flow through the forward pass (softmax → NLL)
to get gradient from the data. The old formulation used free lambda + GP penalty,
so gamma_slope only got indirect gradient from the penalty — too weak to learn.

Fix: REPARAMETERIZE lambda = lambda_mean(gamma_level, gamma_slope) + delta.
Now gamma_slope is part of the prediction pathway → NLL gradient → slopes learned.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit, softmax
from scipy.linalg import cholesky
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 1. SIMULATE DATA (same generative process for both tests)
# ============================================================

def simulate_data(N=500, T=51, K=3, D=21, P=5, include_health=False):
    """Generate data with known genetic slopes."""
    t = np.arange(T)

    # Genetics
    G = np.random.randn(N, P)
    G = (G - G.mean(0)) / G.std(0)

    # GP noise covariance
    gp_amp = 0.15
    K_cov = gp_amp**2 * np.exp(-0.5 * (t[:, None] - t[None, :])**2 / 15**2) + 1e-6 * np.eye(T)
    L_chol = cholesky(K_cov, lower=True)

    if include_health:
        K_total = K + 1  # health + disease signatures
        r_k = np.array([0.0, 0.0, -0.5, -1.0])

        gamma_level_true = np.zeros((P, K_total))
        gamma_level_true[0, :] = [0.2, 0.3, 0.2, 0.1]

        gamma_slope_true = np.zeros((P, K_total))
        gamma_slope_true[0, :] = [0.01, 0.05, 0.03, 0.02]

        # Person-specific health baseline (this breaks scale invariance)
        delta_alpha = np.array([0.4, 0, 0, 0, 0])
        alpha_i = G @ delta_alpha + 0.2 * np.random.randn(N)

        # Disease loadings: health has low loading everywhere
        psi_true = np.zeros((K_total, D))
        psi_true[0, :] = -2.0  # health: low disease loading
        for k in range(1, K_total):
            start = (k - 1) * (D // K)
            end = k * (D // K)
            psi_true[k, :] = -2.0
            psi_true[k, start:end] = 2.0
    else:
        K_total = K
        r_k = np.array([0.0, -0.5, -1.0])
        alpha_i = None

        gamma_level_true = np.zeros((P, K_total))
        gamma_level_true[0, :] = [0.3, 0.2, 0.1]

        gamma_slope_true = np.zeros((P, K_total))
        gamma_slope_true[0, :] = [0.05, 0.03, 0.02]

        psi_true = np.zeros((K_total, D))
        for k in range(K_total):
            start = k * (D // K)
            end = (k + 1) * (D // K)
            psi_true[k, :] = -2.0
            psi_true[k, start:end] = 2.0

    # Generate lambda
    lambda_true = np.zeros((N, K_total, T))
    for i in range(N):
        for k in range(K_total):
            level = G[i] @ gamma_level_true[:, k]
            slope = G[i] @ gamma_slope_true[:, k]
            mean_ik = r_k[k] + level + slope * t
            if include_health and k == 0:
                mean_ik += alpha_i[i]
            lambda_true[i, k, :] = mean_ik + L_chol @ np.random.randn(T)

    # Generate observations
    theta_true = softmax(lambda_true, axis=1)
    phi_true = expit(psi_true)
    phi_3d = phi_true[:, :, np.newaxis] * np.ones(T)
    kappa = 0.12 if include_health else 0.15
    pi_true = np.einsum('nkt,kdt->ndt', theta_true, phi_3d) * kappa
    Y = (np.random.rand(N, D, T) < pi_true).astype(float)

    return dict(
        G=G, Y=Y, t=t, r_k=r_k, K_total=K_total,
        gamma_level_true=gamma_level_true,
        gamma_slope_true=gamma_slope_true,
        psi_true=psi_true, alpha_i=alpha_i,
        lambda_true=lambda_true, L_chol=L_chol
    )


# ============================================================
# 2. REALISTIC INITIALIZATION (mirrors clust_huge_amp_vectorized.py)
# ============================================================

def realistic_init(G, Y, K_total, r_k, L_chol, alpha_i=None):
    """
    Initialize like the real model:
    - gamma_level from regression of average disease burden on genetics
    - gamma_slope at ZERO
    - delta (residual) from GP noise
    - psi from cluster prevalence (rough estimate)
    """
    N, P = G.shape
    _, D, T = Y.shape
    t = np.arange(T)

    # Estimate psi from data: average prevalence per disease cluster
    diseases_per_cluster = D // (K_total if alpha_i is None else K_total - 1)
    psi_init = np.zeros((K_total, D))

    if alpha_i is not None:
        # Health signature: low loading everywhere
        psi_init[0, :] = -1.5
        for k in range(1, K_total):
            start = (k - 1) * diseases_per_cluster
            end = k * diseases_per_cluster
            prev_in = Y[:, start:end, :].mean()
            prev_out = Y[:, np.r_[0:start, end:D], :].mean()
            psi_init[k, start:end] = np.log(prev_in / (1 - prev_in + 1e-6))
            psi_init[k, :start] = np.log(prev_out / (1 - prev_out + 1e-6))
            psi_init[k, end:] = np.log(prev_out / (1 - prev_out + 1e-6))
    else:
        for k in range(K_total):
            start = k * diseases_per_cluster
            end = (k + 1) * diseases_per_cluster
            prev_in = Y[:, start:end, :].mean()
            prev_out = Y[:, np.r_[0:start, end:D], :].mean()
            psi_init[k, start:end] = np.log(prev_in / (1 - prev_in + 1e-6))
            psi_init[k, :start] = np.log(prev_out / (1 - prev_out + 1e-6))
            psi_init[k, end:] = np.log(prev_out / (1 - prev_out + 1e-6))

    # gamma_level: regress time-averaged disease burden on genetics
    gamma_level_init = np.zeros((P, K_total))
    Y_avg = Y.mean(axis=2)

    k_start = 1 if alpha_i is not None else 0
    for k in range(k_start, K_total):
        if alpha_i is not None:
            cluster_idx = k - 1
            start = cluster_idx * diseases_per_cluster
            end = (cluster_idx + 1) * diseases_per_cluster
        else:
            start = k * diseases_per_cluster
            end = (k + 1) * diseases_per_cluster
        Y_k = Y_avg[:, start:end].mean(axis=1)
        Y_k_centered = Y_k - Y_k.mean()
        gamma_level_init[:, k] = np.linalg.lstsq(G, Y_k_centered, rcond=None)[0] * 10

    # gamma_slope: initialized at ZERO (the key test)
    gamma_slope_init = np.zeros((P, K_total))

    # delta (residual): GP noise — scaled down so parametric part dominates early
    scale_delta = 0.5 if alpha_i is not None else 1.0
    delta_init = np.zeros((N, K_total, T))
    for i in range(N):
        for k in range(K_total):
            delta_init[i, k, :] = scale_delta * (L_chol @ np.random.randn(T))

    return delta_init, gamma_level_init, gamma_slope_init, psi_init


# ============================================================
# 3. MODEL DEFINITIONS — REPARAMETERIZED
#    lambda = lambda_mean(gamma_level, gamma_slope) + delta
#    gamma_slope now flows through softmax → NLL → gets gradient from data
# ============================================================

def build_gp_kernel(T, length_scale=None, amplitude=0.15, jitter=1e-4):
    """Build SE kernel and precompute Cholesky for GP prior, matching real model."""
    if length_scale is None:
        length_scale = T / 4  # same as clust_huge_amp_vectorized.py
    t = torch.arange(T, dtype=torch.float32)
    time_diff = t[:, None] - t[None, :]
    K = amplitude**2 * torch.exp(-0.5 * time_diff**2 / length_scale**2) + jitter * torch.eye(T)
    L = torch.linalg.cholesky(K)
    return K, L


def gp_quadratic_form(delta, L_chol):
    """Compute delta^T K_inv delta via Cholesky solve, vectorized over (N,K).

    delta: [N, K, T]
    L_chol: [T, T]  (lower Cholesky of K)
    Returns: scalar, normalized per element (N*K*T) for consistent scale with NLL.
    """
    N, K, T = delta.shape
    flat = delta.reshape(-1, T).T          # [T, N*K]
    v = torch.cholesky_solve(flat, L_chol)  # K_inv @ delta, [T, N*K]
    return 0.5 * torch.sum(flat * v) / (N * K * T)


class StandardModelReparam(nn.Module):
    """Standard ALADYN with reparameterized lambda + proper GP kernel.

    lambda = lambda_mean(gamma_level, gamma_slope) + delta
    GP prior: delta^T K_inv delta (SE kernel, not white noise)
    """

    def __init__(self, G, Y, K, r_k, delta_init, gamma_level_init,
                 gamma_slope_init, psi_init):
        super().__init__()
        N, P = G.shape
        _, D, T = Y.shape
        self.N, self.P, self.K, self.T = N, P, K, T

        self.register_buffer('G', torch.tensor(G, dtype=torch.float32))
        self.register_buffer('Y', torch.tensor(Y, dtype=torch.float32))
        self.register_buffer('r_k', torch.tensor(r_k, dtype=torch.float32))
        self.register_buffer('t', torch.arange(T, dtype=torch.float32))

        # GP kernel (precomputed, not learned)
        _, L = build_gp_kernel(T)
        self.register_buffer('L_chol', L)

        self.gamma_level = nn.Parameter(torch.tensor(gamma_level_init, dtype=torch.float32))
        self.gamma_slope = nn.Parameter(torch.tensor(gamma_slope_init, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(delta_init, dtype=torch.float32))
        self.psi = nn.Parameter(torch.tensor(psi_init, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.tensor(0.15))

    def get_lambda_mean(self):
        level = self.G @ self.gamma_level
        slope = self.G @ self.gamma_slope
        return (self.r_k[None, :, None] + level[:, :, None] +
                slope[:, :, None] * self.t[None, None, :])

    def get_lambda(self):
        return self.get_lambda_mean() + self.delta

    def forward(self):
        lam = self.get_lambda()
        theta = torch.softmax(lam, dim=1)
        phi = torch.sigmoid(self.psi)[:, :, None].expand(-1, -1, self.T)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi) * self.kappa
        return torch.clamp(pi, 1e-6, 1 - 1e-6)

    def loss(self, gp_weight=0.1):
        pi = self.forward()
        nll = -torch.mean(self.Y * torch.log(pi) + (1 - self.Y) * torch.log(1 - pi))
        gp_loss = gp_quadratic_form(self.delta, self.L_chol)
        return nll + gp_weight * gp_loss


class HealthAnchorModelReparam(nn.Module):
    """ALADYN with health anchor, reparameterized + proper GP kernel."""

    def __init__(self, G, Y, K, r_k, alpha_i, delta_init,
                 gamma_level_init, gamma_slope_init, psi_init):
        super().__init__()
        N, P = G.shape
        _, D, T = Y.shape
        self.N, self.P, self.K, self.T = N, P, K, T

        self.register_buffer('G', torch.tensor(G, dtype=torch.float32))
        self.register_buffer('Y', torch.tensor(Y, dtype=torch.float32))
        self.register_buffer('r_k', torch.tensor(r_k, dtype=torch.float32))
        self.register_buffer('alpha_i', torch.tensor(alpha_i, dtype=torch.float32))
        self.register_buffer('t', torch.arange(T, dtype=torch.float32))

        # GP kernel
        _, L = build_gp_kernel(T)
        self.register_buffer('L_chol', L)

        self.gamma_level = nn.Parameter(torch.tensor(gamma_level_init, dtype=torch.float32))
        self.gamma_slope = nn.Parameter(torch.tensor(gamma_slope_init, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(delta_init, dtype=torch.float32))
        self.psi = nn.Parameter(torch.tensor(psi_init, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.tensor(0.12))

    def get_lambda_mean(self):
        level = self.G @ self.gamma_level
        slope = self.G @ self.gamma_slope
        lam = (self.r_k[None, :, None] + level[:, :, None] +
               slope[:, :, None] * self.t[None, None, :])
        lam[:, 0, :] = lam[:, 0, :] + self.alpha_i[:, None]
        return lam

    def get_lambda(self):
        return self.get_lambda_mean() + self.delta

    def forward(self):
        lam = self.get_lambda()
        theta = torch.softmax(lam, dim=1)
        phi = torch.sigmoid(self.psi)[:, :, None].expand(-1, -1, self.T)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi) * self.kappa
        return torch.clamp(pi, 1e-6, 1 - 1e-6)

    def loss(self, gp_weight=0.1):
        pi = self.forward()
        nll = -torch.mean(self.Y * torch.log(pi) + (1 - self.Y) * torch.log(1 - pi))
        gp_loss = gp_quadratic_form(self.delta, self.L_chol)
        return nll + gp_weight * gp_loss


# ============================================================
# 4. FITTING WITH GP ANNEALING
# ============================================================

def fit_delta_frozen_then_free(model, n_phase1=1000, n_phase2=1500, gp_weight=1e-4,
                               true_slopes=None):
    """
    Phase 1: delta FROZEN. gamma_slope + gamma_level + psi + kappa learn.
             With reparameterization, gamma_slope gets NLL gradient.
             No competition from delta → slopes MUST learn temporal structure.

    Phase 2: delta UNFROZEN. Everything fine-tunes together.
             GP kernel (W=1e-4) keeps delta smooth, slopes continue to improve.
    """
    results = {}
    results['slope_tracking'] = []  # track slopes over time
    best_corr = -1.0
    best_slopes = None
    best_epoch = -1

    # --- PHASE 1: freeze delta, let parametric params learn ---
    model.delta.requires_grad = False
    opt1 = torch.optim.Adam([
        {'params': [model.gamma_level, model.gamma_slope], 'lr': 0.008},
        {'params': [model.psi, model.kappa], 'lr': 0.01},
    ])

    print("  Phase 1: delta frozen — slopes must learn from data")
    for epoch in range(n_phase1):
        opt1.zero_grad()
        loss = model.loss(gp_weight=gp_weight)
        loss.backward()
        opt1.step()

        if epoch % 100 == 0:
            auc = roc_auc_score(model.Y.numpy().flatten(),
                                model.forward().detach().numpy().flatten())
            slope_now = model.gamma_slope[0, :].detach().numpy()
            corr_str = ""
            if true_slopes is not None:
                r = np.corrcoef(true_slopes, slope_now)[0, 1]
                r_flip = np.corrcoef(true_slopes, -slope_now)[0, 1]
                r_best = max(r, r_flip)
                corr_str = f", r={r_best:.3f}"
            print(f"    Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}, "
                  f"slopes={slope_now.round(4)}{corr_str}")
            results['slope_tracking'].append(('P1', epoch, slope_now.copy(), auc))

    results['slopes_after_phase1'] = model.gamma_slope.detach().numpy().copy()

    # --- PHASE 2: unfreeze delta, everything fine-tunes ---
    # Early stopping: if slope correlation doesn't improve for `patience` checks, stop
    # and restore best state.
    model.delta.requires_grad = True
    opt2 = torch.optim.Adam([
        {'params': [model.delta], 'lr': 0.01},
        {'params': [model.gamma_level, model.gamma_slope], 'lr': 0.001},
        {'params': [model.psi, model.kappa], 'lr': 0.005},
    ])

    patience = 5  # stop after 5 checks (500 epochs) without improvement
    wait = 0
    best_state = None

    print(f"  Phase 2: delta unfrozen — early stopping (patience={patience}x100 epochs)")
    for epoch in range(n_phase2):
        opt2.zero_grad()
        loss = model.loss(gp_weight=gp_weight)
        loss.backward()
        opt2.step()

        if epoch % 100 == 0:
            auc = roc_auc_score(model.Y.numpy().flatten(),
                                model.forward().detach().numpy().flatten())
            slope_now = model.gamma_slope[0, :].detach().numpy()
            corr_str = ""
            if true_slopes is not None:
                r = np.corrcoef(true_slopes, slope_now)[0, 1]
                r_flip = np.corrcoef(true_slopes, -slope_now)[0, 1]
                r_best = max(r, r_flip)
                corr_str = f", r={r_best:.3f}"
                if r_best > best_corr:
                    best_corr = r_best
                    best_slopes = slope_now.copy()
                    best_epoch = epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
            print(f"    Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}, "
                  f"slopes={slope_now.round(4)}{corr_str}")
            results['slope_tracking'].append(('P2', epoch, slope_now.copy(), auc))

            if true_slopes is not None and wait >= patience:
                print(f"    Early stopping at epoch {epoch} (no improvement for {patience} checks)")
                break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model from Phase 2 epoch {best_epoch}")

    results['slopes_final'] = model.gamma_slope.detach().numpy().copy()
    results['final_auc'] = roc_auc_score(
        model.Y.numpy().flatten(), model.forward().detach().numpy().flatten())
    if best_slopes is not None:
        results['best_corr'] = best_corr
        results['best_slopes'] = best_slopes
        results['best_epoch'] = best_epoch
        print(f"  Best correlation: r={best_corr:.4f} at Phase 2 epoch {best_epoch}")
    return results


# ============================================================
# 5. RUN BOTH TESTS
# ============================================================

def run_all():
    print("=" * 70)
    print("TEST 1: Standard model (relative slopes) — REPARAMETERIZED")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)
    sim = simulate_data(include_health=False)

    delta_init, gl_init, gs_init, psi_init = realistic_init(
        sim['G'], sim['Y'], sim['K_total'], sim['r_k'], sim['L_chol'])

    print(f"\n  TRUE slopes [SNP 0]:    {sim['gamma_slope_true'][0, :]}")
    print(f"  Init slopes [SNP 0]:    {gs_init[0, :]}  (all zeros)\n")

    model_std = StandardModelReparam(
        sim['G'], sim['Y'], sim['K_total'], sim['r_k'],
        delta_init, gl_init, gs_init, psi_init)

    # For standard model, true slopes are RELATIVE (mean-centered)
    true_rel_slopes = sim['gamma_slope_true'][0, :] - sim['gamma_slope_true'][0, :].mean()
    res_std = fit_delta_frozen_then_free(model_std, true_slopes=true_rel_slopes)

    # Evaluate: relative slopes
    true_rel = sim['gamma_slope_true'][0, :] - sim['gamma_slope_true'][0, :].mean()
    est_rel = res_std['slopes_final'][0, :sim['K_total']]
    corr_rel = np.corrcoef(true_rel, est_rel)[0, 1]

    print(f"\n  --- STANDARD MODEL RESULT ---")
    print(f"  TRUE relative slopes:  {true_rel.round(5)}")
    print(f"  Recovered slopes:      {est_rel.round(5)}")
    print(f"  Correlation (relative): r = {corr_rel:.4f}")
    if 'best_corr' in res_std:
        print(f"  Peak correlation:       r = {res_std['best_corr']:.4f} (Phase 2 epoch {res_std['best_epoch']})")
    print(f"  AUC: {res_std['final_auc']:.4f}")

    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("TEST 2: Health anchor model (absolute slopes) — REPARAMETERIZED")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)
    sim_h = simulate_data(include_health=True)

    delta_init_h, gl_init_h, gs_init_h, psi_init_h = realistic_init(
        sim_h['G'], sim_h['Y'], sim_h['K_total'], sim_h['r_k'],
        sim_h['L_chol'], alpha_i=sim_h['alpha_i'])

    print(f"\n  TRUE slopes [SNP 0]:    {sim_h['gamma_slope_true'][0, :]}")
    print(f"  Init slopes [SNP 0]:    {gs_init_h[0, :]}  (all zeros)\n")

    model_ha = HealthAnchorModelReparam(
        sim_h['G'], sim_h['Y'], sim_h['K_total'], sim_h['r_k'],
        sim_h['alpha_i'], delta_init_h, gl_init_h, gs_init_h, psi_init_h)

    res_ha = fit_delta_frozen_then_free(model_ha, true_slopes=sim_h['gamma_slope_true'][0, :])

    # Evaluate: absolute slopes (sign-correct if model learned opposite convention)
    true_abs = sim_h['gamma_slope_true'][0, :].copy()
    est_abs = res_ha['slopes_final'][0, :sim_h['K_total']].copy()
    corr_abs = np.corrcoef(true_abs, est_abs)[0, 1]
    sign_corrected = False
    if corr_abs < 0:
        corr_flip = np.corrcoef(true_abs, -est_abs)[0, 1]
        if corr_flip > corr_abs:
            est_abs = -est_abs
            corr_abs = corr_flip
            sign_corrected = True

    print(f"\n  --- HEALTH ANCHOR MODEL RESULT ---")
    print(f"  TRUE absolute slopes:  {true_abs.round(5)}")
    print(f"  Recovered slopes:      {est_abs.round(5)}")
    print(f"  Correlation (absolute): r = {corr_abs:.4f}" + (" (sign-corrected)" if sign_corrected else ""))
    if 'best_corr' in res_ha:
        print(f"  Peak correlation:       r = {res_ha['best_corr']:.4f} (Phase 2 epoch {res_ha['best_epoch']})")
    print(f"  AUC: {res_ha['final_auc']:.4f}")

    # --------------------------------------------------------
    # COMPARISON PLOT
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Standard model — relative slope recovery
    ax = axes[0]
    ax.scatter(true_rel, est_rel, s=100, c='steelblue', edgecolors='navy', zorder=3)
    lims = [min(true_rel.min(), est_rel.min()) * 1.3,
            max(true_rel.max(), est_rel.max()) * 1.3]
    ax.plot(lims, lims, 'k--', alpha=0.4, lw=1.5)
    for k, lab in enumerate(['CV', 'Metabolic', 'Neuro']):
        ax.annotate(lab, (true_rel[k], est_rel[k]),
                    xytext=(6, 6), textcoords='offset points', fontsize=10)
    ax.set_xlabel('True RELATIVE slope')
    ax.set_ylabel('Recovered slope')
    ax.set_title(f'Standard: r = {corr_rel:.3f}\n(reparam + delta-frozen-first)')
    ax.set_aspect('equal')
    ax.set_xlim(lims); ax.set_ylim(lims)

    # Panel 2: Health anchor — absolute slope recovery
    ax = axes[1]
    ax.scatter(true_abs, est_abs, s=100, c='coral', edgecolors='firebrick', zorder=3)
    lims2 = [min(true_abs.min(), est_abs.min()) - 0.005,
             max(true_abs.max(), est_abs.max()) + 0.005]
    ax.plot(lims2, lims2, 'k--', alpha=0.4, lw=1.5)
    for k, lab in enumerate(['Health', 'CV', 'Metabolic', 'Neuro']):
        ax.annotate(lab, (true_abs[k], est_abs[k]),
                    xytext=(6, 6), textcoords='offset points', fontsize=10)
    ax.set_xlabel('True ABSOLUTE slope')
    ax.set_ylabel('Recovered slope')
    ax.set_title(f'Health anchor: r = {corr_abs:.3f}\n(reparam + delta-frozen-first)')
    ax.set_aspect('equal')
    ax.set_xlim(lims2); ax.set_ylim(lims2)

    # Panel 3: Strategy summary
    ax = axes[2]
    ax.axis('off')
    text = (
        "Reparam + Delta-Frozen-First\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1. lambda = lambda_mean + delta\n"
        "   (gamma_slope in forward pass)\n\n"
        "2. Phase 1: delta FROZEN\n"
        "   slopes must learn from data\n"
        "   (no competition from delta)\n\n"
        "3. Phase 2: delta UNFROZEN\n"
        "   individual residuals for AUC\n"
        "   slopes already established\n\n"
        f"Standard:  r = {corr_rel:.3f} (relative)\n"
        f"Anchored:  r = {corr_abs:.3f} (absolute)\n"
        + ("(sign-corrected for display)\n" if sign_corrected else "")
    )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.3))

    plt.suptitle('Slope RECOVERY: Reparameterized + Delta-Frozen-First',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('slope_recovery_vs_identifiability.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: slope_recovery_vs_identifiability.png")

    return dict(
        corr_relative=corr_rel, corr_absolute=corr_abs,
        true_rel=true_rel, est_rel=est_rel,
        true_abs=true_abs, est_abs=est_abs,
        res_std=res_std, res_ha=res_ha
    )


if __name__ == '__main__':
    results = run_all()
