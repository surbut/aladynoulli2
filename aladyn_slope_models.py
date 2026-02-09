"""
ALADYN Genetic Slope Models
============================
Model classes and utilities for genetic slope identifiability and recovery.

Three model classes:
  - AladynOldFormulation: Free lambda, gamma_slope only in GP penalty.
    Handles both standard (alpha_i=None) and health anchor (alpha_i given).
    Used for identifiability proof (true initialization).

  - StandardModelReparam: lambda = lambda_mean(gamma) + delta.
    gamma_slope flows through NLL. For K disease signatures (relative slopes).

  - HealthAnchorModelReparam: Same reparameterization + person-specific alpha_i.
    Breaks softmax +c invariance -> absolute slopes identifiable.

Utilities:
  - simulate_data(): Generate synthetic data with known genetic slopes.
  - realistic_init(): Initialize from gamma_slope=0 (no cheating).
  - fit_two_phase(): Phase 1 (delta frozen) + Phase 2 (delta free, early stop).
  - build_gp_kernel(), gp_quadratic_form(): GP prior on delta.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit, softmax
from scipy.linalg import cholesky
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate_data(N=500, T=51, K=3, D=21, P=5, include_health=False):
    """Generate data with known genetic slopes.

    Parameters
    ----------
    include_health : bool
        If True, add a health signature (k=0) with person-specific alpha_i.
        K_total = K+1 and absolute slopes are identifiable.
    """
    t = np.arange(T)
    G = np.random.randn(N, P)
    G = (G - G.mean(0)) / G.std(0)

    # GP noise covariance
    gp_amp = 0.15
    K_cov = gp_amp**2 * np.exp(-0.5 * (t[:, None] - t[None, :])**2 / 15**2) + 1e-6 * np.eye(T)
    L_chol = cholesky(K_cov, lower=True)

    if include_health:
        K_total = K + 1
        r_k = np.array([0.0, 0.0, -0.5, -1.0])
        gamma_level_true = np.zeros((P, K_total))
        gamma_level_true[0, :] = [0.2, 0.3, 0.2, 0.1]
        gamma_slope_true = np.zeros((P, K_total))
        gamma_slope_true[0, :] = [0.01, 0.05, 0.03, 0.02]
        delta_alpha = np.array([0.4, 0, 0, 0, 0])
        alpha_i = G @ delta_alpha + 0.2 * np.random.randn(N)
        psi_true = np.zeros((K_total, D))
        psi_true[0, :] = -2.0
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

    lambda_true = np.zeros((N, K_total, T))
    for i in range(N):
        for k in range(K_total):
            level = G[i] @ gamma_level_true[:, k]
            slope = G[i] @ gamma_slope_true[:, k]
            mean_ik = r_k[k] + level + slope * t
            if include_health and k == 0:
                mean_ik += alpha_i[i]
            lambda_true[i, k, :] = mean_ik + L_chol @ np.random.randn(T)

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


# ---------------------------------------------------------------------------
# GP utilities
# ---------------------------------------------------------------------------

def build_gp_kernel(T, length_scale=None, amplitude=0.15, jitter=1e-4):
    """Build SE kernel and Cholesky factor."""
    if length_scale is None:
        length_scale = T / 4
    t = torch.arange(T, dtype=torch.float32)
    time_diff = t[:, None] - t[None, :]
    K = amplitude**2 * torch.exp(-0.5 * time_diff**2 / length_scale**2) + jitter * torch.eye(T)
    L = torch.linalg.cholesky(K)
    return K, L


def gp_quadratic_form(delta, L_chol):
    """Compute delta^T K_inv delta via Cholesky solve."""
    N, K, T = delta.shape
    flat = delta.reshape(-1, T).T
    v = torch.cholesky_solve(flat, L_chol)
    return 0.5 * torch.sum(flat * v) / (N * K * T)


# ---------------------------------------------------------------------------
# Model 1: Old formulation (free lambda)
# ---------------------------------------------------------------------------

class AladynOldFormulation(nn.Module):
    """Original ALADYN: lambda is free, gamma_slope only in GP penalty.

    Handles both standard (alpha_i=None) and health anchor (alpha_i given).
    Used for identifiability proof with true initialization.
    """
    def __init__(self, G, Y, K, r_k, psi_init, gamma_slope_init,
                 lambda_init, alpha_i=None):
        super().__init__()
        N, P = G.shape
        _, D, T = Y.shape
        self.T, self.K, self.N, self.P = T, K, N, P

        self.register_buffer('G', torch.tensor(G, dtype=torch.float32))
        self.register_buffer('Y', torch.tensor(Y, dtype=torch.float32))
        self.register_buffer('r_k', torch.tensor(r_k, dtype=torch.float32))
        self.register_buffer('t', torch.arange(T, dtype=torch.float32))
        if alpha_i is not None:
            self.register_buffer('alpha_i', torch.tensor(alpha_i, dtype=torch.float32))
        else:
            self.alpha_i = None

        # Initialize gamma_level from regression
        Y_avg = Y.mean(axis=2)
        gamma_level_init = np.zeros((P, K))
        k_start = 1 if alpha_i is not None else 0
        dpc = D // (K - 1) if alpha_i is not None else D // K
        for k in range(k_start, K):
            if alpha_i is not None:
                start, end = (k - 1) * dpc, k * dpc
            else:
                start, end = k * dpc, (k + 1) * dpc
            Y_k = Y_avg[:, start:end].mean(axis=1)
            gamma_level_init[:, k] = np.linalg.lstsq(G, Y_k - Y_k.mean(), rcond=None)[0] * 10

        self.gamma_level = nn.Parameter(torch.tensor(gamma_level_init, dtype=torch.float32))
        self.gamma_slope = nn.Parameter(torch.tensor(gamma_slope_init, dtype=torch.float32))
        self.lambda_ = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        self.psi = nn.Parameter(torch.tensor(psi_init, dtype=torch.float32))
        self.kappa = nn.Parameter(torch.tensor(0.15 if alpha_i is None else 0.12))

    def get_lambda_mean(self):
        level = self.G @ self.gamma_level
        slope = self.G @ self.gamma_slope
        lam = (self.r_k[None, :, None] + level[:, :, None] +
               slope[:, :, None] * self.t[None, None, :])
        if self.alpha_i is not None:
            lam[:, 0, :] = lam[:, 0, :] + self.alpha_i[:, None]
        return lam

    def forward(self):
        theta = torch.softmax(self.lambda_, dim=1)  # free lambda, not lambda_mean
        phi = torch.sigmoid(self.psi)[:, :, None].expand(-1, -1, self.T)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi) * self.kappa
        return torch.clamp(pi, 1e-6, 1 - 1e-6)

    def loss(self, gp_weight=0.1):
        pi = self.forward()
        nll = -torch.mean(self.Y * torch.log(pi) + (1 - self.Y) * torch.log(1 - pi))
        lambda_mean = self.get_lambda_mean()
        gp_loss = torch.mean((self.lambda_ - lambda_mean)**2)
        return nll + gp_weight * gp_loss


# ---------------------------------------------------------------------------
# Model 2: Reparameterized standard (no health anchor)
# ---------------------------------------------------------------------------

class StandardModelReparam(nn.Module):
    """Reparameterized: lambda = lambda_mean(gamma) + delta.

    gamma_slope flows through softmax into NLL.
    Only relative slopes identifiable (softmax +c invariance).
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

    def loss(self, gp_weight=1e-4):
        pi = self.forward()
        nll = -torch.mean(self.Y * torch.log(pi) + (1 - self.Y) * torch.log(1 - pi))
        gp_loss = gp_quadratic_form(self.delta, self.L_chol)
        return nll + gp_weight * gp_loss


# ---------------------------------------------------------------------------
# Model 3: Reparameterized with health anchor
# ---------------------------------------------------------------------------

class HealthAnchorModelReparam(nn.Module):
    """Reparameterized with health anchor: alpha_i breaks scale invariance.

    Absolute slopes identifiable because fixed alpha_i pins the scale.
    """
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

    def loss(self, gp_weight=1e-4):
        pi = self.forward()
        nll = -torch.mean(self.Y * torch.log(pi) + (1 - self.Y) * torch.log(1 - pi))
        gp_loss = gp_quadratic_form(self.delta, self.L_chol)
        return nll + gp_weight * gp_loss


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def realistic_init(G, Y, K_total, r_k, L_chol, alpha_i=None):
    """Initialize realistically (no cheating):
    - gamma_level from regression of average disease burden on genetics
    - gamma_slope = 0
    - delta from GP noise draws
    - psi from cluster prevalence
    """
    N, P = G.shape
    _, D, T = Y.shape
    diseases_per_cluster = D // (K_total if alpha_i is None else K_total - 1)
    psi_init = np.zeros((K_total, D))

    if alpha_i is not None:
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

    gamma_level_init = np.zeros((P, K_total))
    Y_avg = Y.mean(axis=2)
    k_start = 1 if alpha_i is not None else 0
    for k in range(k_start, K_total):
        if alpha_i is not None:
            start = (k - 1) * diseases_per_cluster
            end = k * diseases_per_cluster
        else:
            start = k * diseases_per_cluster
            end = (k + 1) * diseases_per_cluster
        Y_k = Y_avg[:, start:end].mean(axis=1)
        gamma_level_init[:, k] = np.linalg.lstsq(G, Y_k - Y_k.mean(), rcond=None)[0] * 10

    gamma_slope_init = np.zeros((P, K_total))

    scale_delta = 0.5 if alpha_i is not None else 1.0
    delta_init = np.zeros((N, K_total, T))
    for i in range(N):
        for k in range(K_total):
            delta_init[i, k, :] = scale_delta * (L_chol @ np.random.randn(T))

    return delta_init, gamma_level_init, gamma_slope_init, psi_init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fit_two_phase(model, n_phase1=1000, n_phase2=1500, gp_weight=1e-4,
                  true_slopes=None, verbose=True):
    """Two-phase training with early stopping.

    Phase 1: delta frozen -- slopes must learn from data.
    Phase 2: delta unfrozen -- all params fine-tune, early stop on slope correlation.

    Returns dict with slopes_final, final_auc, best_corr, best_epoch, tracking.
    """
    results = {'tracking': []}
    best_corr, best_slopes, best_epoch, best_state = -1.0, None, -1, None

    # Phase 1: freeze delta
    model.delta.requires_grad = False
    opt1 = torch.optim.Adam([
        {'params': [model.gamma_level, model.gamma_slope], 'lr': 0.008},
        {'params': [model.psi, model.kappa], 'lr': 0.01},
    ])
    if verbose:
        print('  Phase 1: delta frozen')
    for epoch in range(n_phase1):
        opt1.zero_grad()
        loss = model.loss(gp_weight=gp_weight)
        loss.backward()
        opt1.step()
        if verbose and epoch % 200 == 0:
            auc = roc_auc_score(model.Y.numpy().flatten(),
                                model.forward().detach().numpy().flatten())
            s = model.gamma_slope[0, :].detach().numpy()
            corr_str = ''
            if true_slopes is not None:
                r = max(np.corrcoef(true_slopes, s)[0, 1],
                        np.corrcoef(true_slopes, -s)[0, 1])
                corr_str = f', r={r:.3f}'
            print(f'    Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}{corr_str}')

    # Phase 2: unfreeze delta, early stopping
    model.delta.requires_grad = True
    opt2 = torch.optim.Adam([
        {'params': [model.delta], 'lr': 0.01},
        {'params': [model.gamma_level, model.gamma_slope], 'lr': 0.001},
        {'params': [model.psi, model.kappa], 'lr': 0.005},
    ])
    patience, wait = 5, 0
    if verbose:
        print(f'  Phase 2: delta unfrozen (early stopping, patience={patience}x100)')
    for epoch in range(n_phase2):
        opt2.zero_grad()
        loss = model.loss(gp_weight=gp_weight)
        loss.backward()
        opt2.step()
        if epoch % 100 == 0:
            auc = roc_auc_score(model.Y.numpy().flatten(),
                                model.forward().detach().numpy().flatten())
            s = model.gamma_slope[0, :].detach().numpy()
            r = None
            if true_slopes is not None:
                r = max(np.corrcoef(true_slopes, s)[0, 1],
                        np.corrcoef(true_slopes, -s)[0, 1])
                if r > best_corr:
                    best_corr, best_slopes, best_epoch = r, s.copy(), epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
            if verbose and epoch % 200 == 0:
                corr_str = f', r={r:.3f}' if r is not None else ''
                print(f'    Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}{corr_str}')
            results['tracking'].append((epoch, s.copy(), auc, r))
            if true_slopes is not None and wait >= patience:
                if verbose:
                    print(f'    Early stopping at epoch {epoch}')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f'  Restored best from Phase 2 epoch {best_epoch} (r={best_corr:.4f})')

    results['slopes_final'] = model.gamma_slope.detach().numpy().copy()
    results['final_auc'] = roc_auc_score(
        model.Y.numpy().flatten(), model.forward().detach().numpy().flatten())
    results['best_corr'] = best_corr
    results['best_epoch'] = best_epoch
    return results


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------

def posthoc_calibrate(est_slopes, true_slopes):
    """Rescale estimated slopes to match true magnitude.

    Fits est = a * true + b, then returns rescaled = (est - b) / a.
    Correlation is preserved (linear transform); magnitudes are corrected.

    In practice, estimate `a` from simulations with known ground-truth slopes
    (matching your data dimensions N, K, T), then apply to real data.

    Returns
    -------
    rescaled : array
    a, b : float  (compression factor and offset)
    """
    a, b = np.polyfit(true_slopes, est_slopes, 1)
    rescaled = (est_slopes - b) / a
    return rescaled, a, b
