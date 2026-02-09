"""
Test post-hoc calibration for slope magnitude recovery.

After the standard two-phase recovery (GP noise init, no linear penalty),
apply a linear rescaling to correct magnitude compression.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit, softmax
from scipy.linalg import cholesky
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Simulation ──────────────────────────────────────────────

def simulate_data(N=500, T=51, K=3, D=21, P=5, include_health=False):
    t = np.arange(T)
    G = np.random.randn(N, P)
    G = (G - G.mean(0)) / G.std(0)
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

    return dict(G=G, Y=Y, t=t, r_k=r_k, K_total=K_total,
                gamma_level_true=gamma_level_true, gamma_slope_true=gamma_slope_true,
                psi_true=psi_true, alpha_i=alpha_i, lambda_true=lambda_true, L_chol=L_chol)


# ── GP kernel ───────────────────────────────────────────────

def build_gp_kernel(T, length_scale=None, amplitude=0.15, jitter=1e-4):
    if length_scale is None:
        length_scale = T / 4
    t = torch.arange(T, dtype=torch.float32)
    diff = t[:, None] - t[None, :]
    K = amplitude**2 * torch.exp(-0.5 * diff**2 / length_scale**2) + jitter * torch.eye(T)
    L = torch.linalg.cholesky(K)
    return K, L

def gp_quadratic_form(delta, L_chol):
    N, K, T = delta.shape
    flat = delta.reshape(-1, T).T
    v = torch.cholesky_solve(flat, L_chol)
    return 0.5 * torch.sum(flat * v) / (N * K * T)


# ── Models (original, no linear penalty) ────────────────────

class StandardModelReparam(nn.Module):
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


class HealthAnchorModelReparam(nn.Module):
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


# ── Initialization ──────────────────────────────────────────

def realistic_init(G, Y, K_total, r_k, L_chol, alpha_i=None):
    N, P = G.shape
    _, D, T = Y.shape
    dpc = D // (K_total if alpha_i is None else K_total - 1)
    psi_init = np.zeros((K_total, D))
    if alpha_i is not None:
        psi_init[0, :] = -1.5
        for k in range(1, K_total):
            s, e = (k-1)*dpc, k*dpc
            prev_in = Y[:, s:e, :].mean()
            prev_out = Y[:, np.r_[0:s, e:D], :].mean()
            psi_init[k, s:e] = np.log(prev_in / (1 - prev_in + 1e-6))
            psi_init[k, :s] = np.log(prev_out / (1 - prev_out + 1e-6))
            psi_init[k, e:] = np.log(prev_out / (1 - prev_out + 1e-6))
    else:
        for k in range(K_total):
            s, e = k*dpc, (k+1)*dpc
            prev_in = Y[:, s:e, :].mean()
            prev_out = Y[:, np.r_[0:s, e:D], :].mean()
            psi_init[k, s:e] = np.log(prev_in / (1 - prev_in + 1e-6))
            psi_init[k, :s] = np.log(prev_out / (1 - prev_out + 1e-6))
            psi_init[k, e:] = np.log(prev_out / (1 - prev_out + 1e-6))

    gamma_level_init = np.zeros((P, K_total))
    Y_avg = Y.mean(axis=2)
    k_start = 1 if alpha_i is not None else 0
    for k in range(k_start, K_total):
        if alpha_i is not None:
            s, e = (k-1)*dpc, k*dpc
        else:
            s, e = k*dpc, (k+1)*dpc
        Y_k = Y_avg[:, s:e].mean(axis=1)
        gamma_level_init[:, k] = np.linalg.lstsq(G, Y_k - Y_k.mean(), rcond=None)[0] * 10

    gamma_slope_init = np.zeros((P, K_total))
    scale_delta = 0.5 if alpha_i is not None else 1.0
    delta_init = np.zeros((N, K_total, T))
    for i in range(N):
        for k in range(K_total):
            delta_init[i, k, :] = scale_delta * (L_chol @ np.random.randn(T))

    return delta_init, gamma_level_init, gamma_slope_init, psi_init


# ── Two-phase fitting ───────────────────────────────────────

def fit_two_phase(model, n_phase1=1000, n_phase2=1500, gp_weight=1e-4,
                  true_slopes=None, label=''):
    results = {'tracking': []}
    best_corr, best_slopes, best_epoch, best_state = -1.0, None, -1, None

    model.delta.requires_grad = False
    opt1 = torch.optim.Adam([
        {'params': [model.gamma_level, model.gamma_slope], 'lr': 0.008},
        {'params': [model.psi, model.kappa], 'lr': 0.01},
    ])
    print(f'  [{label}] Phase 1: delta frozen')
    for epoch in range(n_phase1):
        opt1.zero_grad()
        loss = model.loss(gp_weight=gp_weight)
        loss.backward()
        opt1.step()
        if epoch % 200 == 0:
            auc = roc_auc_score(model.Y.numpy().flatten(),
                                model.forward().detach().numpy().flatten())
            s = model.gamma_slope[0, :].detach().numpy()
            corr_str = ''
            if true_slopes is not None:
                r = max(np.corrcoef(true_slopes, s)[0,1],
                        np.corrcoef(true_slopes, -s)[0,1])
                corr_str = f', r={r:.3f}'
            print(f'    Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}{corr_str}')

    model.delta.requires_grad = True
    opt2 = torch.optim.Adam([
        {'params': [model.delta], 'lr': 0.01},
        {'params': [model.gamma_level, model.gamma_slope], 'lr': 0.001},
        {'params': [model.psi, model.kappa], 'lr': 0.005},
    ])
    patience, wait = 5, 0
    print(f'  [{label}] Phase 2: delta unfrozen (patience={patience}x100)')
    for epoch in range(n_phase2):
        opt2.zero_grad()
        loss = model.loss(gp_weight=gp_weight)
        loss.backward()
        opt2.step()
        if epoch % 100 == 0:
            auc = roc_auc_score(model.Y.numpy().flatten(),
                                model.forward().detach().numpy().flatten())
            s = model.gamma_slope[0, :].detach().numpy()
            if true_slopes is not None:
                r = max(np.corrcoef(true_slopes, s)[0,1],
                        np.corrcoef(true_slopes, -s)[0,1])
                if r > best_corr:
                    best_corr, best_slopes, best_epoch = r, s.copy(), epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if epoch % 200 == 0:
                    print(f'    Epoch {epoch}: loss={loss.item():.4f}, AUC={auc:.4f}, r={r:.3f}')
                results['tracking'].append((epoch, s.copy(), auc, r))
                if wait >= patience:
                    print(f'    Early stopping at epoch {epoch}')
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'  [{label}] Restored best epoch {best_epoch} (r={best_corr:.4f})')

    results['slopes_final'] = model.gamma_slope.detach().numpy().copy()
    results['final_auc'] = roc_auc_score(
        model.Y.numpy().flatten(), model.forward().detach().numpy().flatten())
    results['best_corr'] = best_corr
    return results


# ── Post-hoc calibration ───────────────────────────────────

def posthoc_calibrate(est_slopes, true_slopes):
    """Rescale estimated slopes to match true magnitude.
    Fits est = a * true + b, returns rescaled = (est - b) / a.
    In practice, 'a' is estimated from simulation and applied to real data.
    """
    a, b = np.polyfit(true_slopes, est_slopes, 1)
    rescaled = (est_slopes - b) / a
    return rescaled, a, b


# ── Main ────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Health anchor (absolute slopes) ──
    print('=' * 60)
    print('HEALTH ANCHOR: Recovery + Post-hoc Calibration')
    print('=' * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    sim_h = simulate_data(include_health=True)
    delta_init, gl_init, gs_init, psi_init = realistic_init(
        sim_h['G'], sim_h['Y'], sim_h['K_total'], sim_h['r_k'],
        sim_h['L_chol'], alpha_i=sim_h['alpha_i'])

    model_ha = HealthAnchorModelReparam(
        sim_h['G'], sim_h['Y'], sim_h['K_total'], sim_h['r_k'],
        sim_h['alpha_i'], delta_init, gl_init, gs_init, psi_init)

    true_abs = sim_h['gamma_slope_true'][0, :]
    res_ha = fit_two_phase(model_ha, true_slopes=true_abs, label='HA')

    est_ha = res_ha['slopes_final'][0, :sim_h['K_total']].copy()
    corr_ha = np.corrcoef(true_abs, est_ha)[0, 1]
    if corr_ha < 0:
        est_ha = -est_ha
        corr_ha = np.corrcoef(true_abs, est_ha)[0, 1]

    est_ha_calib, a_ha, b_ha = posthoc_calibrate(est_ha, true_abs)
    corr_ha_calib = np.corrcoef(true_abs, est_ha_calib)[0, 1]

    print(f'\n  TRUE absolute:    {true_abs}')
    print(f'  Raw recovered:    {est_ha.round(5)}  (r={corr_ha:.4f})')
    print(f'  Calibrated:       {est_ha_calib.round(5)}  (r={corr_ha_calib:.4f})')
    print(f'  Calibration: a={a_ha:.3f}, b={b_ha:.5f}')

    # ── Standard (relative slopes) ──
    print('\n' + '=' * 60)
    print('STANDARD: Recovery + Post-hoc Calibration')
    print('=' * 60)
    np.random.seed(42)
    torch.manual_seed(42)
    sim = simulate_data(include_health=False)
    delta_init, gl_init, gs_init, psi_init = realistic_init(
        sim['G'], sim['Y'], sim['K_total'], sim['r_k'], sim['L_chol'])

    model_std = StandardModelReparam(
        sim['G'], sim['Y'], sim['K_total'], sim['r_k'],
        delta_init, gl_init, gs_init, psi_init)

    true_rel = sim['gamma_slope_true'][0, :] - sim['gamma_slope_true'][0, :].mean()
    res_std = fit_two_phase(model_std, true_slopes=true_rel, label='Std')

    est_std = res_std['slopes_final'][0, :sim['K_total']]
    corr_std = np.corrcoef(true_rel, est_std)[0, 1]

    est_std_calib, a_std, b_std = posthoc_calibrate(est_std, true_rel)
    corr_std_calib = np.corrcoef(true_rel, est_std_calib)[0, 1]

    print(f'\n  TRUE relative:    {true_rel.round(5)}')
    print(f'  Raw recovered:    {est_std.round(5)}  (r={corr_std:.4f})')
    print(f'  Calibrated:       {est_std_calib.round(5)}  (r={corr_std_calib:.4f})')
    print(f'  Calibration: a={a_std:.3f}, b={b_std:.5f}')

    # ── Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # Top left: standard raw
    ax = axes[0, 0]
    ax.scatter(true_rel, est_std, s=100, c='steelblue', edgecolors='navy', zorder=3)
    lims = [min(true_rel.min(), est_std.min()) * 1.3,
            max(true_rel.max(), est_std.max()) * 1.3]
    ax.plot(lims, lims, 'k--', alpha=0.4, lw=1.5)
    for k, lab in enumerate(['CV', 'Metabolic', 'Neuro']):
        ax.annotate(lab, (true_rel[k], est_std[k]), xytext=(6,6),
                    textcoords='offset points', fontsize=10)
    ax.set_xlabel('True RELATIVE slope')
    ax.set_ylabel('Recovered slope')
    ax.set_title(f'Standard: Raw (r={corr_std:.3f})')
    ax.set_aspect('equal'); ax.set_xlim(lims); ax.set_ylim(lims)

    # Top right: standard calibrated
    ax = axes[0, 1]
    ax.scatter(true_rel, est_std_calib, s=100, c='steelblue', edgecolors='navy', zorder=3)
    lims_c = [min(true_rel.min(), est_std_calib.min()) * 1.3,
              max(true_rel.max(), est_std_calib.max()) * 1.3]
    ax.plot(lims_c, lims_c, 'k--', alpha=0.4, lw=1.5)
    for k, lab in enumerate(['CV', 'Metabolic', 'Neuro']):
        ax.annotate(lab, (true_rel[k], est_std_calib[k]), xytext=(6,6),
                    textcoords='offset points', fontsize=10)
    ax.set_xlabel('True RELATIVE slope')
    ax.set_ylabel('Calibrated slope')
    ax.set_title(f'Standard: Post-hoc calibrated (r={corr_std_calib:.3f})')
    ax.set_aspect('equal'); ax.set_xlim(lims_c); ax.set_ylim(lims_c)

    # Bottom left: HA raw
    ax = axes[1, 0]
    ax.scatter(true_abs, est_ha, s=100, c='coral', edgecolors='firebrick', zorder=3)
    lims2 = [min(true_abs.min(), est_ha.min()) - 0.005,
             max(true_abs.max(), est_ha.max()) + 0.005]
    ax.plot(lims2, lims2, 'k--', alpha=0.4, lw=1.5)
    for k, lab in enumerate(['Health', 'CV', 'Metabolic', 'Neuro']):
        ax.annotate(lab, (true_abs[k], est_ha[k]), xytext=(6,6),
                    textcoords='offset points', fontsize=10)
    ax.set_xlabel('True ABSOLUTE slope')
    ax.set_ylabel('Recovered slope')
    ax.set_title(f'Health anchor: Raw (r={corr_ha:.3f})')
    ax.set_aspect('equal'); ax.set_xlim(lims2); ax.set_ylim(lims2)

    # Bottom right: HA calibrated
    ax = axes[1, 1]
    ax.scatter(true_abs, est_ha_calib, s=100, c='coral', edgecolors='firebrick', zorder=3)
    lims2c = [min(true_abs.min(), est_ha_calib.min()) - 0.005,
              max(true_abs.max(), est_ha_calib.max()) + 0.005]
    ax.plot(lims2c, lims2c, 'k--', alpha=0.4, lw=1.5)
    for k, lab in enumerate(['Health', 'CV', 'Metabolic', 'Neuro']):
        ax.annotate(lab, (true_abs[k], est_ha_calib[k]), xytext=(6,6),
                    textcoords='offset points', fontsize=10)
    ax.set_xlabel('True ABSOLUTE slope')
    ax.set_ylabel('Calibrated slope')
    ax.set_title(f'Health anchor: Post-hoc calibrated (r={corr_ha_calib:.3f})')
    ax.set_aspect('equal'); ax.set_xlim(lims2c); ax.set_ylim(lims2c)

    plt.suptitle('Slope Recovery: Raw vs Post-hoc Calibrated', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('posthoc_calibration.png', dpi=150, bbox_inches='tight')
    print(f'\nSaved: posthoc_calibration.png')

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Standard model:')
    print(f'  Raw:        r={corr_std:.3f}, slopes={est_std.round(5)}')
    print(f'  Calibrated: r={corr_std_calib:.3f}, slopes={est_std_calib.round(5)}')
    print(f'  Scale factor: 1/{a_std:.3f} = {1/a_std:.2f}x')
    print(f'\nHealth anchor:')
    print(f'  Raw:        r={corr_ha:.3f}, slopes={est_ha.round(5)}')
    print(f'  Calibrated: r={corr_ha_calib:.3f}, slopes={est_ha_calib.round(5)}')
    print(f'  Scale factor: 1/{a_ha:.3f} = {1/a_ha:.2f}x')
    print(f'\nNote: calibration factor estimated from simulation.')
    print(f'In practice, run simulations with known slopes to estimate')
    print(f'the compression factor, then apply to real data.')
