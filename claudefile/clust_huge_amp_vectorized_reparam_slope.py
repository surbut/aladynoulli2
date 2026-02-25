"""
Reparameterized model WITH genetic slopes: gamma_level + gamma_slope * t.

Extends the nokappa v3 reparameterization:
  lambda = mean_lambda(gamma_level, gamma_slope) + delta

where:
  mean_lambda[:, k, t] = sig_ref[k] + scale * (G @ gamma_level[:, k])
                        + t * scale * (G @ gamma_slope[:, k])

  For health signature (k=K):
    mean_lambda[:, K, t] = healthy_ref + alpha_i
    where alpha_i = G @ gamma_health is a FIXED person-specific baseline
    (estimated from pretrained delta, not optimized).
    This breaks softmax scale invariance -> absolute slopes identifiable.

Three ingredients for slope recovery (from genetic_slope_recovery.ipynb):
  1. Reparameterization: gamma in the forward pass (NLL gradient)
  2. Two-phase training: Phase 1 freezes delta so gamma_slope must learn
  3. GP kernel on delta: penalizes temporal trends in residuals

Person-specific health anchor (from genetic_slope_health_signature_math.tex):
  Without alpha_i, softmax is invariant to shifting all lambdas by +c,
  so only RELATIVE slopes are identifiable. With alpha_i fixed, the
  health-vs-disease balance is pinned -> ABSOLUTE slopes identifiable.

Can warm-start from existing nokappa v3 checkpoints (gamma -> gamma_level).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import SpectralClustering


class AladynSurvivalReparamWithSlope(nn.Module):
    """
    lambda = sig_ref + scale * (G @ gamma_level) + t * scale * (G @ gamma_slope) + delta
    phi = logit_prev + psi + epsilon

    Health signature: lambda_health = healthy_ref + alpha_i + delta_health
      where alpha_i = G @ gamma_health (fixed buffer, not optimized).

    gamma_level, gamma_slope, psi flow through NLL.
    delta, epsilon have GP priors.
    kappa is fixed at 1 (nokappa).
    alpha_i breaks softmax scale invariance -> absolute slopes identifiable.
    """

    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 signature_references=None, healthy_reference=None, disease_names=None,
                 pretrained_gamma=None, pretrained_psi=None, pretrained_delta=None,
                 pretrained_epsilon=None, gamma_health=None):
        super().__init__()
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.gpweight = W
        self.jitter = 1e-6
        self.kappa = torch.ones(1)  # fixed at 1
        self.lrtpen = R
        self.lambda_length_scale = T / 4
        self.phi_length_scale = T / 3
        self.init_amplitude = 1.0
        self.lambda_amplitude_init = init_sd_scaler
        self.phi_amplitude_init = init_sd_scaler

        time_points = torch.arange(T, dtype=torch.float32)
        time_diff = time_points[:, None] - time_points[None, :]
        self.base_K_lambda = torch.exp(-0.5 * (time_diff**2) / (self.lambda_length_scale**2))
        self.base_K_phi = torch.exp(-0.5 * (time_diff**2) / (self.phi_length_scale**2))
        self.K_lambda_init = (self.lambda_amplitude_init**2) * self.base_K_lambda + self.jitter * torch.eye(T)
        self.K_phi_init = (self.phi_amplitude_init**2) * self.base_K_phi + self.jitter * torch.eye(T)
        self.phi_amplitude = 1
        self.lambda_amplitude = 1
        self.K_phi = (self.phi_amplitude ** 2) * self.base_K_phi + self.jitter * torch.eye(self.T)
        self.K_lambda = (self.lambda_amplitude ** 2) * self.base_K_lambda + self.jitter * torch.eye(self.T)

        # Normalized time grid [0, 1] for slope term
        self.register_buffer('time_grid', torch.arange(T, dtype=torch.float32) / (T - 1))

        self.disease_names = disease_names

        if signature_references is None:
            raise ValueError("signature_references must be provided")
        self.signature_refs = signature_references.clone().detach().to(torch.float32) if torch.is_tensor(signature_references) else torch.tensor(signature_references, dtype=torch.float32)
        self.genetic_scale = genetic_scale
        if healthy_reference is not None:
            self.healthy_ref = torch.tensor(-5.0, dtype=torch.float32)
        else:
            self.healthy_ref = None

        G_arr = np.array(G)
        G_centered = G_arr - G_arr.mean(axis=0, keepdims=True)
        G_scaled = G_centered / G_centered.std(axis=0, keepdims=True)
        self.G = torch.tensor(G_scaled, dtype=torch.float32)
        self.Y = torch.tensor(np.array(Y), dtype=torch.float32)
        self.prevalence_t = torch.tensor(np.array(prevalence_t), dtype=torch.float32)
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )

        self._init_params(pretrained_gamma, pretrained_psi,
                          pretrained_delta, pretrained_epsilon,
                          gamma_health=gamma_health)

    def _init_params(self, pretrained_gamma, pretrained_psi,
                     pretrained_delta, pretrained_epsilon,
                     gamma_health=None):
        """Initialize parameters, optionally warm-starting from nokappa v3 checkpoint."""

        if pretrained_gamma is not None and pretrained_psi is not None:
            # Warm start from existing nokappa v3
            gamma_level_init = pretrained_gamma.clone().detach()
            psi_init = pretrained_psi.clone().detach()

            if pretrained_delta is not None:
                delta_init = pretrained_delta.clone().detach()
            else:
                delta_init = torch.zeros((self.N, self.K_total, self.T))

            if pretrained_epsilon is not None:
                epsilon_init = pretrained_epsilon.clone().detach()
            else:
                epsilon_init = torch.zeros((self.K_total, self.D, self.T))

            print(f"Warm-starting from pretrained gamma ({gamma_level_init.shape}), psi ({psi_init.shape})")
        else:
            # Cold start: spectral clustering + lstsq init (same as original)
            gamma_level_init, psi_init, delta_init, epsilon_init = self._cold_init()

        # --- Person-specific health baseline (alpha_i) ---
        # Breaks softmax scale invariance -> absolute slopes identifiable.
        # alpha_i is a FIXED buffer (not optimized), computed from genetics.
        if self.healthy_ref is not None:
            if gamma_health is not None:
                # Use provided gamma_health (e.g., from training model for holdout)
                gh = gamma_health.clone().detach() if isinstance(gamma_health, torch.Tensor) else torch.tensor(gamma_health, dtype=torch.float32)
                self.register_buffer('gamma_health', gh)
                print(f"Using provided gamma_health for alpha_i")
            elif pretrained_delta is not None:
                # Estimate gamma_health by regressing health-signature delta means on G
                health_delta_means = pretrained_delta[:, self.K, :].mean(dim=1)  # (N,)
                gh = torch.linalg.lstsq(self.G, health_delta_means.unsqueeze(1)).solution.squeeze()  # (P,)
                self.register_buffer('gamma_health', gh)
                print(f"Estimated gamma_health from pretrained delta (health sig mean -> G regression)")
            else:
                # No info available — alpha_i will be zero
                self.register_buffer('gamma_health', torch.zeros(self.P))
                print(f"No pretrained delta or gamma_health; alpha_i = 0")

            alpha_i = self.G @ self.gamma_health  # (N,)
            self.register_buffer('alpha_i', alpha_i)

            # Adjust health signature's delta: remove alpha_i so delta stays mean-zero
            delta_init[:, self.K, :] -= alpha_i.unsqueeze(1)

            print(f"  alpha_i: mean={alpha_i.mean():.4f}, std={alpha_i.std():.4f}, "
                  f"range=[{alpha_i.min():.4f}, {alpha_i.max():.4f}]")
        else:
            self.alpha_i = None
            self.gamma_health = None

        # gamma_slope always starts at zero
        gamma_slope_init = torch.zeros((self.P, self.K_total))

        self.gamma_level = nn.Parameter(gamma_level_init)
        self.gamma_slope = nn.Parameter(gamma_slope_init)
        self.psi = nn.Parameter(psi_init)
        self.delta = nn.Parameter(delta_init)
        self.epsilon = nn.Parameter(epsilon_init)

        print(f"Slope model init: gamma_level {self.gamma_level.shape}, "
              f"gamma_slope {self.gamma_slope.shape} (zeros), "
              f"delta {self.delta.shape}, epsilon {self.epsilon.shape}")

    def _cold_init(self):
        """Cold initialization (no pretrained params): spectral clustering + lstsq."""
        Y_avg = torch.mean(self.Y, dim=2)
        eps_num = 1e-6
        Y_avg = torch.clamp(Y_avg, eps_num, 1.0 - eps_num)
        Y_avg = torch.log(Y_avg / (1 - Y_avg))

        Y_corr = torch.corrcoef(Y_avg.T)
        Y_corr = torch.nan_to_num(Y_corr, nan=0.0)
        similarity = (Y_corr + 1) / 2
        spectral = SpectralClustering(
            n_clusters=self.K, assign_labels='kmeans',
            affinity='precomputed', n_init=10, random_state=42
        ).fit(similarity.numpy())
        self.clusters = spectral.labels_

        psi_init = torch.zeros((self.K_total, self.D))
        for k in range(self.K):
            cluster_mask = (self.clusters == k)
            psi_init[k, cluster_mask] = 1.0 + 0.1 * torch.randn(cluster_mask.sum())
            psi_init[k, ~cluster_mask] = -2.0 + 0.01 * torch.randn((~cluster_mask).sum())
        if self.healthy_ref is not None:
            psi_init[self.K, :] = -5.0 + 0.01 * torch.randn(self.D)

        gamma_init = torch.zeros((self.P, self.K_total))
        lambda_init = torch.zeros((self.N, self.K_total, self.T))
        phi_init = torch.zeros((self.K_total, self.D, self.T))

        for k in range(self.K):
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[k, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps

        for k in range(self.K):
            cluster_diseases = (self.clusters == k)
            base_value = Y_avg[:, cluster_diseases].mean(dim=1)
            base_value_centered = base_value - base_value.mean()
            gamma_init[:, k] = torch.linalg.lstsq(self.G, base_value_centered.unsqueeze(1)).solution.squeeze()
            lambda_means = self.genetic_scale * (self.G @ gamma_init[:, k])
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = self.signature_refs[k] + lambda_means[i] + eps

        if self.healthy_ref is not None:
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                mean_phi = self.logit_prev_t[d, :] + psi_init[self.K, d]
                eps = L_phi @ torch.randn(self.T)
                phi_init[self.K, d, :] = mean_phi + eps
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, self.K, :] = self.healthy_ref + eps
            gamma_init[:, self.K] = 0.0

        # delta = lambda - mean_lambda (with slope=0, same as original)
        mean_lambda_init = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        for k in range(self.K):
            mean_lambda_init[:, k, :] = self.signature_refs[k].unsqueeze(0).unsqueeze(1) + \
                self.genetic_scale * (self.G @ gamma_init[:, k]).unsqueeze(1)
        if self.healthy_ref is not None:
            mean_lambda_init[:, self.K, :] = self.healthy_ref.unsqueeze(0).unsqueeze(1)

        mean_phi_init = self.logit_prev_t.unsqueeze(0) + psi_init.unsqueeze(2)
        delta_init = lambda_init - mean_lambda_init
        epsilon_init = phi_init - mean_phi_init

        return gamma_init, psi_init, delta_init, epsilon_init

    def get_mean_lambda(self):
        """Mean of lambda GP: sig_ref + scale * (G @ gamma_level) + t * scale * (G @ gamma_slope)
        Health signature: healthy_ref + alpha_i (person-specific, fixed)."""
        mean_lambda = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        t = self.time_grid  # (T,)
        for k in range(self.K):
            level = self.genetic_scale * (self.G @ self.gamma_level[:, k])  # (N,)
            slope = self.genetic_scale * (self.G @ self.gamma_slope[:, k])  # (N,)
            mean_lambda[:, k, :] = (self.signature_refs[k]
                                    + level.unsqueeze(1)
                                    + slope.unsqueeze(1) * t.unsqueeze(0))
        if self.healthy_ref is not None:
            if self.alpha_i is not None:
                # Person-specific health baseline: breaks softmax scale invariance
                mean_lambda[:, self.K, :] = self.healthy_ref + self.alpha_i.unsqueeze(1)
            else:
                mean_lambda[:, self.K, :] = self.healthy_ref.unsqueeze(0).unsqueeze(1)
        return mean_lambda

    def get_mean_phi(self):
        return self.logit_prev_t.unsqueeze(0) + self.psi.unsqueeze(2)

    def get_lambda(self):
        return self.get_mean_lambda() + self.delta

    def get_phi(self):
        return self.get_mean_phi() + self.epsilon

    @property
    def lambda_(self):
        return self.get_lambda()

    @property
    def phi(self):
        return self.get_phi()

    # For compatibility: gamma property returns gamma_level
    @property
    def gamma(self):
        return self.gamma_level

    def forward(self):
        lam = self.get_lambda()
        theta = torch.softmax(lam, dim=1)
        phi_val = self.get_phi()
        phi_prob = torch.sigmoid(phi_val)
        eps_clamp = 1e-6
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, eps_clamp, 1 - eps_clamp)
        return pi, theta, phi_prob

    def compute_loss(self, event_times):
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        N, D, T = self.Y.shape
        event_times_tensor = event_times.clone().detach().long() if torch.is_tensor(event_times) else torch.tensor(event_times, dtype=torch.long)
        event_times_expanded = event_times_tensor.unsqueeze(-1)
        time_grid = torch.arange(T, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        mask_before_event = (time_grid < event_times_expanded).float()
        mask_at_event = (time_grid == event_times_expanded).float()

        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))
        total_data_loss = (loss_censored + loss_event + loss_no_event) / self.N

        if self.gpweight > 0:
            gp_loss = self.compute_gp_prior_loss()
        else:
            gp_loss = 0.0

        total_loss = total_data_loss + self.gpweight * gp_loss
        return total_loss

    def compute_nll_only(self, event_times):
        """NLL without GP prior — for holdout evaluation."""
        pi, _, _ = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        N, D, T = self.Y.shape
        event_times_tensor = event_times.clone().detach().long() if torch.is_tensor(event_times) else torch.tensor(event_times, dtype=torch.long)
        event_times_expanded = event_times_tensor.unsqueeze(-1)
        time_grid = torch.arange(T, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        mask_before_event = (time_grid < event_times_expanded).float()
        mask_at_event = (time_grid == event_times_expanded).float()
        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))
        return (loss_censored + loss_event + loss_no_event) / self.N

    def compute_gp_prior_loss(self):
        """GP prior on delta and epsilon only."""
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        L_phi = torch.linalg.cholesky(self.K_phi)
        deviations_flat = self.delta.reshape(-1, self.T)
        v_flat = torch.cholesky_solve(deviations_flat.T, L_lambda).T
        gp_loss_lambda = 0.5 * torch.sum(deviations_flat * v_flat)
        deviations_phi_flat = self.epsilon.reshape(-1, self.T)
        v_phi_flat = torch.cholesky_solve(deviations_phi_flat.T, L_phi).T
        gp_loss_phi = 0.5 * torch.sum(deviations_phi_flat * v_phi_flat)
        return gp_loss_lambda / self.N + gp_loss_phi / self.D

    def fit_two_phase(self, event_times, num_epochs_phase1=100, num_epochs_phase2=200,
                      learning_rate=0.1, lr_slope=None, verbose_every=10):
        """
        Two-phase training for genetic slope recovery.

        Phase 1: delta FROZEN. gamma_slope must learn from NLL gradient.
                 gamma_level, psi, epsilon also train.
        Phase 2: delta UNFROZEN. All parameters fine-tune.

        Returns dict with losses and gamma_slope history.
        """
        if lr_slope is None:
            lr_slope = learning_rate

        history = {
            'phase1_losses': [], 'phase2_losses': [],
            'gamma_slope_norm': [],
            'phase1_nll': [], 'phase2_nll': [],
        }

        # ── Phase 1: freeze delta ──
        print(f'{"="*60}')
        print(f'PHASE 1: delta frozen, learning gamma_slope ({num_epochs_phase1} epochs)')
        print(f'{"="*60}')
        self.delta.requires_grad_(False)

        param_groups_p1 = [
            {'params': [self.gamma_level], 'lr': learning_rate},
            {'params': [self.gamma_slope], 'lr': lr_slope},
            {'params': [self.psi], 'lr': learning_rate * 0.1},
            {'params': [self.epsilon], 'lr': learning_rate * 0.1},
        ]
        optimizer_p1 = optim.Adam(param_groups_p1)

        for epoch in range(num_epochs_phase1):
            optimizer_p1.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            optimizer_p1.step()
            history['phase1_losses'].append(loss.item())

            with torch.no_grad():
                nll = self.compute_nll_only(event_times).item()
                history['phase1_nll'].append(nll)
                slope_norm = self.gamma_slope.data.norm().item()
                history['gamma_slope_norm'].append(slope_norm)

            if epoch % verbose_every == 0 or epoch == num_epochs_phase1 - 1:
                print(f'  P1 Epoch {epoch:4d}: loss={loss.item():.2f}, '
                      f'NLL={nll:.2f}, |gamma_slope|={slope_norm:.4f}')

        # ── Phase 2: unfreeze delta ──
        print(f'\n{"="*60}')
        print(f'PHASE 2: delta unfrozen, all params train ({num_epochs_phase2} epochs)')
        print(f'{"="*60}')
        self.delta.requires_grad_(True)

        param_groups_p2 = [
            {'params': [self.delta], 'lr': learning_rate},
            {'params': [self.gamma_level], 'lr': learning_rate},
            {'params': [self.gamma_slope], 'lr': lr_slope * 0.1},  # lower LR for slopes in P2
            {'params': [self.psi], 'lr': learning_rate * 0.1},
            {'params': [self.epsilon], 'lr': learning_rate * 0.1},
        ]
        optimizer_p2 = optim.Adam(param_groups_p2)

        for epoch in range(num_epochs_phase2):
            optimizer_p2.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            optimizer_p2.step()
            history['phase2_losses'].append(loss.item())

            with torch.no_grad():
                nll = self.compute_nll_only(event_times).item()
                history['phase2_nll'].append(nll)
                slope_norm = self.gamma_slope.data.norm().item()
                history['gamma_slope_norm'].append(slope_norm)

            if epoch % verbose_every == 0 or epoch == num_epochs_phase2 - 1:
                print(f'  P2 Epoch {epoch:4d}: loss={loss.item():.2f}, '
                      f'NLL={nll:.2f}, |gamma_slope|={slope_norm:.4f}')

        return history

    def fit_single_phase(self, event_times, num_epochs=300,
                         learning_rate=0.1, lr_slope=None, verbose_every=10):
        """
        Single-phase training: ALL parameters train together from the start.
        Ablation control for fit_two_phase — tests whether freezing delta
        is necessary for gamma_slope recovery.

        Same total epochs, same LR structure as phase 2 of two-phase.
        """
        if lr_slope is None:
            lr_slope = learning_rate

        history = {
            'losses': [],
            'nll': [],
            'gamma_slope_norm': [],
        }

        print(f'{"="*60}')
        print(f'SINGLE PHASE: all params train together ({num_epochs} epochs)')
        print(f'{"="*60}')

        param_groups = [
            {'params': [self.delta], 'lr': learning_rate},
            {'params': [self.gamma_level], 'lr': learning_rate},
            {'params': [self.gamma_slope], 'lr': lr_slope},
            {'params': [self.psi], 'lr': learning_rate * 0.1},
            {'params': [self.epsilon], 'lr': learning_rate * 0.1},
        ]
        optimizer = optim.Adam(param_groups)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
            optimizer.step()
            history['losses'].append(loss.item())

            with torch.no_grad():
                nll = self.compute_nll_only(event_times).item()
                history['nll'].append(nll)
                slope_norm = self.gamma_slope.data.norm().item()
                history['gamma_slope_norm'].append(slope_norm)

            if epoch % verbose_every == 0 or epoch == num_epochs - 1:
                print(f'  Epoch {epoch:4d}: loss={loss.item():.2f}, '
                      f'NLL={nll:.2f}, |gamma_slope|={slope_norm:.4f}')

        return history


def subset_data(Y, E, G, start_index, end_index):
    indices = list(range(start_index, end_index))
    Y_subset = Y[indices]
    E_subset = E[indices]
    G_subset = G[indices]
    return Y_subset, E_subset, G_subset, indices
