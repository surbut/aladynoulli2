"""
Fixed phi, gamma, kappa with REPARAMETERIZED lambda = mean(gamma) + delta.

For prediction with pooled reparam params: gamma is IN the forward pass.
Optimize delta; lambda = mean(gamma) + delta at every step.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Import base for shared structure we override key methods
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa import (
    AladynSurvivalFixedPhiFixedGammaFixedKappa as _BaseFixedGk,
)


class AladynSurvivalFixedPhiFixedGammaFixedKappaReparam(_BaseFixedGk):
    """
    Reparam version: lambda = mean(gamma) + delta.
    Gamma is IN the forward pass. Optimize delta only.
    """
    def initialize_params(self, **kwargs):
        """Initialize delta (residual); lambda = mean(gamma) + delta."""
        mean_lambda = self._get_mean_lambda()
        # delta = 0 initially so lambda starts at mean(gamma)
        delta_init = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        self.delta = nn.Parameter(delta_init)
        if self.healthy_ref is not None:
            print(f"Initializing delta (reparam) with {self.K} disease + 1 healthy")
        else:
            print(f"Initializing delta (reparam) with {self.K} disease states")
        print("  lambda = mean(gamma) + delta (gamma in forward)")

    def _get_mean_lambda(self):
        """Mean of lambda GP: r_k + G @ gamma (gamma in forward)."""
        mean_lambda = torch.zeros((self.N, self.K_total, self.T), dtype=torch.float32)
        for k in range(self.K):
            mean_lambda[:, k, :] = self.signature_refs[k].unsqueeze(0).unsqueeze(1) + \
                self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
        if self.healthy_ref is not None:
            mean_lambda[:, self.K, :] = self.healthy_ref.unsqueeze(0).unsqueeze(1)
        return mean_lambda

    def get_lambda(self):
        """lambda = mean(gamma) + delta -- gamma in forward."""
        return self._get_mean_lambda() + self.delta

    @property
    def lambda_(self):
        """For compatibility with code expecting self.lambda_."""
        return self.get_lambda()

    def forward(self):
        # Gamma used here via get_lambda
        lam = self.get_lambda()
        # Clamp lam to prevent softmax overflow (repram gamma can be large)
        lam = torch.clamp(lam, -50.0, 50.0)
        theta = torch.softmax(lam, dim=1)
        epsilon = 1e-6
        phi_prob = torch.sigmoid(self.phi)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        return pi, theta, phi_prob

    def compute_gp_prior_loss(self):
        """GP prior on delta only (delta ~ GP(0, Omega))."""
        L_lambda = torch.linalg.cholesky(self.K_lambda)
        deviations_flat = self.delta.reshape(-1, self.T)
        deviations_flat_T = deviations_flat.T
        v_flat_T = torch.cholesky_solve(deviations_flat_T, L_lambda)
        v_flat = v_flat_T.T
        gp_loss_delta = 0.5 * torch.sum(deviations_flat * v_flat)
        return gp_loss_delta / self.N + self.phi_gp_loss

    def compute_loss(self, event_times):
        """Same as base but uses get_lambda() for LRT (lambda_at_diagnosis)."""
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-6
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        N, D, T = self.Y.shape

        if not isinstance(event_times, torch.Tensor):
            event_times_tensor = torch.tensor(event_times, dtype=torch.long)
        else:
            event_times_tensor = event_times.long()

        if len(event_times_tensor.shape) == 2:
            event_times_expanded = event_times_tensor.unsqueeze(-1)
        elif len(event_times_tensor.shape) == 1:
            event_times_expanded = event_times_tensor.unsqueeze(-1).expand(N, D).unsqueeze(-1)
        else:
            raise ValueError(f"event_times must be 1D [N] or 2D [N, D]")

        time_grid = torch.arange(T, dtype=torch.long, device=event_times_expanded.device)
        time_grid = time_grid.unsqueeze(0).unsqueeze(0).expand(N, D, T)
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

        signature_update_loss = 0.0
        if self.lrtpen > 0:
            lam = self.get_lambda()
            diagnoses = self.Y
            phi_avg = phi_prob.mean(dim=2)
            for d in range(self.D):
                if torch.any(diagnoses[:, d, :]):
                    spec_d = phi_avg[:, d]
                    max_sig = torch.argmax(spec_d)
                    other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (self.K_total - 1)
                    lr = spec_d[max_sig] / (other_mean + epsilon)
                    if lr > 2:
                        diagnosis_mask = diagnoses[:, d, :].bool()
                        patient_idx, time_idx = torch.where(diagnosis_mask)
                        lambda_at_diagnosis = lam[patient_idx, max_sig, time_idx]
                        target_value = 2.0
                        disease_prevalence = diagnoses[:, d, :].float().mean() + epsilon
                        prevalence_scaling = min(0.1 / disease_prevalence, 10.0)
                        signature_update_loss += torch.sum(
                            torch.log(lr) * prevalence_scaling * (target_value - lambda_at_diagnosis)
                        )

        total_loss = total_data_loss + self.gpweight * gp_loss + self.lrtpen * signature_update_loss / (self.N * self.T)
        return total_loss

    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01, grad_clip=5.0,
            patience=50, min_improvement=1e-4):
        """Fit delta only (gamma fixed).

        Includes cosine annealing LR schedule and early stopping.
        """
        optimizer = optim.Adam([{'params': [self.delta], 'lr': learning_rate}])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
        losses = []
        best_loss = float('inf')
        best_epoch = 0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            if torch.isnan(loss):
                # Diagnose: check lambda, pi for NaN
                with torch.no_grad():
                    lam = self.get_lambda()
                    pi, theta, _ = self.forward()
                    nan_lam = torch.isnan(lam).any().item()
                    nan_pi = torch.isnan(pi).any().item()
                    print(f"Epoch {epoch} (REPARAM): Loss=nan (stopping) [nan in lambda={nan_lam}, pi={nan_pi}, kappa={self.kappa.item():.4f}]")
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.delta], max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                self.delta.data.clamp_(-20.0, 20.0)
            cur_loss = loss.item()
            losses.append(cur_loss)
            if cur_loss < best_loss * (1 - min_improvement):
                best_loss = cur_loss
                best_epoch = epoch
            if epoch % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} (REPARAM): Loss={cur_loss:.4f}, LR={lr_now:.1e}")
            if epoch - best_epoch > patience and epoch > 50:
                print(f"Epoch {epoch} (REPARAM): Early stop (no improvement for {patience} epochs, best={best_loss:.4f} at epoch {best_epoch})")
                break
        return losses

    def predict(self, new_G=None, new_indices=None):
        """Predict: for new_G, use mean(gamma) only (no delta - no data to fit)."""
        with torch.no_grad():
            if new_G is not None:
                new_G_tensor = torch.tensor(new_G, dtype=torch.float32)
                if self.G.shape[0] > 1 and not torch.allclose(self.G[0:1], self.G):
                    new_G_scaled = (new_G_tensor - self.G.mean(dim=0)) / self.G.std(dim=0)
                else:
                    new_G_scaled = new_G_tensor
                N_new = new_G_scaled.shape[0]
                new_mean = torch.zeros((N_new, self.K_total, self.T))
                for k in range(self.K):
                    means = self.genetic_scale * (new_G_scaled @ self.gamma[:, k])
                    new_mean[:, k, :] = self.signature_refs[k] + means.unsqueeze(1)
                if self.healthy_ref is not None:
                    new_mean[:, self.K, :] = self.healthy_ref
                new_theta = torch.softmax(new_mean, dim=1)
                phi_prob = torch.sigmoid(self.phi)
                new_pi = torch.einsum('nkt,kdt->ndt', new_theta, phi_prob) * self.kappa
                return new_pi, new_theta
            elif new_indices is not None:
                lam = self.get_lambda()
                subset_theta = torch.softmax(lam[new_indices], dim=1)
                phi_prob = torch.sigmoid(self.phi)
                subset_pi = torch.einsum('nkt,kdt->ndt', subset_theta, phi_prob) * self.kappa
                return subset_pi, subset_theta
            else:
                raise ValueError("Either new_G or new_indices must be provided")
