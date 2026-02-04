"""
Test ALADYN Slope Identifiability

This script tests whether the FULL ALADYN model can recover genetic slope effects.

Pipeline:
1. Simulate TRUE data with known gamma_level and gamma_slope
2. Generate Y (disease outcomes) from the full generative model:
   - λ ~ GP(r_k + G @ gamma_level + t * G @ gamma_slope, Ω_λ)
   - θ = softmax(λ)  
   - φ = sigmoid(ψ + GP)
   - π = θ × φ × κ
   - Y ~ Bernoulli(π)
3. Fit ALADYN WITH slope and WITHOUT slope
4. Compare parameter recovery
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.special import expit, softmax
import sys
sys.path.insert(0, '/Users/sarahurbut/aladynoulli2')

np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("ALADYN SLOPE IDENTIFIABILITY TEST")
print("="*70)

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
N = 300   # Individuals (smaller for speed)
T = 51    # Timepoints (ages 30-80)
K = 3     # Signatures
D = 20    # Diseases
P = 5     # Genetic features

ages = np.arange(30, 81)
t_centered = ages - 30  # 0 to 50

print(f"\nSetup: N={N}, T={T}, K={K}, D={D}, P={P}")

# ============================================================
# GENERATE GENETIC DATA
# ============================================================
G = np.random.randn(N, P)
G = (G - G.mean(axis=0)) / G.std(axis=0)

prs = G[:, 0]
high_risk_idx = prs > 0.5
low_risk_idx = prs < -0.5

print(f"High-risk (n={high_risk_idx.sum()}), Low-risk (n={low_risk_idx.sum()})")

# ============================================================
# TRUE PARAMETERS
# ============================================================
# Signature references (baseline intercepts)
r_k = np.array([0.0, -0.5, -1.0])

# Genetic effects - only first genetic feature has effects
gamma_level_true = np.zeros((P, K))
gamma_level_true[0, :] = [0.2, 0.1, 0.05]  # Small level effect

gamma_slope_true = np.zeros((P, K))
gamma_slope_true[0, :] = [0.02, 0.01, 0.005]  # Slope effect (per year)

# Disease-signature associations (psi)
# Each signature "owns" some diseases
diseases_per_sig = D // K
psi_true = np.zeros((K, D))
for k in range(K):
    start = k * diseases_per_sig
    end = (k + 1) * diseases_per_sig if k < K - 1 else D
    psi_true[k, start:end] = 1.5  # High association
    # Small association with other diseases
    for j in range(D):
        if j < start or j >= end:
            psi_true[k, j] = -1.0

# Disease prevalence (baseline)
base_prevalence = 0.02 * np.ones(D)

print(f"\nTRUE PARAMETERS:")
print(f"  gamma_level[0,:] = {gamma_level_true[0, :]}")
print(f"  gamma_slope[0,:] = {gamma_slope_true[0, :]} per year")
print(f"  Over 50 years, slope adds: {gamma_slope_true[0, :] * 50}")

# ============================================================
# GENERATE TRUE LAMBDA (with GP noise)
# ============================================================
print("\nGenerating true lambda trajectories...")

# GP kernel for lambda
lambda_length_scale = T / 4
lambda_amplitude = 0.3
K_lambda = lambda_amplitude**2 * np.exp(-0.5 * (t_centered[:, None] - t_centered[None, :])**2 / lambda_length_scale**2)
K_lambda += 1e-6 * np.eye(T)
L_lambda = cholesky(K_lambda, lower=True)

# Generate lambda for each individual and signature
lambda_true = np.zeros((N, K, T))
for i in range(N):
    for k in range(K):
        # Mean = r_k + level_effect + slope_effect * t
        level_effect = G[i, :] @ gamma_level_true[:, k]
        slope_effect = G[i, :] @ gamma_slope_true[:, k]
        mean_ik = r_k[k] + level_effect + slope_effect * t_centered
        
        # Add GP noise
        noise = L_lambda @ np.random.randn(T)
        lambda_true[i, k, :] = mean_ik + noise

# ============================================================
# GENERATE THETA (softmax of lambda)
# ============================================================
theta_true = softmax(lambda_true, axis=1)  # (N, K, T)

# ============================================================
# GENERATE PHI (sigmoid of psi, no time variation for simplicity)
# ============================================================
phi_true = expit(psi_true)  # (K, D)
# Expand to (K, D, T) - same at all times
phi_true_3d = np.repeat(phi_true[:, :, np.newaxis], T, axis=2)

# ============================================================
# GENERATE PI and Y
# ============================================================
kappa = 0.1  # Scaling factor

# pi = theta @ phi * kappa
# Using einsum: pi[n,d,t] = sum_k theta[n,k,t] * phi[k,d,t]
pi_true = np.einsum('nkt,kdt->ndt', theta_true, phi_true_3d) * kappa
pi_true = np.clip(pi_true, 1e-6, 1 - 1e-6)

# Generate Y ~ Bernoulli(pi)
Y = (np.random.rand(N, D, T) < pi_true).astype(float)

print(f"Disease prevalence: {Y.mean():.4f}")

# Event times (when first diagnosis occurs, or T if never)
E = np.full((N, D), T, dtype=int)
for i in range(N):
    for d in range(D):
        diagnosed = np.where(Y[i, d, :] > 0)[0]
        if len(diagnosed) > 0:
            E[i, d] = diagnosed[0]

# Prevalence over time
prevalence_t = Y.mean(axis=0)  # (D, T)

print(f"Generated Y shape: {Y.shape}")
print(f"Mean event time: {E.mean():.1f}")

# ============================================================
# VISUALIZE TRUE DATA
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Lambda trajectories by risk group
ax = axes[0, 0]
high_mean = lambda_true[high_risk_idx, 0, :].mean(axis=0)
low_mean = lambda_true[low_risk_idx, 0, :].mean(axis=0)
ax.plot(ages, high_mean, 'r-', linewidth=3, label='High-risk mean')
ax.plot(ages, low_mean, 'b-', linewidth=3, label='Low-risk mean')
for i in np.where(high_risk_idx)[0][:5]:
    ax.plot(ages, lambda_true[i, 0, :], 'r-', alpha=0.2)
for i in np.where(low_risk_idx)[0][:5]:
    ax.plot(ages, lambda_true[i, 0, :], 'b-', alpha=0.2)
ax.set_xlabel('Age')
ax.set_ylabel('λ (Signature 0)')
ax.set_title('TRUE Lambda: Groups Diverge Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Divergence
ax = axes[0, 1]
divergence = high_mean - low_mean
ax.fill_between(ages, 0, divergence, alpha=0.3, color='purple')
ax.plot(ages, divergence, 'purple', linewidth=2)
ax.set_xlabel('Age')
ax.set_ylabel('High-risk - Low-risk')
ax.set_title('Divergence Grows (Slope Effect)')
ax.grid(True, alpha=0.3)

# Plot 3: Disease probability by risk group
ax = axes[1, 0]
high_pi = pi_true[high_risk_idx, 0, :].mean(axis=0)
low_pi = pi_true[low_risk_idx, 0, :].mean(axis=0)
ax.plot(ages, high_pi, 'r-', linewidth=2, label='High-risk')
ax.plot(ages, low_pi, 'b-', linewidth=2, label='Low-risk')
ax.set_xlabel('Age')
ax.set_ylabel('P(Disease 0)')
ax.set_title('Disease Probability Also Diverges')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: True gamma parameters
ax = axes[1, 1]
x = np.arange(K)
width = 0.35
ax.bar(x - width/2, gamma_level_true[0, :], width, label='γ_level', color='blue', alpha=0.7)
ax.bar(x + width/2, gamma_slope_true[0, :] * 50, width, label='γ_slope × 50', color='red', alpha=0.7)
ax.set_xlabel('Signature')
ax.set_ylabel('Effect size')
ax.set_title('TRUE Parameters\n(slope scaled by 50 years)')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/aladyn_slope_true_data.png', dpi=150)
print("\n✓ Saved true data plot")

# ============================================================
# FIT ALADYN MODEL WITH SLOPE
# ============================================================
print("\n" + "="*70)
print("FITTING ALADYN WITH SLOPE")
print("="*70)

from clust_huge_amp_vectorized_with_slope import AladynWithGeneticSlope

# Create model
model_with_slope = AladynWithGeneticSlope(
    N=N, D=D, T=T, K=K, P=P,
    G=G, Y=Y,
    R=0.001,  # Regularization
    W=0.1,    # GP weight
    prevalence_t=prevalence_t,
    init_sd_scaler=0.3,
    genetic_scale=1.0,
    signature_references=r_k,
    healthy_reference=None,
    disease_names=None,
    flat_lambda=False,
    learn_kappa=True,
    learn_slope=True  # KEY: learn the slope
)

# Fit
print("\nTraining (this may take a few minutes)...")
losses_with_slope, _ = model_with_slope.fit(E, num_epochs=100, learning_rate=0.01, lambda_reg=0.001)

# ============================================================
# FIT ALADYN MODEL WITHOUT SLOPE
# ============================================================
print("\n" + "="*70)
print("FITTING ALADYN WITHOUT SLOPE")
print("="*70)

model_no_slope = AladynWithGeneticSlope(
    N=N, D=D, T=T, K=K, P=P,
    G=G, Y=Y,
    R=0.001,
    W=0.1,
    prevalence_t=prevalence_t,
    init_sd_scaler=0.3,
    genetic_scale=1.0,
    signature_references=r_k,
    healthy_reference=None,
    disease_names=None,
    flat_lambda=False,
    learn_kappa=True,
    learn_slope=False  # KEY: do NOT learn slope
)

print("\nTraining...")
losses_no_slope, _ = model_no_slope.fit(E, num_epochs=100, learning_rate=0.01, lambda_reg=0.001)

# ============================================================
# COMPARE RESULTS
# ============================================================
print("\n" + "="*70)
print("PARAMETER RECOVERY")
print("="*70)

est_level_with = model_with_slope.gamma_level.detach().numpy()[0, :K]
est_slope_with = model_with_slope.gamma_slope.detach().numpy()[0, :K]
est_level_no = model_no_slope.gamma_level.detach().numpy()[0, :K]

print(f"\n--- γ_level ---")
print(f"TRUE:        {gamma_level_true[0, :]}")
print(f"With slope:  {est_level_with.round(4)}")
print(f"No slope:    {est_level_no.round(4)}")

print(f"\n--- γ_slope ---")
print(f"TRUE:        {gamma_slope_true[0, :]}")
print(f"With slope:  {est_slope_with.round(4)}")
print(f"No slope:    N/A")

print(f"\n--- Final Loss ---")
print(f"With slope:  {losses_with_slope[-1]:.4f}")
print(f"No slope:    {losses_no_slope[-1]:.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curves
ax = axes[0, 0]
ax.plot(losses_with_slope, 'r-', label='With slope', linewidth=2)
ax.plot(losses_no_slope, 'b-', label='No slope', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Parameter recovery - level
ax = axes[0, 1]
x = np.arange(K)
width = 0.25
ax.bar(x - width, gamma_level_true[0, :], width, label='TRUE', color='green', alpha=0.8)
ax.bar(x, est_level_with, width, label='With slope', color='red', alpha=0.8)
ax.bar(x + width, est_level_no, width, label='No slope', color='blue', alpha=0.8)
ax.set_xlabel('Signature')
ax.set_ylabel('γ_level')
ax.set_title('Level Parameter Recovery')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Parameter recovery - slope
ax = axes[1, 0]
ax.bar(x - width/2, gamma_slope_true[0, :], width, label='TRUE', color='green', alpha=0.8)
ax.bar(x + width/2, est_slope_with, width, label='Estimated', color='red', alpha=0.8)
ax.set_xlabel('Signature')
ax.set_ylabel('γ_slope')
ax.set_title('Slope Parameter Recovery (With Slope Model)')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Lambda trajectories - model predictions
ax = axes[1, 1]
with torch.no_grad():
    lambda_with = model_with_slope.lambda_.numpy()
    lambda_no = model_no_slope.lambda_.numpy()

# High-risk individual
high_i = np.where(high_risk_idx)[0][0]
ax.plot(ages, lambda_true[high_i, 0, :], 'k-', linewidth=3, label='TRUE')
ax.plot(ages, lambda_with[high_i, 0, :], 'r--', linewidth=2, label='With slope')
ax.plot(ages, lambda_no[high_i, 0, :], 'b--', linewidth=2, label='No slope')
ax.set_xlabel('Age')
ax.set_ylabel('λ (Signature 0)')
ax.set_title(f'High-Risk Individual (PRS={prs[high_i]:.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/aladyn_slope_results.png', dpi=150)
print("\n✓ Saved results plot")

print("\n" + "="*70)
print("DONE")
print("="*70)
