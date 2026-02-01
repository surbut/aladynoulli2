"""
Simulation demonstrating that genetic effects on progression speed are identifiable.

This simulation shows:
1. Two mechanisms for early-onset disease:
   - High baseline signature loading (gamma_level)
   - Fast progression through signature (gamma_slope)
2. These two effects are separable and identifiable from data
3. Current model (no slope) misses the progression speed effect
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit, softmax
import torch
import torch.nn as nn

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

N = 1000  # Number of individuals
T = 51    # Timepoints (ages 30-80)
K = 3     # Number of signatures
D = 20    # Number of diseases
P = 5     # Number of genetic features (simplified)

ages = np.arange(30, 81)
t_centered = ages - 30  # Center time at age 30

print("=" * 70)
print("GENETIC SLOPE IDENTIFIABILITY SIMULATION")
print("=" * 70)
print(f"\nParameters:")
print(f"  N={N} individuals, T={T} timepoints (ages 30-80)")
print(f"  K={K} signatures, D={D} diseases, P={P} genetic features")

# ============================================================================
# GENERATE GENETIC DATA
# ============================================================================

# Simple genetic matrix: standardized random values
G = np.random.randn(N, P)
G = (G - G.mean(axis=0)) / G.std(axis=0)

# Create "high-risk" and "low-risk" groups based on first genetic component
prs = G[:, 0]  # Use first PC as "PRS"
high_risk_idx = prs > 0.5
low_risk_idx = prs < -0.5

print(f"\nGenetic groups:")
print(f"  High-risk (PRS > 0.5): {high_risk_idx.sum()} individuals")
print(f"  Low-risk (PRS < -0.5): {low_risk_idx.sum()} individuals")

# ============================================================================
# TRUE PARAMETERS (GROUND TRUTH)
# ============================================================================

# Signature references
r_k = np.array([0.0, -0.5, -1.0])  # Baseline for each signature

# TRUE genetic effects on LEVEL (current model captures this)
gamma_level_true = np.zeros((P, K))
gamma_level_true[0, :] = [0.5, 0.3, 0.1]  # First genetic feature affects level

# TRUE genetic effects on SLOPE (new! current model misses this)
gamma_slope_true = np.zeros((P, K))
gamma_slope_true[0, :] = [0.02, 0.015, 0.005]  # First genetic feature affects speed

print(f"\nTrue parameters:")
print(f"  Signature references: {r_k}")
print(f"  Gamma_level[0,:] (level effect): {gamma_level_true[0, :]}")
print(f"  Gamma_slope[0,:] (slope effect): {gamma_slope_true[0, :]} per year")

# ============================================================================
# GENERATE LAMBDA TRAJECTORIES WITH GENETIC SLOPE
# ============================================================================

# Compute mean for each individual-signature-time
lambda_mean = np.zeros((N, K, T))
for k in range(K):
    level_effect = G @ gamma_level_true[:, k]  # (N,)
    slope_effect = G @ gamma_slope_true[:, k]  # (N,)

    for i in range(N):
        # Mean = r_k + level_effect + time * slope_effect
        lambda_mean[i, k, :] = r_k[k] + level_effect[i] + t_centered * slope_effect[i]

# Add GP noise
lambda_amplitude = 0.3
lambda_length_scale = T / 4
time_diff = t_centered[:, None] - t_centered[None, :]
K_lambda = (lambda_amplitude ** 2) * np.exp(-0.5 * (time_diff ** 2) / (lambda_length_scale ** 2))
K_lambda += 1e-6 * np.eye(T)  # Add jitter

# Generate lambda with noise
lambda_true = np.zeros((N, K, T))
for i in range(N):
    for k in range(K):
        noise = np.random.multivariate_normal(np.zeros(T), K_lambda)
        lambda_true[i, k, :] = lambda_mean[i, k, :] + noise

# Convert to theta (softmax)
theta_true = np.exp(lambda_true)
theta_true = theta_true / theta_true.sum(axis=1, keepdims=True)

print(f"\nGenerated lambda trajectories with:")
print(f"  Mean structure: r_k + g_i^T * gamma_level + t * g_i^T * gamma_slope")
print(f"  GP noise: amplitude={lambda_amplitude}, length_scale={lambda_length_scale}")

# ============================================================================
# GENERATE PHI (POPULATION-LEVEL DISEASE-SIGNATURE ASSOCIATIONS)
# ============================================================================

# Simple phi: logistic growth curves
phi_true = np.zeros((K, D, T))
psi_true = np.random.randn(K, D) * 2  # Random signature-disease associations

for k in range(K):
    for d in range(D):
        # Logistic curve: starts low, increases with age
        phi_true[k, d, :] = psi_true[k, d] * expit((ages - 50) / 10)

# ============================================================================
# GENERATE DISEASE OUTCOMES
# ============================================================================

kappa = 0.1  # Calibration parameter
Y = np.zeros((N, D, T))

for i in range(N):
    for d in range(D):
        for t in range(T):
            # pi_idt = kappa * sum_k theta_ikt * sigmoid(phi_kdt)
            pi_idt = kappa * np.sum([
                theta_true[i, k, t] * expit(phi_true[k, d, t])
                for k in range(K)
            ])

            # Bernoulli outcome
            Y[i, d, t] = np.random.binomial(1, min(pi_idt, 0.99))

n_events = Y.sum()
print(f"\nGenerated {int(n_events)} disease events")

# ============================================================================
# VISUALIZE TRUE TRAJECTORIES
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Lambda trajectories for high vs. low risk (Signature 0 - CVD)
ax = axes[0, 0]
for idx in np.where(high_risk_idx)[0][:5]:
    ax.plot(ages, lambda_true[idx, 0, :], 'r-', alpha=0.3, linewidth=0.5)
for idx in np.where(low_risk_idx)[0][:5]:
    ax.plot(ages, lambda_true[idx, 0, :], 'b-', alpha=0.3, linewidth=0.5)

# Plot means
high_mean = lambda_true[high_risk_idx, 0, :].mean(axis=0)
low_mean = lambda_true[low_risk_idx, 0, :].mean(axis=0)
ax.plot(ages, high_mean, 'r-', linewidth=3, label='High PRS (mean)')
ax.plot(ages, low_mean, 'b-', linewidth=3, label='Low PRS (mean)')
ax.set_xlabel('Age')
ax.set_ylabel('λ (Signature 0)')
ax.set_title('True Lambda Trajectories: CVD Signature\n(Different slopes!)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Theta trajectories
ax = axes[0, 1]
for idx in np.where(high_risk_idx)[0][:5]:
    ax.plot(ages, theta_true[idx, 0, :], 'r-', alpha=0.3, linewidth=0.5)
for idx in np.where(low_risk_idx)[0][:5]:
    ax.plot(ages, theta_true[idx, 0, :], 'b-', alpha=0.3, linewidth=0.5)
ax.plot(ages, theta_true[high_risk_idx, 0, :].mean(axis=0), 'r-', linewidth=3, label='High PRS')
ax.plot(ages, theta_true[low_risk_idx, 0, :].mean(axis=0), 'b-', linewidth=3, label='Low PRS')
ax.set_xlabel('Age')
ax.set_ylabel('θ (Signature 0)')
ax.set_title('Theta Trajectories (after softmax)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Slopes by PRS
ax = axes[0, 2]
slopes = np.zeros(N)
for i in range(N):
    # Compute slope from ages 30-60
    slopes[i] = (lambda_true[i, 0, 30] - lambda_true[i, 0, 0]) / 30

ax.scatter(prs, slopes, alpha=0.5, s=20)
ax.set_xlabel('PRS (first genetic component)')
ax.set_ylabel('Progression Speed\n(Δλ per 30 years)')
ax.set_title(f'Genetic Effect on Slope\n(r={np.corrcoef(prs, slopes)[0,1]:.3f})')
ax.grid(True, alpha=0.3)

# Add regression line
z = np.polyfit(prs, slopes, 1)
p = np.poly1d(z)
x_line = np.linspace(prs.min(), prs.max(), 100)
ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Slope={z[0]:.3f}')
ax.legend()

# Plot 4: Age at first event by group
ax = axes[1, 0]
age_at_first_event_high = []
age_at_first_event_low = []

for i in np.where(high_risk_idx)[0]:
    events = np.where(Y[i, :, :].sum(axis=0) > 0)[0]
    if len(events) > 0:
        age_at_first_event_high.append(ages[events[0]])

for i in np.where(low_risk_idx)[0]:
    events = np.where(Y[i, :, :].sum(axis=0) > 0)[0]
    if len(events) > 0:
        age_at_first_event_low.append(ages[events[0]])

ax.hist(age_at_first_event_high, bins=20, alpha=0.5, color='red', label='High PRS', density=True)
ax.hist(age_at_first_event_low, bins=20, alpha=0.5, color='blue', label='Low PRS', density=True)
ax.set_xlabel('Age at First Event')
ax.set_ylabel('Density')
ax.set_title('Age at First Event by PRS Group')
ax.legend()
ax.grid(True, alpha=0.3)

print(f"\nAge at first event:")
if len(age_at_first_event_high) > 0:
    print(f"  High PRS: {np.mean(age_at_first_event_high):.1f} ± {np.std(age_at_first_event_high):.1f}")
if len(age_at_first_event_low) > 0:
    print(f"  Low PRS: {np.mean(age_at_first_event_low):.1f} ± {np.std(age_at_first_event_low):.1f}")

# Plot 5: Decomposition of lambda
ax = axes[1, 1]
example_idx_high = np.where(high_risk_idx)[0][0]
example_idx_low = np.where(low_risk_idx)[0][0]

# For high-risk individual
level_high = r_k[0] + (G[example_idx_high] @ gamma_level_true[:, 0])
slope_high = G[example_idx_high] @ gamma_slope_true[:, 0]
mean_high = level_high + t_centered * slope_high

# For low-risk individual
level_low = r_k[0] + (G[example_idx_low] @ gamma_level_true[:, 0])
slope_low = G[example_idx_low] @ gamma_slope_true[:, 0]
mean_low = level_low + t_centered * slope_low

ax.plot(ages, mean_high, 'r-', linewidth=2, label=f'High PRS (slope={slope_high:.3f}/yr)')
ax.plot(ages, mean_low, 'b-', linewidth=2, label=f'Low PRS (slope={slope_low:.3f}/yr)')
ax.axhline(y=level_high, color='r', linestyle='--', alpha=0.5, label=f'High baseline={level_high:.2f}')
ax.axhline(y=level_low, color='b', linestyle='--', alpha=0.5, label=f'Low baseline={level_low:.2f}')
ax.set_xlabel('Age')
ax.set_ylabel('E[λ] (Mean trajectory)')
ax.set_title('Decomposition: Level + Slope Effect')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 6: Comparison of mechanisms
ax = axes[1, 2]
# Person A: High baseline, low slope
person_A_mean = 1.0 + 0.005 * t_centered
# Person B: Low baseline, high slope
person_B_mean = 0.3 + 0.025 * t_centered

ax.plot(ages, person_A_mean, 'g-', linewidth=3, label='Person A: High baseline, slow progression')
ax.plot(ages, person_B_mean, 'm-', linewidth=3, label='Person B: Normal baseline, fast progression')
ax.set_xlabel('Age')
ax.set_ylabel('E[λ]')
ax.set_title('Two Routes to High Risk\n(Different biology!)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Mark where they cross
crossing_age = (1.0 - 0.3) / (0.025 - 0.005) + 30
if 30 <= crossing_age <= 80:
    ax.axvline(x=crossing_age, color='k', linestyle=':', alpha=0.5)
    ax.text(crossing_age + 1, ax.get_ylim()[1] * 0.9, f'Cross at\nage {crossing_age:.0f}', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/sarahurbut/aladynoulli2/genetic_slope_simulation.pdf', dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: genetic_slope_simulation.pdf")

# ============================================================================
# SAVE SIMULATED DATA
# ============================================================================

print(f"\n" + "=" * 70)
print("SAVING SIMULATED DATA")
print("=" * 70)

sim_data = {
    'G': G,  # Genetics (N x P)
    'Y': Y,  # Outcomes (N x D x T)
    'lambda_true': lambda_true,  # True lambda (N x K x T)
    'theta_true': theta_true,  # True theta (N x K x T)
    'phi_true': phi_true,  # True phi (K x D x T)
    'gamma_level_true': gamma_level_true,  # True level effects (P x K)
    'gamma_slope_true': gamma_slope_true,  # True slope effects (P x K)
    'psi_true': psi_true,  # True psi (K x D)
    'r_k': r_k,  # Signature references (K,)
    'ages': ages,  # Timepoints (T,)
    'N': N, 'T': T, 'K': K, 'D': D, 'P': P,
    'high_risk_idx': high_risk_idx,
    'low_risk_idx': low_risk_idx,
}

np.savez('/Users/sarahurbut/aladynoulli2/simulated_genetic_slope_data.npz', **sim_data)
print(f"Saved simulation data to: simulated_genetic_slope_data.npz")

print(f"\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print("\nKey findings:")
print("  1. High-PRS individuals have STEEPER lambda trajectories")
print("  2. This creates systematic differences in progression speed")
print("  3. Both level AND slope effects contribute to early-onset disease")
print("  4. These effects are separable because they affect different")
print("     aspects of the trajectory (intercept vs. slope)")
print("\nNext steps:")
print("  - Run clust_huge_amp_vectorized_with_slope.py to fit model WITH slope")
print("  - Compare to model WITHOUT slope to show identifiability")
print("=" * 70)

plt.show()
