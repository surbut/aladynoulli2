"""
Simulation comparing Latent Factor Analysis vs Mixture Modeling for Binary Data

Demonstrates:
1. Gradient flow differences
2. Learning rate requirements
3. Parameter recovery
4. Optimization stability
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_mixture_data(N=1000, D=20, K=3, prevalence_range=(0.01, 0.1)):
    """
    Generate binary data from a TRUE mixture model:
    P(Y_nd = 1) = Σ_k θ_nk × p_kd

    where θ_nk are mixture weights and p_kd are component probabilities
    """
    # True mixture weights (sum to 1 for each person)
    theta_true = np.random.dirichlet(np.ones(K) * 2, size=N)  # [N x K]

    # True component probabilities (varying prevalence)
    p_true = np.random.uniform(prevalence_range[0], prevalence_range[1], size=(K, D))

    # Compute true probabilities
    pi_true = theta_true @ p_true  # [N x D]

    # Generate binary outcomes
    Y = (np.random.rand(N, D) < pi_true).astype(float)

    return Y, theta_true, p_true, pi_true


# ============================================================================
# MODEL 1: LATENT FACTOR MODEL (sigmoid outside)
# ============================================================================

class LatentFactorModel(nn.Module):
    """
    Y = sigmoid(Σ_k l_k × f_k)

    - l_nk: person-specific loadings [N x K]
    - f_kd: factor values [K x D]
    """
    def __init__(self, N, D, K):
        super().__init__()
        self.N, self.D, self.K = N, D, K

        # Initialize parameters (small random values)
        self.loadings = nn.Parameter(torch.randn(N, K) * 0.1)
        self.factors = nn.Parameter(torch.randn(K, D) * 0.1)

    def forward(self):
        # Linear combination THEN sigmoid
        eta = self.loadings @ self.factors  # [N x D]
        pi = torch.sigmoid(eta)
        return pi

    def compute_loss(self, Y):
        pi = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)

        # Binary cross-entropy
        loss = -torch.mean(Y * torch.log(pi) + (1 - Y) * torch.log(1 - pi))
        return loss


# ============================================================================
# MODEL 2: MIXTURE MODEL (sigmoid inside)
# ============================================================================

class MixtureModel(nn.Module):
    """
    Y = Σ_k θ_k × sigmoid(φ_k)

    - λ_nk (unconstrained) -> θ_nk = softmax(λ) mixture weights [N x K]
    - φ_kd: component logits [K x D]
    """
    def __init__(self, N, D, K):
        super().__init__()
        self.N, self.D, self.K = N, D, K

        # Initialize parameters (small random values)
        self.lambda_ = nn.Parameter(torch.randn(N, K) * 0.1)
        self.phi = nn.Parameter(torch.randn(K, D) * 0.1)

    def forward(self):
        # Mixture weights (sum to 1)
        theta = torch.softmax(self.lambda_, dim=1)  # [N x K]

        # Component probabilities
        p = torch.sigmoid(self.phi)  # [K x D]

        # Mixture: Σ_k θ_k × p_k
        pi = theta @ p  # [N x D]
        return pi

    def compute_loss(self, Y):
        pi = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)

        # Binary cross-entropy
        loss = -torch.mean(Y * torch.log(pi) + (1 - Y) * torch.log(1 - pi))
        return loss


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(model, Y_tensor, lr=0.001, n_epochs=1000, verbose=False):
    """Train model and track loss + gradient norms"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    grad_norms = []

    pbar = tqdm(range(n_epochs), disable=not verbose)
    for epoch in pbar:
        optimizer.zero_grad()
        loss = model.compute_loss(Y_tensor)
        loss.backward()

        # Track gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(total_norm)

        if verbose and epoch % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}, Grad: {total_norm:.4f}")

    return losses, grad_norms


# ============================================================================
# EXPERIMENT: COMPARE MODELS WITH DIFFERENT LRs
# ============================================================================

def run_comparison(N=1000, D=20, K=3, n_epochs=1000):
    """Compare LFA vs Mixture models with different learning rates"""

    print("="*70)
    print("GENERATING DATA FROM TRUE MIXTURE MODEL")
    print("="*70)
    Y, theta_true, p_true, pi_true = generate_mixture_data(N, D, K)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    print(f"Data: {N} samples, {D} features, {K} components")
    print(f"True prevalence range: {pi_true.min():.3f} - {pi_true.max():.3f}")
    print(f"Sparsity: {(Y.sum() / Y.size):.3f}")
    print()

    # Test different learning rates
    learning_rates = [0.001, 0.003, 0.01, 0.03]

    results = {}

    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"LEARNING RATE: {lr}")
        print(f"{'='*70}")

        # Train Latent Factor Model
        print(f"\nTraining Latent Factor Model (sigmoid outside)...")
        lfa_model = LatentFactorModel(N, D, K)
        lfa_losses, lfa_grads = train_model(lfa_model, Y_tensor, lr=lr, n_epochs=n_epochs, verbose=True)

        # Train Mixture Model
        print(f"\nTraining Mixture Model (sigmoid inside)...")
        mix_model = MixtureModel(N, D, K)
        mix_losses, mix_grads = train_model(mix_model, Y_tensor, lr=lr, n_epochs=n_epochs, verbose=True)

        results[lr] = {
            'lfa_losses': lfa_losses,
            'lfa_grads': lfa_grads,
            'mix_losses': mix_losses,
            'mix_grads': mix_grads,
            'lfa_final': lfa_losses[-1],
            'mix_final': mix_losses[-1]
        }

        print(f"\nFinal Losses:")
        print(f"  LFA:     {lfa_losses[-1]:.6f}")
        print(f"  Mixture: {mix_losses[-1]:.6f}")

    return results, Y, theta_true, p_true


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results):
    """Plot training curves for different LRs"""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    learning_rates = list(results.keys())

    for idx, lr in enumerate(learning_rates):
        res = results[lr]

        # Loss curves
        ax = axes[0, idx]
        ax.plot(res['lfa_losses'], label='LFA (sigmoid outside)', linewidth=2)
        ax.plot(res['mix_losses'], label='Mixture (sigmoid inside)', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Learning Rate = {lr}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Gradient norms
        ax = axes[1, idx]
        ax.plot(res['lfa_grads'], label='LFA', linewidth=2)
        ax.plot(res['mix_grads'], label='Mixture', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/Users/sarahurbut/aladynoulli2/lfa_vs_mixture_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: /Users/sarahurbut/aladynoulli2/lfa_vs_mixture_comparison.pdf")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LATENT FACTOR ANALYSIS vs MIXTURE MODELING")
    print("Comparison for Binary Data")
    print("="*70 + "\n")

    # Run comparison
    results, Y, theta_true, p_true = run_comparison(
        N=1000,
        D=20,
        K=3,
        n_epochs=1000
    )

    # Plot results
    plot_results(results)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nFinal Losses by Learning Rate:")
    print(f"{'LR':<10} {'LFA':<12} {'Mixture':<12} {'Winner'}")
    print("-" * 50)
    for lr in results.keys():
        lfa_loss = results[lr]['lfa_final']
        mix_loss = results[lr]['mix_final']
        winner = 'LFA' if lfa_loss < mix_loss else 'Mixture'
        print(f"{lr:<10} {lfa_loss:<12.6f} {mix_loss:<12.6f} {winner}")
