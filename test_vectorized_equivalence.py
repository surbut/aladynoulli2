#!/usr/bin/env python
"""
Test script to verify clust_huge_amp.py and clust_huge_amp_vectorized.py
produce identical results.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'pyScripts_forPublish'))

# Import both versions
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest as ModelLoop
from clust_huge_amp_vectorized import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest as ModelVectorized

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Create small test data
N, D, T, K, P = 100, 20, 10, 3, 5

# Generate synthetic data
Y = torch.rand(N, D, T) < 0.1  # Binary outcomes
E_matrix = torch.randint(0, T, (N, D))  # Event times matrix (N x D)
# For compute_loss, we need per-patient event times (1D array)
# Use max across diseases, or censor at T-1
event_times = torch.clamp(E_matrix.max(dim=1)[0], max=T-1).numpy()  # Per-patient event times
G = torch.randn(N, P)  # Genetic data
prevalence_t = torch.rand(D, T) * 0.01  # Small prevalences
signature_refs = torch.randn(K)

print("="*80)
print("TESTING EQUIVALENCE: Loop vs Vectorized GP Prior Loss")
print("="*80)

# Initialize both models with same random seed
print("\n1. Initializing models...")
torch.manual_seed(0)
np.random.seed(0)
model_loop = ModelLoop(
    N=N, D=D, T=T, K=K, P=P,
    G=G.numpy(),
    Y=Y.numpy(),
    R=0, W=0.001,
    prevalence_t=prevalence_t.numpy(),
    init_sd_scaler=0.1,
    genetic_scale=1.0,
    signature_references=signature_refs.numpy(),
    healthy_reference=True,
    disease_names=[f"Disease_{i}" for i in range(D)]
)

torch.manual_seed(0)
np.random.seed(0)
model_vectorized = ModelVectorized(
    N=N, D=D, T=T, K=K, P=P,
    G=G.numpy(),
    Y=Y.numpy(),
    R=0, W=0.001,
    prevalence_t=prevalence_t.numpy(),
    init_sd_scaler=0.1,
    genetic_scale=1.0,
    signature_references=signature_refs.numpy(),
    healthy_reference=True,
    disease_names=[f"Disease_{i}" for i in range(D)]
)

print("   ✓ Models initialized")

# Check initial parameters match
print("\n2. Checking initial parameters match...")
params_match = True
for name, param_loop in model_loop.named_parameters():
    param_vec = dict(model_vectorized.named_parameters())[name]
    if not torch.allclose(param_loop, param_vec, atol=1e-6):
        print(f"   ✗ Mismatch in {name}: max diff = {(param_loop - param_vec).abs().max().item():.2e}")
        params_match = False

if params_match:
    print("   ✓ All initial parameters match!")

# Test forward pass
print("\n3. Testing forward pass...")
with torch.no_grad():
    pi_loop, theta_loop, phi_prob_loop = model_loop.forward()
    pi_vec, theta_vec, phi_prob_vec = model_vectorized.forward()

forward_match = (
    torch.allclose(pi_loop, pi_vec, atol=1e-6) and
    torch.allclose(theta_loop, theta_vec, atol=1e-6) and
    torch.allclose(phi_prob_loop, phi_prob_vec, atol=1e-6)
)

if forward_match:
    print("   ✓ Forward pass outputs match!")
else:
    print("   ✗ Forward pass outputs differ:")
    print(f"     pi diff: {(pi_loop - pi_vec).abs().max().item():.2e}")
    print(f"     theta diff: {(theta_loop - theta_vec).abs().max().item():.2e}")
    print(f"     phi_prob diff: {(phi_prob_loop - phi_prob_vec).abs().max().item():.2e}")

# Test GP prior loss computation
print("\n4. Testing GP prior loss computation...")
with torch.no_grad():
    gp_loss_loop = model_loop.compute_gp_prior_loss()
    gp_loss_vec = model_vectorized.compute_gp_prior_loss()

gp_loss_match = torch.allclose(gp_loss_loop, gp_loss_vec, atol=1e-5)

if gp_loss_match:
    print(f"   ✓ GP prior loss matches!")
    print(f"     Loop version: {gp_loss_loop.item():.6f}")
    print(f"     Vectorized:   {gp_loss_vec.item():.6f}")
    print(f"     Difference:   {(gp_loss_loop - gp_loss_vec).abs().item():.2e}")
else:
    print(f"   ✗ GP prior loss differs:")
    print(f"     Loop version: {gp_loss_loop.item():.6f}")
    print(f"     Vectorized:   {gp_loss_vec.item():.6f}")
    print(f"     Difference:   {(gp_loss_loop - gp_loss_vec).abs().item():.2e}")

# Test full loss computation
print("\n5. Testing full loss computation...")
with torch.no_grad():
    loss_loop = model_loop.compute_loss(event_times)
    loss_vec = model_vectorized.compute_loss(event_times)

loss_match = torch.allclose(loss_loop, loss_vec, atol=1e-5)

if loss_match:
    print(f"   ✓ Full loss matches!")
    print(f"     Loop version: {loss_loop.item():.6f}")
    print(f"     Vectorized:   {loss_vec.item():.6f}")
    print(f"     Difference:   {(loss_loop - loss_vec).abs().item():.2e}")
else:
    print(f"   ✗ Full loss differs:")
    print(f"     Loop version: {loss_loop.item():.6f}")
    print(f"     Vectorized:   {loss_vec.item():.6f}")
    print(f"     Difference:   {(loss_loop - loss_vec).abs().item():.2e}")

# Test training for a few epochs
print("\n6. Testing training (3 epochs)...")
torch.manual_seed(1)
np.random.seed(1)
losses_loop, _ = model_loop.fit(event_times, num_epochs=3, learning_rate=0.01, lambda_reg=0.01)

torch.manual_seed(1)
np.random.seed(1)
losses_vec, _ = model_vectorized.fit(event_times, num_epochs=3, learning_rate=0.01, lambda_reg=0.01)

# Check final parameters after training
print("\n7. Checking parameters after training...")
final_params_match = True
for name, param_loop in model_loop.named_parameters():
    param_vec = dict(model_vectorized.named_parameters())[name]
    max_diff = (param_loop - param_vec).abs().max().item()
    if max_diff > 1e-5:
        print(f"   ✗ Mismatch in {name}: max diff = {max_diff:.2e}")
        final_params_match = False

if final_params_match:
    print("   ✓ All parameters match after training!")

# Check final losses
final_loss_match = abs(losses_loop[-1] - losses_vec[-1]) < 1e-5
if final_loss_match:
    print(f"   ✓ Final losses match: {losses_loop[-1]:.6f} vs {losses_vec[-1]:.6f}")
else:
    print(f"   ✗ Final losses differ: {losses_loop[-1]:.6f} vs {losses_vec[-1]:.6f}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
all_match = (
    params_match and forward_match and gp_loss_match and 
    loss_match and final_params_match and final_loss_match
)

if all_match:
    print("✓ ALL TESTS PASSED - Models are equivalent!")
else:
    print("✗ SOME TESTS FAILED - Models differ!")
    print("\nFailed checks:")
    if not params_match:
        print("  - Initial parameters")
    if not forward_match:
        print("  - Forward pass")
    if not gp_loss_match:
        print("  - GP prior loss")
    if not loss_match:
        print("  - Full loss")
    if not final_params_match:
        print("  - Parameters after training")
    if not final_loss_match:
        print("  - Final loss after training")

print("="*80)

