#!/usr/bin/env python3
"""
Test script to verify that the vectorized GP loss computation produces identical results
to the original loop-based version.
"""

import numpy as np
import torch
import sys
import os

# Add path for imports
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts_forPublish')

from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest as ModelOriginal
from clust_huge_amp_vectorized import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest as ModelVectorized

def create_test_data(N=100, D=10, T=20, K=3, P=5):
    """Create small test dataset"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create random data
    Y = np.random.binomial(1, 0.1, size=(N, D, T)).astype(np.float32)
    G = np.random.randn(N, P).astype(np.float32)
    prevalence_t = np.random.uniform(0.01, 0.1, size=(D, T)).astype(np.float32)
    event_times = np.random.randint(5, T-1, size=N)
    
    # Create signature references
    signature_references = np.random.randn(K).astype(np.float32)
    
    return Y, G, prevalence_t, event_times, signature_references

def test_models():
    """Test that both models produce identical results"""
    print("="*80)
    print("TESTING VECTORIZED VS ORIGINAL GP LOSS COMPUTATION")
    print("="*80)
    
    # Create test data
    N, D, T, K, P = 100, 10, 20, 3, 5
    Y, G, prevalence_t, event_times, signature_references = create_test_data(N, D, T, K, P)
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"\nTest parameters:")
    print(f"  N={N}, D={D}, T={T}, K={K}, P={P}")
    
    # Initialize original model
    print("\n" + "="*80)
    print("Initializing ORIGINAL model...")
    print("="*80)
    torch.manual_seed(42)
    np.random.seed(42)
    model_orig = ModelOriginal(
        N=N, D=D, T=T, K=K, P=P, G=G, Y=Y, R=0.01, W=0.001,
        prevalence_t=prevalence_t, init_sd_scaler=0.1, genetic_scale=1.0,
        signature_references=signature_references, healthy_reference=True,
        disease_names=[f"Disease_{i}" for i in range(D)], learn_kappa=False
    )
    
    # Initialize vectorized model with same seed
    print("\n" + "="*80)
    print("Initializing VECTORIZED model...")
    print("="*80)
    torch.manual_seed(42)
    np.random.seed(42)
    model_vec = ModelVectorized(
        N=N, D=D, T=T, K=K, P=P, G=G, Y=Y, R=0.01, W=0.001,
        prevalence_t=prevalence_t, init_sd_scaler=0.1, genetic_scale=1.0,
        signature_references=signature_references, healthy_reference=True,
        disease_names=[f"Disease_{i}" for i in range(D)], learn_kappa=False
    )
    
    # Copy parameters from original to vectorized to ensure identical initialization
    print("\n" + "="*80)
    print("Copying parameters to ensure identical initialization...")
    print("="*80)
    model_vec.lambda_.data = model_orig.lambda_.data.clone()
    model_vec.phi.data = model_orig.phi.data.clone()
    model_vec.psi.data = model_orig.psi.data.clone()
    model_vec.gamma.data = model_orig.gamma.data.clone()
    model_vec.kappa.data = model_orig.kappa.data.clone()
    
    # Test GP loss computation
    print("\n" + "="*80)
    print("TESTING GP PRIOR LOSS COMPUTATION")
    print("="*80)
    
    # Set models to eval mode and disable gradients for comparison
    model_orig.eval()
    model_vec.eval()
    
    with torch.no_grad():
        gp_loss_orig = model_orig.compute_gp_prior_loss()
        gp_loss_vec = model_vec.compute_gp_prior_loss()
    
    print(f"\nGP Loss Comparison:")
    print(f"  Original:  {gp_loss_orig.item():.10f}")
    print(f"  Vectorized: {gp_loss_vec.item():.10f}")
    print(f"  Difference: {abs(gp_loss_orig.item() - gp_loss_vec.item()):.2e}")
    
    if torch.allclose(gp_loss_orig, gp_loss_vec, rtol=1e-5, atol=1e-6):
        print("  ✓ GP losses match!")
    else:
        print("  ✗ GP losses DO NOT match!")
        return False
    
    # Test full loss computation
    print("\n" + "="*80)
    print("TESTING FULL LOSS COMPUTATION")
    print("="*80)
    
    model_orig.train()
    model_vec.train()
    
    loss_orig = model_orig.compute_loss(event_times)
    loss_vec = model_vec.compute_loss(event_times)
    
    print(f"\nFull Loss Comparison:")
    print(f"  Original:  {loss_orig.item():.10f}")
    print(f"  Vectorized: {loss_vec.item():.10f}")
    print(f"  Difference: {abs(loss_orig.item() - loss_vec.item()):.2e}")
    
    if torch.allclose(loss_orig, loss_vec, rtol=1e-5, atol=1e-6):
        print("  ✓ Full losses match!")
    else:
        print("  ✗ Full losses DO NOT match!")
        return False
    
    # Test gradients
    print("\n" + "="*80)
    print("TESTING GRADIENTS")
    print("="*80)
    
    # Zero gradients
    model_orig.zero_grad()
    model_vec.zero_grad()
    
    # Compute losses and backprop
    loss_orig = model_orig.compute_loss(event_times)
    loss_vec = model_vec.compute_loss(event_times)
    
    loss_orig.backward()
    loss_vec.backward()
    
    # Compare gradients
    grad_params = ['lambda_', 'phi', 'psi', 'gamma']
    all_match = True
    
    for param_name in grad_params:
        grad_orig = getattr(model_orig, param_name).grad
        grad_vec = getattr(model_vec, param_name).grad
        
        if grad_orig is None or grad_vec is None:
            print(f"  {param_name}: No gradients")
            continue
        
        max_diff = torch.max(torch.abs(grad_orig - grad_vec)).item()
        mean_diff = torch.mean(torch.abs(grad_orig - grad_vec)).item()
        
        print(f"\n  {param_name} gradients:")
        print(f"    Max difference: {max_diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        
        if torch.allclose(grad_orig, grad_vec, rtol=1e-4, atol=1e-5):
            print(f"    ✓ Gradients match!")
        else:
            print(f"    ✗ Gradients DO NOT match!")
            all_match = False
    
    # Test a few training steps
    print("\n" + "="*80)
    print("TESTING TRAINING STEP (1 epoch)")
    print("="*80)
    
    # Reset models to same state
    model_orig.lambda_.data = model_orig.lambda_.data.clone()
    model_vec.lambda_.data = model_orig.lambda_.data.clone()
    model_orig.phi.data = model_orig.phi.data.clone()
    model_vec.phi.data = model_orig.phi.data.clone()
    model_orig.psi.data = model_orig.psi.data.clone()
    model_vec.psi.data = model_orig.psi.data.clone()
    model_orig.gamma.data = model_orig.gamma.data.clone()
    model_vec.gamma.data = model_orig.gamma.data.clone()
    
    # Create optimizers with same learning rate
    optimizer_orig = torch.optim.Adam(model_orig.parameters(), lr=0.01)
    optimizer_vec = torch.optim.Adam(model_vec.parameters(), lr=0.01)
    
    # One training step
    optimizer_orig.zero_grad()
    optimizer_vec.zero_grad()
    
    loss_orig = model_orig.compute_loss(event_times)
    loss_vec = model_vec.compute_loss(event_times)
    
    loss_orig.backward()
    loss_vec.backward()
    
    optimizer_orig.step()
    optimizer_vec.step()
    
    # Compare parameters after one step
    print(f"\nParameter differences after 1 training step:")
    for param_name in grad_params:
        param_orig = getattr(model_orig, param_name).data
        param_vec = getattr(model_vec, param_name).data
        
        max_diff = torch.max(torch.abs(param_orig - param_vec)).item()
        print(f"  {param_name}: max diff = {max_diff:.2e}")
        
        if not torch.allclose(param_orig, param_vec, rtol=1e-4, atol=1e-5):
            print(f"    ⚠ Warning: Parameters differ after training step")
            all_match = False
    
    print("\n" + "="*80)
    if all_match:
        print("✓ ALL TESTS PASSED - Vectorized version produces identical results!")
        print("="*80)
        return True
    else:
        print("✗ SOME TESTS FAILED - Check differences above")
        print("="*80)
        return False

if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)

