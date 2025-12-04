"""
Test script to verify that vectorized and non-vectorized versions
produce identical results.
"""
import numpy as np
import torch
import sys
import os

# Add the directory to path so we can import both versions
sys.path.insert(0, os.path.dirname(__file__))

from clust_huge_amp_fixedPhi import AladynSurvivalFixedPhi as ModelOriginal
from clust_huge_amp_fixedPhi_vectorized import AladynSurvivalFixedPhi as ModelVectorized

def create_test_data():
    """Create small synthetic data for testing"""
    N, D, T, K, P = 100, 20, 10, 5, 10
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data
    G = np.random.randn(N, P)
    Y = np.random.binomial(1, 0.1, size=(N, D, T)).astype(float)
    
    # Create synthetic phi and psi
    # Note: In actual use case, phi has K_total entries (K + healthy state)
    # But for testing, we create K entries and let the model handle healthy state
    phi = np.random.randn(K, D, T) * 0.1
    psi = np.random.randn(K, D) * 0.1
    
    # Create prevalence
    prevalence_t = np.random.uniform(0.01, 0.1, size=(D, T))
    
    # Signature references
    signature_refs = np.random.randn(K) * 0.1
    
    return {
        'N': N, 'D': D, 'T': T, 'K': K, 'P': P,
        'G': G, 'Y': Y,
        'phi': phi, 'psi': psi,
        'prevalence_t': prevalence_t,
        'signature_refs': signature_refs
    }

def test_gp_loss_equivalence():
    """Test that GP prior loss is identical between versions"""
    print("=" * 60)
    print("Testing GP Prior Loss Equivalence")
    print("=" * 60)
    
    data = create_test_data()
    
    # Set same random seed for both models
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create original model
    model_orig = ModelOriginal(
        N=data['N'],
        D=data['D'],
        T=data['T'],
        K=data['K'],
        P=data['P'],
        G=data['G'],
        Y=data['Y'],
        R=0,
        W=0.0001,
        prevalence_t=data['prevalence_t'],
        init_sd_scaler=0.1,
        genetic_scale=1.0,
        pretrained_phi=data['phi'],
        pretrained_psi=data['psi'],
        signature_references=data['signature_refs'],
        healthy_reference=True
    )
    
    # Set same random seed again for vectorized model
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create vectorized model
    model_vec = ModelVectorized(
        N=data['N'],
        D=data['D'],
        T=data['T'],
        K=data['K'],
        P=data['P'],
        G=data['G'],
        Y=data['Y'],
        R=0,
        W=0.0001,
        prevalence_t=data['prevalence_t'],
        init_sd_scaler=0.1,
        genetic_scale=1.0,
        pretrained_phi=data['phi'],
        pretrained_psi=data['psi'],
        signature_references=data['signature_refs'],
        healthy_reference=True
    )
    
    # Copy parameters from original to vectorized to ensure they're identical
    model_vec.lambda_.data = model_orig.lambda_.data.clone()
    model_vec.gamma.data = model_orig.gamma.data.clone()
    model_vec.kappa.data = model_orig.kappa.data.clone()
    
    # Compute GP losses
    with torch.no_grad():
        loss_orig = model_orig.compute_gp_prior_loss()
        loss_vec = model_vec.compute_gp_prior_loss()
    
    print(f"\nOriginal GP Loss:  {loss_orig.item():.10f}")
    print(f"Vectorized GP Loss: {loss_vec.item():.10f}")
    print(f"Difference:         {abs(loss_orig.item() - loss_vec.item()):.2e}")
    
    # Check if they're close (within numerical precision)
    tolerance = 1e-6
    if torch.allclose(loss_orig, loss_vec, atol=tolerance, rtol=tolerance):
        print(f"\n✓ PASS: Losses are identical (within {tolerance:.0e} tolerance)")
        return True
    else:
        print(f"\n✗ FAIL: Losses differ by more than {tolerance:.0e}")
        return False

def test_full_loss_equivalence():
    """Test that full loss computation is identical"""
    print("\n" + "=" * 60)
    print("Testing Full Loss Equivalence")
    print("=" * 60)
    
    data = create_test_data()
    
    # Create event times
    E = torch.randint(0, data['T'], size=(data['N'], data['D']))
    
    # Set same random seed for both models
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_orig = ModelOriginal(
        N=data['N'], D=data['D'], T=data['T'], K=data['K'], P=data['P'],
        G=data['G'], Y=data['Y'], R=0, W=0.0001,
        prevalence_t=data['prevalence_t'], init_sd_scaler=0.1, genetic_scale=1.0,
        pretrained_phi=data['phi'], pretrained_psi=data['psi'],
        signature_references=data['signature_refs'], healthy_reference=True
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_vec = ModelVectorized(
        N=data['N'], D=data['D'], T=data['T'], K=data['K'], P=data['P'],
        G=data['G'], Y=data['Y'], R=0, W=0.0001,
        prevalence_t=data['prevalence_t'], init_sd_scaler=0.1, genetic_scale=1.0,
        pretrained_phi=data['phi'], pretrained_psi=data['psi'],
        signature_references=data['signature_refs'], healthy_reference=True
    )
    
    # Copy parameters
    model_vec.lambda_.data = model_orig.lambda_.data.clone()
    model_vec.gamma.data = model_orig.gamma.data.clone()
    model_vec.kappa.data = model_orig.kappa.data.clone()
    
    # Compute full losses
    with torch.no_grad():
        loss_orig = model_orig.compute_loss(E.numpy())
        loss_vec = model_vec.compute_loss(E.numpy())
    
    print(f"\nOriginal Full Loss:  {loss_orig.item():.10f}")
    print(f"Vectorized Full Loss: {loss_vec.item():.10f}")
    print(f"Difference:           {abs(loss_orig.item() - loss_vec.item()):.2e}")
    
    tolerance = 1e-6
    if torch.allclose(loss_orig, loss_vec, atol=tolerance, rtol=tolerance):
        print(f"\n✓ PASS: Full losses are identical (within {tolerance:.0e} tolerance)")
        return True
    else:
        print(f"\n✗ FAIL: Full losses differ by more than {tolerance:.0e}")
        return False

def test_gradient_equivalence():
    """Test that gradients are identical after one training step"""
    print("\n" + "=" * 60)
    print("Testing Gradient Equivalence")
    print("=" * 60)
    
    data = create_test_data()
    E = torch.randint(0, data['T'], size=(data['N'], data['D']))
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_orig = ModelOriginal(
        N=data['N'], D=data['D'], T=data['T'], K=data['K'], P=data['P'],
        G=data['G'], Y=data['Y'], R=0, W=0.0001,
        prevalence_t=data['prevalence_t'], init_sd_scaler=0.1, genetic_scale=1.0,
        pretrained_phi=data['phi'], pretrained_psi=data['psi'],
        signature_references=data['signature_refs'], healthy_reference=True
    )
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_vec = ModelVectorized(
        N=data['N'], D=data['D'], T=data['T'], K=data['K'], P=data['P'],
        G=data['G'], Y=data['Y'], R=0, W=0.0001,
        prevalence_t=data['prevalence_t'], init_sd_scaler=0.1, genetic_scale=1.0,
        pretrained_phi=data['phi'], pretrained_psi=data['psi'],
        signature_references=data['signature_refs'], healthy_reference=True
    )
    
    # Copy parameters
    model_vec.lambda_.data = model_orig.lambda_.data.clone()
    model_vec.gamma.data = model_orig.gamma.data.clone()
    model_vec.kappa.data = model_orig.kappa.data.clone()
    
    # Compute loss and gradients
    loss_orig = model_orig.compute_loss(E.numpy())
    loss_orig.backward()
    
    loss_vec = model_vec.compute_loss(E.numpy())
    loss_vec.backward()
    
    # Compare gradients
    lambda_grad_diff = torch.norm(model_orig.lambda_.grad - model_vec.lambda_.grad).item()
    gamma_grad_diff = torch.norm(model_orig.gamma.grad - model_vec.gamma.grad).item()
    kappa_grad_diff = torch.norm(model_orig.kappa.grad - model_vec.kappa.grad).item()
    
    print(f"\nLambda gradient difference: {lambda_grad_diff:.2e}")
    print(f"Gamma gradient difference:   {gamma_grad_diff:.2e}")
    print(f"Kappa gradient difference:  {kappa_grad_diff:.2e}")
    
    tolerance = 1e-5
    all_close = (lambda_grad_diff < tolerance and 
                 gamma_grad_diff < tolerance and 
                 kappa_grad_diff < tolerance)
    
    if all_close:
        print(f"\n✓ PASS: Gradients are identical (within {tolerance:.0e} tolerance)")
        return True
    else:
        print(f"\n✗ FAIL: Gradients differ by more than {tolerance:.0e}")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Vectorization Equivalence Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: GP loss only
    results.append(test_gp_loss_equivalence())
    
    # Test 2: Full loss
    results.append(test_full_loss_equivalence())
    
    # Test 3: Gradients
    results.append(test_gradient_equivalence())
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"GP Loss Test:      {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"Full Loss Test:    {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"Gradient Test:     {'✓ PASS' if results[2] else '✗ FAIL'}")
    
    if all(results):
        print("\n🎉 All tests passed! Vectorized version is mathematically equivalent.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please investigate.")
        sys.exit(1)

