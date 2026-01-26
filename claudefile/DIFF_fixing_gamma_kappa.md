# Exact Differences: Fixing Gamma and Kappa

This document shows **ONLY** the changes made to fix gamma and kappa (from `clust_huge_amp_fixedPhi_vectorized.py` to `clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa.py`).

## 1. `__init__` Method Signature

**ORIGINAL (gamma & kappa learnable):**
```python
def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
             pretrained_phi, pretrained_psi, signature_references=None, healthy_reference=None, 
             disease_names=None, flat_lambda=False):
```

**FIXED (gamma & kappa provided):**
```python
def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
             pretrained_phi, pretrained_psi, pretrained_gamma, pretrained_kappa, 
             signature_references=None, healthy_reference=None, disease_names=None, flat_lambda=False):
```

**CHANGE:** Added `pretrained_gamma` and `pretrained_kappa` as required parameters.

---

## 2. Kappa Initialization

**ORIGINAL (learnable parameter):**
```python
self.kappa = nn.Parameter(torch.ones(1))  # Single global calibration parameter
```

**FIXED (fixed buffer):**
```python
# Store pretrained kappa as a buffer (not trainable)
if pretrained_kappa is None:
    raise ValueError("pretrained_kappa must be provided for fixed-kappa model")
if isinstance(pretrained_kappa, (int, float)):
    self.register_buffer('kappa', torch.tensor(float(pretrained_kappa), dtype=torch.float32))
elif torch.is_tensor(pretrained_kappa):
    if pretrained_kappa.numel() == 1:
        self.register_buffer('kappa', pretrained_kappa.clone().detach().squeeze())
    else:
        raise ValueError(f"kappa must be a scalar, got shape {pretrained_kappa.shape}")
else:
    self.register_buffer('kappa', torch.tensor(float(pretrained_kappa), dtype=torch.float32))
```

**CHANGE:** 
- `nn.Parameter` → `register_buffer` (not trainable)
- Must be provided, not initialized to 1.0

---

## 3. Gamma Initialization

**ORIGINAL (learnable parameter, initialized in `initialize_params()`):**
```python
# In initialize_params():
gamma_init = torch.zeros((self.P, self.K_total))
# ... compute gamma_init from data ...
self.gamma = nn.Parameter(gamma_init)  # Line 185
```

**FIXED (fixed buffer, provided in `__init__`):**
```python
# In __init__ (after phi/psi):
# Store pretrained gamma as a buffer (not trainable)
if pretrained_gamma is None:
    raise ValueError("pretrained_gamma must be provided for fixed-gamma model")
if torch.is_tensor(pretrained_gamma):
    self.register_buffer('gamma', pretrained_gamma.clone().detach())
else:
    self.register_buffer('gamma', torch.tensor(pretrained_gamma, dtype=torch.float32))
```

**CHANGE:**
- Moved from `initialize_params()` to `__init__`
- `nn.Parameter` → `register_buffer` (not trainable)
- Must be provided, not computed from data

---

## 4. `initialize_params()` Method

**ORIGINAL (initializes both lambda and gamma):**
```python
def initialize_params(self, init_psi=None, init_gamma=None, **kwargs):
    """Initialize only individual-specific parameters (lambda, gamma)"""
    # Initialize gamma for disease clusters
    gamma_init = torch.zeros((self.P, self.K_total))
    lambda_init = torch.zeros((self.N, self.K_total, self.T))
    
    # ... compute gamma_init from Y_avg or use init_gamma ...
    
    # Only make lambda and gamma trainable parameters
    self.gamma = nn.Parameter(gamma_init)  # ← LEARNABLE
    self.lambda_ = nn.Parameter(lambda_init)
```

**FIXED (only initializes lambda, uses fixed gamma):**
```python
def initialize_params(self, **kwargs):
    """Initialize only lambda (gamma and kappa are fixed from pretrained values)"""
    lambda_init = torch.zeros((self.N, self.K_total, self.T))
    
    # Initialize lambda using the fixed gamma
    for k in range(self.K):
        lambda_means = self.genetic_scale * (self.G @ self.gamma[:, k])  # ← Uses fixed gamma
        # ... rest of initialization ...
    
    # Only make lambda trainable (gamma and kappa are fixed as buffers)
    self.lambda_ = nn.Parameter(lambda_init)  # ← Only lambda is learnable
```

**CHANGE:**
- Removed all gamma initialization logic
- Uses `self.gamma` (fixed buffer) instead of computing `gamma_init`
- No `self.gamma = nn.Parameter(...)` line

---

## 5. `fit()` Method - Optimizer

**ORIGINAL (optimizes lambda, gamma, and kappa):**
```python
def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01):
    """Modified fit method that only updates lambda and gamma"""
    
    optimizer = optim.Adam([
        {'params': [self.lambda_], 'lr': learning_rate},      # e.g. 1e-2
        {'params': [self.kappa], 'lr': learning_rate},        # ← KAPPA LEARNABLE
        {'params': [self.gamma], 'lr': learning_rate}         # ← GAMMA LEARNABLE
    ])
```

**FIXED (only optimizes lambda):**
```python
def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01, verbose=True, analyze_every=20):
    """Modified fit method that only updates lambda (gamma and kappa are fixed)"""
    
    # Only lambda is trainable - gamma and kappa are fixed buffers
    optimizer = optim.Adam([
        {'params': [self.lambda_], 'lr': learning_rate},      # e.g. 1e-2
        # gamma and kappa are NOT in optimizer - they're fixed buffers
    ])
```

**CHANGE:**
- Removed `self.kappa` and `self.gamma` from optimizer
- Only `self.lambda_` is optimized

---

## 6. `compute_gp_prior_loss()` Method

**ORIGINAL:**
```python
mean_lambda_k = self.signature_refs[k].unsqueeze(0) + \
            self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
```

**FIXED:**
```python
mean_lambda_k = self.signature_refs[k].unsqueeze(0) + \
            self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
```

**CHANGE:** None - both use `self.gamma`, but in fixed version it's a buffer (not updated)

---

## 7. `forward()` Method

**ORIGINAL:**
```python
pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
```

**FIXED:**
```python
pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
```

**CHANGE:** None - both use `self.kappa`, but in fixed version it's a buffer (not updated)

---

## Summary

**Key Changes:**
1. **Kappa**: `nn.Parameter(torch.ones(1))` → `register_buffer('kappa', pretrained_kappa)`
2. **Gamma**: `nn.Parameter(gamma_init)` in `initialize_params()` → `register_buffer('gamma', pretrained_gamma)` in `__init__()`
3. **Optimizer**: Removed `self.kappa` and `self.gamma` from optimizer parameters
4. **Initialization**: Removed gamma computation logic from `initialize_params()`

**What stays the same:**
- `forward()` method (uses `self.kappa` and `self.gamma` the same way)
- `compute_loss()` method
- `compute_gp_prior_loss()` method (uses `self.gamma` the same way)
- All lambda-related logic

**Effect on AUC:**
- Fixing **kappa** prevents per-batch calibration overfitting → may reduce AUC slightly
- Fixing **gamma** prevents per-batch genetic effect overfitting → may reduce AUC slightly
- The combined effect is what you're seeing in the comparison
