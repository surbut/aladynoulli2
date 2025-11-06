# Signature Correlation: Phi vs Lambda

## Definitions

### Lambda (λ): Patient-Signature Logits
- **Shape**: `λ[n, k, t]` → `(N, K, T)`
- **Meaning**: Logit of signature k's activity for patient n at time t
- **Range**: Real numbers (typically -∞ to +∞)
- **Interpretation**: How much patient n expresses signature k at time t (before softmax)

### Theta (θ): Patient-Signature Proportions
- **Shape**: `θ[n, k, t]` → `(N, K, T)`
- **Relationship**: `θ[n, k, t] = softmax(λ[n, :, t])[k]`
- **Meaning**: Proportion of signature k for patient n at time t
- **Range**: [0, 1], sums to 1 across k for each (n, t)
- **Interpretation**: Normalized signature activity (what we usually visualize)

### Phi (φ): Signature-Disease Associations
- **Shape**: `φ[k, d, t]` → `(K, D, T)`
- **Meaning**: Log-odds that signature k is associated with disease d at time t
- **Range**: Real numbers (typically -∞ to +∞)
- **Interpretation**: How signature k relates to disease d over time (biological mechanism)

---

## Correlation Methods

### Method 1: Correlation Using Phi (φ)

**What it measures**: Which signatures are associated with similar diseases over time

**Formula**:
```
Corr_Phi(k1, k2) = Correlation(
    φ[k1, :, :].flatten(),  # All diseases × all timepoints for signature k1
    φ[k2, :, :].flatten()   # All diseases × all timepoints for signature k2
)
```

**Matrix form**:
```
R_Phi = K×K correlation matrix
R_Phi[k1, k2] = Corr(φ[k1, d, t], φ[k2, d, t])  # Correlated across (d, t) pairs
```

**Biological interpretation**:
- High correlation → Signatures k1 and k2 are associated with similar diseases
- Example: If Sig 5 (inflammatory) and Sig 12 (metabolic) both associate with diabetes, they'll have positive correlation
- This captures **biological similarity** (which diseases each signature affects)

**Python code**:
```python
# Phi shape: (K, D, T)
K, D, T = phi.shape

# Flatten each signature's disease-time matrix
phi_flat = phi.reshape(K, D * T)  # (K, D*T)

# Compute K×K correlation matrix
corr_phi = np.corrcoef(phi_flat)  # (K, K)
```

---

### Method 2: Correlation Using Lambda (λ)

**What it measures**: Which signatures co-occur in the same patients over time

**Formula**:
```
Corr_Lambda(k1, k2) = Correlation(
    λ[:, k1, :].flatten(),  # All patients × all timepoints for signature k1
    λ[:, k2, :].flatten()   # All patients × all timepoints for signature k2
)
```

**Matrix form**:
```
R_Lambda = K×K correlation matrix
R_Lambda[k1, k2] = Corr(λ[n, k1, t], λ[n, k2, t])  # Correlated across (n, t) pairs
```

**Biological interpretation**:
- High correlation → Signatures k1 and k2 tend to be active in the same patients
- Example: If patients with high Sig 5 (inflammatory) also have high Sig 12 (metabolic), they'll have positive correlation
- This captures **co-occurrence patterns** (which signatures appear together in patients)

**Python code**:
```python
# Lambda shape: (N, K, T)
N, K, T = lambda_.shape

# Flatten each signature's patient-time matrix
lambda_flat = lambda_.transpose(1, 0, 2).reshape(K, N * T)  # (K, N*T)

# Compute K×K correlation matrix
corr_lambda = np.corrcoef(lambda_flat)  # (K, K)
```

---

### Method 3: Correlation Using Theta (θ) - Alternative to Lambda

**What it measures**: Which signatures co-occur in the same patients over time (normalized)

**Formula**:
```
Corr_Theta(k1, k2) = Correlation(
    θ[:, k1, :].flatten(),  # All patients × all timepoints for signature k1
    θ[:, k2, :].flatten()   # All patients × all timepoints for signature k2
)
```

**Note**: Since θ sums to 1 across k for each (n, t), correlations will be **negatively biased** (if one signature increases, others must decrease). This is a constraint of the softmax normalization.

**Python code**:
```python
# Theta shape: (N, K, T)
N, K, T = thetas.shape

# Flatten each signature's patient-time matrix
theta_flat = thetas.transpose(1, 0, 2).reshape(K, N * T)  # (K, N*T)

# Compute K×K correlation matrix
corr_theta = np.corrcoef(theta_flat)  # (K, K)
```

---

## Key Differences

### 1. **What They Measure**

| Method | Measures | Dimension Correlated Over |
|--------|----------|---------------------------|
| **Phi** | Biological similarity (disease associations) | Diseases × Time (D×T) |
| **Lambda** | Patient co-occurrence (who has what) | Patients × Time (N×T) |
| **Theta** | Patient co-occurrence (normalized) | Patients × Time (N×T) |

### 2. **Biological Meaning**

**Phi correlations**:
- "Which signatures affect similar diseases?"
- Example: Inflammatory and metabolic signatures both associated with cardiovascular disease
- **Population-level** biological relationships

**Lambda/Theta correlations**:
- "Which signatures appear together in patients?"
- Example: Patients with high inflammatory activity also have high metabolic activity
- **Individual-level** co-occurrence patterns

### 3. **Mathematical Relationship**

**Important**: Phi and Lambda are **independent** in the model structure:
- Lambda determines **which signatures are active** in each patient
- Phi determines **what diseases each signature causes**

However, they interact through the disease probability:
```
P(Y[n, d, t] = 1) = σ(Σ_k θ[n, k, t] * φ[k, d, t])
```

Where:
- `θ[n, k, t] = softmax(λ[n, :, t])[k]` (patient signature activity)
- `φ[k, d, t]` (signature-disease association)

So correlations in Lambda/Theta can **indirectly** relate to Phi through this interaction, but they measure different things.

---

## Example: Why They're Different

### Scenario: Two Signatures

**Signature 5 (Inflammatory)**:
- Phi: High association with autoimmune diseases, low with metabolic diseases
- Lambda: High in patients with autoimmune conditions

**Signature 12 (Metabolic)**:
- Phi: High association with metabolic diseases, low with autoimmune diseases
- Lambda: High in patients with metabolic conditions

### Correlation Results:

**Phi correlation (Sig 5, Sig 12)**:
- **Low/negative**: They associate with different diseases
- Measures: Biological mechanism similarity

**Lambda correlation (Sig 5, Sig 12)**:
- **Could be positive**: If patients often have both inflammatory AND metabolic issues
- Measures: Patient co-occurrence

**Conclusion**: Two signatures can have:
- **Low Phi correlation** (different disease associations)
- **High Lambda correlation** (co-occur in same patients)

This is biologically plausible! A patient might have both inflammatory and metabolic pathways active, even though they affect different diseases.

---

## When to Use Which?

### Use **Phi correlations** when:
- You want to understand **biological mechanisms**
- You want to group signatures by **disease associations**
- You're interested in **population-level** signature relationships
- You want to identify signatures with similar **functional roles**

### Use **Lambda/Theta correlations** when:
- You want to understand **patient-level patterns**
- You want to identify signatures that **co-occur** in individuals
- You're interested in **comorbidity patterns**
- You want to find signatures that are **mutually exclusive** or **complementary** in patients

---

## Code Implementation

```python
import numpy as np
from scipy.stats import pearsonr

def compute_phi_correlation(phi):
    """
    Compute K×K correlation matrix using Phi (signature-disease associations)
    
    Parameters:
    -----------
    phi : np.ndarray, shape (K, D, T)
        Signature-disease association matrix
    
    Returns:
    --------
    corr_matrix : np.ndarray, shape (K, K)
        Correlation matrix where corr_matrix[k1, k2] = correlation between
        phi[k1, :, :] and phi[k2, :, :] across all (disease, time) pairs
    """
    K, D, T = phi.shape
    
    # Flatten each signature: (K, D*T)
    phi_flat = phi.reshape(K, D * T)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(phi_flat)
    
    return corr_matrix


def compute_lambda_correlation(lambda_):
    """
    Compute K×K correlation matrix using Lambda (patient-signature logits)
    
    Parameters:
    -----------
    lambda_ : np.ndarray, shape (N, K, T)
        Patient-signature logit matrix
    
    Returns:
    --------
    corr_matrix : np.ndarray, shape (K, K)
        Correlation matrix where corr_matrix[k1, k2] = correlation between
        lambda_[:, k1, :] and lambda_[:, k2, :] across all (patient, time) pairs
    """
    N, K, T = lambda_.shape
    
    # Reshape: (K, N*T) - each row is one signature across all patients and times
    lambda_flat = lambda_.transpose(1, 0, 2).reshape(K, N * T)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(lambda_flat)
    
    return corr_matrix


def compute_theta_correlation(thetas):
    """
    Compute K×K correlation matrix using Theta (patient-signature proportions)
    
    Note: Theta correlations will be negatively biased due to softmax constraint
    (sum to 1 across signatures for each patient-time)
    
    Parameters:
    -----------
    thetas : np.ndarray, shape (N, K, T)
        Patient-signature proportion matrix (from softmax of lambda)
    
    Returns:
    --------
    corr_matrix : np.ndarray, shape (K, K)
        Correlation matrix where corr_matrix[k1, k2] = correlation between
        thetas[:, k1, :] and thetas[:, k2, :] across all (patient, time) pairs
    """
    N, K, T = thetas.shape
    
    # Reshape: (K, N*T)
    theta_flat = thetas.transpose(1, 0, 2).reshape(K, N * T)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(theta_flat)
    
    return corr_matrix


# Example usage:
# corr_phi = compute_phi_correlation(phi)      # (K, K) - biological similarity
# corr_lambda = compute_lambda_correlation(lambda_)  # (K, K) - patient co-occurrence
# corr_theta = compute_theta_correlation(thetas)     # (K, K) - patient co-occurrence (normalized)
```

---

## Summary

**Phi correlations** and **Lambda/Theta correlations** measure **different things**:

1. **Phi**: Which signatures affect similar diseases? (Biological mechanisms)
2. **Lambda/Theta**: Which signatures appear together in patients? (Co-occurrence patterns)

They are **not the same** and can give very different results. Choose based on your research question!

