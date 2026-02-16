"""
Shared prediction logic for nokappa models.
Used by nokappa_v3_pipeline and holdout_predict_and_auc_nokappa_v3_b20_30.
Ensures identical fit + pi extraction — no NaN in output.
"""

import numpy as np
import torch
from pathlib import Path

# Import fixed-gamma model
_path = Path(__file__).parent
import sys
sys.path.insert(0, str(_path / 'aws_offsetmaster'))
from clust_huge_amp_fixedPhi_vectorized_fixed_gamma_fixed_kappa_REPARAM import (
    AladynSurvivalFixedPhiFixedGammaFixedKappaReparam,
)


def fit_and_extract_pi(Y_batch, E_batch, G_with_sex, phi, psi, kappa, gamma,
                       signature_refs, prevalence_t, disease_names,
                       num_epochs=200, learning_rate=0.1):
    """Fit reparam prediction model and extract pi. Same logic as pipeline.
    Handles NaN/Inf so output pi has no NaN."""
    torch.manual_seed(42)
    np.random.seed(42)

    N, D, T = Y_batch.shape
    K = phi.shape[0] - 1 if phi.shape[0] == 21 else phi.shape[0]
    P = G_with_sex.shape[1]

    model = AladynSurvivalFixedPhiFixedGammaFixedKappaReparam(
        N=N, D=D, T=T, K=K, P=P,
        G=G_with_sex, Y=Y_batch,
        R=0, W=0.0001, prevalence_t=prevalence_t,
        init_sd_scaler=1e-1, genetic_scale=1,
        pretrained_phi=phi, pretrained_psi=psi,
        pretrained_gamma=gamma, pretrained_kappa=kappa,
        signature_references=signature_refs, healthy_reference=True,
        disease_names=disease_names,
    )

    result = model.fit(E_batch, num_epochs=num_epochs, learning_rate=learning_rate)
    losses = result[0] if isinstance(result, tuple) else result

    with torch.no_grad():
        pi, _, _ = model.forward()

    # Same NaN/Inf handling as pipeline — ensures no NaN in output
    n_nan = torch.isnan(pi).sum().item()
    n_inf = torch.isinf(pi).sum().item()
    if n_nan > 0 or n_inf > 0:
        print(f'    WARNING: {n_nan} NaN, {n_inf} Inf in pi — fixing')
        pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    pi = torch.clamp(pi, 1e-8, 1 - 1e-8)

    return pi, losses[-1] if losses else float('nan')
