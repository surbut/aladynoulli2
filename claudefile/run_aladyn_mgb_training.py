#!/usr/bin/env python
"""
Train MGB model with corrected E and prevalence
Uses vectorized code for faster training
"""

import numpy as np
import torch
import warnings
import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'pyScripts_forPublish'))

from clust_huge_amp_vectorized import *
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser(description='Train MGB model with corrected E')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization parameter')
    parser.add_argument('--W', type=float, default=0.0001,
                       help='W parameter')
    parser.add_argument('--output_path', type=str,
                       default='/Users/sarahurbut/aladynoulli2/mgb_model_trained_correctedE.pt',
                       help='Output path for trained model')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(7)
    np.random.seed(4)

    print(f"\n{'='*60}")
    print(f"Training MGB model with corrected E and prevalence")
    print(f"{'='*60}\n")

    # Load data
    data_path_mgb = "/Users/sarahurbut/Dropbox-Personal/mgbbtopic/"
    
    print("Loading Y and E...")
    Y_mgb = np.array(robjects.r['readRDS'](os.path.join(data_path_mgb, 'Y_sub.rds')))
    E_mgb = np.array(robjects.r['readRDS'](os.path.join(data_path_mgb, 'E_sub.rds')))
    E_mgb = E_mgb.astype(int)
    Y_tensor_mgb = torch.FloatTensor(Y_mgb)
    E_tensor_mgb = torch.FloatTensor(E_mgb)
    
    print(f"Y shape: {Y_tensor_mgb.shape}")
    print(f"E shape: {E_tensor_mgb.shape}")

    # Load corrected E
    print("Loading corrected E...")
    E_corrected_mgb = torch.load('/Users/sarahurbut/aladynoulli2/mgb_E_corrected.pt')
    print(f"Corrected E shape: {E_corrected_mgb.shape}")

    # Load corrected prevalence
    print("Loading corrected prevalence...")
    prevalence_t_mgb = torch.load('/Users/sarahurbut/aladynoulli2/mgb_prevalence_corrected_E.pt')
    print(f"Prevalence shape: {prevalence_t_mgb.shape}")

    # Load initialized model to get clusters, G, signature_refs, etc.
    print("Loading initialized model...")
    initialized_checkpoint = torch.load('/Users/sarahurbut/aladynoulli2/mgb_model_initialized.pt')
    
    initial_clusters_mgb = initialized_checkpoint['clusters']
    if isinstance(initial_clusters_mgb, torch.Tensor):
        initial_clusters_mgb = initial_clusters_mgb.numpy()
    else:
        initial_clusters_mgb = np.array(initial_clusters_mgb)
    
    # Load G from old checkpoint (not saved in initialized model)
    old_checkpoint = torch.load('/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt', map_location='cpu')
    G_mgb = old_checkpoint['G']
    
    signature_refs_mgb = initialized_checkpoint['signature_refs']
    disease_names_mgb = initialized_checkpoint['disease_names']
    K_mgb = int(initial_clusters_mgb.max() + 1)
    
    print(f"K={K_mgb} signatures")
    print(f"G shape: {G_mgb.shape}")

    # Initialize model
    print(f"\nInitializing model...")
    model_mgb = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_tensor_mgb.shape[0],
        D=Y_tensor_mgb.shape[1],
        T=Y_tensor_mgb.shape[2],
        K=K_mgb,
        P=G_mgb.shape[1],
        init_sd_scaler=1e-1,
        G=G_mgb,
        Y=Y_tensor_mgb,
        genetic_scale=1,
        W=args.W,
        R=0,
        prevalence_t=prevalence_t_mgb,
        signature_references=signature_refs_mgb,
        healthy_reference=True,
        disease_names=disease_names_mgb
    )

    # Set clusters and initialize with psi_config (like notebook)
    # This uses the old clusters and initializes psi where:
    # - Diseases IN the cluster get positive values (in_cluster=1)
    # - Diseases OUT of the cluster get negative values (out_cluster=-2)
    print("Setting clusters and initializing with psi_config...")
    torch.manual_seed(0)
    np.random.seed(0)
    
    model_mgb.clusters = initial_clusters_mgb
    psi_config = {'in_cluster': 1, 'out_cluster': -2, 'noise_in': 0.1, 'noise_out': 0.01}
    model_mgb.initialize_params(psi_config=psi_config)
    
    # Verify clusters match
    clusters_match = np.array_equal(initial_clusters_mgb, model_mgb.clusters)
    print(f"  Clusters match exactly: {clusters_match}")
    print(f"  Psi config: in_cluster={psi_config['in_cluster']}, out_cluster={psi_config['out_cluster']}")
    print(f"✓ Model initialized with psi_config (like notebook)")

    # Train the model
    print(f"\nTraining model for {args.num_epochs} epochs...")
    print(f"Learning rate: {args.learning_rate}, Lambda: {args.lambda_reg}")

    history = model_mgb.fit(E_corrected_mgb,
                           num_epochs=args.num_epochs,
                           learning_rate=args.learning_rate,
                           lambda_reg=args.lambda_reg)

    # Save trained model (like batch script)
    print(f"\nSaving trained model to {args.output_path}...")
    torch.save({
        'model_state_dict': model_mgb.state_dict(),
        'phi': model_mgb.phi,
        'Y': model_mgb.Y,
        'prevalence_t': prevalence_t_mgb,
        'logit_prevalence_t': model_mgb.logit_prev_t,
        'G': model_mgb.G,
        'args': vars(args),
        'clusters': initial_clusters_mgb,  # Save initial_clusters directly
        'version': 'VECTORIZED_TRAINED',  # Mark as trained version
    }, args.output_path)

    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to:")
    print(f"{args.output_path}")
    print(f"{'='*60}\n")

    if history and len(history) > 0 and 'loss' in history[-1]:
        print(f"Final loss: {history[-1]['loss']:.4f}")


if __name__ == '__main__':
    main()

