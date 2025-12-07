#!/usr/bin/env python
"""
Train AOU model with corrected E and prevalence
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
    parser = argparse.ArgumentParser(description='Train AOU model with corrected E')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-1,
                       help='Learning rate')
    parser.add_argument('--lambda_reg', type=float, default=1e-2,
                       help='Regularization parameter')
    parser.add_argument('--W', type=float, default=0.0001,
                       help='W parameter')
    parser.add_argument('--output_path', type=str,
                       default='/Users/sarahurbut/aladynoulli2/aou_model_trained_correctedE.pt',
                       help='Output path for trained model')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(7)
    np.random.seed(4)

    print(f"\n{'='*60}")
    print(f"Training AOU model with corrected E and prevalence")
    print(f"{'='*60}\n")

    # Load data
    data_path = "/Users/sarahurbut/Library/CloudStorage/DB_backup_5132025941p/aou_fromdl/"
    
    print("Loading Y and E...")
    Y_aou = np.array(robjects.r['readRDS'](os.path.join(data_path, 'Y_binary.rds')))
    E_aou = np.array(robjects.r['readRDS'](os.path.join(data_path, 'E_binary.rds')))
    E_aou = E_aou.astype(int)
    Y_tensor_aou = torch.FloatTensor(Y_aou)
    E_tensor_aou = torch.FloatTensor(E_aou)
    
    print(f"Y shape: {Y_tensor_aou.shape}")
    print(f"E shape: {E_tensor_aou.shape}")

    # Load corrected E
    print("Loading corrected E...")
    E_corrected_aou = torch.load('/Users/sarahurbut/aladynoulli2/aou_E_corrected.pt')
    print(f"Corrected E shape: {E_corrected_aou.shape}")

    # Load corrected prevalence
    print("Loading corrected prevalence...")
    prevalence_t_aou = torch.load('/Users/sarahurbut/aladynoulli2/aou_prevalence_corrected_E.pt')
    print(f"Prevalence shape: {prevalence_t_aou.shape}")

    # Load initialized model to get clusters, G, signature_refs, etc.
    print("Loading initialized model...")
    initialized_checkpoint = torch.load('/Users/sarahurbut/aladynoulli2/aou_model_initialized.pt')
    
    initial_clusters_aou = initialized_checkpoint['clusters']
    if isinstance(initial_clusters_aou, torch.Tensor):
        initial_clusters_aou = initial_clusters_aou.numpy()
    else:
        initial_clusters_aou = np.array(initial_clusters_aou)
    
    # Load G from old checkpoint (not saved in initialized model)
    old_checkpoint = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt', map_location='cpu')
    G_aou = old_checkpoint['G']
    
    signature_refs_aou = initialized_checkpoint['signature_refs']
    disease_names_aou = initialized_checkpoint['disease_names']
    K_aou = int(initial_clusters_aou.max() + 1)
    
    print(f"K={K_aou} signatures")
    print(f"G shape: {G_aou.shape}")

    # Initialize model
    print(f"\nInitializing model...")
    model_aou = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y_tensor_aou.shape[0],
        D=Y_tensor_aou.shape[1],
        T=Y_tensor_aou.shape[2],
        K=K_aou,
        P=G_aou.shape[1],
        init_sd_scaler=1e-1,
        G=G_aou,
        Y=Y_tensor_aou,
        genetic_scale=1,
        W=args.W,
        R=0,
        prevalence_t=prevalence_t_aou,
        signature_references=signature_refs_aou,
        healthy_reference=True,
        disease_names=disease_names_aou
    )

    # Set clusters and initialize
    model_aou.clusters = initial_clusters_aou
    psi_config = initialized_checkpoint.get('psi_config', {'in_cluster': 1, 'out_cluster': -2, 'noise_in': 0.1, 'noise_out': 0.01})
    model_aou.initialize_params(psi_config=psi_config)

    print(f"✓ Model initialized")

    # Train the model
    print(f"\nTraining model for {args.num_epochs} epochs...")
    print(f"Learning rate: {args.learning_rate}, Lambda: {args.lambda_reg}")

    history = model_aou.fit(E_corrected_aou,
                           num_epochs=args.num_epochs,
                           learning_rate=args.learning_rate,
                           lambda_reg=args.lambda_reg)

    # Save trained model
    print(f"\nSaving trained model to {args.output_path}...")
    torch.save({
        'model_state_dict': model_aou.state_dict(),
        'phi': model_aou.phi,
        'Y': model_aou.Y,
        'prevalence_t': prevalence_t_aou,
        'clusters': initial_clusters_aou,
        'signature_refs': signature_refs_aou,
        'disease_names': disease_names_aou,
        'G': G_aou,
        'args': vars(args),
        'history': history[-10:] if history else None,  # Save last 10 losses
    }, args.output_path)

    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to:")
    print(f"{args.output_path}")
    print(f"{'='*60}\n")

    if history and len(history) > 0 and 'loss' in history[-1]:
        print(f"Final loss: {history[-1]['loss']:.4f}")


if __name__ == '__main__':
    main()

