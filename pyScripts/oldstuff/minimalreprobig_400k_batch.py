#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import seaborn as s
import pandas as pd
# Assuming cluster_g_logit_init is in the same directory
from cluster_g_logit_init import *

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AladynSurvival model in batch mode')
    
    parser.add_argument('--base-path', type=str, required=True,
                       help='Base path for loading model components')
    parser.add_argument('--n-samples', type=int, default=100000,
                       help='Number of samples to use (default: 100000)')
    parser.add_argument('--num-epochs', type=int, default=1000,
                       help='Number of epochs for training (default: 1000)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate for training (default: 0.0001)')
    parser.add_argument('--lambda-reg', type=float, default=1e-2,
                       help='Regularization parameter (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save outputs (default: ./output)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    
    return parser.parse_args()

def load_model_essentials(base_path):
    """Load all essential model components."""
    print("Loading components...")
    
    try:
        # Load large matrices
        Y = torch.load(base_path + '/Y_tensor.pt')
        E = torch.load(base_path + '/E_matrix.pt')
        G = torch.load(base_path + '/G_matrix.pt')
        
        # Load other components
        essentials = torch.load(base_path + '/model_essentials.pt')
        
        print("Loaded all components successfully!")
        
        return Y, E, G, essentials
    except Exception as e:
        print(f"Error loading model components: {str(e)}")
        sys.exit(1)

def subset_data(Y, E, G, n_samples, seed):
    """Subset the data to n_samples individuals while maintaining consistency."""
    torch.manual_seed(seed)
    
    # Get total number of individuals
    N = Y.shape[0]
    
    # Randomly select n_samples indices
    indices = torch.randperm(N)[:n_samples]
    
    # Subset all matrices using the same indices
    Y_sub = Y[indices]
    E_sub = E[indices]
    G_sub = G[indices]
    
    print(f"Original shapes: Y={Y.shape}, E={E.shape}, G={G.shape}")
    print(f"New shapes: Y={Y_sub.shape}, E={E_sub.shape}, G={G_sub.shape}")
    
    return Y_sub, E_sub, G_sub, indices

def save_visualizations(model, disease_names, output_dir):
    """Generate and save visualization plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different types of disease_names input
    if isinstance(disease_names, pd.DataFrame):
        disease_names_list = disease_names[0].tolist()
    elif isinstance(disease_names, pd.Series):
        disease_names_list = disease_names.tolist()
    elif isinstance(disease_names, list):
        disease_names_list = disease_names
    elif isinstance(disease_names, np.ndarray):
        disease_names_list = disease_names.tolist()
    else:
        print("Warning: disease_names not in expected format, skipping visualization")
        return
    
    # Print some debug info
    print(f"Number of disease names: {len(disease_names_list)}")
    print(f"First few disease names: {disease_names_list[:5]}")
    
    try:
        # Save cluster visualization
        plt.figure(figsize=(12, 8))
        model.visualize_clusters(disease_names_list)
        plt.savefig(os.path.join(output_dir, 'clusters_visualization.png'))
        plt.close()
        
        # Save initialization visualization
        plt.figure(figsize=(12, 8))
        model.visualize_initialization()
        plt.savefig(os.path.join(output_dir, 'initialization_visualization.png'))
        plt.close()
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        print("Continuing without visualizations...")

def main():
    args = parse_arguments()
    
    # Load model components
    Y, E, G, essentials = load_model_essentials(args.base_path)
    
    # Subset the data
    Y_sub, E_sub, G_sub, indices = subset_data(Y, E, G, args.n_samples, args.seed)
    

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize model with subsetted data
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit(
        N=Y_sub.shape[0],
        D=Y_sub.shape[1],
        T=Y_sub.shape[2],
        K=essentials['K'],
        P=essentials['P'],
        G=G_sub,
        Y=Y_sub,
        prevalence_t=essentials['prevalence_t']
    )
    
    # Save initial state
    initial_psi = model.psi.detach().clone()
    
    # Generate visualizations if requested
    if args.visualize:
        save_visualizations(model, essentials['disease_names'], args.output_dir)
    
    # Train the model
    print(f"\nStarting training for {args.num_epochs} epochs...")


    
    history = model.fit(E_sub, 
                       num_epochs=args.num_epochs,
                       learning_rate=args.learning_rate,
                       lambda_reg=args.lambda_reg)
    
    # Save the trained model and history
    torch.save({
        'model_state': model.state_dict(),
        'person_indices': indices,
        'initial_psi': initial_psi,
        'final_psi': model.psi.detach(),
        'training_history': history
    }, f"{args.output_dir}/trained_model.pt")
    
    print(f"\nTraining completed. Model and results saved to {args.output_dir}")

if __name__ == "__main__":
    main()