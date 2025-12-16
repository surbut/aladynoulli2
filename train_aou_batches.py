#!/usr/bin/env python3
"""
Batch training script for AOU model with corrected E_corrected
Run with: nohup python train_aou_batches.py > train_aou_batches.log 2>&1 &
"""

import sys
import os
import gc
import numpy as np
import torch
from pathlib import Path

# Add path for model import
sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts_forPublish')
from clust_huge_amp_vectorized import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

# ============================================================================
# Configuration
# ============================================================================
BATCH_SIZE = 10000
START_BATCH = 0  # Start from beginning

# Paths
Y_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/Y_binary.pt'
E_CORRECTED_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/E_corrected.pt'
G_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/G_dummy.pt'
PREVALENCE_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/prevalence_t_corrected.pt'
SIGNATURE_REFS_PATH = '/Users/sarahurbut/aladynoulli2/aou_signature_refs.pt'
CHECKPOINT_PATH = '/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt'
SAVE_DIR = os.path.expanduser('~/Library/CloudStorage/Dropbox/aou_batches')

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameters
HYPERPARAMS = {
    'init_sd_scaler': 1e-1,
    'genetic_scale': 0,
    'W': 0.0001,
    'R': 0,
    'num_epochs': 200,
    'learning_rate': 1e-1,
    'lambda_reg': 1e-2,
}

def main():
    print("="*80)
    print("AOU BATCH TRAINING WITH CORRECTED E_CORRECTED")
    print("="*80)
    print(f"Start batch: {START_BATCH}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Output directory: {SAVE_DIR}")
    print(f"Hyperparameters: {HYPERPARAMS}")
    print("="*80)
    
    # Load all data
    print("\nLoading data...")
    Y_tensor_aou = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/Y_binary.pt')
    E_corrected_aou = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/E_corrected.pt')
    G_aou = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/G_dummy.pt')
    prevalence_t = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/prevalence_t_corrected.pt')
    signature_refs_aou = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/allofusbigdata/aou_signature_refs.pt')

    
    print(f"  Y shape: {Y_tensor_aou.shape}")
    print(f"  E_corrected shape: {E_corrected_aou.shape}")
    print(f"  G shape: {G_aou.shape}")
    print(f"  Prevalence shape: {prevalence_t.shape}")
    
    # Verify E_corrected has no negative values
    min_e = E_corrected_aou.min().item() if torch.is_tensor(E_corrected_aou) else E_corrected_aou.min()
    if min_e < 0:
        print(f"\n⚠️ WARNING: E_corrected has negative values (min={min_e})! Clipping to 0...")
        if torch.is_tensor(E_corrected_aou):
            E_corrected_aou = torch.clamp(E_corrected_aou, min=0)
        else:
            E_corrected_aou = np.clip(E_corrected_aou, 0, None)
        print(f"  ✓ Clipped. New min: {E_corrected_aou.min().item() if torch.is_tensor(E_corrected_aou) else E_corrected_aou.min()}")
    else:
        print(f"  ✓ E_corrected is non-negative (min={min_e})")
    
    # Convert to numpy for easier slicing
    if torch.is_tensor(Y_tensor_aou):
        Y_np = Y_tensor_aou.numpy()
    else:
        Y_np = Y_tensor_aou
    
    if torch.is_tensor(E_corrected_aou):
        E_corrected_np = E_corrected_aou.numpy()
    else:
        E_corrected_np = E_corrected_aou
    
    if torch.is_tensor(G_aou):
        G_np = G_aou.numpy()
    else:
        G_np = G_aou
    
    # Load old AOU checkpoint for clusters
    print("\nLoading checkpoint for cluster information...")
    aou_checkpoint_old = torch.load(CHECKPOINT_PATH, map_location='cpu')
    initial_clusters_aou = aou_checkpoint_old['clusters']
    if isinstance(initial_clusters_aou, torch.Tensor):
        initial_clusters_aou = initial_clusters_aou.numpy()
    else:
        initial_clusters_aou = np.array(initial_clusters_aou)
    
    K_aou = int(initial_clusters_aou.max() + 1)
    disease_names_aou = aou_checkpoint_old['disease_names']
    print(f"  K (number of signatures): {K_aou}")
    print(f"  Number of diseases: {len(disease_names_aou)}")
    
    # Calculate batches
    N_total = Y_np.shape[0]
    n_batches = int(np.ceil(N_total / BATCH_SIZE))
    
    print(f"\n{'='*80}")
    print("BATCH TRAINING SETUP")
    print(f"{'='*80}")
    print(f"Total patients: {N_total:,}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Number of batches: {n_batches}")
    print(f"Will train batches: {START_BATCH} to {n_batches - 1}")
    print(f"{'='*80}")
    
    # Train each batch
    for batch_idx in range(START_BATCH, n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, N_total)
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1} / {n_batches} (batch_idx={batch_idx})")
        print(f"{'='*80}")
        print(f"Patients: {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} patients)")
        
        try:
            # Extract batch
            Y_batch = torch.FloatTensor(Y_np[start_idx:end_idx, :, :])
            E_batch = E_corrected_np[start_idx:end_idx, :]
            G_batch = torch.FloatTensor(G_np[start_idx:end_idx, :])
            
            print(f"Y_batch shape: {Y_batch.shape}")
            print(f"E_batch shape: {E_batch.shape}")
            print(f"G_batch shape: {G_batch.shape}")
            
            # Create model for this batch
            print(f"\nInitializing model for batch {batch_idx + 1}...")
            model_batch = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
                N=Y_batch.shape[0],
                D=Y_batch.shape[1], 
                T=Y_batch.shape[2], 
                K=K_aou,
                P=G_batch.shape[1],
                init_sd_scaler=HYPERPARAMS['init_sd_scaler'],
                G=G_batch, 
                Y=Y_batch,
                genetic_scale=HYPERPARAMS['genetic_scale'],
                W=HYPERPARAMS['W'],
                R=HYPERPARAMS['R'],
                prevalence_t=prevalence_t,
                signature_references=signature_refs_aou,
                healthy_reference=True,
                disease_names=disease_names_aou
            )
            
            # Set clusters and initialize
            model_batch.clusters = initial_clusters_aou
            psi_config = {'in_cluster': 1, 'out_cluster': -2, 'noise_in': 0.1, 'noise_out': 0.01}
            model_batch.initialize_params(psi_config=psi_config)
            
            # After model initialization, set gamma to zeros since genetic_scale=0
            with torch.no_grad():
                model_batch.gamma.zero_()
            
            # Train
            print(f"\nTraining batch {batch_idx + 1}...")
            history = model_batch.fit(
                E_batch, 
                num_epochs=HYPERPARAMS['num_epochs'], 
                learning_rate=HYPERPARAMS['learning_rate'], 
                lambda_reg=HYPERPARAMS['lambda_reg']
            )
            
            # Save batch model
            save_dict_batch = {
                'model_state_dict': model_batch.state_dict(),
                'clusters': initial_clusters_aou,
                'signature_refs': signature_refs_aou,
                'psi_config': psi_config,
                'hyperparameters': {
                    'N': Y_batch.shape[0],
                    'D': Y_batch.shape[1],
                    'T': Y_batch.shape[2],
                    'K': K_aou,
                    'P': G_batch.shape[1],
                    **HYPERPARAMS
                },
                'prevalence_t': prevalence_t,
                'disease_names': disease_names_aou,
                'batch_idx': batch_idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
            }
            
            batch_save_path = os.path.join(SAVE_DIR, f'aou_model_batch_{batch_idx}_{start_idx}_{end_idx}.pt')
            torch.save(save_dict_batch, batch_save_path)
            print(f"✓ Saved batch {batch_idx + 1} to: {batch_save_path}")
            
            # Clean up: delete model and history, then garbage collect
            del model_batch
            del history
            del Y_batch
            del E_batch
            del G_batch
            gc.collect()
            
            print(f"✓ Batch {batch_idx + 1} complete and memory cleaned")
            
        except Exception as e:
            print(f"\n❌ ERROR in batch {batch_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Continuing with next batch...")
            continue
    
    print(f"\n{'='*80}")
    print("ALL BATCHES COMPLETE")
    print(f"{'='*80}")
    print(f"Trained batches {START_BATCH} to {n_batches - 1}")
    print(f"Models saved in: {SAVE_DIR}")
    print(f"Models saved with prefix: aou_model_batch_*")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

