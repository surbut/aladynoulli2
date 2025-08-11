#!/usr/bin/env python3
"""
AWS-compatible survival model training script with configurable batch indices.
Usage: python train_survival_model.py --start_index 0 --end_index 10000 --output_dir s3://your-bucket/results/
"""

## this can be run using E_enrollment using the censoring time capped at minimum of event time or age of recruitment for prediction tasks, whicever comes first, or using the full E matrix for learning and discovery tasks. 
# It is designed to be run with a fixed phi and psi, and a fixed starting age of 40, so that everyone can have 30 years of predictions using the externally valuidated phi and psi.

import argparse
import os
import sys
import gc
import logging
from pathlib import Path
import cProfile
import pstats
from pstats import SortKey

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

# Assuming these are your custom modules - adjust paths as needed
try:
    from utils import *
    from clust_huge_amp_fixedPhi import *
except ImportError as e:
    logging.error(f"Failed to import custom modules: {e}")
    logging.error("Make sure utils.py and clust_huge_amp_fixedPhi.py are in your Python path")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SurvivalModelTrainer:
    def __init__(self, base_data_path, output_dir, start_index=0, end_index=10000):
        """
        Initialize the survival model trainer.
        
        Args:
            base_data_path (str): Path to the data directory (local or S3)
            output_dir (str): Output directory for results (local or S3)
            start_index (int): Starting index for data subset
            end_index (int): Ending index for data subset
        """
        self.base_data_path = base_data_path
        self.output_dir = output_dir
        self.start_index = start_index
        self.end_index = end_index
        
        # Create output directory if it doesn't exist
        if not output_dir.startswith('s3://'):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        self.set_random_seeds()
        
    def set_random_seeds(self, seed=42):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        logger.info(f"Set random seeds to {seed}")

    def load_model_essentials(self):
        """Load all essential components for the model."""
        logger.info("Loading model components...")
        
        try:
            # Load large matrices
            Y = torch.load(os.path.join(self.base_data_path, 'Y_tensor.pt'))
            E = torch.load(os.path.join(self.base_data_path, 'E_matrix.pt'))
            G = torch.load(os.path.join(self.base_data_path, 'G_matrix.pt'))
            
            # Load other components
            essentials = torch.load(os.path.join(self.base_data_path, 'model_essentials.pt'))
            
            logger.info("Loaded all components successfully!")
            return Y, E, G, essentials
            
        except Exception as e:
            logger.error(f"Failed to load model essentials: {e}")
            raise

    def load_and_prepare_data(self):
        """Load and prepare all data for training."""
        logger.info(f"Preparing data for indices {self.start_index} to {self.end_index}")
        
        # Load model essentials
        Y, E, G, essentials = self.load_model_essentials()
        
        # Subset the data
        Y_subset, E_subset, G_subset, indices = subset_data(
            Y, E, G, 
            start_index=self.start_index, 
            end_index=self.end_index
        )
        
        # Clean up large Y tensor from memory
        del Y
        gc.collect()
        
        # Load references
        refs_path = os.path.join(self.base_data_path, 'reference_trajectories.pt')
        refs = torch.load(refs_path)
        signature_refs = refs['signature_refs']
        
        # Load demographic data
        fh_processed_path = os.path.join(self.base_data_path, 'baselinagefamh.csv')
        fh_processed = pd.read_csv(fh_processed_path)
        
        # Subset demographic data
        pce_df_subset = fh_processed.iloc[self.start_index:self.end_index].reset_index(drop=True)
        sex = pce_df_subset['sex'].values
        G_with_sex = np.column_stack([G_subset, sex])
        
        # Load pretrained phi and psi
        total_fit_path = os.path.join(self.base_data_path, 'enrollment_model_W0.0001_fulldata_sexspecific.pt')
        total_checkpoint = torch.load(total_fit_path, map_location='cpu')
        phi_total = total_checkpoint['model_state_dict']['phi'].cpu().numpy()
        psi_total = total_checkpoint['model_state_dict']['psi'].cpu().numpy()
        
        logger.info(f"Data preparation complete. Subset shape: {Y_subset.shape}")
        
        return {
            'Y_subset': Y_subset,
            'E_subset': E_subset,
            'G_with_sex': G_with_sex,
            'essentials': essentials,
            'signature_refs': signature_refs,
            'phi_total': phi_total,
            'psi_total': psi_total,
            'pce_df_subset': pce_df_subset
        }

    def create_age_specific_events(self, E_original, pce_df_subset, age_offset):
        """Create age-specific event times for given age offset.
        
        Now uses fixed starting age of 40 (10 years from 30) and applies age_offset
        for up to 30 years (so age_offset 0 = age 40, age_offset 30 = age 70).
        """
        E_age_specific = E_original.clone()
        
        total_times_changed = 0
        max_cap_applied = 0
        min_cap_applied = float('inf')
        
        # Fixed starting age is 40 (which is 10 years from 30)
        fixed_starting_age = 40
        time_since_30_fixed_start = fixed_starting_age - 30  # This is 10
        
        for patient_idx, row in enumerate(pce_df_subset.itertuples()):
            if patient_idx >= E_age_specific.shape[0]:
                break
                
            # Current age = fixed starting age (40) + age_offset
            current_age = fixed_starting_age + age_offset
            time_since_30 = max(0, current_age - 30)
            
            max_cap_applied = max(max_cap_applied, time_since_30)
            min_cap_applied = min(min_cap_applied, time_since_30)
            
            # Store original times for this patient
            original_times = E_age_specific[patient_idx, :].clone()
            
            # Cap event times to current age
            E_age_specific[patient_idx, :] = torch.minimum(
                E_age_specific[patient_idx, :],
                torch.full_like(E_age_specific[patient_idx, :], time_since_30)
            )
            
            times_changed = torch.sum(E_age_specific[patient_idx, :] != original_times).item()
            total_times_changed += times_changed
        
        # Log censoring verification
        logger.info(f"Censoring for age offset {age_offset} (age {fixed_starting_age + age_offset}):")
        logger.info(f"  Total event times changed: {total_times_changed}")
        logger.info(f"  Max cap applied: {max_cap_applied:.1f}")
        logger.info(f"  Min cap applied: {min_cap_applied:.1f}")
        
        return E_age_specific

    def train_model_for_age(self, data, age_offset):
        """Train model for specific age offset."""
        logger.info(f"Training model for age offset {age_offset} years")
        
        # Set seeds for reproducibility
        self.set_random_seeds()
        
        # Create age-specific event times
        E_age_specific = self.create_age_specific_events(
            data['E_subset'], data['pce_df_subset'], age_offset
        )
        
        # Initialize model
        model = AladynSurvivalFixedPhi(
            N=data['Y_subset'].shape[0],
            D=data['Y_subset'].shape[1],
            T=data['Y_subset'].shape[2],
            K=20,
            P=data['G_with_sex'].shape[1],
            G=data['G_with_sex'],
            Y=data['Y_subset'],
            R=0,
            W=0.0001,
            prevalence_t=data['essentials']['prevalence_t'],
            init_sd_scaler=1e-1,
            genetic_scale=1,
            pretrained_phi=data['phi_total'],
            pretrained_psi=data['psi_total'],
            signature_references=data['signature_refs'],
            healthy_reference=True,
            disease_names=data['essentials']['disease_names']
        )
        
        # Verify phi and psi match
        if np.allclose(model.phi.cpu().numpy(), data['phi_total']):
            logger.info("phi matches phi_total!")
        else:
            logger.warning("phi does NOT match phi_total!")
            
        if np.allclose(model.psi.cpu().numpy(), data['psi_total']):
            logger.info("psi matches psi_total!")
        else:
            logger.warning("psi does NOT match psi_total!")
        
        # Train model with profiling
        profiler = cProfile.Profile()
        profiler.enable()
        
        history = model.fit(
            E_age_specific,
            num_epochs=200,
            learning_rate=1e-1,
            lambda_reg=1e-2
        )
        
        profiler.disable()
        
        # Log profiling results
        stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(20)
        
        # Generate and save predictions
        with torch.no_grad():
            pi, _, _ = model.forward()
            
            # Save predictions
            pi_filename = f"pi_enroll_fixedphi_age_offset_{age_offset}_sex_{self.start_index}_{self.end_index}.pt"
            pi_path = os.path.join(self.output_dir, pi_filename)
            torch.save(pi, pi_path)
            logger.info(f"Saved predictions to {pi_path}")
            
            # Save model
            model_filename = f"model_enroll_fixedphi_age_offset_{age_offset}_sex_{self.start_index}_{self.end_index}.pt"
            model_path = os.path.join(self.output_dir, model_filename)
            torch.save({
                'model_state_dict': model.state_dict(),
                'E': E_age_specific,
                'prevalence_t': model.prevalence_t,
                'logit_prevalence_t': model.logit_prev_t,
                #'training_history': history,
                'age_offset': age_offset,
                'start_index': self.start_index,
                'end_index': self.end_index
            }, model_path)
            logger.info(f"Saved model to {model_path}")
        
        # Clean up memory
        del pi, model, E_age_specific
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return history

    def train_all_ages(self, max_age_offset=30):
        """Train models for all age offsets.
        
        Now trains for age offsets 0-30, which corresponds to ages 40-70
        (starting from fixed age 40, which is 10 years from 30).
        """
        logger.info(f"Starting training for batch {self.start_index}-{self.end_index}")
        logger.info(f"Training for age offsets 0-{max_age_offset} (ages 40-{40+max_age_offset})")
        
        # Load and prepare data once
        data = self.load_and_prepare_data()
        
        # Store training histories
        all_histories = {}
        
        for age_offset in range(0, max_age_offset + 1):
            logger.info(f"\n=== Processing age offset {age_offset} years (age {40 + age_offset}) ===")
            
            try:
                history = self.train_model_for_age(data, age_offset)
                all_histories[age_offset] = history
                
            except Exception as e:
                logger.error(f"Failed to train model for age offset {age_offset}: {e}")
                continue
        
        # Save combined results
        summary_path = os.path.join(self.output_dir, f"training_summary_{self.start_index}_{self.end_index}.pt")
        torch.save({
            'start_index': self.start_index,
            'end_index': self.end_index,
            'data_shape': data['Y_subset'].shape
        }, summary_path)
        
        logger.info(f"Training complete! Summary saved to {summary_path}")
        return all_histories


def main():
    parser = argparse.ArgumentParser(description='Train survival models for age-specific predictions')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for data subset')
    parser.add_argument('--end_index', type=int, default=10000, help='Ending index for data subset')
    parser.add_argument('--base_data_path', type=str, 
                       default='/opt/ml/input/data/', 
                       help='Base path for input data')
    parser.add_argument('--output_dir', type=str, 
                       default='/opt/ml/output/', 
                       help='Output directory for results')
    parser.add_argument('--max_age_offset', type=int, default=30, 
                       help='Maximum age offset to train for (0-30 corresponds to ages 40-70)')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], 
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    logger.info(f"Training for indices {args.start_index} to {args.end_index}")
    
    # Initialize trainer
    trainer = SurvivalModelTrainer(
        base_data_path=args.base_data_path,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index
    )
    
    # Run training
    try:
        histories = trainer.train_all_ages(max_age_offset=args.max_age_offset)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()