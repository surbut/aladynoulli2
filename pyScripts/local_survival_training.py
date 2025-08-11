#!/usr/bin/env python3
"""
Local survival model training script for age-specific predictions (ages 40-70).
This script trains models for each age from 40-70, censoring events appropriately.
"""

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

class LocalSurvivalTrainer:
    def __init__(self, data_dir, output_dir, start_index=0, end_index=10000):
        """
        Initialize the local survival model trainer.
        
        Args:
            data_dir (str): Path to the data directory
            output_dir (str): Output directory for results
            start_index (int): Starting index for data subset
            end_index (int): Ending index for data subset
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.start_index = start_index
        self.end_index = end_index
        
        # Create output directory if it doesn't exist
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
            # Load large matrices - adjust paths as needed
            Y_path = os.path.join(self.data_dir, 'Y_tensor.pt')
            E_path = os.path.join(self.data_dir, 'E_matrix.pt')
            G_path = os.path.join(self.data_dir, 'G_matrix.pt')
            
            if not os.path.exists(Y_path):
                raise FileNotFoundError(f"Y_tensor.pt not found at {Y_path}")
            if not os.path.exists(E_path):
                raise FileNotFoundError(f"E_matrix.pt not found at {E_path}")
            if not os.path.exists(G_path):
                raise FileNotFoundError(f"G_matrix.pt not found at {G_path}")
            
            Y = torch.load(Y_path)
            E = torch.load(E_path)
            G = torch.load(G_path)
            
            # Load other components
            essentials_path = os.path.join(self.data_dir, 'model_essentials.pt')
            if not os.path.exists(essentials_path):
                raise FileNotFoundError(f"model_essentials.pt not found at {essentials_path}")
            
            essentials = torch.load(essentials_path)
            
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
        Y_subset = Y[self.start_index:self.end_index]
        E_subset = E[self.start_index:self.end_index]
        G_subset = G[self.start_index:self.end_index]
        
        # Clean up large Y tensor from memory
        del Y
        gc.collect()
        
        # Load references
        refs_path = os.path.join(self.data_dir, 'reference_trajectories.pt')
        if not os.path.exists(refs_path):
            logger.warning(f"reference_trajectories.pt not found at {refs_path}")
            signature_refs = None
        else:
            refs = torch.load(refs_path)
            signature_refs = refs.get('signature_refs', None)
        
        # Load demographic data for sex information
        fh_processed_path = os.path.join(self.data_dir, 'baselinagefamh.csv')
        if not os.path.exists(fh_processed_path):
            logger.warning(f"baselinagefamh.csv not found at {fh_processed_path}")
            # Create dummy sex data for testing
            sex = np.zeros(self.end_index - self.start_index)
        else:
            fh_processed = pd.read_csv(fh_processed_path)
            pce_df_subset = fh_processed.iloc[self.start_index:self.end_index].reset_index(drop=True)
            sex = pce_df_subset['sex'].values
        
        G_with_sex = np.column_stack([G_subset, sex])
        
        # Load pretrained phi and psi
        total_fit_path = os.path.join(self.data_dir, 'enrollment_model_W0.0001_fulldata_sexspecific.pt')
        if not os.path.exists(total_fit_path):
            logger.warning(f"Pretrained model not found at {total_fit_path}")
            # Create dummy phi and psi for testing
            phi_total = np.random.randn(20, Y_subset.shape[1])
            psi_total = np.random.randn(20, Y_subset.shape[2])
        else:
            total_checkpoint = torch.load(total_fit_path, map_location='cpu')
            phi_total = total_checkpoint['model_state_dict']['phi'].cpu().numpy()
            psi_total = total_checkpoint['model_state_dict']['psi'].cpu().numpy()
        
        # Check for initial parameter files
        initial_psi_path = os.path.join(self.data_dir, 'initial_psi_400k.pt')
        initial_clusters_path = os.path.join(self.data_dir, 'initial_clusters_400k.pt')
        
        if not os.path.exists(initial_psi_path):
            logger.warning(f"initial_psi_400k.pt not found at {initial_psi_path}")
        if not os.path.exists(initial_clusters_path):
            logger.warning(f"initial_clusters_400k.pt not found at {initial_clusters_path}")
        
        logger.info(f"Data preparation complete. Subset shape: {Y_subset.shape}")
        
        return {
            'Y_subset': Y_subset,
            'E_subset': E_subset,
            'G_with_sex': G_with_sex,
            'essentials': essentials,
            'signature_refs': signature_refs,
            'phi_total': phi_total,
            'psi_total': psi_total
        }

    def create_age_specific_events(self, E_original, age_offset):
        """Create age-specific event times for given age offset.
        
        This function implements the censoring logic with 10-year lead-up:
        - For age_offset N, we're predicting at age 40 + N
        - We use data from age (30 + N) to (40 + N) for prediction
        - Events are censored at min(actual_event_age, 40 + N)
        - All times are relative to age 30 (so age 30 = time 0)
        """
        E_age_specific = E_original.clone()
        
        total_times_changed = 0
        max_cap_applied = 0
        min_cap_applied = float('inf')
        
        # Fixed starting age is 40 (which is 10 years from 30)
        fixed_starting_age = 40
        lead_up_years = 10
        
        for patient_idx in range(E_age_specific.shape[0]):
                
            # Current prediction age = fixed starting age (40) + age_offset
            prediction_age = fixed_starting_age + age_offset
            
            # Start of lead-up period = prediction_age - lead_up_years
            lead_up_start_age = prediction_age - lead_up_years
            
            # Convert to times relative to age 30
            prediction_time = prediction_age - 30
            lead_up_start_time = lead_up_start_age - 30
            
            max_cap_applied = max(max_cap_applied, prediction_time)
            min_cap_applied = min(min_cap_applied, prediction_time)
            
            # Store original times for this patient
            original_times = E_age_specific[patient_idx, :].clone()
            
            # Cap event times to prediction age (this is the censoring)
            E_age_specific[patient_idx, :] = torch.minimum(
                E_age_specific[patient_idx, :],
                torch.full_like(E_age_specific[patient_idx, :], prediction_time)
            )
            
            # Also filter out events that happened before the lead-up period
            # (events before lead_up_start_time are set to 0 or very small value)
            E_age_specific[patient_idx, :] = torch.where(
                E_age_specific[patient_idx, :] < lead_up_start_time,
                torch.zeros_like(E_age_specific[patient_idx, :]),
                E_age_specific[patient_idx, :]
            )
            
            times_changed = torch.sum(E_age_specific[patient_idx, :] != original_times).item()
            total_times_changed += times_changed
        
        # Log censoring verification
        logger.info(f"Censoring for age offset {age_offset} (prediction at age {fixed_starting_age + age_offset}):")
        logger.info(f"  Lead-up period: age {fixed_starting_age + age_offset - lead_up_years} to {fixed_starting_age + age_offset}")
        logger.info(f"  Time window: {lead_up_start_time:.1f} to {prediction_time:.1f}")
        logger.info(f"  Total event times changed: {total_times_changed}")
        logger.info(f"  Max cap applied: {max_cap_applied:.1f}")
        logger.info(f"  Min cap applied: {min_cap_applied:.1f}")
        
        # Test verification for a few patients
        self.verify_censoring(E_age_specific, age_offset, prediction_time, lead_up_start_time)
        
        return E_age_specific

    def verify_censoring(self, E_age_specific, age_offset, prediction_time, lead_up_start_time):
        """Verify that censoring was applied correctly."""
        logger.info("  Verifying censoring for test patients...")
        
        # Check a few specific patients
        test_patients = [0, 1, 100]  # Check patients 0, 1, and 100
        for test_idx in test_patients:
            if test_idx < E_age_specific.shape[0]:
                # Check max value in this patient's event times
                max_time = torch.max(E_age_specific[test_idx, :]).item()
                
                logger.info(f"    Patient {test_idx}: max_event_time={max_time:.1f}, cap={prediction_time:.1f}")
                
                # Verify cap was applied correctly
                if max_time > prediction_time + 0.01:  # Small tolerance
                    logger.warning(f"      WARNING: Max time {max_time:.1f} exceeds cap {prediction_time:.1f}!")
                else:
                    logger.info(f"      ✓ Censoring applied correctly")
                
                # Check if lead-up filtering was applied
                min_nonzero_time = torch.min(E_age_specific[test_idx, :][E_age_specific[test_idx, :] > 0]).item() if torch.any(E_age_specific[test_idx, :] > 0) else float('inf')
                if min_nonzero_time < lead_up_start_time - 0.01:
                    logger.warning(f"      WARNING: Found event at time {min_nonzero_time:.1f} before lead-up start {lead_up_start_time:.1f}!")
                else:
                    logger.info(f"      ✓ Lead-up filtering applied correctly")

    def train_model_for_age(self, data, age_offset):
        """Train model for specific age offset."""
        logger.info(f"Training model for age offset {age_offset} years")
        
        # Set seeds for reproducibility
        self.set_random_seeds(42)
        
        # Create age-specific event times
        E_age_specific = self.create_age_specific_events(
            data['E_subset'], age_offset
        )
        
        # Initialize fresh model for this age
        logger.info("Initializing model...")
        
        # Import the model class - you'll need to adjust this path
        try:
            from clust_huge_amp_fixedPhi import AladynSurvivalFixedPhi
        except ImportError:
            logger.error("Could not import AladynSurvivalFixedPhi")
            logger.error("Please make sure clust_huge_amp_fixedPhi.py is in your Python path")
            return {"age_offset": age_offset, "status": "failed", "error": "Import error"}
        
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
        
        # Verify phi and psi match (they should be fixed)
        if np.allclose(model.phi.cpu().numpy(), data['phi_total']):
            logger.info("phi matches phi_total!")
        else:
            logger.warning("phi does NOT match phi_total!")
            
        if np.allclose(model.psi.cpu().numpy(), data['psi_total']):
            logger.info("psi matches psi_total!")
        else:
            logger.warning("psi does NOT match phi_total!")
        
        # Train model for this specific age
        logger.info(f"Training model for age offset {age_offset}...")
        
        try:
            history_new = model.fit(
                E_age_specific, 
                num_epochs=200, 
                learning_rate=1e-1, 
                lambda_reg=1e-2
            )
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"age_offset": age_offset, "status": "failed", "error": str(e)}
        
        # Get predictions for this age
        logger.info("Generating predictions...")
        with torch.no_grad():
            pi, _, _ = model.forward()
            
            # Save age-specific predictions
            pi_filename = f"pi_enroll_age_offset_{age_offset}_indices_{self.start_index}_{self.end_index}.pt"
            pi_path = os.path.join(self.output_dir, pi_filename)
            torch.save(pi, pi_path)
            logger.info(f"Saved predictions to {pi_path}")
            
            # Save model
            model_filename = f"model_enroll_age_offset_{age_offset}_indices_{self.start_index}_{self.end_index}.pt"
            model_path = os.path.join(self.output_dir, model_filename)
            torch.save({
                'model_state_dict': model.state_dict(),
                'E': E_age_specific,
                'prevalence_t': model.prevalence_t,
                'logit_prevalence_t': model.logit_prev_t,
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
        
        return {"age_offset": age_offset, "status": "completed", "history": history_new}

    def train_all_ages(self, max_age_offset=30):
        """Train models for all age offsets.
        
        Trains for age offsets 0-30, which corresponds to ages 40-70
        (starting from fixed age 40, which is 10 years from 30).
        """
        logger.info(f"Starting training for batch {self.start_index}-{self.end_index}")
        logger.info(f"Training for age offsets 0-{max_age_offset} (ages 40-{40+max_age_offset}, times 10-{10+max_age_offset})")
        logger.info(f"Each prediction uses 10-year lead-up data window")
        
        # Load and prepare data once
        data = self.load_and_prepare_data()
        
        # Store training histories
        all_histories = {}
        
        for age_offset in range(15, max_age_offset + 1):
            prediction_age = 40 + age_offset
            lead_up_start = 30 + age_offset
            logger.info(f"\n=== Processing age offset {age_offset} years ===")
            logger.info(f"  Prediction at age {prediction_age} using data from age {lead_up_start} to {prediction_age}")
            
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
            'data_shape': data['Y_subset'].shape,
            'histories': all_histories
        }, summary_path)
        
        logger.info(f"Training complete! Summary saved to {summary_path}")
        return all_histories


def main():
    # Set default paths for local execution
    default_data_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running"  # Adjust this path
    default_output_dir = "/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/aladynoulli2_results_offsets_40_70/"  # Adjust this path
    

    
    # Initialize trainer
    trainer = LocalSurvivalTrainer(
        data_dir=default_data_dir,
        output_dir=default_output_dir,
        start_index=0,
        end_index=10000  # Full range
    )
    
    # Run training
    try:
        histories = trainer.train_all_ages(max_age_offset=30)  # Full age range 40-70
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
