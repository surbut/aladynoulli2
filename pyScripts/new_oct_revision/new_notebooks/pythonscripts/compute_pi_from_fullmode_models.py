"""
Compute pi predictions from full-mode models (joint lambda and phi estimation).

These models are saved at:
  Dropbox/enrollment_retrospective_full/enrollment_model_W0.0001_batch_350000_360000.pt

Formula: pi = kappa * sigmoid(phi) * softmax(lambda)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Add path to import utils
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent.parent / 'pyScripts'))
from utils import calculate_pi_pred, softmax_by_k

def load_fullmode_model(model_path):
    """Load a full-mode model checkpoint and extract lambda, phi, kappa."""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    
    # Extract parameters
    lambda_ = state_dict['lambda_']  # Shape: (N, K, T)
    phi = state_dict['phi']  # Shape: (K, D, T) 
    kappa = state_dict['kappa']  # Scalar or tensor
    
    # Handle kappa if it's a tensor
    if torch.is_tensor(kappa):
        if kappa.numel() == 1:
            kappa = kappa.item()
        else:
            # If kappa is multi-dimensional, take mean or handle appropriately
            kappa = kappa.mean().item() if kappa.dim() > 0 else kappa.item()
    
    print(f"  Lambda shape: {lambda_.shape}")
    print(f"  Phi shape: {phi.shape}")
    print(f"  Kappa: {kappa}")
    
    return lambda_, phi, kappa

def compute_pi_from_fullmode(model_path, output_path=None):
    """
    Compute pi predictions from a full-mode model.
    
    Args:
        model_path: Path to the model checkpoint
        output_path: Optional path to save pi predictions
    
    Returns:
        pi: Tensor of shape (N, D, T)
    """
    # Load model
    lambda_, phi, kappa = load_fullmode_model(model_path)
    
    # Compute pi using the formula: kappa * sigmoid(phi) * softmax(lambda)
    print("\nComputing pi predictions...")
    pi = calculate_pi_pred(lambda_, phi, kappa)
    
    print(f"  Pi shape: {pi.shape}")
    print(f"  Pi range: [{pi.min().item():.6f}, {pi.max().item():.6f}]")
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving pi predictions to: {output_path}")
        torch.save(pi, output_path)
        print("✓ Saved!")
    
    return pi

def process_all_batches(base_dir, start_batch=0, end_batch=40, output_base_dir=None):
    """
    Process all batches and compute pi predictions.
    
    Args:
        base_dir: Base directory containing model files
        start_batch: Starting batch index
        end_batch: Ending batch index (inclusive)
        output_base_dir: Directory to save pi predictions
    """
    base_path = Path(base_dir)
    if output_base_dir:
        output_base = Path(output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)
    
    all_pi = []
    
    for batch_idx in range(start_batch, end_batch + 1):
        start_idx = batch_idx * 10000
        end_idx = (batch_idx + 1) * 10000
        
        model_path = base_path / f"enrollment_model_W0.0001_batch_{start_idx}_{end_idx}.pt"
        
        if not model_path.exists():
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing batch {batch_idx}: {start_idx} to {end_idx}")
        print(f"{'='*80}")
        
        # Compute pi
        pi = compute_pi_from_fullmode(model_path)
        all_pi.append(pi)
        
        # Save individual batch pi
        if output_base_dir:
            output_path = output_base / f"pi_fullmode_batch_{start_idx}_{end_idx}.pt"
            torch.save(pi, output_path)
            print(f"✓ Saved batch pi to: {output_path}")
    
    # Concatenate all batches
    if len(all_pi) > 0:
        print(f"\n{'='*80}")
        print("Concatenating all batches...")
        print(f"{'='*80}")
        pi_full = torch.cat(all_pi, dim=0)
        print(f"Full pi shape: {pi_full.shape}")
        
        if output_base_dir:
            output_path = output_base / "pi_fullmode_400k.pt"
            torch.save(pi_full, output_path)
            print(f"✓ Saved full pi to: {output_path}")
        
        return pi_full
    else:
        print("\n⚠️  No batches processed!")
        return None

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute pi from full-mode models")
    parser.add_argument("--model_path", type=str, 
                       help="Path to a single model checkpoint")
    parser.add_argument("--base_dir", type=str,
                       default="/Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_retrospective_full",
                       help="Base directory containing model files")
    parser.add_argument("--start_batch", type=int, default=0,
                       help="Starting batch index")
    parser.add_argument("--end_batch", type=int, default=40,
                       help="Ending batch index")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save pi predictions")
    parser.add_argument("--single_batch", action="store_true",
                       help="Process a single batch instead of all batches")
    
    args = parser.parse_args()
    
    if args.single_batch and args.model_path:
        # Process single model
        output_path = None
        if args.output_dir:
            output_path = Path(args.output_dir) / Path(args.model_path).stem.replace("model", "pi") + ".pt"
        
        pi = compute_pi_from_fullmode(args.model_path, output_path)
    else:
        # Process all batches
        pi_full = process_all_batches(
            args.base_dir,
            args.start_batch,
            args.end_batch,
            args.output_dir
        )

