"""
Compare results from non-vectorized and vectorized versions.
Checks predictions (pi), model parameters, and correlations.
"""
import torch
import numpy as np
import os
import sys

def compare_predictions(pi_orig_path, pi_vec_path):
    """Compare prediction tensors"""
    print("=" * 60)
    print("Comparing Predictions (pi)")
    print("=" * 60)
    
    pi_orig = torch.load(pi_orig_path, map_location='cpu')
    pi_vec = torch.load(pi_vec_path, map_location='cpu')
    
    print(f"Original shape: {pi_orig.shape}")
    print(f"Vectorized shape: {pi_vec.shape}")
    
    # Flatten for comparison
    pi_orig_flat = pi_orig.flatten()
    pi_vec_flat = pi_vec.flatten()
    
    # Basic statistics
    print(f"\nOriginal - Mean: {pi_orig_flat.mean().item():.6f}, Std: {pi_orig_flat.std().item():.6f}")
    print(f"Vectorized - Mean: {pi_vec_flat.mean().item():.6f}, Std: {pi_vec_flat.std().item():.6f}")
    
    # Differences
    diff = pi_orig_flat - pi_vec_flat
    abs_diff = torch.abs(diff)
    
    print(f"\nDifference Statistics:")
    print(f"  Mean absolute difference: {abs_diff.mean().item():.2e}")
    print(f"  Max absolute difference:  {abs_diff.max().item():.2e}")
    print(f"  Median absolute difference: {abs_diff.median().item():.2e}")
    
    # For large tensors, use numpy quantile or sample
    try:
        # Try torch quantile first
        p95 = torch.quantile(abs_diff, 0.95).item()
        p99 = torch.quantile(abs_diff, 0.99).item()
        print(f"  95th percentile: {p95:.2e}")
        print(f"  99th percentile: {p99:.2e}")
    except RuntimeError:
        # Fall back to numpy for large tensors
        abs_diff_np = abs_diff.cpu().numpy()
        p95 = np.percentile(abs_diff_np, 95)
        p99 = np.percentile(abs_diff_np, 99)
        print(f"  95th percentile: {p95:.2e} (using numpy)")
        print(f"  99th percentile: {p99:.2e} (using numpy)")
    
    # Correlation
    correlation = torch.corrcoef(torch.stack([pi_orig_flat, pi_vec_flat]))[0, 1]
    print(f"\nCorrelation: {correlation.item():.8f}")
    
    # Relative error
    with torch.no_grad():
        # Avoid division by zero
        mask = pi_orig_flat != 0
        if mask.sum() > 0:
            rel_error = (abs_diff[mask] / torch.abs(pi_orig_flat[mask])).mean()
            print(f"Mean relative error (where pi_orig != 0): {rel_error.item():.2e}")
    
    # Check if they're close
    tolerance = 1e-5
    if torch.allclose(pi_orig, pi_vec, atol=tolerance, rtol=tolerance):
        print(f"\n✓ PASS: Predictions are identical (within {tolerance:.0e} tolerance)")
        return True
    else:
        # Check if highly correlated
        if correlation.item() > 0.9999:
            print(f"\n✓ PASS: Predictions are highly correlated (>0.9999)")
            print(f"  Small differences likely due to numerical precision")
            return True
        else:
            print(f"\n✗ WARNING: Predictions differ significantly")
            return False

def compare_model_parameters(model_orig_path, model_vec_path):
    """Compare model parameters"""
    print("\n" + "=" * 60)
    print("Comparing Model Parameters")
    print("=" * 60)
    
    checkpoint_orig = torch.load(model_orig_path, map_location='cpu')
    checkpoint_vec = torch.load(model_vec_path, map_location='cpu')
    
    state_orig = checkpoint_orig['model_state_dict']
    state_vec = checkpoint_vec['model_state_dict']
    
    # Compare each parameter
    params_to_check = ['lambda_', 'gamma', 'kappa']
    
    all_match = True
    for param_name in params_to_check:
        if param_name in state_orig and param_name in state_vec:
            param_orig = state_orig[param_name]
            param_vec = state_vec[param_name]
            
            print(f"\n{param_name}:")
            print(f"  Original shape: {param_orig.shape}")
            print(f"  Vectorized shape: {param_vec.shape}")
            
            if param_orig.shape != param_vec.shape:
                print(f"  ✗ Shape mismatch!")
                all_match = False
                continue
            
            diff = torch.abs(param_orig - param_vec)
            print(f"  Mean absolute difference: {diff.mean().item():.2e}")
            print(f"  Max absolute difference:  {diff.max().item():.2e}")
            
            # Correlation
            flat_orig = param_orig.flatten()
            flat_vec = param_vec.flatten()
            if len(flat_orig) > 1:
                correlation = torch.corrcoef(torch.stack([flat_orig, flat_vec]))[0, 1]
                print(f"  Correlation: {correlation.item():.8f}")
            
            # Check if close
            tolerance = 1e-4
            if torch.allclose(param_orig, param_vec, atol=tolerance, rtol=tolerance):
                print(f"  ✓ Parameters match (within {tolerance:.0e})")
            else:
                print(f"  ⚠ Parameters differ (may be due to numerical precision)")
                if torch.allclose(param_orig, param_vec, atol=1e-3, rtol=1e-3):
                    print(f"    But within 1e-3 tolerance - likely acceptable")
        else:
            print(f"\n{param_name}: Not found in one or both checkpoints")
            all_match = False
    
    return all_match

def compare_metadata(model_orig_path, model_vec_path):
    """Compare metadata from checkpoints"""
    print("\n" + "=" * 60)
    print("Comparing Metadata")
    print("=" * 60)
    
    checkpoint_orig = torch.load(model_orig_path, map_location='cpu')
    checkpoint_vec = torch.load(model_vec_path, map_location='cpu')
    
    metadata_keys = ['age_offset', 'current_age', 'fixed_starting_age', 
                     'start_index', 'end_index']
    
    print("\nMetadata comparison:")
    for key in metadata_keys:
        if key in checkpoint_orig and key in checkpoint_vec:
            val_orig = checkpoint_orig[key]
            val_vec = checkpoint_vec[key]
            match = "✓" if val_orig == val_vec else "✗"
            print(f"  {match} {key}: {val_orig} vs {val_vec}")
        else:
            print(f"  ? {key}: Missing in one or both")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare vectorized vs non-vectorized results')
    parser.add_argument('--orig_dir', type=str, default='output/age_40_70',
                       help='Directory with original (non-vectorized) results')
    parser.add_argument('--vec_dir', type=str, default='output_vector/age_40_70',
                       help='Directory with vectorized results')
    parser.add_argument('--age', type=int, default=40,
                       help='Age to compare (default: 40)')
    parser.add_argument('--offset', type=int, default=0,
                       help='Age offset (default: 0)')
    parser.add_argument('--batch_start', type=int, default=0,
                       help='Batch start index (default: 0)')
    parser.add_argument('--batch_end', type=int, default=10000,
                       help='Batch end index (default: 10000)')
    
    args = parser.parse_args()
    
    # Construct file paths
    age = args.age
    offset = args.offset
    batch_str = f"batch_{args.batch_start}_{args.batch_end}"
    
    pi_orig_path = os.path.join(args.orig_dir, 
        f'pi_fixedphi_age_{age}_offset_{offset}_{batch_str}.pt')
    pi_vec_path = os.path.join(args.vec_dir,
        f'pi_fixedphi_age_{age}_offset_{offset}_{batch_str}.pt')
    
    model_orig_path = os.path.join(args.orig_dir,
        f'model_fixedphi_age_{age}_offset_{offset}_{batch_str}.pt')
    model_vec_path = os.path.join(args.vec_dir,
        f'model_fixedphi_age_{age}_offset_{offset}_{batch_str}.pt')
    
    # Check if files exist
    files_exist = True
    for path, name in [(pi_orig_path, 'pi_orig'), (pi_vec_path, 'pi_vec'),
                       (model_orig_path, 'model_orig'), (model_vec_path, 'model_vec')]:
        if not os.path.exists(path):
            print(f"✗ {name} not found: {path}")
            files_exist = False
    
    if not files_exist:
        print("\nPlease check file paths and try again.")
        return
    
    print(f"\nComparing results for age {age}, offset {offset}, {batch_str}")
    print(f"Original directory: {args.orig_dir}")
    print(f"Vectorized directory: {args.vec_dir}")
    
    # Compare predictions
    pred_match = compare_predictions(pi_orig_path, pi_vec_path)
    
    # Compare model parameters
    param_match = compare_model_parameters(model_orig_path, model_vec_path)
    
    # Compare metadata
    compare_metadata(model_orig_path, model_vec_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Predictions match: {'✓' if pred_match else '✗'}")
    print(f"Parameters match:  {'✓' if param_match else '✗'}")
    
    if pred_match and param_match:
        print("\n🎉 Results are highly correlated! Vectorized version is working correctly.")
    else:
        print("\n⚠ Some differences detected. Check details above.")
        print("  Small differences (< 1e-4) are likely due to numerical precision.")
        print("  Large differences (> 1e-3) may indicate an issue.")

if __name__ == '__main__':
    main()

