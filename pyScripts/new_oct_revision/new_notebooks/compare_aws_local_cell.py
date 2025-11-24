# Compare AWS vs Local results per batch
import pandas as pd
import numpy as np

def extract_aucs_from_results(results_list):
    """Extract AUCs from results list into a dictionary by batch and disease"""
    aucs_by_batch = {}
    for result in results_list:
        batch_idx = result['batch_idx']
        if batch_idx not in aucs_by_batch:
            aucs_by_batch[batch_idx] = {}
        for disease, metrics in result.items():
            if disease not in ['batch_idx', 'analysis_type'] and isinstance(metrics, dict):
                if 'auc' in metrics:
                    aucs_by_batch[batch_idx][disease] = metrics['auc']
    return aucs_by_batch

# Extract AUCs for all result types
aws_10yr_aucs = extract_aucs_from_results(aws_10yr_results)
aws_30yr_aucs = extract_aucs_from_results(aws_30yr_results)
aws_static_10yr_aucs = extract_aucs_from_results(aws_static_10yr_results)

local_10yr_aucs = extract_aucs_from_results(fixed_retrospective_10yr_results)
local_30yr_aucs = extract_aucs_from_results(fixed_retrospective_30yr_results)
local_static_10yr_aucs = extract_aucs_from_results(fixed_retrospective_static_10yr_results)

def compare_results(aws_aucs, local_aucs, title):
    """Compare AWS vs Local AUCs per batch"""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    
    all_differences = []
    
    for batch_idx in sorted(set(list(aws_aucs.keys()) + list(local_aucs.keys()))):
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx}")
        print(f"{'='*80}")
        
        aws_batch = aws_aucs.get(batch_idx, {})
        local_batch = local_aucs.get(batch_idx, {})
        
        common_diseases = set(aws_batch.keys()) & set(local_batch.keys())
        
        if not common_diseases:
            print("No common diseases found")
            continue
        
        print(f"\n{'Disease':<30} {'AWS':<12} {'Local':<12} {'Difference':<12} {'Match':<8}")
        print("-"*80)
        
        differences = []
        for disease in sorted(common_diseases):
            aws_auc = aws_batch[disease]
            local_auc = local_batch[disease]
            diff = abs(aws_auc - local_auc)
            differences.append(diff)
            all_differences.append(diff)
            match = "✓" if diff < 0.01 else "⚠" if diff < 0.05 else "✗"
            print(f"{disease:<30} {aws_auc:<12.4f} {local_auc:<12.4f} {diff:<12.4f} {match:<8}")
        
        print(f"\nBatch {batch_idx} Summary:")
        print(f"  Mean difference: {sum(differences)/len(differences):.4f}")
        print(f"  Max difference: {max(differences):.4f}")
        print(f"  Min difference: {min(differences):.4f}")
        print(f"  Diseases with diff < 0.01: {sum(1 for d in differences if d < 0.01)}/{len(differences)}")
        print(f"  Diseases with diff < 0.05: {sum(1 for d in differences if d < 0.05)}/{len(differences)}")
    
    if all_differences:
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY ({title})")
        print(f"{'='*80}")
        print(f"Mean difference: {sum(all_differences)/len(all_differences):.4f}")
        print(f"Max difference: {max(all_differences):.4f}")
        print(f"Min difference: {min(all_differences):.4f}")
        print(f"Median difference: {np.median(all_differences):.4f}")
        print(f"Std difference: {np.std(all_differences):.4f}")
        print(f"Total comparisons: {len(all_differences)}")
        print(f"Diseases with diff < 0.01: {sum(1 for d in all_differences if d < 0.01)}/{len(all_differences)} ({100*sum(1 for d in all_differences if d < 0.01)/len(all_differences):.1f}%)")
        print(f"Diseases with diff < 0.05: {sum(1 for d in all_differences if d < 0.05)}/{len(all_differences)} ({100*sum(1 for d in all_differences if d < 0.05)/len(all_differences):.1f}%)")

# Compare 10-year predictions
compare_results(aws_10yr_aucs, local_10yr_aucs, "AWS vs LOCAL - 10-YEAR PREDICTIONS")

# Compare 30-year predictions
compare_results(aws_30yr_aucs, local_30yr_aucs, "AWS vs LOCAL - 30-YEAR PREDICTIONS")

# Compare static 10-year predictions
compare_results(aws_static_10yr_aucs, local_static_10yr_aucs, "AWS vs LOCAL - STATIC 10-YEAR PREDICTIONS")

