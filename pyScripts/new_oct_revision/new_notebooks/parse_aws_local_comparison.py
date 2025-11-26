#!/usr/bin/env python
"""
Parse compare_aws_tolocal file and extract AWS vs Local comparisons
"""

from collections import defaultdict
import re

# Read the file
with open('compare_aws_tolocal', 'r') as f:
    lines = f.readlines()

results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

current_batch = None
current_source = None  # 'aws' or 'local'
current_timeframe = None  # '10yr' or '30yr'
current_section = None  # Track which section we're in

i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Detect batch start
    if 'Processing batch' in line:
        match = re.search(r'Processing batch (\d+):', line)
        if match:
            current_batch = int(match.group(1))
            current_source = None
            current_timeframe = None
    
    # Detect AWS section start
    elif '--- Fixed Phi (retrospective AWS) ---' in line:
        current_source = 'aws'
        current_timeframe = None
    
    # Detect Local section start
    elif '--- Fixed Phi (RETROSPECTIVE) ---' in line:
        current_source = 'local'
        current_timeframe = None
    
    # Detect 10-year predictions
    elif '10 year predictions' in line:
        if current_source:
            current_timeframe = '10yr'
    
    # Detect 30-year predictions
    elif '30 year predictions' in line:
        if current_source:
            current_timeframe = '30yr'
    
    # Detect disease evaluation
    elif line.startswith('Evaluating ') and 'Dynamic' in line:
        # Extract disease name
        disease_match = re.search(r'Evaluating (\w+(?:_\w+)*)', line)
        if disease_match:
            disease = disease_match.group(1)
            
            # Look ahead for AUC line
            if i + 1 < len(lines):
                auc_line = lines[i + 1].strip()
                auc_match = re.search(r'AUC: ([\d.]+)', auc_line)
                if auc_match and current_batch is not None and current_source and current_timeframe:
                    auc = float(auc_match.group(1))
                    results[current_batch][current_source][current_timeframe][disease] = auc
    
    i += 1

# Print comparison
print("="*100)
print("AWS vs LOCAL COMPARISON - 10-YEAR PREDICTIONS")
print("="*100)

all_10yr_diffs = []

for batch_num in sorted(results.keys()):
    print(f"\n{'='*100}")
    print(f"BATCH {batch_num}")
    print(f"{'='*100}")
    
    aws_10yr = results[batch_num]['aws']['10yr']
    local_10yr = results[batch_num]['local']['10yr']
    
    # Get common diseases
    common_diseases = set(aws_10yr.keys()) & set(local_10yr.keys())
    
    if not common_diseases:
        print("No common diseases found")
        continue
    
    print(f"\n{'Disease':<30} {'AWS':<12} {'Local':<12} {'Difference':<12} {'Match':<8}")
    print("-"*100)
    
    differences = []
    for disease in sorted(common_diseases):
        aws_auc = aws_10yr[disease]
        local_auc = local_10yr[disease]
        diff = abs(aws_auc - local_auc)
        differences.append(diff)
        all_10yr_diffs.append(diff)
        match = "✓" if diff < 0.01 else "⚠" if diff < 0.05 else "✗"
        print(f"{disease:<30} {aws_auc:<12.4f} {local_auc:<12.4f} {diff:<12.4f} {match:<8}")
    
    print(f"\nMean difference: {sum(differences)/len(differences):.4f}")
    print(f"Max difference: {max(differences):.4f}")
    print(f"Min difference: {min(differences):.4f}")
    print(f"Diseases with diff < 0.01: {sum(1 for d in differences if d < 0.01)}/{len(differences)}")
    print(f"Diseases with diff < 0.05: {sum(1 for d in differences if d < 0.05)}/{len(differences)}")

if all_10yr_diffs:
    print(f"\n{'='*100}")
    print("OVERALL 10-YEAR SUMMARY (across all batches)")
    print(f"{'='*100}")
    print(f"Mean difference: {sum(all_10yr_diffs)/len(all_10yr_diffs):.4f}")
    print(f"Max difference: {max(all_10yr_diffs):.4f}")
    print(f"Min difference: {min(all_10yr_diffs):.4f}")

print("\n" + "="*100)
print("AWS vs LOCAL COMPARISON - 30-YEAR PREDICTIONS")
print("="*100)

all_30yr_diffs = []

for batch_num in sorted(results.keys()):
    print(f"\n{'='*100}")
    print(f"BATCH {batch_num}")
    print(f"{'='*100}")
    
    aws_30yr = results[batch_num]['aws']['30yr']
    local_30yr = results[batch_num]['local']['30yr']
    
    # Get common diseases
    common_diseases = set(aws_30yr.keys()) & set(local_30yr.keys())
    
    if not common_diseases:
        print("No common diseases found")
        continue
    
    print(f"\n{'Disease':<30} {'AWS':<12} {'Local':<12} {'Difference':<12} {'Match':<8}")
    print("-"*100)
    
    differences = []
    for disease in sorted(common_diseases):
        aws_auc = aws_30yr[disease]
        local_auc = local_30yr[disease]
        diff = abs(aws_auc - local_auc)
        differences.append(diff)
        all_30yr_diffs.append(diff)
        match = "✓" if diff < 0.01 else "⚠" if diff < 0.05 else "✗"
        print(f"{disease:<30} {aws_auc:<12.4f} {local_auc:<12.4f} {diff:<12.4f} {match:<8}")
    
    print(f"\nMean difference: {sum(differences)/len(differences):.4f}")
    print(f"Max difference: {max(differences):.4f}")
    print(f"Min difference: {min(differences):.4f}")
    print(f"Diseases with diff < 0.01: {sum(1 for d in differences if d < 0.01)}/{len(differences)}")
    print(f"Diseases with diff < 0.05: {sum(1 for d in differences if d < 0.05)}/{len(differences)}")

if all_30yr_diffs:
    print(f"\n{'='*100}")
    print("OVERALL 30-YEAR SUMMARY (across all batches)")
    print(f"{'='*100}")
    print(f"Mean difference: {sum(all_30yr_diffs)/len(all_30yr_diffs):.4f}")
    print(f"Max difference: {max(all_30yr_diffs):.4f}")
    print(f"Min difference: {min(all_30yr_diffs):.4f}")
