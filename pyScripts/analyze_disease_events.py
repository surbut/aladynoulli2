import torch
import numpy as np
from event_analysis import (
    analyze_disease_event,
    plot_event_analysis,
    find_matched_pairs,
    plot_matched_pairs
)

# Load your trained model
model_path = '/Users/sarahurbut/Library/CloudStorage/Dropbox/resultshighamp/results/output_0_10000/model.pt'
checkpoint = torch.load(model_path)

# Initialize model (using your existing initialization code)
from clust_huge_amp import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest

# Load references
refs = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/reference_trajectories.pt')
signature_refs = refs['signature_refs']

# Create model
model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=checkpoint['hyperparameters']['N'],
    D=checkpoint['hyperparameters']['D'],
    T=checkpoint['hyperparameters']['T'],
    K=20,
    P=checkpoint['hyperparameters']['P'],
    G=checkpoint['G'],
    Y=checkpoint['Y'],
    prevalence_t=checkpoint['prevalence_t'],
    disease_names=checkpoint['disease_names'],
    init_sd_scaler=1e-1,
    genetic_scale=1,
    W=0.0001,
    R=0,
    signature_references=signature_refs,
    healthy_reference=True
)

# Load initial psi and clusters
initial_psi = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/initial_psi_400k.pt')
initial_clusters = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox/data_for_running/initial_clusters_400k.pt')

# Initialize model
model.initialize_params(true_psi=initial_psi)
model.clusters = initial_clusters
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Example: Analyze a specific disease
disease_idx = 112  # Example: Myocardial Infarction
disease_name = model.disease_names[disease_idx]

print(f"\nAnalyzing disease: {disease_name}")

# 1. Analyze disease events
print("\nAnalyzing disease events...")
event_results = analyze_disease_event(model, disease_idx, window_size=5)
if event_results:
    plot_event_analysis(event_results, disease_name, n_samples=5)

# 2. Find and analyze matched pairs
print("\nFinding matched pairs...")
matched_results = find_matched_pairs(model, disease_idx, baseline_time=0, n_neighbors=5)
if matched_results:
    plot_matched_pairs(matched_results, disease_name, n_samples=5)

# Print summary statistics
if event_results and matched_results:
    print("\nSummary Statistics:")
    print(f"Number of disease events: {len(event_results['patients'])}")
    print(f"Number of matched pairs: {len(matched_results['pairs'])}")
    print(f"Most specific signature: {event_results['specific_signature']}")
    
    # Calculate average change in signature proportion
    case_trajs = np.array([p['case_trajectory'] for p in matched_results['pairs']])
    control_trajs = np.array([p['control_trajectory'] for p in matched_results['pairs']])
    
    # Calculate change at event time
    event_changes = []
    for pair in matched_results['pairs']:
        event_time = pair['event_time']
        case_change = pair['case_trajectory'][event_time] - pair['case_trajectory'][event_time-1]
        control_change = pair['control_trajectory'][event_time] - pair['control_trajectory'][event_time-1]
        event_changes.append(case_change - control_change)
    
    print(f"\nAverage change in signature proportion at event time: {np.mean(event_changes):.3f} ± {np.std(event_changes):.3f}") 