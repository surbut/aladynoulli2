import numpy as np
import torch
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

def find_similar_individuals(model, baseline_time=0, n_neighbors=5):
    """
    Find individuals with similar baseline signature profiles
    
    Parameters:
    -----------
    model : AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
        The trained model
    baseline_time : int
        Time point to use for baseline comparison
    n_neighbors : int
        Number of nearest neighbors to find
        
    Returns:
    --------
    nn : NearestNeighbors
        Fitted nearest neighbors model
    """
    # Get baseline theta values (signature proportions)
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
        baseline_thetas = theta[:, :, baseline_time].numpy()
    
    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
    nn.fit(baseline_thetas)
    
    return nn, baseline_thetas

def analyze_disease_event(model, disease_idx, window_size=5):
    """
    Analyze how signature profiles change around a disease event
    
    Parameters:
    -----------
    model : AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
        The trained model
    disease_idx : int
        Index of the disease to analyze
    window_size : int
        Number of time points before/after event to analyze
        
    Returns:
    --------
    dict with analysis results
    """
    # Get disease events
    Y = model.Y
    event_times = torch.where(Y[:, disease_idx, :])[1]
    patients_with_disease = torch.where(Y[:, disease_idx, :])[0]
    
    if len(event_times) == 0:
        print(f"No events found for disease {disease_idx}")
        return None
    
    # Get signature trajectories
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
    
    # Find most specific signature for this disease
    phi_avg = phi_prob.mean(dim=2)
    spec_d = phi_avg[:, disease_idx]
    max_sig = torch.argmax(spec_d)
    
    # Analyze trajectories around events
    results = {
        'event_times': event_times.numpy(),
        'patients': patients_with_disease.numpy(),
        'specific_signature': max_sig.item(),
        'trajectories': [],
        'baseline_thetas': [],
        'time_windows': []  # Store the actual time windows for each trajectory
    }
    
    for i, (patient, event_time) in enumerate(zip(patients_with_disease, event_times)):
        # Get window around event
        start = max(0, event_time - window_size)
        end = min(model.T, event_time + window_size + 1)
        
        # Get theta trajectory
        traj = theta[patient, :, start:end].numpy()
        results['trajectories'].append(traj)
        results['time_windows'].append((start, end))
        
        # Store baseline theta
        baseline = theta[patient, :, start].numpy()
        results['baseline_thetas'].append(baseline)
    
    return results

def plot_event_analysis(results, disease_name, n_samples=5):
    """
    Plot signature trajectories around disease events
    
    Parameters:
    -----------
    results : dict
        Results from analyze_disease_event
    disease_name : str
        Name of the disease
    n_samples : int
        Number of random samples to plot
    """
    if results is None:
        return
    
    # Sample random patients if we have more than n_samples
    n_patients = len(results['patients'])
    if n_patients > n_samples:
        sample_idx = np.random.choice(n_patients, n_samples, replace=False)
    else:
        sample_idx = np.arange(n_patients)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot individual trajectories
    for i in sample_idx:
        traj = results['trajectories'][i]
        start, end = results['time_windows'][i]
        # Create time points relative to event
        time_points = np.arange(start - results['event_times'][i], 
                              end - results['event_times'][i])
        # Plot specific signature
        axes[0].plot(time_points, traj[results['specific_signature']], 
                    alpha=0.7, label=f'Patient {results["patients"][i]}')
    
    axes[0].axvline(x=0, color='r', linestyle='--', label='Disease Event')
    axes[0].set_title(f'Signature {results["specific_signature"]} Trajectories\nAround {disease_name} Events')
    axes[0].set_xlabel('Time Relative to Event')
    axes[0].set_ylabel('Signature Proportion')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot average trajectory
    # First, create a common time grid
    min_time = min(t[0] - e for t, e in zip(results['time_windows'], results['event_times']))
    max_time = max(t[1] - e for t, e in zip(results['time_windows'], results['event_times']))
    time_grid = np.arange(min_time, max_time + 1)
    
    # Interpolate each trajectory to the common time grid
    interpolated_trajs = []
    for traj, (start, end), event_time in zip(results['trajectories'], 
                                            results['time_windows'], 
                                            results['event_times']):
        traj_times = np.arange(start - event_time, end - event_time)
        traj_values = traj[results['specific_signature']]
        # Interpolate to common time grid
        interp_values = np.interp(time_grid, traj_times, traj_values)
        interpolated_trajs.append(interp_values)
    
    # Convert to array and compute statistics
    all_trajs = np.array(interpolated_trajs)
    mean_traj = all_trajs.mean(axis=0)
    std_traj = all_trajs.std(axis=0)
    
    axes[1].plot(time_grid, mean_traj, 'k-', label='Mean')
    axes[1].fill_between(time_grid, 
                        mean_traj - std_traj,
                        mean_traj + std_traj,
                        alpha=0.2)
    axes[1].axvline(x=0, color='r', linestyle='--', label='Disease Event')
    axes[1].set_title('Average Signature Trajectory')
    axes[1].set_xlabel('Time Relative to Event')
    axes[1].set_ylabel('Signature Proportion')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def find_matched_pairs(model, disease_idx, baseline_time=0, n_neighbors=5):
    """
    Find matched pairs of individuals (one with disease, one without)
    based on baseline signature profiles
    
    Parameters:
    -----------
    model : AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest
        The trained model
    disease_idx : int
        Index of the disease to analyze
    baseline_time : int
        Time point to use for baseline comparison
    n_neighbors : int
        Number of nearest neighbors to find
        
    Returns:
    --------
    dict with matched pairs and their trajectories
    """
    # Get disease events
    Y = model.Y
    event_times = torch.where(Y[:, disease_idx, :])[1]
    patients_with_disease = torch.where(Y[:, disease_idx, :])[0]
    
    if len(event_times) == 0:
        print(f"No events found for disease {disease_idx}")
        return None
    
    # Get baseline theta values
    with torch.no_grad():
        pi, theta, phi_prob = model.forward()
        baseline_thetas = theta[:, :, baseline_time].numpy()
    
    # Find most specific signature
    phi_avg = phi_prob.mean(dim=2)
    spec_d = phi_avg[:, disease_idx]
    max_sig = torch.argmax(spec_d)
    
    # Find matched controls
    nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric='euclidean')
    nn.fit(baseline_thetas)
    
    matched_pairs = []
    for i, (patient, event_time) in enumerate(zip(patients_with_disease, event_times)):
        # Get patient's baseline theta
        patient_theta = baseline_thetas[patient]
        
        # Find nearest neighbors
        distances, indices = nn.kneighbors(patient_theta.reshape(1, -1))
        
        # Find first neighbor that doesn't have the disease
        for idx in indices[0][1:]:  # Skip first (self)
            if not Y[idx, disease_idx, :].any():
                matched_pairs.append({
                    'case': patient.item(),
                    'control': idx,
                    'event_time': event_time.item(),
                    'case_trajectory': theta[patient, max_sig, :].numpy(),
                    'control_trajectory': theta[idx, max_sig, :].numpy()
                })
                break
    
    return {
        'pairs': matched_pairs,
        'specific_signature': max_sig.item()
    }

def plot_matched_pairs(results, disease_name, n_samples=5, T=None):
    """
    Plot trajectories for matched case-control pairs
    
    Parameters:
    -----------
    results : dict
        Results from find_matched_pairs
    disease_name : str
        Name of the disease
    n_samples : int
        Number of random pairs to plot
    T : int
        Number of timepoints (required)
    """
    if results is None or not results['pairs']:
        return
    
    if T is None:
        raise ValueError("You must provide the number of timepoints T.")
    
    # Sample random pairs if we have more than n_samples
    n_pairs = len(results['pairs'])
    if n_pairs > n_samples:
        sample_idx = np.random.choice(n_pairs, n_samples, replace=False)
    else:
        sample_idx = np.arange(n_pairs)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot individual pairs
    time_points = np.arange(T)
    
    for i in sample_idx:
        pair = results['pairs'][i]
        # Plot case trajectory
        axes[0].plot(time_points, pair['case_trajectory'], 
                    alpha=0.7, label=f'Case {pair["case"]}')
        # Plot control trajectory
        axes[0].plot(time_points, pair['control_trajectory'], 
                    alpha=0.7, label=f'Control {pair["control"]}')
        # Mark event time
        axes[0].axvline(x=pair['event_time'], color='r', linestyle='--')
    
    axes[0].set_title(f'Signature {results["specific_signature"]} Trajectories\nfor Matched Pairs ({disease_name})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Signature Proportion')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot average trajectories
    case_trajs = np.array([p['case_trajectory'] for p in results['pairs']])
    control_trajs = np.array([p['control_trajectory'] for p in results['pairs']])
    
    case_mean = case_trajs.mean(axis=0)
    case_std = case_trajs.std(axis=0)
    control_mean = control_trajs.mean(axis=0)
    control_std = control_trajs.std(axis=0)
    
    axes[1].plot(time_points, case_mean, 'r-', label='Cases')
    axes[1].fill_between(time_points, case_mean - case_std, case_mean + case_std, 
                        color='r', alpha=0.2)
    axes[1].plot(time_points, control_mean, 'b-', label='Controls')
    axes[1].fill_between(time_points, control_mean - control_std, control_mean + control_std,
                        color='b', alpha=0.2)
    
    axes[1].set_title('Average Trajectories for Cases and Controls')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Signature Proportion')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 