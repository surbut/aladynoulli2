import torch
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.special import logit
from sklearn.cluster import SpectralClustering
from clust_newlambda import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest, generate_clustered_survival_data

def create_reference_trajectories(Y, initial_clusters, K, healthy_prop=0, frac=0.3):
    """Create reference trajectories using LOWESS smoothing on logit scale"""
    T = Y.shape[2]
    
    # Get raw counts and proportions
    Y_counts = Y.sum(dim=0)  
    signature_props = torch.zeros(K, T)
    total_counts = Y_counts.sum(dim=0) + 1e-8
    
    for k in range(K):
        cluster_mask = (initial_clusters == k)
        signature_props[k] = Y_counts[cluster_mask].sum(dim=0) / total_counts
    
    # Normalize and clamp
    signature_props = torch.clamp(signature_props, min=1e-8, max=1-1e-8)
    signature_props = signature_props / signature_props.sum(dim=0, keepdim=True)
    signature_props *= (1 - healthy_prop)
    
    # Convert to logit and smooth
    logit_props = torch.tensor(logit(signature_props.numpy()))
    signature_refs = torch.zeros_like(logit_props)
    
    times = np.arange(T)
    for k in range(K):
        smoothed = lowess(
            logit_props[k].numpy(), 
            times,
            frac=frac,
            it=3,
            delta=0.0,
            return_sorted=False
        )
        signature_refs[k] = torch.tensor(smoothed)
    
    healthy_ref = torch.ones(T) * logit(torch.tensor(healthy_prop))
    
    return signature_refs, healthy_ref

def analyze_lambda_responses(model, Y, E):
    """
    Analyze if lambdas increase for disease-specific signatures when diagnoses occur
    """
    # Get model predictions
    lambda_vals = model.lambda_.detach()  # N x K x T
    pi, theta, phi_prob = model.forward()
    N, K, T = lambda_vals.shape
    
    print("\nAnalyzing lambda responses to disease events:")
    
    # For each disease
    for d in range(Y.shape[1]):
        # Find the most specific signature for this disease
        spec_d = phi_prob[:, d, :].mean(dim=1)  # Average over time
        max_sig = torch.argmax(spec_d)
        other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (K - 1)
        specificity_ratio = spec_d[max_sig] / (other_mean + 1e-8)
        
        if specificity_ratio > 2:  # Same threshold as in loss
            # Track lambda changes around disease events
            lambda_changes = []
            lambda_changes_other = []  # Changes in other signatures
            
            # For each patient with this disease
            for n in range(N):
                event_time = E[n,d].item()
                if event_time < T-1 and event_time > 0:  # Real event, not censoring
                    if Y[n,d,event_time] == 1:  # Verify it's a real event
                        # Get lambda changes for specific signature
                        before = lambda_vals[n, max_sig, event_time-1]
                        after = lambda_vals[n, max_sig, event_time]
                        change = after - before
                        lambda_changes.append(change.item())
                        
                        # Get average lambda changes for other signatures
                        other_sigs = [k for k in range(K) if k != max_sig]
                        other_changes = []
                        for k in other_sigs:
                            before = lambda_vals[n, k, event_time-1]
                            after = lambda_vals[n, k, event_time]
                            other_changes.append((after - before).item())
                        lambda_changes_other.append(np.mean(other_changes))
            
            if lambda_changes:
                changes = torch.tensor(lambda_changes)
                changes_other = torch.tensor(lambda_changes_other)
                
                print(f"\nDisease {d} (most specific signature: {max_sig}, specificity ratio: {specificity_ratio:.2f}):")
                print(f"Specific signature lambda changes:")
                print(f"  Mean change: {changes.mean():.3e}")
                print(f"  Percent positive: {(changes > 0).float().mean()*100:.1f}%")
                print(f"Other signatures average changes:")
                print(f"  Mean change: {changes_other.mean():.3e}")
                print(f"  Percent positive: {(changes_other > 0).float().mean()*100:.1f}%")
                print(f"Number of events analyzed: {len(changes)}")

def test_model():
    """Test model with proper initialization from synthetic data"""
    # Generate synthetic data with known patterns
    data = generate_clustered_survival_data(N=1000, D=20, T=50, K=5, P=10)
    
    # Convert to tensors
    Y = torch.tensor(data['Y'], dtype=torch.float32)
    G = torch.tensor(data['G'], dtype=torch.float32)
    E = torch.tensor(data['event_times'], dtype=torch.long)
    
    # Get disease correlations for clustering
    Y_avg = torch.mean(Y, dim=2)  # Average over time
    Y_corr = torch.corrcoef(Y_avg.T)
    similarity = (Y_corr + 1) / 2
    
    # Do spectral clustering
    clustering = SpectralClustering(
        n_clusters=5,
        affinity='precomputed',
        random_state=42
    ).fit(similarity.numpy())
    
    initial_clusters = clustering.labels_

    # Initialize and train model
    model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
        N=Y.shape[0], D=Y.shape[1], T=Y.shape[2], K=5,
        P=G.shape[1], G=G, Y=Y,
        prevalence_t=Y.mean(dim=0),
        init_var_scaler=0.1,
        genetic_scale=1.0,
        flat_lambda=True
    )

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        loss = model.compute_loss(E)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            # Add lambda response analysis every 10 epochs
            analyze_lambda_responses(model, Y, E)
            
    return model, losses, data

if __name__ == "__main__":
    model, losses, true_data = test_model()