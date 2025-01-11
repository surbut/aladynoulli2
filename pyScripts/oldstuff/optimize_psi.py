import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from cluster_g_logit_init_acceptpsi import AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest


class PsiOptimizer:
    def __init__(self, N, D, T, K, P, G, Y, prevalence_t, disease_names=None):
        self.N = N
        self.D = D
        self.T = T
        self.K = K
        self.P = P
        self.G = G
        self.Y = Y
        self.prevalence_t = prevalence_t
        self.disease_names = disease_names
        
    def compute_metrics(self, model, val_data):
        """Compute both prediction accuracy and cluster separation metrics"""
        with torch.no_grad():
            pi, theta, phi = model()
            
            # Prediction metrics
            val_r2 = self.compute_r2(val_data, pi)
            
            # Cluster separation metrics
            cluster_sep = self.compute_cluster_separation(model.psi)
            
            return {
                'r2': val_r2,
                'cluster_separation': cluster_sep,
                'combined_score': val_r2 + 0.2 * cluster_sep  # Weighted combination
            }
    
    def compute_r2(self, Y_true, pi):
        """Compute calibrated R² score"""
        ss_res = torch.sum((Y_true - pi) ** 2)
        ss_tot = torch.sum((Y_true - torch.mean(Y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot).item()
        # Apply calibration factor if needed
        calibrated_r2 = r2 * 0.95  # Example calibration factor
        return calibrated_r2
    
    def compute_cluster_separation(self, psi):
        """Measure how well-separated the clusters are"""
        psi_np = psi.detach().cpu().numpy()
        return np.mean(np.max(psi_np, axis=0) - np.min(psi_np, axis=0))
    
    def quick_train(self, model, train_data, val_data, num_epochs=50, param_change_threshold=1e-5):
        """Quick training to evaluate configuration"""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        best_val_score = float('-inf')
        best_metrics = None
        consecutive_small_changes = 0
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = model.compute_loss(train_data)
            loss.backward()
            
            # Check parameter changes
            max_param_change = max(
                (p.grad.abs().max().item() for p in model.parameters() if p.grad is not None),
                default=0
            )
            
            if max_param_change < param_change_threshold:
                consecutive_small_changes += 1
                if consecutive_small_changes >= 3:
                    print(f"Early stopping at epoch {epoch} due to small parameter changes.")
                    break
            else:
                consecutive_small_changes = 0
            
            optimizer.step()
            
            # Evaluate
            if epoch % 10 == 0:
                metrics = self.compute_metrics(model, val_data)
                if metrics['combined_score'] > best_val_score:
                    best_val_score = metrics['combined_score']
                    best_metrics = metrics
        
        return best_metrics
    
    def optimize_psi(self, n_trials=5, quick_epochs=50):
        """Try different psi configurations and find the best"""
        # Split data into train/val
        train_idx, val_idx = train_test_split(range(self.N), test_size=0.2)
        
        # Convert Y to tensor if it's not already
        Y_tensor = torch.tensor(self.Y) if isinstance(self.Y, np.ndarray) else self.Y
        
        # Create event times from Y data
        event_times = torch.full((self.N, self.D), self.T)  # Initialize with T (censoring)
        for n in range(self.N):
            for d in range(self.D):
                events = (Y_tensor[n, d, :] == 1).nonzero()
                if len(events) > 0:
                    event_times[n, d] = events[0]
    
        # Split both Y and event_times
        train_Y = Y_tensor[train_idx]
        val_Y = Y_tensor[val_idx]
        train_events = event_times[train_idx]
        val_events = event_times[val_idx]
        
        # Convert G to tensor if needed
        G_tensor = torch.tensor(self.G) if isinstance(self.G, np.ndarray) else self.G
        
        # Define psi configurations to try
        psi_configs = [
            {'in_cluster': 1.0, 'out_cluster': -3.0, 'noise_in': 0.1, 'noise_out': 0.01},
            {'in_cluster': 2.0, 'out_cluster': -4.0, 'noise_in': 0.1, 'noise_out': 0.01},
            {'in_cluster': 1.5, 'out_cluster': -3.5, 'noise_in': 0.05, 'noise_out': 0.01},
            {'in_cluster': 2.5, 'out_cluster': -3.0, 'noise_in': 0.1, 'noise_out': 0.01},
        ]
        
        results = []
        for config in psi_configs:
            print(f"\nTrying config: {config}")
            
            # Initialize model with this config
            model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
                N=len(train_idx), D=self.D, T=self.T, K=self.K, P=self.P,
                G=G_tensor[train_idx], Y=train_Y, prevalence_t=self.prevalence_t,
                disease_names=self.disease_names
            )
            model.initialize_params(psi_config=config)
            
            # Quick training with event times instead of Y
            metrics = self.quick_train(model, train_events, val_events, num_epochs=quick_epochs)
            
            results.append({
                'config': config,
                'metrics': metrics
            })
            
            print(f"R²: {metrics['r2']:.3f}")
            print(f"Cluster separation: {metrics['cluster_separation']:.3f}")
            print(f"Combined score: {metrics['combined_score']:.3f}")
        
        # Find best config
        best_result = max(results, key=lambda x: x['metrics']['combined_score'])
        
        self.plot_results(results)
        return best_result['config'], results
    
    def plot_results(self, results):
        """Visualize results from different configurations"""
        plt.figure(figsize=(12, 6))
        
        # Plot metrics
        configs = [str(r['config']) for r in results]
        r2_scores = [r['metrics']['r2'] for r in results]
        sep_scores = [r['metrics']['cluster_separation'] for r in results]
        combined_scores = [r['metrics']['combined_score'] for r in results]
        
        x = np.arange(len(configs))
        width = 0.25
        
        plt.bar(x - width, r2_scores, width, label='R²')
        plt.bar(x, sep_scores, width, label='Cluster Separation')
        plt.bar(x + width, combined_scores, width, label='Combined Score')
        
        plt.xlabel('Configuration')
        plt.ylabel('Score')
        plt.title('Comparison of Psi Configurations')
        plt.xticks(x, [f'Config {i+1}' for i in range(len(configs))], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
