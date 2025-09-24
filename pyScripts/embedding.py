import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.special import expit, softmax
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HierarchicalEmbeddingModel(nn.Module):
    def __init__(self, N, D, T, K, P, G, Y, R, W, prevalence_t, init_sd_scaler, genetic_scale,
                 embedding_dim=32, signature_references=None, healthy_reference=None, true_psi=None):
        super().__init__()
        
        # Store dimensions
        self.N, self.D, self.T, self.K = N, D, T, K
        self.K_total = K + 1 if healthy_reference is not None else K
        self.P = P
        self.embedding_dim = embedding_dim
        self.gpweight = W
        self.lrtpen = R
        self.jitter = 1e-6
        
        # Fixed kernel parameters (same as your original model)
        self.lambda_length_scale = T/4
        self.phi_length_scale = T/3
        self.init_amplitude = init_sd_scaler
        
        # Store base kernel matrices
        time_points = torch.arange(T, dtype=torch.float32)
        time_diff = time_points[:, None] - time_points[None, :]
        self.base_K_lambda = torch.exp(-0.5 * (time_diff**2) / (self.lambda_length_scale**2))
        self.base_K_phi = torch.exp(-0.5 * (time_diff**2) / (self.phi_length_scale**2))
        
        # Initialize kernels
        self.K_lambda_init = (self.init_amplitude**2) * self.base_K_lambda + self.jitter * torch.eye(T)
        self.K_phi_init = (self.init_amplitude**2) * self.base_K_phi + self.jitter * torch.eye(T)
        
        # HIERARCHICAL EMBEDDINGS
        self.disease_embeddings = nn.Embedding(D, embedding_dim)
        self.signature_embeddings = nn.Embedding(self.K_total, embedding_dim)
        
        # Attention mechanism
        self.attention_matrix = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        
        # Simple affine mapping from attention A to psi per signature
        # psi_{k,d} = psi_scale[k] * A_{d,k} + psi_bias[k]
        self.psi_scale = nn.Parameter(torch.ones(self.K_total))
        self.psi_bias = nn.Parameter(torch.zeros(self.K_total))
        
        # Other parameters (same as original)
        self.G = torch.tensor(G, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        
        # Compute logit prevalence
        # Check if prevalence_t is already logits (negative values) or probabilities (0-1)
        if torch.all(self.prevalence_t <= 0):  # Already logits
            self.logit_prev_t = self.prevalence_t
        else:  # Probabilities, convert to logits
            epsilon = 1e-8
            self.logit_prev_t = torch.log(
                (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
            )
        
        # Store true psi for initialization
        self.true_psi = true_psi
        
        # Initialize other parameters
        self.initialize_other_params(signature_references, genetic_scale, healthy_reference)

        # If we have ground-truth psi, initialize embeddings/attention/projection close to it
        if self.true_psi is not None:
            self._initialize_embeddings_from_true_psi()
        else:
            # No ground truth: use spectral clustering like ALADYNOULLI
            self._initialize_embeddings_from_spectral_clustering()
    
    def initialize_other_params(self, signature_references, genetic_scale, healthy_reference):
        """Initialize lambda, phi, gamma parameters"""
        if signature_references is None:
            self.signature_refs = torch.zeros(self.K)
        else:
            self.signature_refs = torch.tensor(signature_references, dtype=torch.float32)
        
        self.genetic_scale = genetic_scale
        
        if healthy_reference is not None:
            self.healthy_ref = torch.tensor(-5.0, dtype=torch.float32)
        else:
            self.healthy_ref = None
        
        # Initialize other parameters
        gamma_init = torch.zeros((self.P, self.K_total))
        lambda_init = torch.zeros((self.N, self.K_total, self.T))
        phi_init = torch.zeros((self.K_total, self.D, self.T))
        
        # Simple, robust initialization like the simulation
        # Initialize lambda with small random values
        for k in range(self.K):
            for i in range(self.N):
                lambda_init[i, k, :] = torch.randn(self.T) * 0.1  # Simple random initialization
        
        # Initialize phi with small random values
        for k in range(self.K_total):
            for d in range(self.D):
                if self.true_psi is not None:
                    # Use true_psi but keep it small
                    psi_contribution = torch.clamp(self.true_psi[k, d], -1.0, 1.0)
                    phi_init[k, d, :] = self.logit_prev_t[d, :] + psi_contribution + torch.randn(self.T) * 0.1
                else:
                    phi_init[k, d, :] = self.logit_prev_t[d, :] + torch.randn(self.T) * 0.1
                
                # Ensure no extreme values
                phi_init[k, d, :] = torch.clamp(phi_init[k, d, :], -3.0, 3.0)
        
        if self.healthy_ref is not None:
            # Simple initialization for healthy reference too
            for i in range(self.N):
                lambda_init[i, self.K, :] = torch.randn(self.T) * 0.1
            gamma_init[:, self.K] = 0.0
        
        self.gamma = nn.Parameter(gamma_init)
        self.lambda_ = nn.Parameter(lambda_init)
        self.phi = nn.Parameter(phi_init)
        self.kappa = nn.Parameter(torch.ones(1))
    
    def compute_psi_from_embeddings(self):
        """Compute psi using attention only (no contextualization):
        psi_{k,d} = psi_scale[k] * A_{d,k} + psi_bias[k]
        """
        # Get embeddings
        E_d = self.disease_embeddings(torch.arange(self.D))  # [D, L]
        E_k = self.signature_embeddings(torch.arange(self.K_total))  # [K_total, L]
        
        # Compute attention weights
        attention_scores = torch.matmul(
            torch.matmul(E_d, self.attention_matrix),  # [D, L] @ [L, L] = [D, L]
            E_k.T  # [L, K_total]
        ) / np.sqrt(self.embedding_dim)  # [D, K_total]
        
        A = torch.softmax(attention_scores, dim=1)  # [D, K_total]
        # Map attention directly to psi with per-signature affine parameters
        psi = A * self.psi_scale.unsqueeze(0) + self.psi_bias.unsqueeze(0)  # [D, K_total]
        return psi.T  # [K_total, D]

    def _initialize_embeddings_from_true_psi(self):
        """Warm-start embeddings and psi head to approximate true_psi.
        We factorize the attention score matrix S (D x K_total) via truncated SVD so that
        (E_d @ W_a @ E_k^T) ≈ S, with W_a initialized to identity. Then we fit the
        projection vector so that psi ≈ true_psi via least squares on features A_{d,k} * E_d[d].
        """
        with torch.no_grad():
            K_total = self.K_total
            D = self.D
            L = self.embedding_dim
            # Target score matrix S: use true psi transposed (D x K_total)
            true_psi_np = self.true_psi.detach().cpu().numpy() if isinstance(self.true_psi, torch.Tensor) else np.asarray(self.true_psi)
            # If healthy column is missing, pad zeros
            if true_psi_np.shape[0] < K_total:
                pad_rows = K_total - true_psi_np.shape[0]
                true_psi_np = np.vstack([true_psi_np, np.zeros((pad_rows, true_psi_np.shape[1]))])
            S = true_psi_np.T  # [D, K_total]
            # Truncated SVD
            r = min(L, min(D, K_total))
            U, Svals, Vt = np.linalg.svd(S, full_matrices=False)
            U_r = U[:, :r]
            S_r = np.diag(Svals[:r])
            V_r = Vt[:r, :].T  # [K_total, r]
            E_d_init = U_r @ np.sqrt(S_r)
            E_k_init = V_r @ np.sqrt(S_r)
            # If embedding_dim > r, pad with small noise
            if L > r:
                E_d_init = np.concatenate([E_d_init, 0.01 * np.random.randn(D, L - r)], axis=1)
                E_k_init = np.concatenate([E_k_init, 0.01 * np.random.randn(K_total, L - r)], axis=1)
            # Set embeddings and W_a ≈ identity
            self.disease_embeddings.weight.copy_(torch.tensor(E_d_init, dtype=torch.float32))
            self.signature_embeddings.weight.copy_(torch.tensor(E_k_init, dtype=torch.float32))
            eye = np.eye(L, dtype=np.float32)
            self.attention_matrix.copy_(torch.tensor(eye))
            # Build attention A from current params
            E_d = self.disease_embeddings(torch.arange(D))
            E_k = self.signature_embeddings(torch.arange(K_total))
            scores = (E_d @ self.attention_matrix) @ E_k.T / np.sqrt(L)
            A = torch.softmax(scores, dim=1)  # [D, K_total]
            # Fit psi_scale and psi_bias per signature using simple linear regression
            A_np = A.detach().cpu().numpy()  # [D, K_total]
            for k in range(K_total):
                a_k = A_np[:, k]
                y_k = true_psi_np[k, :]
                a_mean = a_k.mean()
                y_mean = y_k.mean()
                var_a = np.var(a_k) + 1e-8
                cov_ay = np.mean((a_k - a_mean) * (y_k - y_mean))
                s_k = cov_ay / var_a
                b_k = y_mean - s_k * a_mean
                self.psi_scale.data[k] = torch.tensor(s_k, dtype=torch.float32)
                self.psi_bias.data[k] = torch.tensor(b_k, dtype=torch.float32)

    def _initialize_embeddings_from_spectral_clustering(self):
        """Initialize embeddings using spectral clustering on disease co-occurrence, like ALADYNOULLI.
        1. Build disease co-occurrence matrix from Y
        2. Spectral clustering to assign diseases to K signatures
        3. Set psi with 1/-3 contrast based on assignments
        4. Initialize embeddings to approximate this psi structure
        """
        with torch.no_grad():
            K_total = self.K_total
            D = self.D
            L = self.embedding_dim
            
            # 1. Build disease co-occurrence matrix (D x D)
            Y_np = self.Y.detach().cpu().numpy()  # [N, D, T]
            # Sum over patients and time to get co-occurrence counts
            cooccur = np.zeros((D, D))
            for i in range(D):
                for j in range(D):
                    if i != j:
                        # Count how many patients have both diseases i and j at any time
                        cooccur[i, j] = np.sum((Y_np[:, i, :].sum(axis=1) > 0) & (Y_np[:, j, :].sum(axis=1) > 0))
                    else:
                        cooccur[i, j] = np.sum(Y_np[:, i, :])  # Self-occurrence
            
            # Normalize by disease prevalence to get relative co-occurrence
            disease_counts = np.sum(Y_np, axis=(0, 2))  # [D]
            disease_counts = np.maximum(disease_counts, 1)  # Avoid division by zero
            cooccur_norm = cooccur / (disease_counts[:, None] + disease_counts[None, :] - cooccur)
            
            # 2. Spectral clustering on co-occurrence matrix
            from sklearn.cluster import SpectralClustering
            from sklearn.preprocessing import StandardScaler
            
            # Use only the K disease signatures (not healthy)
            K_disease = self.K
            
            # Symmetrize and make positive semi-definite
            cooccur_sym = (cooccur_norm + cooccur_norm.T) / 2
            cooccur_sym = np.maximum(cooccur_sym, 0)  # Ensure non-negative
            
            # Spectral clustering
            spectral = SpectralClustering(n_clusters=K_disease, random_state=42, affinity='precomputed')
            disease_labels = spectral.fit_predict(cooccur_sym)
            
            # 3. Set psi with 1/-3 contrast based on assignments
            psi_init = np.full((K_total, D), -3.0, dtype=np.float32)  # Default low
            for k in range(K_disease):
                # Diseases assigned to signature k get high value (1.0)
                mask = (disease_labels == k)
                psi_init[k, mask] = 1.0
            
            # 4. Initialize embeddings to approximate this psi structure
            # Use the same SVD approach as in _initialize_embeddings_from_true_psi
            S = psi_init.T  # [D, K_total]
            r = min(L, min(D, K_total))
            U, Svals, Vt = np.linalg.svd(S, full_matrices=False)
            U_r = U[:, :r]
            S_r = np.diag(Svals[:r])
            V_r = Vt[:r, :].T  # [K_total, r]
            E_d_init = U_r @ np.sqrt(S_r)
            E_k_init = V_r @ np.sqrt(S_r)
            
            # Pad if needed
            if L > r:
                E_d_init = np.concatenate([E_d_init, 0.01 * np.random.randn(D, L - r)], axis=1)
                E_k_init = np.concatenate([E_k_init, 0.01 * np.random.randn(K_total, L - r)], axis=1)
            
            # Set embeddings
            self.disease_embeddings.weight.copy_(torch.tensor(E_d_init, dtype=torch.float32))
            self.signature_embeddings.weight.copy_(torch.tensor(E_k_init, dtype=torch.float32))
            
            # Initialize attention matrix as identity
            eye = np.eye(L, dtype=np.float32)
            self.attention_matrix.copy_(torch.tensor(eye))
            
            # Fit psi_scale and psi_bias to match the spectral clustering psi
            E_d = self.disease_embeddings(torch.arange(D))
            E_k = self.signature_embeddings(torch.arange(K_total))
            scores = (E_d @ self.attention_matrix) @ E_k.T / np.sqrt(L)
            A = torch.softmax(scores, dim=1)  # [D, K_total]
            A_np = A.detach().cpu().numpy()
            for k in range(K_total):
                a_k = A_np[:, k]
                y_k = psi_init[k, :]
                a_mean = a_k.mean()
                y_mean = y_k.mean()
                var_a = np.var(a_k) + 1e-8
                cov_ay = np.mean((a_k - a_mean) * (y_k - y_mean))
                s_k = cov_ay / var_a
                b_k = y_mean - s_k * a_mean
                self.psi_scale.data[k] = torch.tensor(s_k, dtype=torch.float32)
                self.psi_bias.data[k] = torch.tensor(b_k, dtype=torch.float32)
    
    def compute_psi(self):
        """Wrapper for psi computation"""
        return self.compute_psi_from_embeddings()
    
    def forward(self):
        """Forward pass"""
        theta = torch.softmax(self.lambda_, dim=1)
        epsilon = 1e-6
        phi_prob = torch.sigmoid(self.phi)
        pi = torch.einsum('nkt,kdt->ndt', theta, phi_prob) * self.kappa
        pi = torch.clamp(pi, epsilon, 1-epsilon)
        return pi, theta, phi_prob
    
    def compute_loss(self, event_times, simple: bool = False):
        """Loss function.
        If simple=True, use only data loss + light embedding regularization.
        Otherwise, include GP prior and LRT penalty to mirror clust_huge_amp.
        """
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        
        # Original survival loss components (same as clust_huge_amp)
        N, D, T = self.Y.shape
        event_times_tensor = torch.tensor(event_times, dtype=torch.long)
        event_times_expanded = event_times_tensor.unsqueeze(-1)
        time_grid = torch.arange(T).unsqueeze(0).unsqueeze(0)
        mask_before_event = (time_grid < event_times_expanded).float()
        mask_at_event = (time_grid == event_times_expanded).float()
        
        loss_censored = -torch.sum(torch.log(1 - pi) * mask_before_event)
        loss_event = -torch.sum(torch.log(pi) * mask_at_event * self.Y)
        loss_no_event = -torch.sum(torch.log(1 - pi) * mask_at_event * (1 - self.Y))
        total_data_loss = (loss_censored + loss_event + loss_no_event) / self.N

        # Embedding regularization (small addition)
        embedding_reg = 0.001 * (torch.norm(self.disease_embeddings.weight) + 
                                torch.norm(self.signature_embeddings.weight))

        if simple:
            total_loss = total_data_loss + embedding_reg
            return total_loss

        # GP prior loss (same as clust_huge_amp)
        if self.gpweight > 0:
            gp_loss = self.compute_gp_prior_loss()
        else:
            gp_loss = 0.0
            
        # LRT penalty (same as clust_huge_amp)
        signature_update_loss = 0.0
        if self.lrtpen > 0:
            diagnoses = self.Y  # [N x D x T]
            phi_avg = phi_prob.mean(dim=2)  
            for d in range(self.D):
                if torch.any(diagnoses[:, d, :]):
                    spec_d = phi_avg[:, d]
                    max_sig = torch.argmax(spec_d)
                    
                    other_mean = (torch.sum(spec_d) - spec_d[max_sig]) / (self.K_total - 1)
                    lr = spec_d[max_sig] / (other_mean + epsilon)
                    
                    if lr > 2:
                        diagnosis_mask = diagnoses[:, d, :].bool()
                        patient_idx, time_idx = torch.where(diagnosis_mask)
                        lambda_at_diagnosis = self.lambda_[patient_idx, max_sig, time_idx]
                        
                        target_value = 2.0  # This should give ~0.4 theta share
                        disease_prevalence = diagnoses[:, d, :].float().mean() + epsilon
                        prevalence_scaling = min(0.1 / disease_prevalence, 10.0)
                        
                        signature_update_loss += torch.sum(
                            torch.log(lr) * prevalence_scaling * (target_value - lambda_at_diagnosis)
                        )
        
        total_loss = total_data_loss + self.gpweight*gp_loss + self.lrtpen*signature_update_loss / (self.N * self.T) + embedding_reg
        
        return total_loss
    
    def compute_gp_prior_loss(self):
        """GP prior loss matching clust_huge_amp.py"""
        # Initialize losses
        gp_loss_lambda = 0.0
        gp_loss_phi = 0.0
        
        # Compute Cholesky once
        L_lambda = torch.linalg.cholesky(self.K_lambda_init)
        
        # Lambda GP prior
        for k in range(self.K_total):
            lambda_k = self.lambda_[:, k, :]  # N x T
            
            if k == self.K and self.healthy_ref is not None:  # Healthy state
                mean_lambda_k = self.healthy_ref.unsqueeze(0)
            else:  # Disease signatures
                mean_lambda_k = self.signature_refs[k].unsqueeze(0) + \
                            self.genetic_scale * (self.G @ self.gamma[:, k]).unsqueeze(1)
            
            deviations_lambda = lambda_k - mean_lambda_k
            for i in range(self.N):
                dev_i = deviations_lambda[i:i+1].T
                v_i = torch.cholesky_solve(dev_i, L_lambda)
                gp_loss_lambda += 0.5 * torch.sum(v_i.T @ dev_i)
            
        # Phi GP prior (uses learned psi from embeddings)
        L_phi = torch.linalg.cholesky(self.K_phi_init)
        learned_psi = self.compute_psi()  # [K_total, D]
        for k in range(self.K_total):
            phi_k = self.phi[k]  # D x T
            for d in range(self.D):
                mean_phi_d = self.logit_prev_t[d, :] + learned_psi[k, d]
                dev_d = (phi_k[d:d+1, :] - mean_phi_d).T
                v_d = torch.cholesky_solve(dev_d, L_phi)
                gp_loss_phi += 0.5 * torch.sum(v_d.T @ dev_d)
        
        # Return combined loss with appropriate scaling
        return gp_loss_lambda / self.N + gp_loss_phi / self.D
    
    def fit(self, event_times, num_epochs=100, learning_rate=0.01, lambda_reg=0.01, simple_loss: bool = True):
        """Fit method. If simple_loss=True, use simplified loss path."""
        
        # Create parameter groups matching clust_huge_amp
        param_groups = [
            {'params': [self.lambda_], 'lr': learning_rate},
            {'params': [self.phi], 'lr': learning_rate * 0.1},
            {'params': [self.gamma], 'weight_decay': lambda_reg, 'lr': learning_rate},
            # Add embedding parameters
            {'params': list(self.disease_embeddings.parameters()) + list(self.signature_embeddings.parameters()), 'lr': learning_rate * 0.1},
            {'params': [self.attention_matrix, self.psi_scale, self.psi_bias], 'lr': learning_rate * 0.1},
        ]
        
        # Add kappa if it's learnable
        if isinstance(self.kappa, nn.Parameter):
            param_groups.append({'params': [self.kappa], 'lr': learning_rate})
        
        optimizer = torch.optim.Adam(param_groups)
        
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times, simple=simple_loss)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, stopping training")
                break
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        return losses

def run_hierarchical_embedding_simulation():
    """Run a complete simulation demonstrating hierarchical embeddings"""
    
    print("�� Starting Hierarchical Embedding Simulation")
    print("=" * 50)
    
    # Simulation parameters
    N, D, T, K, P = 200, 20, 10, 3, 5
    embedding_dim = 16
    
    print(f"📊 Simulation Setup:")
    print(f"   - {N} patients, {D} diseases, {T} time points")
    print(f"   - {K} signatures, {P} genetic components")
    print(f"   - {embedding_dim}D embedding space")
    
    # Generate synthetic data using your function
    print("\n🔬 Generating synthetic data...")
    
    # Create some realistic signature references
    signature_refs = np.array([
        [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],  # Signature 0: decreasing
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2],  # Signature 1: peak in middle
        [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # Signature 2: increasing
    ])
    
    # Create prevalence patterns
    logit_prev_t = np.random.randn(D, T) * 0.5 - 2.0
    
    # Create gamma (genetic effects)
    gamma = np.random.randn(P, K) * 0.3
    
    # Create psi (disease-signature associations) with stronger dissimilarity
    # Block-sparse structure: each signature has its own disease block with high loadings
    def make_block_sparse_psi(num_signatures, num_diseases,
                              high=1.0, low=-3.0, noise_sd=0.01,
                              crossload_frac=0.05, seed=42):
        rng = np.random.default_rng(seed)
        psi_mat = np.full((num_signatures, num_diseases), low, dtype=float)
        # Equal-size blocks (last block may be larger by remainder)
        base = num_diseases // num_signatures
        remainder = num_diseases % num_signatures
        start = 0
        for k in range(num_signatures):
            size = base + (1 if k < remainder else 0)
            end = start + size
            psi_mat[k, start:end] = high
            start = end
        # Add a few cross-loadings per signature to avoid being perfectly separable
        num_cross = max(1, int(crossload_frac * num_diseases))
        for k in range(num_signatures):
            cross_idx = rng.choice(num_diseases, size=num_cross, replace=False)
            psi_mat[k, cross_idx] = np.maximum(psi_mat[k, cross_idx], high * 0.5)
        # Small noise for realism
        psi_mat += rng.normal(0.0, noise_sd, size=psi_mat.shape)
        return psi_mat

    psi = make_block_sparse_psi(K, D, high=1.0, low=-3.0, noise_sd=0.01, crossload_frac=0.07, seed=42)
    
    # Generate data
    data = generate_clustered_survival_data_from_real(
        N=N, D=D, T=T, K=K, P=P,
        real_signature_refs=signature_refs,
        real_logit_prev_t=logit_prev_t,
        real_gamma=gamma,
        real_psi=psi,
        init_sd_scaler=0.1,
        signature_scale=0.5
    )
    
    print(f"✅ Generated data with {data['Y'].sum()} total events")
    
    # Initialize model
    print("\n🧠 Initializing hierarchical embedding model...")
    model = HierarchicalEmbeddingModel(
        N=N, D=D, T=T, K=K, P=P,
        G=data['G'], Y=data['Y'], R=0, W=0,
        prevalence_t=data['logit_prev_t'],
        init_sd_scaler=0.1, genetic_scale=1.0,
        signature_references=signature_refs,
        healthy_reference=True,
        embedding_dim=embedding_dim
    )
    
    # Train model
    print("\n�� Training model...")
    losses = model.fit(data['event_times'], num_epochs=100, learning_rate=0.01)
    
    # Analyze results
    print("\n�� Analyzing results...")
    
    # Get learned embeddings
    with torch.no_grad():
        disease_embeddings = model.disease_embeddings.weight.numpy()
        signature_embeddings = model.signature_embeddings.weight.numpy()
        learned_psi = model.compute_psi().numpy()
        attention_weights = torch.softmax(
            torch.matmul(
                torch.matmul(model.disease_embeddings.weight, model.attention_matrix),
                model.signature_embeddings.weight.T
            ) / np.sqrt(embedding_dim), dim=1
        ).numpy()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # 2. Disease embeddings (t-SNE)
    tsne = TSNE(n_components=2, random_state=42)
    disease_2d = tsne.fit_transform(disease_embeddings)
    
    scatter = axes[0, 1].scatter(disease_2d[:, 0], disease_2d[:, 1], 
                                c=range(D), cmap='tab10', s=100, alpha=0.7)
    axes[0, 1].set_title('Disease Embeddings (t-SNE)')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    
    # Add disease labels
    for i in range(D):
        axes[0, 1].annotate(f'D{i}', (disease_2d[i, 0], disease_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Signature embeddings
    signature_2d = tsne.fit_transform(signature_embeddings)
    axes[0, 2].scatter(signature_2d[:, 0], signature_2d[:, 1], 
                      c=range(model.K_total), cmap='Set1', s=200, alpha=0.7)
    axes[0, 2].set_title('Signature Embeddings (t-SNE)')
    axes[0, 2].set_xlabel('t-SNE 1')
    axes[0, 2].set_ylabel('t-SNE 2')
    
    # Add signature labels
    sig_labels = [f'Sig{i}' for i in range(model.K)] + ['Healthy']
    for i, label in enumerate(sig_labels):
        axes[0, 2].annotate(label, (signature_2d[i, 0], signature_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # 4. Attention weights heatmap
    im = axes[1, 0].imshow(attention_weights, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Attention Weights (Diseases × Signatures)')
    axes[1, 0].set_xlabel('Signature')
    axes[1, 0].set_ylabel('Disease')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 5. Learned vs True Psi
    true_psi = data['psi']
    axes[1, 1].scatter(true_psi.flatten(), learned_psi.flatten(), alpha=0.6)
    axes[1, 1].plot([true_psi.min(), true_psi.max()], [true_psi.min(), true_psi.max()], 'r--')
    axes[1, 1].set_title('Learned vs True Psi')
    axes[1, 1].set_xlabel('True Psi')
    axes[1, 1].set_ylabel('Learned Psi')
    
    # 6. Attention weights for top diseases
    top_diseases = np.argsort(attention_weights.max(axis=1))[-5:]
    axes[1, 2].bar(range(len(top_diseases)), attention_weights[top_diseases].max(axis=1))
    axes[1, 2].set_title('Max Attention Weights (Top 5 Diseases)')
    axes[1, 2].set_xlabel('Disease Rank')
    axes[1, 2].set_ylabel('Max Attention Weight')
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("\n🔍 Analysis Results:")
    print(f"   - Final loss: {losses[-1]:.4f}")
    print(f"   - Psi correlation: {np.corrcoef(true_psi.flatten(), learned_psi.flatten())[0, 1]:.3f}")
    
    # Find most attended diseases for each signature
    print("\n🎯 Top Diseases per Signature:")
    for k in range(model.K_total):
        top_diseases = np.argsort(attention_weights[:, k])[-3:]
        sig_name = f"Signature {k}" if k < model.K else "Healthy"
        print(f"   {sig_name}: Diseases {top_diseases} (attention: {attention_weights[top_diseases, k]})")
    
    # Show embedding statistics
    print(f"\n📊 Embedding Statistics:")
    print(f"   - Disease embedding norm: {np.linalg.norm(disease_embeddings, axis=1).mean():.3f} ± {np.linalg.norm(disease_embeddings, axis=1).std():.3f}")
    print(f"   - Signature embedding norm: {np.linalg.norm(signature_embeddings, axis=1).mean():.3f} ± {np.linalg.norm(signature_embeddings, axis=1).std():.3f}")
    
    return model, data, losses

# Run the simulation
if __name__ == "__main__":
    model, data, losses = run_hierarchical_embedding_simulation()
    
    print("\n🎉 Simulation Complete!")
    print("The model learned hierarchical embeddings that capture disease-signature relationships!")


def generate_embedding_based_data(N, D, T, K, P, embedding_dim=32, init_sd_scaler=0.1, signature_scale=0.5):
    """Generate data using embedding-based psi computation"""
    
    # Set random seed
    np.random.seed(42)
    
    # Create embeddings with much smaller values to avoid extreme psi
    disease_embeddings = np.random.randn(D, embedding_dim) #* 0.01
    signature_embeddings = np.random.randn(K, embedding_dim) #* 0.01
    
    # Create attention matrix with smaller values
    attention_matrix = np.random.randn(embedding_dim, embedding_dim) * 0.01
    
    # Create projection layer with smaller values
    psi_projection_weight = np.random.randn(1, embedding_dim) * 0.01
    psi_projection_bias = np.random.randn(1) * 0.1
    
    # Compute psi using embedding structure (like the model does)
    E_d = disease_embeddings  # [D, embedding_dim]
    E_k = signature_embeddings  # [K, embedding_dim]

    # Sharper attention via temperature
    temperature = 0.5
    attention_scores = (E_d @ attention_matrix) @ E_k.T
    attention_scores = attention_scores / (np.sqrt(embedding_dim) * temperature)  # [D,K]
    A = np.exp(attention_scores)
    A /= A.sum(axis=1, keepdims=True)

    # Informative psi via bilinear interaction (bounded)
# 3) Larger ψ from bilinear scores (was 0.5 * tanh(.))
    psi_scores = (E_k @ E_d.T) / (np.sqrt(embedding_dim) * 0.5)  # temp=0.5
    psi = 1.5 * np.tanh(psi_scores)  # yields roughly [-1.5, 1.5]
        
    # The function expects logit_prev_t to have more time points than T
    T_original = max(T + 10, 50)  # Generate more time points than needed
    
    # Create other required parameters for generate_clustered_survival_data_from_real
    # Generate signature references (simple patterns) - also need more time points
    signature_refs = np.zeros((K, T_original))
    for k in range(K):
        # Different patterns for each signature
        if k == 0:
            signature_refs[k, :] = np.linspace(1.0, 0.0, T_original)  # Decreasing
        elif k == 1:
            signature_refs[k, :] = np.linspace(0.0, 1.0, T_original)  # Increasing
        elif k == 2:
            # Peak in middle
            peak = T_original // 2
            signature_refs[k, :] = np.exp(-0.5 * ((np.arange(T_original) - peak) / (T_original/4))**2)
        elif k == 3:
            # Two peaks
            peak1, peak2 = T_original // 4, 3 * T_original // 4
            signature_refs[k, :] = (np.exp(-0.5 * ((np.arange(T_original) - peak1) / (T_original/6))**2) + 
                                  np.exp(-0.5 * ((np.arange(T_original) - peak2) / (T_original/6))**2)) / 2
        else:
            # Random pattern
            signature_refs[k, :] = np.random.randn(T_original) * 0.3 + 0.5
    
    # Generate logit prevalence trajectories
    logit_prev_t = np.zeros((D, T_original))
    for d in range(D):
        base_rate = np.random.uniform(-3, -1)
        slope = np.random.uniform(-0.02, 0.02)
        logit_prev_t[d, :] = base_rate + slope * np.arange(T_original)
    
    # Generate gamma (genetic effects)
    gamma = np.random.randn(P, K) * 0.3
    
    # Now generate the data using generate_clustered_survival_data_from_real
    from new_clust import generate_clustered_survival_data_from_real
    
    # Debug: print shapes
    print(f"Debug - signature_refs shape: {signature_refs.shape}")
    print(f"Debug - logit_prev_t shape: {logit_prev_t.shape}")
    print(f"Debug - gamma shape: {gamma.shape}")
    print(f"Debug - psi shape: {psi.shape}")
    print(f"Debug - T: {T}, D: {D}, K: {K}")
    
    data = generate_clustered_survival_data_from_real(
        N=N, D=D, T=T, K=K, P=P,
        real_signature_refs=signature_refs,
        real_logit_prev_t=logit_prev_t,
        real_gamma=gamma,
        real_psi=psi,
        init_sd_scaler=init_sd_scaler,
        signature_scale=signature_scale
    )
    
    # Store the embedding components for reference
    data['disease_embeddings'] = disease_embeddings
    data['signature_embeddings'] = signature_embeddings
    data['attention_matrix'] = attention_matrix
    data['psi_projection_weight'] = psi_projection_weight
    data['psi_projection_bias'] = psi_projection_bias
    
    return data

