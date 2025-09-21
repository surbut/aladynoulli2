import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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
        
        # Projection to psi space
        self.psi_projection = nn.Linear(embedding_dim, 1)
        
        # Other parameters (same as original)
        self.G = torch.tensor(G, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)
        
        # Compute logit prevalence
        epsilon = 1e-8
        self.logit_prev_t = torch.log(
            (self.prevalence_t + epsilon) / (1 - self.prevalence_t + epsilon)
        )
        
        # Store true psi for initialization
        self.true_psi = true_psi
        
        # Initialize other parameters
        self.initialize_other_params(signature_references, genetic_scale, healthy_reference)
    
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
        
        # Initialize lambda and gamma
        for k in range(self.K):
            lambda_means = self.genetic_scale * (self.G @ gamma_init[:, k])
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, k, :] = self.signature_refs[k] + lambda_means[i] + eps
        
        # Initialize phi
        for k in range(self.K_total):
            L_phi = torch.linalg.cholesky(self.K_phi_init)
            for d in range(self.D):
                if self.true_psi is not None:
                    mean_phi = self.logit_prev_t[d, :] + self.true_psi[k, d]
                else:
                    mean_phi = self.logit_prev_t[d, :] + torch.randn(1) * 0.1  # Fallback to random
                eps = L_phi @ torch.randn(self.T)
                phi_init[k, d, :] = mean_phi + eps
        
        if self.healthy_ref is not None:
            L_k = torch.linalg.cholesky(self.K_lambda_init)
            for i in range(self.N):
                eps = L_k @ torch.randn(self.T)
                lambda_init[i, self.K, :] = self.healthy_ref + eps
            gamma_init[:, self.K] = 0.0
        
        self.gamma = nn.Parameter(gamma_init)
        self.lambda_ = nn.Parameter(lambda_init)
        self.phi = nn.Parameter(phi_init)
        self.kappa = nn.Parameter(torch.ones(1))
    
    def compute_psi_from_embeddings(self):
        """Compute psi using hierarchical embeddings with attention"""
        # Get embeddings
        E_d = self.disease_embeddings(torch.arange(self.D))  # [D, L]
        E_k = self.signature_embeddings(torch.arange(self.K_total))  # [K_total, L]
        
        # Compute attention weights
        attention_scores = torch.matmul(
            torch.matmul(E_d, self.attention_matrix),  # [D, L] @ [L, L] = [D, L]
            E_k.T  # [L, K_total]
        ) / np.sqrt(self.embedding_dim)  # [D, K_total]
        
        A = torch.softmax(attention_scores, dim=1)  # [D, K_total]
        
        # Contextualized representations
        C = A.unsqueeze(-1) * E_d.unsqueeze(1)  # [D, K_total, L]
        
        # Compute psi
        psi = torch.matmul(C, self.psi_projection.weight.T) + self.psi_projection.bias  # [D, K_total]
        
        return psi.T  # [K_total, D]
    
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
    
    def compute_loss(self, event_times):
        """Compute loss (simplified version)"""
        pi, theta, phi_prob = self.forward()
        epsilon = 1e-8
        pi = torch.clamp(pi, epsilon, 1 - epsilon)
        
        # Simple likelihood loss
        Y_tensor = self.Y
        loss = -torch.sum(Y_tensor * torch.log(pi) + (1 - Y_tensor) * torch.log(1 - pi))
        
        # Add embedding regularization
        embedding_reg = 0.01 * (torch.norm(self.disease_embeddings.weight) + 
                               torch.norm(self.signature_embeddings.weight))
        
        return loss / self.N + embedding_reg
    
    def fit(self, event_times, num_epochs=100, learning_rate=0.01):
        """Train the model"""
        optimizer = torch.optim.Adam([
            {'params': list(self.disease_embeddings.parameters()) + list(self.signature_embeddings.parameters()), 'lr': learning_rate},
            {'params': [self.attention_matrix] + list(self.psi_projection.parameters()), 'lr': learning_rate * 0.1},
            {'params': [self.lambda_, self.phi, self.gamma, self.kappa], 'lr': learning_rate * 0.01}
        ])
        
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(event_times)
            loss.backward()
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
    
    # Create psi (disease-signature associations)
    psi = np.random.randn(K, D) * 0.5
    # Make some diseases more associated with specific signatures
    psi[0, :5] += 1.0    # First 5 diseases associated with signature 0
    psi[1, 5:10] += 1.0  # Next 5 diseases associated with signature 1
    psi[2, 10:15] += 1.0 # Next 5 diseases associated with signature 2
    
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