import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import expit, softmax
from scipy.linalg import cho_factor, cho_solve, solve
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class Aladynoulli:
    def __init__(self, n_topics, n_individuals, n_diseases, n_timepoints, n_genetics, 
                 length_scales, amplitudes):
        self.K = n_topics
        self.N = n_individuals
        self.D = n_diseases
        self.T = n_timepoints
        self.P = n_genetics
        
        # Setup fixed kernels for each topic
        times = np.arange(self.T)
        self.sq_dists = squareform(pdist(times.reshape(-1, 1))**2)
        
        self.K_lambda = []
        self.K_phi = []
        self.L_lambda = []
        self.L_phi = []
        
        for k in range(self.K):
            # Kernel for individual trajectories in topic k
            K_lambda_k = amplitudes[k] * np.exp(-0.5 * self.sq_dists / (length_scales[k]**2))
            self.K_lambda.append(K_lambda_k + 1e-6 * np.eye(self.T))  # Add small diagonal term
            
            # Cholesky decomposition for faster inversion
            L_lambda_k = np.linalg.cholesky(self.K_lambda[k])
            self.L_lambda.append(L_lambda_k)
            
            # Kernel for disease trajectories in topic k
            self.K_phi.append(self.K_lambda[k])  # Reuse same kernel for simplicity
            self.L_phi.append(L_lambda_k)

    def stable_softmax(self, x, axis=1):
        """Numerically stable softmax implementation"""
        max_x = np.max(x, axis=axis, keepdims=True)
        z = x - max_x
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=axis, keepdims=True) + 1e-10
        return numerator / denominator

    def stable_expit(self, x):
        """Numerically stable sigmoid implementation"""
        return expit(np.clip(x, -15, 15))

    def initialize_params(self, Y, G, prevalence):
        """Initialize parameters using SVD"""
        self.G = G
        Y_avg = Y.mean(axis=2)  # N x D
        
        # SVD initialization
        U, S, Vh = np.linalg.svd(Y_avg, full_matrices=False)
        
        # Get lambda initial values
        lambda_init = U[:, :self.K] @ np.diag(np.sqrt(S[:self.K]))  # N x K
        self.gamma = np.linalg.lstsq(G, lambda_init, rcond=None)[0]  # P x K
        lambda_means = G @ self.gamma  # N x K
        
        # Initialize lambda: N x K x T (pre-softmax values)
        self.lambda_ = np.random.randn(self.N, self.K, self.T)
        
        # Initialize phi: K x D x T (pre-expit values)
        self.logit_prev = np.log(prevalence / (1 - prevalence))  # D
        self.phi_ = np.random.randn(self.K, self.D, self.T)
        
        # Draw from GP priors for each topic
        for k in range(self.K):
            # For topic k:
            # Draw N individual trajectories from shared kernel
            for i in range(self.N):
                mean = lambda_means[i,k]
                eps = np.random.multivariate_normal(np.zeros(self.T), self.K_lambda[k])
                self.lambda_[i,k,:] = mean + eps
            
            # Draw D disease trajectories from shared kernel
            for d in range(self.D):
                mean = self.logit_prev[d]
                eps = np.random.multivariate_normal(np.zeros(self.T), self.K_phi[k])
                self.phi_[k,d,:] = mean + eps
        
        # Clip initial values
        self.lambda_ = np.clip(self.lambda_, -10, 10)
        self.phi_ = np.clip(self.phi_, -10, 10)
        
        self.visualize_initializations()

    def compute_gradients(self, Y, event_times):
        """Compute gradients with numerical stability"""
        # Compute theta with stable softmax
        theta = self.stable_softmax(self.lambda_, axis=1)
        
        # Compute phi probabilities with stable sigmoid
        phi_prob = self.stable_expit(self.phi_)
        
        # Compute pi with clipping to prevent 0/1
        pi = np.clip(np.einsum('nkt,kdt->ndt', theta, phi_prob), 1e-10, 1-1e-10)
        
        # Initialize gradients
        grad_lambda = np.zeros_like(self.lambda_)
        grad_phi = np.zeros_like(self.phi_)
        
        # Compute dL/dpi with stability checks
        dL_dpi = np.zeros_like(pi)
        for n in range(self.N):
            for d in range(self.D):
                t = event_times[n, d]
                if t < self.T:  # Event occurred
                    dL_dpi[n, d, :t] = 1.0 / (1.0 - pi[n, d, :t] + 1e-10)
                    dL_dpi[n, d, t] = -1.0 / (pi[n, d, t] + 1e-10)
                else:  # Censored
                    dL_dpi[n, d, :] = 1.0 / (1.0 - pi[n, d, :] + 1e-10)
        
        # Clip extreme values
        dL_dpi = np.clip(dL_dpi, -1e6, 1e6)
        
        # Compute gradient for lambda
        temp = np.einsum('ndt,kdt->nkt', dL_dpi, phi_prob)
        grad_theta = temp - (theta * np.sum(temp, axis=1, keepdims=True))
        grad_lambda = grad_theta * theta * (1.0 - theta)
        
        # Compute gradient for phi
        temp_phi = np.einsum('ndt,nkt->kdt', dL_dpi, theta)
        grad_phi = temp_phi * phi_prob * (1.0 - phi_prob)
        
        # Final gradient clipping
        grad_lambda = np.clip(grad_lambda, -1e6, 1e6)
        grad_phi = np.clip(grad_phi, -1e6, 1e6)
        
        return grad_lambda, grad_phi

    def compute_mvn_gradients(self, lambda_, phi_, gamma, G, logit_prev):
        """Compute MVN prior gradients with robust error handling"""
        grad_lambda = np.zeros_like(lambda_)
        grad_phi = np.zeros_like(phi_)
        grad_gamma = np.zeros_like(gamma)
        
        lambda_means = G @ gamma
        
        for k in range(self.K):
            # Handle lambda gradients
            deviations_lambda = lambda_[:,k,:] - lambda_means[:,k,None]
            K_reg = self.K_lambda[k] + 1e-6 * np.eye(self.T)
            
            try:
                L = np.linalg.cholesky(K_reg)
                grad_lambda[:,k,:] = -cho_solve((L, True), deviations_lambda.T).T
            except np.linalg.LinAlgError:
                # Fallback to more stable but slower method
                grad_lambda[:,k,:] = -np.linalg.solve(K_reg + 1e-6 * np.eye(self.T), 
                                                    deviations_lambda.T).T
            
            # Handle phi gradients
            deviations_phi = phi_[k,:,:] - logit_prev[:,None]
            try:
                grad_phi[k,:,:] = -cho_solve((L, True), deviations_phi.T).T
            except np.linalg.LinAlgError:
                grad_phi[k,:,:] = -np.linalg.solve(K_reg + 1e-6 * np.eye(self.T), 
                                                 deviations_phi.T).T
            
            # Compute gamma gradients
            grad_gamma[:,k] = G.T @ grad_lambda[:,k,:].sum(axis=1)
        
        # Clip gradients
        grad_lambda = np.clip(grad_lambda, -1e6, 1e6)
        grad_phi = np.clip(grad_phi, -1e6, 1e6)
        grad_gamma = np.clip(grad_gamma, -1e6, 1e6)
        
        return grad_lambda, grad_phi, grad_gamma

    def clip_gradients(self, grad, max_norm=1e3):
        """Clip gradient norm"""
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad

    def compute_loss(self, pi, Y, event_times):
        """Compute loss with numerical stability"""
        pi = np.clip(pi, 1e-10, 1-1e-10)
        nll_loss = 0.0
        
        for n in range(self.N):
            for d in range(self.D):
                t = event_times[n, d]
                if t < self.T:  # Event occurred
                    nll_loss -= np.sum(np.log(1.0 - pi[n, d, :t] + 1e-10))
                    nll_loss -= np.log(pi[n, d, t] + 1e-10)
                else:  # Censored
                    nll_loss -= np.sum(np.log(1.0 - pi[n, d, :] + 1e-10))
        
        gp_loss = self.compute_gp_prior_loss()
        
        # Add smoothness penalty
        smoothness_penalty = 0.0
        for k in range(self.K):
            for i in range(self.N):
                diff = np.diff(self.lambda_[i,k,:])
                smoothness_penalty += 0.1 * np.sum(diff**2)
            for d in range(self.D):
                diff = np.diff(self.phi_[k,d,:])
                smoothness_penalty += 0.1 * np.sum(diff**2)
        
        if not np.isfinite(nll_loss) or not np.isfinite(gp_loss):
            return 1e10
        
        return nll_loss + gp_loss + smoothness_penalty

    def compute_gp_prior_loss(self):
        """Compute GP prior loss with numerical stability"""
        gp_loss = 0.0
        lambda_means = self.G @ self.gamma

        for k in range(self.K):
            K_reg = self.K_lambda[k] + 1e-6 * np.eye(self.T)
            
            # Lambda prior
            for i in range(self.N):
                dev = self.lambda_[i,k,:] - lambda_means[i,k]
                try:
                    L = np.linalg.cholesky(K_reg)
                    alpha = cho_solve((L, True), dev)
                    gp_loss += 0.5 * (dev @ alpha + np.sum(np.log(np.diag(L))))
                except np.linalg.LinAlgError:
                    # Fallback method
                    gp_loss += 0.5 * (dev @ np.linalg.solve(K_reg, dev))
            
            # Phi prior
            for d in range(self.D):
                dev = self.phi_[k,d,:] - self.logit_prev[d]
                try:
                    L = np.linalg.cholesky(K_reg)
                    alpha = cho_solve((L, True), dev)
                    gp_loss += 0.5 * (dev @ alpha + np.sum(np.log(np.diag(L))))
                except np.linalg.LinAlgError:
                    gp_loss += 0.5 * (dev @ np.linalg.solve(K_reg, dev))
        
        return gp_loss

    def fit(self, Y, n_epochs=1000, learning_rate=1e-3):
        """Fit the model with adaptive learning rate and early stopping"""
        losses = []
        event_times = np.full((self.N, self.D), self.T)
        
        # Compute event times
        for n in range(self.N):
            for d in range(self.D):
                events = np.where(Y[n, d] == 1)[0]
                if len(events) > 0:
                    event_times[n, d] = events[0]
        
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        min_lr = 1e-6
        
        for epoch in range(n_epochs):
            # Forward pass
            theta = self.stable_softmax(self.lambda_, axis=1)
            phi_prob = self.stable_expit(self.phi_)
            pi = np.clip(np.einsum('nkt,kdt->ndt', theta, phi_prob), 1e-10, 1-1e-10)
            
            # Compute and clip gradients
            grad_lambda, grad_phi = self.compute_gradients(Y, event_times)
            mvn_grad_lambda, mvn_grad_phi, mvn_grad_gamma = self.compute_mvn_gradients(
                self.lambda_, self.phi_, self.gamma, self.G, self.logit_prev)
            
            grad_lambda = self.clip_gradients(grad_lambda)
            grad_phi = self.clip_gradients(grad_phi)
            mvn_grad_gamma = self.clip_gradients(mvn_grad_gamma)
            
            # Parameter updates
            self.lambda_ -= learning_rate * (grad_lambda + mvn_grad_lambda)
            self.phi_ -= learning_rate * (grad_phi + mvn_grad_phi)
            self.gamma -= learning_rate * mvn_grad_gamma
            
            # Clip parameters
            self.lambda_ = np.clip(self.lambda_, -10, 10)
            self.phi_ = np.clip(self.phi_, -10, 10)
            
            # Compute loss and handle early stopping
            loss = self.compute_loss(pi, Y, event_times)
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                learning_rate *= 0.5
                patience_counter = 0
                if learning_rate < min_lr:
                    print(f"Stopping early at epoch {epoch} due to learning rate decay")
                    break
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Learning Rate: {learning_rate:.6f}")
        
        return {
            'losses': losses,
            'lambda': self.lambda_,
            'phi': self.phi_,
            'gamma': self.gamma
        }

    def visualize_initializations(self):
        """Visualize model parameters"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        # Lambda visualization
        for k in range(self.K):
            axes[0].plot(self.lambda_[0, k, :], label=f'Topic {k+1}')
        axes[0].set_title('Initial Lambda Values (First Individual)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Lambda Value')
        axes[0].legend()

        # Phi visualization
        for k in range(self.K):
            axes[1].plot(self.phi_[k, 0, :], label=f'Topic {k+1}')
        axes[1].set_title('Initial Phi Values (First Disease)')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Phi Value')
        axes[1].legend()

        # Gamma visualization
        im = axes[2].imshow(self.gamma, aspect='auto', cmap='coolwarm')
        axes[2].set_title('Initial Gamma Values')
        axes[2].set_xlabel('Topics')
        axes[2].set_ylabel('Genetic Covariates')
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.show()

    def predict(self, G_new=None):
        """Make predictions for new individuals or existing ones"""
        if G_new is None:
            G_new = self.G
        
        # Compute lambda means for new individuals
        lambda_means = G_new @ self.gamma
        
        # Initialize predictions
        N_new = G_new.shape[0]
        predicted_lambda = np.zeros((N_new, self.K, self.T))
        
        # Generate trajectories for each topic
        for k in range(self.K):
            for i in range(N_new):
                predicted_lambda[i, k, :] = lambda_means[i, k]
        
        # Convert to probabilities
        theta = self.stable_softmax(predicted_lambda, axis=1)
        phi_prob = self.stable_expit(self.phi_)
        
        # Compute final probabilities
        pi = np.clip(np.einsum('nkt,kdt->ndt', theta, phi_prob), 1e-10, 1-1e-10)
        
        return pi, theta, phi_prob

def generate_synthetic_data(N=100, D=5, T=20, K=3, P=10):
    """Generate synthetic data for testing"""
    # Generate genetic covariates
    G = np.random.normal(0, 1, (N, P))
    
    # Generate gamma (genetic effects)
    gamma = np.random.normal(0, 0.5, (P, K))
    
    # Generate base disease prevalence
    prevalence = np.exp(np.random.normal(-3, 0.5, D))
    prevalence = np.clip(prevalence, 0.01, 0.2)
    
    # Setup kernels
    length_scales = np.random.uniform(T/4, T/2, K)
    amplitudes = np.random.uniform(0.5, 1.5, K)
    
    # Initialize model
    model = Aladynoulli(K, N, D, T, P, length_scales, amplitudes)
    
    # Generate true parameters
    lambda_true = np.random.normal(0, 1, (N, K, T))
    phi_true = np.random.normal(0, 1, (K, D, T))
    
    # Generate observations
    theta_true = softmax(lambda_true, axis=1)
    pi_true = np.einsum('nkt,kdt->ndt', theta_true, expit(phi_true))
    Y = np.random.binomial(1, pi_true)
    
    return {
        'Y': Y,
        'G': G,
        'prevalence': prevalence,
        'lambda_true': lambda_true,
        'phi_true': phi_true,
        'gamma_true': gamma,
        'pi_true': pi_true,
        'theta_true': theta_true
    }

def plot_model_fit(model, data, num_individuals=3, num_diseases=3):
    """Plot model fit against true data"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot lambda comparison
    for i in range(min(num_individuals, 3)):
        for k in range(model.K):
            axes[0,0].plot(data['lambda_true'][i,k,:], '--', label=f'True λ_{i}{k}')
            axes[0,1].plot(model.lambda_[i,k,:], '-', label=f'Fitted λ_{i}{k}')
    
    axes[0,0].set_title('True Lambda')
    axes[0,1].set_title('Fitted Lambda')
    axes[0,0].legend()
    axes[0,1].legend()
    
    # Plot phi comparison
    for d in range(min(num_diseases, 3)):
        for k in range(model.K):
            axes[1,0].plot(data['phi_true'][k,d,:], '--', label=f'True φ_{k}{d}')
            axes[1,1].plot(model.phi_[k,d,:], '-', label=f'Fitted φ_{k}{d}')
    
    axes[1,0].set_title('True Phi')
    axes[1,1].set_title('Fitted Phi')
    axes[1,0].legend()
    axes[1,1].legend()
    
    # Plot gamma comparison
    im0 = axes[2,0].imshow(data['gamma_true'], aspect='auto', cmap='coolwarm')
    im1 = axes[2,1].imshow(model.gamma, aspect='auto', cmap='coolwarm')
    axes[2,0].set_title('True Gamma')
    axes[2,1].set_title('Fitted Gamma')
    plt.colorbar(im0, ax=axes[2,0])
    plt.colorbar(im1, ax=axes[2,1])
    
    plt.tight_layout()
    plt.show()


def create_increasing_trend(start, end, T):
    """Create a smooth increasing trend using sigmoid function"""
    x = np.linspace(0, 1, T)
    y = start + (end - start) * expit((x - 0.5) * 10)  # Sharper S-shape with expit
    return y

def generate_synthetic_data(N=100, D=5, T=20, K=3, P=10, seed=42):
    """
    Generate synthetic data for testing Aladynoulli model.

    Parameters:
    -----------
    N : int, number of individuals
    D : int, number of diseases
    T : int, number of timepoints
    K : int, number of topics
    P : int, number of genetic covariates
    seed : int, random seed

    Returns:
    --------
    dict containing:
        Y : binary tensor (N x D x T) of disease occurrences
        G : matrix (N x P) of genetic covariates
        event_times : matrix (N x D) of event times
        var_scales_lambda : array (K,) of variance scales for lambda
        length_scales_lambda : array (K,) of length scales for lambda
        var_scales_phi : array (K,) of variance scales for phi
        length_scales_phi : array (K,) of length scales for phi
        mu_d : matrix (D x T) of baseline disease probabilities
        lambda_true : tensor (N x K x T) of true lambda values
        phi_true : tensor (K x D x T) of true phi values
        gamma_true : matrix (P x K) of true gamma values
        pi_true : tensor (N x D x T) of true probabilities
        theta_true : tensor (N x K x T) of true topic weights
        eta_true : tensor (K x D x T) of true disease loadings
    """
    np.random.seed(seed)

    # Generate time points for covariance calculations
    time_points = np.arange(T)
    time_diff = time_points[:, None] - time_points[None, :]

    # Generate kernel parameters
    length_scales_lambda = np.random.uniform(T/3, T/2, K)
    var_scales_lambda = np.random.uniform(0.8, 1.2, K)
    length_scales_phi = np.random.uniform(T/3, T/2, K)
    var_scales_phi = np.random.uniform(0.8, 1.2, K)

    # Generate baseline disease prevalences (mu_d)
    mu_d = np.zeros((D, T))
    for d in range(D):
        # Create increasing trend from low to moderate prevalence
        base_trend = create_increasing_trend(np.log(0.01), np.log(0.05), T)
        # Add random temporal correlation
        cov_matrix_mu = np.exp(-0.5 * 0.1 * (time_diff ** 2))
        random_effect = multivariate_normal.rvs(mean=np.zeros(T), cov=cov_matrix_mu)
        mu_d[d, :] = base_trend + 0.2 * random_effect  # Scale random effect

    # Generate genetic covariates (G)
    G = np.random.randn(N, P)

    # Initialize arrays
    lambda_true = np.zeros((N, K, T))
    phi_true = np.zeros((K, D, T))
    gamma_true = np.random.randn(P, K) * 0.5  # Genetic effects

    # Generate lambda (individual-topic trajectories)
    for k in range(K):
        # Create kernel matrix for topic k
        cov_matrix = var_scales_lambda[k] * np.exp(
            -0.5 * (time_diff ** 2) / length_scales_lambda[k] ** 2
        )
        
        # Generate mean trajectory for each individual based on genetics
        mean_lambda = G @ gamma_true[:, k]
        
        # Generate trajectories with temporal correlation
        for i in range(N):
            lambda_true[i, k, :] = multivariate_normal.rvs(
                mean=mean_lambda[i] * np.ones(T),
                cov=cov_matrix
            )

    # Convert lambda to theta using softmax
    exp_lambda = np.exp(lambda_true)
    theta_true = exp_lambda / np.sum(exp_lambda, axis=1, keepdims=True)

    # Generate phi (topic-disease trajectories)
    for k in range(K):
        # Create kernel matrix for topic k
        cov_matrix = var_scales_phi[k] * np.exp(
            -0.5 * (time_diff ** 2) / length_scales_phi[k] ** 2
        )
        
        # Generate disease loadings with temporal correlation
        for d in range(D):
            phi_true[k, d, :] = multivariate_normal.rvs(
                mean=mu_d[d, :],
                cov=cov_matrix
            )

    # Convert phi to eta using sigmoid
    eta_true = expit(phi_true)

    # Calculate disease probabilities
    pi_true = np.einsum('nkt,kdt->ndt', theta_true, eta_true)
    pi_true = np.clip(pi_true, 1e-10, 1-1e-10)  # Ensure numerical stability

    # Generate binary observations and event times
    Y = np.zeros((N, D, T), dtype=int)
    event_times = np.full((N, D), T)  # Initialize all events as censored

    for i in range(N):
        for d in range(D):
            for t in range(T):
                # Only generate event if no previous event
                if Y[i, d, :t].sum() == 0:
                    if np.random.rand() < pi_true[i, d, t]:
                        Y[i, d, t] = 1
                        event_times[i, d] = t
                        break  # Stop after first event

    return {
        'Y': Y,
        'G': G,
        'event_times': event_times,
        'var_scales_lambda': var_scales_lambda,
        'length_scales_lambda': length_scales_lambda,
        'var_scales_phi': var_scales_phi,
        'length_scales_phi': length_scales_phi,
        'mu_d': mu_d,
        'lambda_true': lambda_true,
        'phi_true': phi_true,
        'gamma_true': gamma_true,
        'pi_true': pi_true,
        'theta_true': theta_true,
        'eta_true': eta_true,
        'prevalence': expit(mu_d[:, 0])  # Initial prevalence
    }

def plot_synthetic_data(data, num_individuals=3, num_diseases=3):
    """
    Visualize the generated synthetic data

    Parameters:
    -----------
    data : dict, output from generate_synthetic_data
    num_individuals : int, number of individuals to plot
    num_diseases : int, number of diseases to plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Plot lambda trajectories
    for i in range(min(num_individuals, data['lambda_true'].shape[0])):
        for k in range(data['lambda_true'].shape[1]):
            axes[0, 0].plot(data['lambda_true'][i, k, :], 
                            label=f'Individual {i}, Topic {k}')
    axes[0, 0].set_title('Lambda Trajectories')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()

    # Plot theta (normalized weights)
    for i in range(min(num_individuals, data['theta_true'].shape[0])):
        for k in range(data['theta_true'].shape[1]):
            axes[0, 1].plot(data['theta_true'][i, k, :],
                            label=f'Individual {i}, Topic {k}')
    axes[0, 1].set_title('Theta (Topic Weights)')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].legend()

    # Plot phi trajectories
    for d in range(min(num_diseases, data['phi_true'].shape[1])):
        for k in range(data['phi_true'].shape[0]):
            axes[1, 0].plot(data['phi_true'][k, d, :],
                            label=f'Disease {d}, Topic {k}')
    axes[1, 0].set_title('Phi Trajectories')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()

    # Plot eta (disease probabilities)
    for d in range(min(num_diseases, data['eta_true'].shape[1])):
        for k in range(data['eta_true'].shape[0]):
            axes[1, 1].plot(data['eta_true'][k, d, :],
                            label=f'Disease {d}, Topic {k}')
    axes[1, 1].set_title('Eta (Disease Probabilities)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].legend()

    # Plot gamma (genetic effects)
    im = axes[2, 0].imshow(data['gamma_true'], aspect='auto', cmap='coolwarm')
    axes[2, 0].set_title('Gamma (Genetic Effects)')
    axes[2, 0].set_xlabel('Topics')
    axes[2, 0].set_ylabel('Genetic Variables')
    plt.colorbar(im, ax=axes[2, 0])

    # Plot event times
    im = axes[2, 1].imshow(data['Y'].sum(axis=2), aspect='auto', cmap='Reds')
    axes[2, 1].set_title('Disease Occurrences')
    axes[2, 1].set_xlabel('Diseases')
    axes[2, 1].set_ylabel('Individuals')
    plt.colorbar(im, ax=axes[2, 1])

    plt.tight_layout()
    plt.show()




