import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax, expit
import gdown  # Add this to requirements.txt
import os

class ModelVisualizer:
    def __init__(self, model_state_dict, G=None, disease_names=None):
        # Get model parameters
        self.lambda_ = model_state_dict['lambda_'].detach().numpy()  # Shape (N, K, T)
        self.phi = model_state_dict['phi'].detach().numpy()  # Shape (K, D, T)
        self.psi = model_state_dict['psi'].detach().numpy()  # Shape (K, D)
        
        # Add G and gamma - handle G differently since it's not a tensor
        self.G = G if G is not None else None  # Remove .numpy() since G is already numpy
        self.gamma = model_state_dict['gamma'].detach().numpy() if 'gamma' in model_state_dict else None
        
        # Store disease names
        if hasattr(disease_names, 'values'):
            self.disease_names = disease_names.values.tolist()
        elif hasattr(disease_names, 'tolist'):
            self.disease_names = disease_names.tolist()
        else:
            self.disease_names = disease_names if disease_names is not None else [f"Disease_{i}" for i in range(self.D)]
        
        # Get dimensions
        self.N, self.K, self.T = self.lambda_.shape
        self.D = self.phi.shape[1]
        
        # Pre-compute theta
        self.theta = softmax(self.lambda_, axis=1)  # Shape (N, K, T)
        
        # Add placeholder for genomic data
        self.genomic_data = None
    
    def set_genomic_data(self, genomic_data):
        """Set genomic data for visualization"""
        self.genomic_data = genomic_data
    
    def compute_person_trajectory(self, person_idx, time_idx):
        # Get theta for specific person and timepoint
        theta_t = self.theta[person_idx, :, time_idx]  # Shape (K,)
        
        # Get phi for specific timepoint
        phi_t = self.phi[:, :, time_idx]  # Shape (K, D)
        
        # Compute disease probabilities
        eta_t = expit(phi_t)  # Shape (K, D)
        pi = np.dot(theta_t, eta_t)  # Shape (D,)
        
        return theta_t, pi
    
    def plot_visualization(self, person_idx, time_idx):
        theta, pi = self.compute_person_trajectory(person_idx, time_idx)
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Plot signature proportions
        plt.subplot(221)
        plt.bar(range(self.K), theta)
        plt.title(f'Signature Proportions (Person {person_idx}, Time {time_idx})')
        plt.xlabel('Signature')
        plt.ylabel('Proportion')
        
        # 2. Plot disease probabilities
        plt.subplot(222)
        plt.hist(pi, bins=30)
        plt.title('Distribution of Disease Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        
        # 3. Plot psi heatmap
        plt.subplot(223)
        sns.heatmap(self.psi, cmap='RdBu_r', center=0)
        plt.title('ψ Values (Signature-Disease Associations)')
        plt.xlabel('Disease')
        plt.ylabel('Signature')
        
        # 4. Plot lambda trajectories for this person
        plt.subplot(224)
        for k in range(self.K):
            plt.plot(self.lambda_[person_idx, k, :], 
                    label=f'Signature {k}')
        plt.axvline(x=time_idx, color='r', linestyle='--')
        plt.title(f'λ Trajectories for Person {person_idx}')
        plt.xlabel('Time')
        plt.ylabel('λ Value')
        plt.legend()
        
        plt.tight_layout()
        return fig

    def plot_theta_comparison(self, person_indices, time_idx):
        """Plot theta comparison for multiple people at a given time"""
        n_people = len(person_indices)
        fig, axes = plt.subplots(n_people, 1, figsize=(12, 3*n_people))
        if n_people == 1:
            axes = [axes]
            
        for idx, person_idx in enumerate(person_indices):
            theta_t = self.theta[person_idx, :, time_idx]
            axes[idx].bar(range(self.K), theta_t)
            axes[idx].set_title(f'Person {person_idx} Signature Proportions (Time {time_idx})')
            axes[idx].set_xlabel('Signature')
            axes[idx].set_ylabel('Proportion')
            
        plt.tight_layout()
        return fig
    
    def plot_genomic_heatmap(self, person_idx):
        """Plot genomic data heatmap for a person"""
        if self.genomic_data is None:
            return None
            
        fig, ax = plt.subplots(figsize=(12, 4))
        person_genomics = self.genomic_data[person_idx]
        sns.heatmap(person_genomics.reshape(1, -1), 
                   cmap='RdBu_r',
                   center=0,
                   ax=ax)
        ax.set_title(f'Genomic Profile - Person {person_idx}')
        ax.set_xlabel('Genetic Features')
        ax.set_yticklabels([])
        plt.tight_layout()
        return fig

    def plot_reference_comparison(self, person_idx, time_range=None):
        """Plot lambda and theta values comparing patient-specific vs reference trajectories"""
        if time_range is None:
            time_range = range(self.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot Lambda Values
        ax1.set_title('Lambda Values\n(dashed=reference, solid=patient-specific)')
        for k in range(self.K):
            # Plot reference trajectory (dashed)
            ref_lambda = np.mean(self.lambda_[:, k, :], axis=0)
            ax1.plot(time_range, ref_lambda[time_range], '--', label=f'Ref {k}', alpha=0.5)
            
            # Plot patient-specific trajectory (solid)
            ax1.plot(time_range, self.lambda_[person_idx, k, time_range], 
                    '-', label=f'Patient {person_idx} - State {k}')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Lambda (logit scale)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot Proportions (Theta)
        ax2.set_title('Final Proportions\n(dashed=reference, solid=patient-specific)')
        for k in range(self.K):
            # Plot reference trajectory (dashed)
            ref_theta = np.mean(self.theta[:, k, :], axis=0)
            ax2.plot(time_range, ref_theta[time_range], '--', label=f'Ref {k}', alpha=0.5)
            
            # Plot patient-specific trajectory (solid)
            ax2.plot(time_range, self.theta[person_idx, k, time_range], 
                    '-', label=f'Patient {person_idx} - State {k}')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Proportion')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig

    def plot_genetic_scores(self, person_idx):
        """Plot genetic scores for a person"""
        if self.G is None or self.gamma is None:
            return None
            
        # Calculate genetic effects
        # G[person_idx] is (36,) and gamma is (36, 21)
        # Need to transpose gamma to get (21, 36) for correct multiplication
        genetic_effects = np.dot(self.G[person_idx], self.gamma)  # Shape (21,)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot raw genetic scores
        sns.heatmap(self.G[person_idx].reshape(1, -1), 
                   cmap='RdBu_r',
                   center=0,
                   ax=ax1)
        ax1.set_title(f'Raw Genetic Scores - Person {person_idx}')
        ax1.set_xlabel('Genetic Features (n=36)')
        ax1.set_yticklabels([])
        
        # Plot genetic effects on signatures
        ax2.bar(range(self.K), genetic_effects)
        ax2.set_title('Genetic Effects on Each Signature')
        ax2.set_xlabel('Signature')
        ax2.set_ylabel('Effect Size')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0 for reference
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        
        plt.tight_layout()
        return fig

    def compute_disease_probabilities(self, person_idx):
        """Compute disease probabilities across all timepoints"""
        # Shape: (T, D)
        pi_t = np.zeros((self.T, self.D))
        for t in range(self.T):
            theta_t = self.theta[person_idx, :, t]  # (K,)
            eta_t = expit(self.phi[:, :, t])  # (K, D)
            pi_t[t] = np.dot(theta_t, eta_t)  # (D,)
        return pi_t
    
    def plot_prediction_explanation(self, person_idx, time_idx):
        """Visualize how predictions are generated"""
        theta_t = self.theta[person_idx, :, time_idx]  # (K,)
        eta_t = expit(self.phi[:, :, time_idx])  # (K, D)
        pi_t = np.dot(theta_t, eta_t)  # (D,)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Show theta (signature proportions)
        axes[0,0].bar(range(self.K), theta_t)
        axes[0,0].set_title('Step 1: Signature Proportions (θ)')
        axes[0,0].set_xlabel('Signature')
        axes[0,0].set_ylabel('Proportion')
        
        # 2. Show eta (signature-disease probabilities)
        sns.heatmap(eta_t, ax=axes[0,1], cmap='RdBu_r',
                   xticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto',
                   yticklabels=[f'Sig {k}' for k in range(self.K)])
        axes[0,1].set_title('Step 2: Signature-Disease Probabilities (η)')
        
        # 3. Show individual contributions from each signature
        contributions = np.zeros((self.K, self.D))
        for k in range(self.K):
            contributions[k] = theta_t[k] * eta_t[k]  # Individual contribution from signature k
        
        sns.heatmap(contributions, ax=axes[1,0], cmap='RdBu_r',
                   xticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto',
                   yticklabels=[f'Sig {k}' for k in range(self.K)])
        axes[1,0].set_title('Step 3: Individual Signature Contributions (θₖ × ηₖ)')
        
        # 4. Show final probabilities (sum of contributions)
        axes[1,1].bar(range(self.D), pi_t)
        axes[1,1].set_title('Step 4: Final Disease Probabilities (π = Σₖ θₖηₖ)')
        axes[1,1].set_xlabel('Disease')
        axes[1,1].set_ylabel('Probability')
        if hasattr(self, 'disease_names'):
            plt.xticks(range(self.D), self.disease_names, rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_disease_covariance(self, person_idx):
        """Plot disease probability covariance over time"""
        # Get all disease probabilities over time
        pi_t = self.compute_disease_probabilities(person_idx)  # (T, D)
        
        # Compute covariance
        cov_matrix = np.cov(pi_t.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot covariance heatmap
        sns.heatmap(cov_matrix, ax=ax1, cmap='RdBu_r', center=0,
                   xticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto',
                   yticklabels=self.disease_names if hasattr(self, 'disease_names') else 'auto')
        ax1.set_title('Disease Probability Covariance')
        
        # Plot disease probability trajectories
        for d in range(self.D):
            ax2.plot(range(self.T), pi_t[:, d], 
                    label=self.disease_names[d] if hasattr(self, 'disease_names') else f'Disease {d}')
        ax2.set_title('Disease Probability Trajectories')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Probability')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig

def plot_phi_evolution(phi, clusters=None, disease_names=None):
    """Plot the evolution of phi values over time for each signature."""
    K, D, T = phi.shape
    
    # Convert disease_names to list if it's a DataFrame or Series
    if hasattr(disease_names, 'values'):
        disease_names = disease_names.values.tolist()
    elif hasattr(disease_names, 'tolist'):
        disease_names = disease_names.tolist()
    
    if disease_names is None:
        disease_names = [f"Disease_{i}" for i in range(D)]
    
    if clusters is None or isinstance(clusters, float):
        disease_indices = {k: np.arange(D) for k in range(K)}
    else:
        # Ensure clusters is a numpy array
        clusters = np.array(clusters)
        disease_indices = {k: np.where(clusters == k)[0] for k in range(K)}
    
    fig, axes = plt.subplots(K, 1, figsize=(15, 4*K))
    if K == 1:
        axes = [axes]
    
    for k in range(K):
        cluster_diseases = disease_indices[k]
        if len(cluster_diseases) > 0:
            phi_k = expit(phi[k, cluster_diseases, :])
            
            # Get disease names for this cluster
            cluster_disease_names = [disease_names[i] for i in cluster_diseases]
            
            sns.heatmap(phi_k, ax=axes[k], cmap='RdBu_r',
                       xticklabels=range(T) if k == K-1 else False,
                       yticklabels=cluster_disease_names,
                       cbar_kws={'label': 'Probability'})
            axes[k].set_title(f'Signature {k} Disease Probabilities Over Time')
            axes[k].set_ylabel(f'Cluster {k} Diseases (n={len(cluster_diseases)})')
            
            # Rotate y-axis labels for better readability
            axes[k].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    return fig

def download_model():
    """Download model from Google Drive if not present"""
    model_path = 'models/model.pt'
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(model_path):
        # Replace with your Google Drive shared link
        url = "YOUR_GOOGLE_DRIVE_SHARED_LINK"
        st.info("Downloading model file... This may take a few minutes.")
        gdown.download(url, model_path, quiet=False)
    return model_path

def main():
    st.title("Disease Trajectory Model Visualization")
    
    try:
        model_path = download_model()
        first_model = torch.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Load model state dict and additional data
    model_state_dict = first_model['model_state_dict']
    
    # Load and convert data types as needed
    clusters = np.array(first_model['clusters'])
    G = first_model['G']
    disease_names = first_model['disease_names']
    
    visualizer = ModelVisualizer(model_state_dict, G=G, disease_names=disease_names)
    
    # TODO: Load and set genomic data
    # genomic_data = np.load('path_to_genomic_data.npy')
    # visualizer.set_genomic_data(genomic_data)
    
    # Sidebar controls
    st.sidebar.header("Controls")
    person_idx = st.sidebar.slider("Select Person", 0, visualizer.N-1, 0)
    time_idx = st.sidebar.slider("Select Time Point", 0, visualizer.T-1, 0)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Individual Trajectories", 
        "Phi Evolution", 
        "Trajectory Comparison",
        "Genetic Effects",
        "Predictions Analysis"
    ])
    
    with tab1:
        # Main visualization
        fig = visualizer.plot_visualization(person_idx, time_idx)
        st.pyplot(fig)
        
        # Original sidebar information
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### How to interpret:
        - **Top Left**: Shows the proportion of each signature at the selected time point
        - **Top Right**: Shows the distribution of disease probabilities
        - **Bottom Left**: Shows the signature-disease associations (ψ values)
        - **Bottom Right**: Shows how λ values change over time for this person
        
        The vertical red line in the bottom right plot shows the current time point.
        """)
    
    with tab2:
        st.markdown("### Evolution of Disease Probabilities by Signature")
        phi_fig = plot_phi_evolution(visualizer.phi, clusters, visualizer.disease_names)
        st.pyplot(phi_fig)
        
        st.markdown("""
        ### How to interpret Phi Evolution:
        - Each row shows cluster-specific diseases for each signature
        - The x-axis shows time progression
        - The y-axis shows different diseases
        - The color intensity indicates the probability of each disease at each time point
        - Darker red indicates higher probability, darker blue indicates lower probability
        """)
    
    with tab3:
        st.markdown("### Compare Patient Trajectories with Reference")
        
        # Time range selector
        time_start = st.slider("Start Time", 0, visualizer.T-1, 0)
        time_end = st.slider("End Time", time_start, visualizer.T-1, visualizer.T-1)
        time_range = range(time_start, time_end + 1)
        
        # Plot comparison
        comparison_fig = visualizer.plot_reference_comparison(person_idx, time_range)
        st.pyplot(comparison_fig)
        
        st.markdown("""
        ### How to interpret:
        - Dashed lines show reference (population average) trajectories
        - Solid lines show patient-specific trajectories
        - Left plot shows lambda values (logit scale)
        - Right plot shows proportions (theta values)
        - Different colors represent different disease states/signatures
        """)

    with tab4:
        st.markdown("### Genetic Scores and Effects")
        genetic_fig = visualizer.plot_genetic_scores(person_idx)
        if genetic_fig is not None:
            st.pyplot(genetic_fig)
            
            st.markdown("""
            ### How to interpret Genetic Effects:
            - Top plot shows raw genetic scores for the selected person
            - Bottom plot shows how these genetic scores influence each signature
            - Positive values indicate genetic predisposition towards that signature
            - Negative values indicate genetic protection against that signature
            """)
        else:
            st.info("Genetic data not available")

    with tab5:
        st.markdown("### Understanding Disease Predictions")
        
        st.markdown("""
        Disease probabilities (π) are generated through a matrix multiplication of:
        1. Individual signature proportions (θ)
        2. Signature-specific disease probabilities (η)
        
        This creates personalized disease probability predictions that account for 
        both individual trajectory patterns and signature-disease associations.
        """)
        
        # Show prediction generation
        pred_fig = visualizer.plot_prediction_explanation(person_idx, time_idx)
        st.pyplot(pred_fig)
        
        st.markdown("### Disease Probability Relationships")
        st.markdown("""
        The covariance structure shows how disease probabilities move together over time:
        - Positive covariance (red) indicates diseases that tend to increase/decrease together
        - Negative covariance (blue) indicates diseases that tend to move oppositely
        - The trajectory plot shows how individual disease probabilities evolve
        """)
        
        # Show covariance structure
        cov_fig = visualizer.plot_disease_covariance(person_idx)
        st.pyplot(cov_fig)

if __name__ == "__main__":
    main()