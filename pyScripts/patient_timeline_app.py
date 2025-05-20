import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax, expit
import os

def load_model():
    """Load the model and return necessary components"""
    model = torch.load('/Users/sarahurbut/Library/Cloudstorage/Dropbox/resultshighamp/results/output_0_10000/model.pt')
    return {
        'model_state_dict': model['model_state_dict'],
        'clusters': np.array(model['clusters']),
        'G': model['G'],
        'disease_names': model['disease_names'],
        'Y': model.get('Y', None)
    }

class PatientTimelineVisualizer:
    def __init__(self, model_data):
        # Extract model parameters
        self.lambda_ = model_data['model_state_dict']['lambda_'].detach().numpy()
        self.phi = model_data['model_state_dict']['phi'].detach().numpy()
        self.psi = model_data['model_state_dict']['psi'].detach().numpy()
        self.G = model_data['G']
        self.gamma = model_data['model_state_dict'].get('gamma', None)
        if self.gamma is not None:
            self.gamma = self.gamma.detach().numpy()
        
        # Store disease names
        self.disease_names = model_data['disease_names']
        if hasattr(self.disease_names, 'values'):
            self.disease_names = self.disease_names.values.tolist()
        
        # Get dimensions
        self.N, self.K, self.T = self.lambda_.shape
        self.D = self.phi.shape[1]
        
        # Pre-compute theta
        self.theta = softmax(self.lambda_, axis=1)
        
        # Store Y if available
        self.Y = model_data.get('Y', None)
        if self.Y is not None and torch.is_tensor(self.Y):
            self.Y = self.Y.detach().numpy()

    def get_diagnosis_times(self, person_idx):
        """Return a dict of diagnosis times for each disease for this person."""
        diagnosis_times = {}
        if self.Y is not None:
            for d in range(self.D):
                times = np.where(self.Y[person_idx, d, :] > 0.5)[0]
                if len(times) > 0:
                    diagnosis_times[d] = times
        return diagnosis_times

    def plot_signature_timeline(self, person_idx, time_window=None):
        """Plot signature proportions over time with genetic impact and overlay diagnosis timing."""
        if time_window is None:
            time_window = range(self.T)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        # Plot 1: Signature proportions over time
        for k in range(self.K):
            ax1.plot(time_window, self.theta[person_idx, k, time_window], 
                    label=f'Signature {k}', linewidth=2)
        ax1.set_title('Signature Proportions Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Proportion')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        # Overlay diagnosis timing (vertical lines or shaded regions)
        diagnosis_times = self.get_diagnosis_times(person_idx)
        all_diag_times = sorted(set(t for times in diagnosis_times.values() for t in times if t in time_window))
        for t in all_diag_times:
            ax1.axvline(x=t, color='orange', linestyle=':', alpha=0.4, linewidth=2)
            ax1.axvspan(t-0.5, t+0.5, color='orange', alpha=0.08, zorder=0)
        # Plot 2: Genetic impact
        if self.gamma is not None:
            genetic_effects = np.dot(self.G[person_idx], self.gamma)
            genetic_impact = np.zeros((self.K, len(time_window)))
            for k in range(self.K):
                genetic_impact[k] = genetic_effects[k] * self.theta[person_idx, k, time_window]
            ax2.stackplot(time_window, genetic_impact, 
                         labels=[f'Sig {k}' for k in range(self.K)],
                         alpha=0.7)
            ax2.set_title('Genetic Impact on Signatures Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Genetic Effect')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            # Overlay diagnosis timing
            for t in all_diag_times:
                ax2.axvline(x=t, color='orange', linestyle=':', alpha=0.4, linewidth=2)
                ax2.axvspan(t-0.5, t+0.5, color='orange', alpha=0.08, zorder=0)
        plt.tight_layout()
        return fig

    def plot_disease_probabilities(self, person_idx, time_window=None):
        """Plot disease probabilities over time with diagnosis timing overlays."""
        if time_window is None:
            time_window = range(self.T)
        if self.Y is not None:
            person_diseases = np.where(self.Y[person_idx, :, :].any(axis=1))[0]
            diagnosis_times = self.get_diagnosis_times(person_idx)
        else:
            pi_t = np.zeros((self.T, self.D))
            for t in range(self.T):
                theta_t = self.theta[person_idx, :, t]
                eta_t = expit(self.phi[:, :, t])
                pi_t[t] = np.dot(theta_t, eta_t)
            person_diseases = np.argsort(np.mean(pi_t, axis=0))[-5:]
            diagnosis_times = {}
        pi_t = np.zeros((self.T, self.D))
        for t in range(self.T):
            theta_t = self.theta[person_idx, :, t]
            eta_t = expit(self.phi[:, :, t])
            pi_t[t] = np.dot(theta_t, eta_t)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        # Plot 1: Disease probabilities over time with diagnosis markers
        for d in person_diseases:
            line = ax1.plot(time_window, pi_t[time_window, d], 
                          label=self.disease_names[d], linewidth=2)
            color = line[0].get_color()
            # Add diagnosis markers if available
            if d in diagnosis_times:
                for t in diagnosis_times[d]:
                    if t in time_window:
                        ax1.axvline(x=t, color=color, linestyle='--', alpha=0.5)
                        ax1.plot(t, pi_t[t, d], 'o', color=color, markersize=8)
                        ax1.axvspan(t-0.5, t+0.5, color=color, alpha=0.08, zorder=0)
        ax1.set_title('Disease Probabilities Over Time\n(Diagnosis overlays: lines, dots, and shaded regions)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Probability')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        # Plot 2: Signature contributions to diseases
        contributions = np.zeros((self.K, len(person_diseases)))
        for i, d in enumerate(person_diseases):
            for k in range(self.K):
                contributions[k, i] = np.mean(self.theta[person_idx, k, time_window] * 
                                           expit(self.phi[k, d, time_window]))
        if self.Y is not None:
            labels = []
            for d in person_diseases:
                if d in diagnosis_times:
                    times = diagnosis_times[d]
                    label = f"{self.disease_names[d]}\n(dx: {', '.join(map(str, times))})"
                else:
                    label = self.disease_names[d]
                labels.append(label)
        else:
            labels = [self.disease_names[d] for d in person_diseases]
        sns.heatmap(contributions, ax=ax2, cmap='YlOrRd',
                   xticklabels=labels,
                   yticklabels=[f'Sig {k}' for k in range(self.K)])
        ax2.set_title('Signature Contributions to Diseases')
        ax2.set_xlabel('Disease')
        ax2.set_ylabel('Signature')
        plt.tight_layout()
        return fig

def main():
    st.set_page_config(layout="wide")
    st.title("Patient Timeline Analysis")
    
    # Load model data
    model_data = load_model()
    visualizer = PatientTimelineVisualizer(model_data)
    
    # Sidebar controls
    st.sidebar.header("Patient Selection")
    person_idx = st.sidebar.slider("Select Patient", 0, visualizer.N-1, 0)
    
    # Time window selection
    st.sidebar.header("Time Window")
    time_start = st.sidebar.slider("Start Time", 0, visualizer.T-1, 0)
    time_end = st.sidebar.slider("End Time", time_start, visualizer.T-1, visualizer.T-1)
    time_window = range(time_start, time_end + 1)
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Signature Evolution and Genetic Impact")
        sig_fig = visualizer.plot_signature_timeline(person_idx, time_window)
        st.pyplot(sig_fig)
        
        st.markdown("""
        **How to interpret:**
        - Top plot shows how signature proportions change over time
        - Orange lines and shaded regions indicate diagnosis events
        - Bottom plot shows the genetic impact on each signature
        - Stacked areas in bottom plot show relative genetic influence
        """)
    
    with col2:
        st.markdown("### Disease Probabilities and Signature Contributions")
        disease_fig = visualizer.plot_disease_probabilities(person_idx, time_window)
        st.pyplot(disease_fig)
        
        st.markdown("""
        **How to interpret:**
        - Top plot shows probability trajectories for patient's diseases
        - Dashed lines, dots, and shaded regions indicate when diseases were diagnosed
        - Bottom heatmap shows how each signature contributes to disease risk
        - Diagnosis times are shown in the disease labels
        """)

if __name__ == "__main__":
    main() 