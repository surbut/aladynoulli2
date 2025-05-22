import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax, expit
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.ticker as mticker

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

def load_prs_names(path='prs_names.csv'):
    return pd.read_csv(path, header=None).iloc[:,0].tolist()

class PatientTimelineVisualizer:
    def __init__(self, model_data, prs_names=None):
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
        # Clean up disease names for display
        def _clean_name(name):
            if isinstance(name, str):
                return name
            elif hasattr(name, 'item'):
                return str(name.item())
            elif isinstance(name, (list, tuple)):
                return ", ".join([str(n) for n in name])
            else:
                return str(name)
        self.disease_names = [_clean_name(n) for n in self.disease_names]
        
        # Get dimensions
        self.N, self.K, self.T = self.lambda_.shape
        self.D = self.phi.shape[1]
        
        # Pre-compute theta
        self.theta = softmax(self.lambda_, axis=1)
        
        # Store Y if available
        self.Y = model_data.get('Y', None)
        if self.Y is not None and torch.is_tensor(self.Y):
            self.Y = self.Y.detach().numpy()
        
        # Store PRS names
        self.prs_names = prs_names
        
        # Add time-varying genetic effects
        self.time_varying_gamma = self._compute_time_varying_genetic_effects()

    def _compute_time_varying_genetic_effects(self):
        """Compute time-varying genetic effects through regression."""
        if self.G is None or self.gamma is None:
            return None
            
        N, K, T = self.lambda_.shape
        P = self.G.shape[1]  # Number of genetic features
        
        # Output array
        time_varying_gamma = np.zeros((P, K, T))
        
        # For each time point
        for t in range(T):
            for k in range(K):
                # Center lambda by removing the mean
                lambda_centered = self.lambda_[:, k, t] - np.mean(self.lambda_[:, k, t])
                
                # Linear regression at this time point
                gamma_t, _, _, _ = np.linalg.lstsq(self.G, lambda_centered, rcond=None)
                
                # Store the result
                time_varying_gamma[:, k, t] = gamma_t
                
        return time_varying_gamma

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
        """Plot signature proportions over time with genetic impact and overlay diagnosis timing and annotate with disease name and top signature."""
        if time_window is None:
            time_window = range(self.T)
        fig = plt.figure(figsize=(15, 12))  # Made taller for genetic effects
        gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])  # Added third subplot
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # New subplot for genetic effects

        # --- Main signature plot ---
        for k in range(self.K):
            color = 'red' if k == 5 else sns.color_palette("tab20", self.K)[k]
            ax1.plot(time_window, self.theta[person_idx, k, time_window], 
                    label=f'Signature {k}', linewidth=2, color=color)
        ax1.set_ylabel('Proportion')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Diagnosis timeline (fig3_utils style, ordered by diagnosis time) ---
        diagnosis_times = self.get_diagnosis_times(person_idx)
        # Get (disease, first diagnosis time) pairs and sort by time
        diag_order = sorted(
            [(d, times[0]) for d, times in diagnosis_times.items() if len(times) > 0],
            key=lambda x: x[1]
        )
        disease_rows = {d: i for i, (d, _) in enumerate(diag_order)}
        base_colors = sns.color_palette("tab20", self.K)
        for d, t_diag in diag_order:
            y = disease_rows[d]
            sig_idx = np.argmax(self.psi[:, d])
            color = 'red' if sig_idx == 5 else base_colors[sig_idx]
            # Horizontal line from start to diagnosis
            ax2.hlines(y, time_window[0], t_diag, colors=color, linestyles='-', alpha=0.5, linewidth=2)
            # Dot at diagnosis
            ax2.scatter(t_diag, y, color=color, s=50, edgecolors='black', zorder=5)
            # Vertical line up to main plot
            ax1.axvline(x=t_diag, color=color, linestyle=':', alpha=0.3, linewidth=1)
        # Print primary sig to left of disease name
        ax2.set_yticks(list(disease_rows.values()))
        ax2.set_yticklabels([f"Sig {np.argmax(self.psi[:, d])} | {self.disease_names[d]}" for d, _ in diag_order], fontsize='small')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Diagnosed Condition')
        ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax2.set_ylim(-0.5, len(disease_rows)-0.5)
        ax2.invert_yaxis()

        # --- Genetic effects plot ---
        if self.time_varying_gamma is not None:
            # Get top genetic features for each signature
            top_features = []
            for k in range(self.K):
                # Average absolute effect across time
                mean_effect = np.mean(np.abs(self.time_varying_gamma[:, k, :]), axis=1)
                top_feature = np.argmax(mean_effect)
                top_features.append((k, top_feature))
            
            # Plot top genetic effects for each signature
            for k, feat_idx in top_features:
                color = 'red' if k == 5 else sns.color_palette("tab20", self.K)[k]
                effect = self.time_varying_gamma[feat_idx, k, time_window]
                ax3.plot(time_window, effect, 
                        label=f'Sig {k}: {self.prs_names[feat_idx] if self.prs_names else f"Feature {feat_idx}"}',
                        color=color, alpha=0.7)
            
            ax3.set_title('Time-Varying Genetic Effects')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Effect Size')
            ax3.grid(True, alpha=0.3)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

        plt.tight_layout()
        return fig

    def plot_disease_probabilities(self, person_idx, time_window=None, disease_idx=None, use_log=False):
        """Plot disease probabilities over time (linear or log10) with diagnosis timing overlays and annotate disease names near their lines."""
        epsilon = 1e-10
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
        if use_log:
            plot_vals = np.log10(pi_t + epsilon)
            ylabel = 'log10(Probability)'
        else:
            plot_vals = pi_t
            ylabel = 'Probability'
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
        # Plot 1: Disease probabilities over time with diagnosis markers
        for d in person_diseases:
            is_selected = (d == disease_idx)
            line = ax1.plot(
                time_window, plot_vals[time_window, d],
                label=self.disease_names[d],
                linewidth=3 if is_selected else 1,
                color='red' if is_selected else f'C{d % 10}',
                alpha=1.0 if is_selected else 0.4
            )
            color = line[0].get_color()
            # Add diagnosis markers if available
            if d in diagnosis_times:
                for t in diagnosis_times[d]:
                    if t in time_window:
                        ax1.axvline(x=t, color=color, linestyle='--', alpha=0.8 if is_selected else 0.3, linewidth=2 if is_selected else 1)
                        ax1.plot(t, plot_vals[t, d], 'o', color=color, markersize=10 if is_selected else 6, zorder=10)
                        if is_selected:
                            ax1.annotate("Dx", (t, plot_vals[t, d]), textcoords="offset points", xytext=(0,10), ha='center', color=color, fontsize=10, fontweight='bold')
            # Annotate disease name at the end of the line
            end_x = time_window[-1]
            end_y = plot_vals[end_x, d]
            ax1.annotate(self.disease_names[d], (end_x, end_y), xytext=(5, 0), textcoords='offset points', color=color, fontsize=9, va='center')
        ax1.set_title(('log10 ' if use_log else '') + 'Disease Probabilities Over Time\n(Diagnosis overlays: lines, dots, and shaded regions)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(ylabel)
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

    def diagnosis_timeline_table(self, person_idx):
        """Table of all diagnosis events for the selected patient."""
        if self.Y is None:
            return pd.DataFrame()
        rows = []
        for d in range(self.D):
            times = np.where(self.Y[person_idx, d, :] > 0.5)[0]
            for t in times:
                sig_idx = np.argmax(self.psi[:, d])
                sig_val = self.theta[person_idx, sig_idx, t]
                pi_val = np.dot(self.theta[person_idx, :, t], expit(self.phi[:, d, t]))
                rows.append({
                    'Disease': self.disease_names[d],
                    'Time': t,
                    'Top Signature': sig_idx,
                    'Signature Value': sig_val,
                    'Probability at Dx': pi_val
                })
        return pd.DataFrame(rows)

    def signature_dominance_table(self, person_idx):
        """Table of dominant signature at each time point."""
        dom_idx = np.argmax(self.theta[person_idx], axis=0)
        dom_val = np.max(self.theta[person_idx], axis=0)
        return pd.DataFrame({
            'Time': np.arange(self.T),
            'Dominant Signature': dom_idx,
            'Value': dom_val
        })

    def genetic_risk_table(self, person_idx):
        """Table of genetic effect sizes for each signature."""
        if self.G is None or self.gamma is None:
            return pd.DataFrame()
        effects = np.dot(self.G[person_idx], self.gamma)
        return pd.DataFrame({
            'Signature': np.arange(self.K),
            'Genetic Effect': effects
        }).sort_values('Genetic Effect', ascending=False)

    def disease_risk_summary_table(self, person_idx, top_n=10):
        """Table of top N diseases by mean probability over time."""
        pi_t = np.zeros((self.T, self.D))
        for t in range(self.T):
            theta_t = self.theta[person_idx, :, t]
            eta_t = expit(self.phi[:, :, t])
            pi_t[t] = np.dot(theta_t, eta_t)
        mean_prob = np.mean(pi_t, axis=0)
        max_prob = np.max(pi_t, axis=0)
        max_time = np.argmax(pi_t, axis=0)
        if self.Y is not None:
            dx_time = [np.where(self.Y[person_idx, d, :] > 0.5)[0] for d in range(self.D)]
            dx_time = [t[0] if len(t) > 0 else None for t in dx_time]
        else:
            dx_time = [None] * self.D
        df = pd.DataFrame({
            'Disease': self.disease_names,
            'Mean Probability': mean_prob,
            'Max Probability': max_prob,
            'Time of Max': max_time,
            'Diagnosis Time': dx_time
        })
        return df.sort_values('Mean Probability', ascending=False).head(top_n)

    def signature_disease_contribution_table(self, person_idx, disease_idx):
        """Table of each signature's average contribution to a disease's risk over time."""
        contrib = [np.mean(self.theta[person_idx, k, :] * expit(self.phi[k, disease_idx, :])) for k in range(self.K)]
        return pd.DataFrame({
            'Signature': np.arange(self.K),
            'Contribution': contrib
        }).sort_values('Contribution', ascending=False)

    def onset_group_table(self, disease_idx):
        """Group patients by onset and show mean signature values and genetic risk."""
        if self.Y is None:
            return pd.DataFrame()
        disease_times = np.argmax(self.Y[:, disease_idx, :] == 1, axis=1)
        early = disease_times < 20
        mid = (disease_times >= 20) & (disease_times < 35)
        late = disease_times >= 35
        def group_stats(mask, label):
            if np.sum(mask) == 0:
                return {'Group': label, 'N': 0}
            mean_theta = np.mean(self.theta[mask], axis=(0,2))
            mean_gen = np.mean(np.dot(self.G[mask], self.gamma), axis=0) if self.G is not None and self.gamma is not None else np.nan
            return {'Group': label, 'N': np.sum(mask), 'Mean θ0': mean_theta[0], 'Mean θ1': mean_theta[1], 'Mean Genetic0': mean_gen[0] if isinstance(mean_gen, np.ndarray) else np.nan}
        rows = [group_stats(early, 'Early'), group_stats(mid, 'Mid'), group_stats(late, 'Late')]
        return pd.DataFrame(rows)

    def prediction_accuracy_table(self, disease_idx):
        """Show model's prediction accuracy (AUC) for each patient for a disease."""
        if self.Y is None:
            return pd.DataFrame()
        aucs = []
        for n in range(self.N):
            y_true = self.Y[n, disease_idx, :]
            pi_t = np.zeros(self.T)
            for t in range(self.T):
                theta_t = self.theta[n, :, t]
                eta_t = expit(self.phi[:, disease_idx, t])
                pi_t[t] = np.dot(theta_t, eta_t)
            if np.any(y_true):
                try:
                    auc = roc_auc_score(y_true, pi_t)
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan
            aucs.append(auc)
        return pd.DataFrame({'Patient': np.arange(self.N), 'AUC': aucs})

    def customizable_table(self, person_idx, columns):
        """Return a table with user-selected columns for the patient."""
        # Available: time, dominant signature, all θ, all π, diagnosis events, genetic effect
        pi_t = np.zeros((self.T, self.D))
        for t in range(self.T):
            theta_t = self.theta[person_idx, :, t]
            eta_t = expit(self.phi[:, :, t])
            pi_t[t] = np.dot(theta_t, eta_t)
        dom_idx = np.argmax(self.theta[person_idx], axis=0)
        dom_val = np.max(self.theta[person_idx], axis=0)
        data = {'Time': np.arange(self.T)}
        if 'Dominant Signature' in columns:
            data['Dominant Signature'] = dom_idx
        if 'Dominant Value' in columns:
            data['Dominant Value'] = dom_val
        if 'Diagnosis Event' in columns and self.Y is not None:
            data['Diagnosis Event'] = [
                ', '.join(str(self.disease_names[d]) for d in range(self.D) if self.Y[person_idx, d, t] > 0.5)
                for t in range(self.T)
            ]
        if 'Genetic Effect 0' in columns and self.G is not None and self.gamma is not None:
            effects = np.dot(self.G[person_idx], self.gamma)
            data['Genetic Effect 0'] = [effects[0]] * self.T
        for k in range(self.K):
            if f'Theta {k}' in columns:
                data[f'Theta {k}'] = self.theta[person_idx, k, :]
        for d in range(self.D):
            if f'Prob {self.disease_names[d]}' in columns:
                data[f'Prob {self.disease_names[d]}'] = pi_t[:, d]
        return pd.DataFrame(data)

    def prs_weights_for_signature(self, signature_idx):
        """Table of PRS feature weights for a given signature."""
        if self.gamma is None or not hasattr(self, 'prs_names') or self.prs_names is None:
            print("gamma or prs_names missing")
            return pd.DataFrame()
        print("gamma shape:", self.gamma.shape)
        print("prs_names length:", len(self.prs_names))
        print("First 5 PRS names:", self.prs_names[:5])
        print("First 5 weights for signature 0:", self.gamma[:5, 0])
        weights = self.gamma[:, signature_idx]
        return pd.DataFrame({
            'PRS Feature': self.prs_names,
            'Weight': weights
        }).sort_values('Weight', key=abs, ascending=False)

    def boxplot_prs_weights(self, prs_idx):
        """Box/strip plot of PRS weights across all signatures for a selected PRS feature."""
        if self.gamma is None or self.prs_names is None:
            return None
        weights = self.gamma[prs_idx, :]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.stripplot(x=np.arange(self.K), y=weights, ax=ax)
        ax.set_title(f"PRS '{self.prs_names[prs_idx]}' Weights Across Signatures")
        ax.set_xlabel("Signature")
        ax.set_ylabel("Weight")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        return fig

    def counterfactual_signature_trajectory(self, person_idx, signature_idx):
        """Plot actual vs. counterfactual signature trajectory (PRS set to zero) for a patient/signature."""
        if self.gamma is None or self.G is None:
            return None
        # Actual
        actual_traj = self.theta[person_idx, signature_idx, :]
        # Counterfactual: set PRS for this signature to zero
        if isinstance(self.G, np.ndarray):
            G_cf = self.G[person_idx].copy()
        else:
            G_cf = self.G[person_idx].clone().detach().numpy()
        gamma_cf = self.gamma.copy()
        # Remove genetic effect for this signature
        lambda_cf = self.lambda_[person_idx, signature_idx, :].copy()
        # Subtract the genetic effect for this signature
        genetic_effect = np.dot(self.G[person_idx], self.gamma[:, signature_idx])
        lambda_cf_cf = lambda_cf - genetic_effect
        # Recompute theta for this signature only
        lambda_all = self.lambda_[person_idx, :, :].copy()
        lambda_all[signature_idx, :] = lambda_cf_cf
        theta_cf = softmax(lambda_all, axis=0)[signature_idx, :]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(actual_traj, label="Actual", linewidth=2)
        ax.plot(theta_cf, label="Counterfactual (PRS=0)", linestyle='--', linewidth=2)
        ax.set_title(f"Signature {signature_idx} Trajectory: Actual vs Counterfactual (PRS=0)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Signature Proportion")
        ax.legend()
        return fig

    def scatter_genetic_vs_signature(self, signature_idx):
        """Scatter plot: genetic effect vs mean signature proportion for all patients for a signature."""
        if self.gamma is None or self.G is None:
            return None
        genetic_effects = np.dot(self.G, self.gamma[:, signature_idx])
        mean_theta = np.mean(self.theta[:, signature_idx, :], axis=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(genetic_effects, mean_theta, alpha=0.5)
        ax.set_title(f"Genetic Effect vs Mean Signature {signature_idx} Proportion")
        ax.set_xlabel("Genetic Effect (G·gamma)")
        ax.set_ylabel("Mean Signature Proportion")
        return fig

    def compute_disease_probabilities(self, person_idx):
        """Compute disease probabilities across all timepoints for a person."""
        pi_t = np.zeros((self.T, self.D))
        for t in range(self.T):
            theta_t = self.theta[person_idx, :, t]
            eta_t = expit(self.phi[:, :, t])
            pi_t[t] = np.dot(theta_t, eta_t)
        return pi_t

    def plot_genetic_impact_on_signatures(self, person_idx, time_window=None):
        """Plot the direct genetic impact on signatures over time for the selected patient only."""
        if self.G is None or self.gamma is None:
            return None
        N, K, T = self.lambda_.shape
        if time_window is None:
            time_window = range(T)
        genetic_effects = np.dot(self.G[person_idx], self.gamma)  # shape: (K,)
        genetic_impact = np.zeros((K, len(time_window)))
        for k in range(K):
            genetic_impact[k] = genetic_effects[k] * self.theta[person_idx, k, time_window]
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["red" if k == 5 else sns.color_palette("tab20", K)[k] for k in range(K)]
        # Stacked area plot for this patient
        ax.stackplot(time_window, genetic_impact, labels=[f'Sig {k}' for k in range(K)], colors=colors, alpha=0.7)
        ax.set_title(f'Genetic Impact on Signatures Over Time (Patient {person_idx})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Genetic Effect')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        return fig

    def plot_time_varying_genetic_effects(self, time_window=None):
        """Plot the regression-based time-varying genetic effects for the top PRS feature of each signature."""
        if self.time_varying_gamma is None:
            return None
        K = self.time_varying_gamma.shape[1]
        T = self.time_varying_gamma.shape[2]
        if time_window is None:
            time_window = range(T)
        fig, ax = plt.subplots(figsize=(12, 5))
        for k in range(K):
            # Top PRS feature for this signature
            mean_effect = np.mean(np.abs(self.time_varying_gamma[:, k, :]), axis=1)
            feat_idx = np.argmax(mean_effect)
            color = 'red' if k == 5 else sns.color_palette("tab20", K)[k]
            label = f'Sig {k}: {self.prs_names[feat_idx] if self.prs_names and feat_idx < len(self.prs_names) else f"Feature {feat_idx}"}'
            ax.plot(time_window, self.time_varying_gamma[feat_idx, k, time_window], label=label, color=color, alpha=0.8)
        ax.set_title('Time-Varying Genetic Effects (Regression Approach)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Effect Size')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        return fig

    def plot_patient_genetic_effect_barplot(self, person_idx):
        """Stacked barplot of overall genetic effect for each signature for the selected patient, colored by PRS feature contribution."""
        if self.G is None or self.gamma is None:
            return None
        # Ensure arrays are numpy
        G_row = self.G[person_idx]
        gamma = self.gamma
        if hasattr(G_row, 'detach'):
            G_row = G_row.detach().cpu().numpy()
        if hasattr(gamma, 'detach'):
            gamma = gamma.detach().cpu().numpy()
        if hasattr(G_row, 'numpy') and not isinstance(G_row, np.ndarray):
            G_row = G_row.numpy()
        if hasattr(gamma, 'numpy') and not isinstance(gamma, np.ndarray):
            gamma = gamma.numpy()
        K = gamma.shape[1]
        P = gamma.shape[0]
        # Each PRS feature's contribution to each signature
        contrib = G_row[:, None] * gamma  # shape: (P, K)
        total = np.sum(contrib, axis=0)  # shape: (K,)
        # Avoid division by zero
        total[total == 0] = 1e-12
        frac = contrib / total  # shape: (P, K)
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = np.zeros(K)
        # Use a color palette for PRS features
        palette = sns.color_palette('tab20', P)
        labels = self.prs_names if self.prs_names and len(self.prs_names) == P else [f'PRS {p}' for p in range(P)]
        for p in range(P):
            ax.bar(np.arange(K), contrib[p], bottom=bottom, color=palette[p], label=labels[p], width=0.8)
            bottom += contrib[p]
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f'Sig {k}' for k in range(K)], rotation=45, ha='right')
        ax.set_ylabel('Genetic Effect (sum of PRS contributions)')
        ax.set_title(f'Genetic Effect for Each Signature (Patient {person_idx})\nColored by PRS Feature Contribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='PRS Feature')
        ax.grid(True, axis='y', alpha=0.3)
        return fig

def name_has(term, name):
    if isinstance(name, str):
        return term in name.lower()
    elif isinstance(name, list):
        return any(term in n.lower() for n in name)
    return False

# --- Helper: Find good example patients ---
def find_good_example_patients(Y, psi, min_diseases=3, min_sigs=2, min_time_spread=10, top_n=50):
    N, D, T = Y.shape
    patient_scores = []
    for n in range(N):
        diag_events = []
        sig_set = set()
        times = []
        for d in range(D):
            diag_times = np.where(Y[n, d, :] > 0.5)[0]
            if len(diag_times) > 0:
                t_diag = diag_times[0]
                times.append(t_diag)
                sig_idx = np.argmax(psi[:, d])
                diag_events.append((d, t_diag, sig_idx))
                sig_set.add(sig_idx)
        if len(diag_events) >= min_diseases and len(sig_set) >= min_sigs:
            if len(times) > 1 and (max(times) - min(times)) >= min_time_spread:
                # Score: more diseases, more sigs, more spread
                score = len(diag_events) + len(sig_set) + (max(times) - min(times)) / 10.0
                patient_scores.append((score, n))
    # Sort by score, descending
    patient_scores.sort(reverse=True)
    # Return top N patient indices
    return [n for _, n in patient_scores[:top_n]]

def find_good_mi_overlap_patients(Y, psi, mi_idx, sig_idx=5, window=3, min_diseases=2):
    N, D, T = Y.shape
    selected = []
    for n in range(N):
        # Must have MI
        mi_times = np.where(Y[n, mi_idx, :] > 0.5)[0]
        if len(mi_times) == 0:
            continue
        # Multi-morbid: at least min_diseases diagnosed
        n_diagnoses = sum(np.any(Y[n, d, :] > 0.5) for d in range(D))
        if n_diagnoses < min_diseases:
            continue
        # Patient-specific signature 5 trajectory
        theta_n = softmax(psi, axis=0) if psi.shape[-1] == T else None
        if theta_n is not None:
            sig5_traj = theta_n[sig_idx, :]
        else:
            # Use patient theta if available
            sig5_traj = None
        # Use patient theta if available
        if hasattr(Y, 'theta'):
            sig5_traj = Y.theta[n, sig_idx, :]
        elif hasattr(Y, 'theta'):
            sig5_traj = Y.theta[n, sig_idx, :]
        else:
            # fallback: use softmax(psi)
            sig5_traj = softmax(psi, axis=0)[sig_idx, :]
        peak_time = int(np.argmax(sig5_traj))
        # Check if any MI diagnosis is within window of peak
        if any(abs(t - peak_time) <= window for t in mi_times):
            selected.append(n)
    return selected

def main():
    st.set_page_config(layout="wide")
    st.title("Patient Timeline Analysis")
    
    # Load model data
    model_data = load_model()
    prs_names = load_prs_names('prs_names.csv')
    visualizer = PatientTimelineVisualizer(model_data, prs_names=prs_names)
    
    # Find MI disease index
    mi_idx = None
    for i, name in enumerate(visualizer.disease_names):
        if "myocardial infarction" in name.lower() or "mi" in name.lower():
            mi_idx = i
            break

    # --- Filter to only multi-morbid MI patients with MI diagnosis near sig 5 peak ---
    good_patients = []
    if mi_idx is not None and visualizer.Y is not None:
        good_patients = find_good_mi_overlap_patients(visualizer.Y, visualizer.psi, mi_idx, sig_idx=5, window=3, min_diseases=2)
    if not good_patients:
        # fallback: all MI patients
        for n in range(visualizer.N):
            if np.any(visualizer.Y[n, mi_idx, :] > 0.5):
                good_patients.append(n)
    if not good_patients:
        good_patients = list(range(visualizer.N))  # fallback to all patients if none with MI

    # Default to first good patient, or 0
    default_person_idx = good_patients[0] if good_patients else 0
    if mi_idx is not None and visualizer.Y is not None:
        best_score = -np.inf
        for n in good_patients:
            mi_times = np.where(visualizer.Y[n, mi_idx, :] > 0.5)[0]
            if len(mi_times) > 0:
                # Get max signature 5 loading at MI diagnosis times
                sig5_loadings = [visualizer.theta[n, 5, t] for t in mi_times]
                max_sig5 = max(sig5_loadings)
                if max_sig5 > best_score:
                    best_score = max_sig5
                    default_person_idx = n
    
    # Sidebar controls
    st.sidebar.header("Patient Selection")
    person_idx = st.sidebar.selectbox(
        "Select Patient",
        options=good_patients,
        index=good_patients.index(default_person_idx) if default_person_idx in good_patients else 0,
        format_func=lambda x: f"Patient {x}"
    )
    
    # Time window selection
    st.sidebar.header("Time Window")
    time_start = st.sidebar.slider("Start Time", 0, visualizer.T-1, 0)
    time_end = st.sidebar.slider("End Time", time_start, visualizer.T-1, visualizer.T-1)
    time_window = range(time_start, time_end + 1)
    
    # Add a new tab for tables
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Plots", "Tables", "Genetics", "Disease Risk Highlights", "Risk Summary & Event Analysis"
    ])
    with tab1:
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
            # Use the most common disease as default for highlighting if not in Risk Summary tab
            if 'disease_idx' in locals():
                disease_fig = visualizer.plot_disease_probabilities(person_idx, time_window, disease_idx, use_log=False)
            else:
                # Use the most common disease
                if visualizer.Y is not None:
                    disease_prevalence = np.sum(visualizer.Y.sum(axis=2) > 0, axis=0)
                    most_common_disease_idx = int(np.argmax(disease_prevalence))
                else:
                    most_common_disease_idx = 0
                disease_fig = visualizer.plot_disease_probabilities(person_idx, time_window, most_common_disease_idx, use_log=False)
            st.pyplot(disease_fig)
            st.markdown("""
            **How to interpret:**
            - Top plot shows probability trajectories for patient's diseases
            - Dashed lines, dots, and shaded regions indicate when diseases were diagnosed
            - Disease names are annotated at the end of each line for clarity
            - Bottom heatmap shows how each signature contributes to disease risk
            - Diagnosis times are shown in the disease labels
            """)
    with tab2:
        st.header("Diagnosis Timeline Table")
        st.dataframe(visualizer.diagnosis_timeline_table(person_idx))

        st.header("Genetic Risk Table")
        st.dataframe(visualizer.genetic_risk_table(person_idx))

        st.header("PRS Weights for Signature")
        sig_idx = st.selectbox("Select Signature for PRS Weights", range(visualizer.K), format_func=lambda x: f"Signature {x}")
        st.dataframe(visualizer.prs_weights_for_signature(sig_idx))

        print("gamma shape:", visualizer.gamma.shape)
        print("prs_names length:", len(visualizer.prs_names))
        print("First 5 PRS names:", visualizer.prs_names[:5])
        print("First 5 weights for signature 0:", visualizer.gamma[:5, 0])

    with tab3:
        #st.header("Box/Strip Plot of PRS Weights Across Signatures")
        #if visualizer.prs_names is not None:
        #    prs_idx = st.selectbox("Select PRS Feature", range(len(visualizer.prs_names)), format_func=lambda x: visualizer.prs_names[x])
        #    fig = visualizer.boxplot_prs_weights(prs_idx)
        #    if fig:
        #        st.pyplot(fig)
        #st.markdown("---")
        st.header("Counterfactual Signature Trajectory (PRS=0)")
        person_idx_cf = st.number_input("Patient Index", min_value=0, max_value=visualizer.N-1, value=536)
        signature_idx_cf = st.number_input("Signature Index", min_value=0, max_value=visualizer.K-1, value=5)
        fig_cf = visualizer.counterfactual_signature_trajectory(person_idx_cf, signature_idx_cf)
        if fig_cf:
            st.pyplot(fig_cf)
        st.markdown("---")
        st.header("Genetic Effect vs Mean Signature Proportion")
        signature_idx_scatter = st.number_input("Signature Index for Scatter", min_value=0, max_value=visualizer.K-1, value=5)
        fig_scatter = visualizer.scatter_genetic_vs_signature(signature_idx_scatter)
        if fig_scatter:
            st.pyplot(fig_scatter)
        st.markdown("---")
        st.header("Genetic Effect for Each Signature (Selected Patient, Stacked by PRS Feature)")
        st.markdown("This stacked barplot shows the overall genetic effect for each signature for the currently selected patient, with each bar colored by the fractional contribution of each PRS feature (G*gamma).")
        fig_bar = visualizer.plot_patient_genetic_effect_barplot(person_idx)
        if fig_bar:
            st.pyplot(fig_bar)
        st.markdown("---")
        st.header("Genetic Impact on Signatures Over Time (Direct Weighting, Selected Patient)")
        st.markdown("This plot shows the direct genetic impact on each signature over time for the currently selected patient, using their PRS and the model's gamma weights.")
        fig_impact = visualizer.plot_genetic_impact_on_signatures(person_idx, time_window)
        if fig_impact:
            st.pyplot(fig_impact)
        st.markdown("---")

    with tab4:
        st.header("Disease Risk Highlights")
        disease_groups = {
            "Cancer": [i for i, name in enumerate(visualizer.disease_names) if name_has("cancer", name) or name_has("neoplasm", name)],
            "Cardiovascular": [i for i, name in enumerate(visualizer.disease_names) if name_has("cardio", name) or name_has("cvd", name) or name_has("heart", name)],
            "Diabetes": [i for i, name in enumerate(visualizer.disease_names) if name_has("diabetes", name) or name_has("t1d", name) or name_has("t2d", name)],
            # Add more groups as you like!
        }
        group = st.selectbox("Select Disease Group", list(disease_groups.keys()))
        indices = disease_groups[group]
        fig, ax = plt.subplots(figsize=(10, 5))
        pi_t = visualizer.compute_disease_probabilities(person_idx)
        for d in indices:
            ax.plot(pi_t[:, d], label=visualizer.disease_names[d])
        ax.set_title(f"{group} Risk Trajectories")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.legend()
        st.pyplot(fig)

    with tab5:
        st.markdown("### Risk Summary & Event Analysis")
        # Only show diseases with at least one diagnosis event for this patient
        if visualizer.Y is not None:
            patient_diagnosed_diseases = [d for d in range(visualizer.D) if np.any(visualizer.Y[person_idx, d, :] > 0.5)]
        else:
            patient_diagnosed_diseases = []
        if not patient_diagnosed_diseases:
            st.info("This patient has no diagnosis events.")
        else:
            # Disease selection (only those diagnosed in this patient)
            disease_idx = st.selectbox(
                "Select Disease for Analysis",
                options=patient_diagnosed_diseases,
                format_func=lambda x: visualizer.disease_names[x]
            )
            disease_name = visualizer.disease_names[disease_idx]
            # Prevalence in dataset
            if visualizer.Y is not None:
                disease_prevalence = np.sum(visualizer.Y.sum(axis=2) > 0, axis=0)
                n_patients_with_event = int(disease_prevalence[disease_idx])
            else:
                n_patients_with_event = 0
            st.markdown(f"**{n_patients_with_event} patients** in the dataset have at least one diagnosis event for **{disease_name}**.")
            # Get disease probabilities
            pi_t = visualizer.compute_disease_probabilities(person_idx)
            disease_probs = pi_t[:, disease_idx]
            epsilon = 1e-10
            log_disease_probs = np.log10(disease_probs + epsilon)
            time_idx = st.sidebar.slider("Current Time Point", 0, visualizer.T-1, 0)
            summary_data = {
                'Metric': ['Minimum log10(Risk)', 'Maximum log10(Risk)', 'Mean log10(Risk)', 'Time of Max log10(Risk)', 'Current log10(Risk)'],
                'Value': [
                    f"{log_disease_probs.min():.3f}",
                    f"{log_disease_probs.max():.3f}",
                    f"{log_disease_probs.mean():.3f}",
                    f"Time {np.argmax(log_disease_probs)}",
                    f"{log_disease_probs[time_idx]:.3f}"
                ]
            }
            st.markdown("#### Risk Summary")
            st.table(pd.DataFrame(summary_data))
            # Event Analysis
            st.markdown("#### Event Analysis")
            Y = visualizer.Y
            diagnoses = Y[person_idx, disease_idx, :]
            diagnosis_times = np.where(diagnoses > 0.5)[0]
            if len(diagnosis_times) > 0:
                st.markdown(f"**Diagnosis Events:** Found {len(diagnosis_times)} events at times {diagnosis_times}")
                # Plot only the selected disease's probability trajectory
                epsilon = 1e-10
                fig, ax = plt.subplots(figsize=(12, 5))
                log_patient = np.log10(disease_probs + epsilon)
                pop_mean = np.mean([visualizer.compute_disease_probabilities(n)[:, disease_idx] for n in range(visualizer.N)], axis=0)
                log_pop_mean = np.log10(pop_mean + epsilon)
                ax.plot(range(visualizer.T), log_patient, color='red', linewidth=2, label=f'Patient Risk ({disease_name})')
                ax.plot(range(visualizer.T), log_pop_mean, color='black', linestyle='--', linewidth=2, label='Population Mean Risk')
                # Mark diagnosis times and annotate fold enrichment (in linear space)
                patient_at_diag = [disease_probs[t] for t in diagnosis_times]
                pop_at_diag = [pop_mean[t] for t in diagnosis_times]
                fold_enrichment = [p / pop if pop > 0 else np.nan for p, pop in zip(patient_at_diag, pop_at_diag)]
                for i, t in enumerate(diagnosis_times):
                    ax.axvline(x=t, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    ax.scatter(t, log_patient[t], color='red', s=80, zorder=10, label='Diagnosis' if i==0 else None)
                    ax.scatter(t, log_pop_mean[t], color='black', marker='x', s=80, zorder=10, label='Population Mean' if i==0 else None)
                    ax.annotate(f"{fold_enrichment[i]:.2f}x", (t, log_patient[t]), textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=10, fontweight='bold')
                ax.set_title(f'log10 Disease Probability Trajectory for {disease_name}')
                ax.set_xlabel('Time')
                ax.set_ylabel('log10(Probability)')
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
                st.pyplot(fig)
                # Table of fold enrichment
                enrichment_df = pd.DataFrame({
                    'Diagnosis Time': diagnosis_times,
                    'Patient Probability': patient_at_diag,
                    'Population Mean': pop_at_diag,
                    'Fold Enrichment': fold_enrichment
                })
                st.markdown('**Fold Enrichment of Disease Probability at Diagnosis**')
                st.table(enrichment_df)
            else:
                st.info(f"No diagnosis events found for **{disease_name}** in the selected patient.")

if __name__ == "__main__":
    main() 