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
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
import io
from datetime import datetime

def load_model():
    """Load the model and return necessary components"""
    st.cache_data.clear()
    #model = torch.load('/Users/sarahurbut/Library/Cloudstorage/Dropbox-Personal/resultshighamp/results/output_0_10000/model.pt')
    model = torch.load('app_patients_compact_nolr.pt', map_location=torch.device('cpu'))
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
        # Extract model parameters and ensure they're numpy arrays
        self.lambda_ = model_data['model_state_dict']['lambda_']
        self.phi = model_data['model_state_dict']['phi']
        self.psi = model_data['model_state_dict']['psi']
        self.G = model_data['G']
        self.gamma = model_data['model_state_dict'].get('gamma', None)
        
        # Convert any remaining tensors to numpy
        if hasattr(self.lambda_, 'detach'):
            self.lambda_ = self.lambda_.detach().numpy()
        if hasattr(self.phi, 'detach'):
            self.phi = self.phi.detach().numpy()
        if hasattr(self.psi, 'detach'):
            self.psi = self.psi.detach().numpy()
        if self.gamma is not None and hasattr(self.gamma, 'detach'):
            self.gamma = self.gamma.detach().numpy()
        if self.G is not None and hasattr(self.G, 'detach'):
            self.G = self.G.detach().numpy()
        
        # Always try to load disease names from CSV if available (it's the source of truth)
        from pathlib import Path
        import os
        
        # Try multiple possible locations for the CSV
        possible_paths = [
            Path('disease_names.csv'),  # Same directory as app
            Path(__file__).parent / 'disease_names.csv',  # App directory
            Path('/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/disease_names.csv'),  # Known location
            Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/disease_names.csv'),  # Data directory
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists():
                csv_path = path
                print(f"✓ Found disease_names.csv at: {csv_path}")
                break
        
        if not csv_path:
            print(f"⚠️ Could not find disease_names.csv in any of these locations:")
            for p in possible_paths:
                print(f"   - {p}")
        
        if csv_path and csv_path.exists():
            try:
                # Read CSV - it has column "x" with disease names
                df = pd.read_csv(csv_path)
                if 'x' in df.columns:
                    self.disease_names = df['x'].dropna().astype(str).tolist()
                    print(f"✓ Loaded {len(self.disease_names)} disease names from CSV column 'x'")
                elif len(df.columns) >= 2:
                    # Use second column (index 1) which should be the disease names
                    self.disease_names = df.iloc[:, 1].dropna().astype(str).tolist()
                    print(f"✓ Loaded {len(self.disease_names)} disease names from CSV second column")
                else:
                    # Fallback: try first column
                    self.disease_names = df.iloc[:, 0].dropna().astype(str).tolist()
                    print(f"✓ Loaded {len(self.disease_names)} disease names from CSV first column")
            except Exception as e:
                print(f"⚠️ Could not load from CSV: {e}, falling back to model data")
                # Fallback to model data if CSV fails
                self.disease_names = model_data['disease_names']
                if hasattr(self.disease_names, 'columns') and 'x' in self.disease_names.columns:
                    self.disease_names = self.disease_names['x'].dropna().astype(str).tolist()
                elif isinstance(self.disease_names, (list, tuple, np.ndarray)):
                    self.disease_names = [str(n) for n in self.disease_names if pd.notna(n)]
                else:
                    self.disease_names = [str(n) for n in list(self.disease_names) if pd.notna(n)]
        else:
            # No CSV available, use model data and try to clean it
            self.disease_names = model_data['disease_names']
            if hasattr(self.disease_names, 'columns'):
                if 'x' in self.disease_names.columns:
                    self.disease_names = self.disease_names['x'].dropna().astype(str).tolist()
                elif len(self.disease_names.columns) >= 2:
                    self.disease_names = self.disease_names.iloc[:, 1].dropna().astype(str).tolist()
                else:
                    self.disease_names = self.disease_names.iloc[:, 0].dropna().astype(str).tolist()
            elif isinstance(self.disease_names, (list, tuple, np.ndarray)):
                self.disease_names = [str(n) for n in self.disease_names if pd.notna(n)]
            else:
                self.disease_names = [str(n) for n in list(self.disease_names) if pd.notna(n)]
        
        print(f"Final disease_names: {len(self.disease_names)} names loaded")
        if len(self.disease_names) > 0:
            print(f"First disease: '{self.disease_names[0]}'")
        else:
            print("⚠️ WARNING: No disease names loaded!")
        
        # Get dimensions
        self.N, self.K, self.T = self.lambda_.shape
        self.D = self.phi.shape[1]
        
        # CRITICAL: Validate that disease_names count matches D
        if len(self.disease_names) != self.D:
            print(f"⚠️⚠️⚠️ WARNING: MISMATCH! disease_names has {len(self.disease_names)} items but D={self.D}")
            print(f"   This means disease indices may not align correctly!")
            print(f"   Trimming or padding disease_names to match D={self.D}")
            if len(self.disease_names) > self.D:
                self.disease_names = self.disease_names[:self.D]
                print(f"   Trimmed disease_names to {len(self.disease_names)} items")
            elif len(self.disease_names) < self.D:
                # Pad with placeholder names
                padding = [f"Disease {i}" for i in range(len(self.disease_names), self.D)]
                self.disease_names.extend(padding)
                print(f"   Padded disease_names to {len(self.disease_names)} items with placeholders")
        else:
            print(f"✓ Validation passed: disease_names count ({len(self.disease_names)}) matches D ({self.D})")
        
        # Pre-compute theta
        self.theta = softmax(self.lambda_, axis=1)
        
        # Store Y if available
        self.Y = model_data.get('Y', None)
        if self.Y is not None and hasattr(self.Y, 'detach'):
            self.Y = self.Y.detach().numpy()
        
        # Store clusters if available
        self.clusters = model_data.get('clusters', None)
        if self.clusters is not None:
            if hasattr(self.clusters, 'numpy'):
                self.clusters = self.clusters.numpy()
            elif torch.is_tensor(self.clusters):
                self.clusters = self.clusters.cpu().numpy()
        
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
                # Handle cases where feat_idx might be >= len(prs_names) (e.g., sex, PCs)
                if self.prs_names and feat_idx < len(self.prs_names):
                    feature_label = self.prs_names[feat_idx]
                elif feat_idx == 36:
                    feature_label = "Sex"
                elif feat_idx > 36:
                    feature_label = f"PC{feat_idx - 36}"
                else:
                    feature_label = f"Feature {feat_idx}"
                ax3.plot(time_window, effect, 
                        label=f'Sig {k}: {feature_label}',
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
        # Handle case where we have 36 PRS names but 47 features (PRS + sex + PCs)
        if self.prs_names and len(self.prs_names) == len(weights):
            feature_names = self.prs_names
        elif self.prs_names and len(self.prs_names) == 36 and len(weights) == 47:
            feature_names = list(self.prs_names) + ["Sex"] + [f"PC{i}" for i in range(1, 11)]
        else:
            feature_names = [f'Feature {p}' for p in range(len(weights))]
        return pd.DataFrame({
            'PRS Feature': feature_names,
            'Weight': weights
        }).sort_values('Weight', key=abs, ascending=False)

    def boxplot_prs_weights(self, prs_idx):
        """Box/strip plot of PRS weights across all signatures for a selected PRS feature."""
        if self.gamma is None or self.prs_names is None:
            return None
        weights = self.gamma[prs_idx, :]
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.stripplot(x=np.arange(self.K), y=weights, ax=ax)
        # Safe indexing check
        prs_label = self.prs_names[prs_idx] if self.prs_names and prs_idx < len(self.prs_names) else f"Feature {prs_idx}"
        ax.set_title(f"PRS '{prs_label}' Weights Across Signatures")
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
        ax.set_title(f'Genetic Impact on Signatures Over Time (Sample Patient {person_idx})')
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
            # Handle cases where feat_idx might be >= len(prs_names) (e.g., sex, PCs)
            if self.prs_names and feat_idx < len(self.prs_names):
                feature_label = self.prs_names[feat_idx]
            elif feat_idx == 36:
                feature_label = "Sex"
            elif feat_idx > 36:
                feature_label = f"PC{feat_idx - 36}"
            else:
                feature_label = f"Feature {feat_idx}"
            label = f'Sig {k}: {feature_label}'
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
        
        # Ensure arrays are numpy (no more .detach() calls needed)
        G_row = self.G[person_idx]
        gamma = self.gamma
        
        # Convert to numpy if they're not already
        if not isinstance(G_row, np.ndarray):
            G_row = np.array(G_row)
        if not isinstance(gamma, np.ndarray):
            gamma = np.array(gamma)
        
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
        # Handle case where we have 36 PRS names but 47 features (PRS + sex + PCs)
        if self.prs_names and len(self.prs_names) == P:
            labels = self.prs_names
        elif self.prs_names and len(self.prs_names) == 36 and P == 47:
            labels = list(self.prs_names) + ["Sex"] + [f"PC{i}" for i in range(1, 11)]
        else:
            labels = [f'Feature {p}' for p in range(P)]
        
        for p in range(P):
            ax.bar(np.arange(K), contrib[p], bottom=bottom, color=palette[p], label=labels[p], width=0.8)
            bottom += contrib[p]
        
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f'Sig {k}' for k in range(K)], rotation=45, ha='right')
        ax.set_ylabel('Genetic Effect (sum of PRS contributions)')
        ax.set_title(f'Genetic Effect for Each Signature (Sample Patient {person_idx})\nColored by PRS Feature Contribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title='PRS Feature')
        ax.grid(True, axis='y', alpha=0.3)
        return fig

    def plot_comprehensive_timeline(self, person_idx, age_offset=30, cluster_assignments=None, figsize=(20, 14)):
        """
        Create comprehensive multi-panel timeline visualization.
        Adapted from realfit_app_nolr.py plot_patient_timeline_comprehensive
        """
        pi = self.compute_disease_probabilities(person_idx)  # [T, D]
        theta = self.theta[person_idx]  # [K, T]
        Y = self.Y[person_idx] if self.Y is not None else np.zeros((self.D, self.T))  # [D, T]
        
        T, D = pi.shape  # pi is [T, D], not [D, T]
        K = theta.shape[0]
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['font.size'] = 10
        
        # Find diagnoses
        diagnosis_times = {}
        for d in range(D):
            event_times = np.where(Y[d, :] > 0.5)[0]
            if len(event_times) > 0:
                diagnosis_times[d] = event_times.tolist()
        
        n_diseases = len(diagnosis_times)
        all_times = []
        for times in diagnosis_times.values():
            all_times.extend(times)
        time_range = (min(all_times), max(all_times)) if all_times else (0, T-1)
        
        # Calculate average theta
        avg_theta = theta.mean(axis=1)  # Shape: (K,)
        
        # Get signature colors
        sig_colors = sns.color_palette("tab20", K)
        
        # Get cluster assignments for diseases
        if cluster_assignments is None:
            # Try to use clusters from model_data
            if hasattr(self, 'clusters') and self.clusters is not None:
                cluster_assignments = self.clusters
            else:
                # Fallback: use argmax of psi averaged over time for each disease
                if hasattr(self, 'psi') and self.psi is not None:
                    if len(self.psi.shape) == 3:  # [K, D, T]
                        cluster_assignments = np.argmax(self.psi.mean(axis=2), axis=0)  # [D]
                    elif len(self.psi.shape) == 2:  # [K, D]
                        cluster_assignments = np.argmax(self.psi, axis=0)  # [D]
                    else:
                        cluster_assignments = np.zeros(D, dtype=int)
                else:
                    cluster_assignments = np.zeros(D, dtype=int)
        
        # Convert timepoints to ages
        ages = np.arange(age_offset, age_offset + T)
        
        # Create figure
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = plt.GridSpec(3, 4, width_ratios=[1.5, 1.5, 1.2, 1.2], 
                          height_ratios=[2, 2.8, 2], 
                          hspace=0.35, wspace=0.25)
        
        # Panel 1: Signature loadings (θ) vs Age
        ax1 = fig.add_subplot(gs[0, :])
        
        for k in range(K):
            ax1.plot(ages, theta[k, :], 
                     label=f'Signature {k}', linewidth=2.3, color=sig_colors[k], alpha=0.85)
        
        # Add vertical lines at diagnosis times
        for d, times in diagnosis_times.items():
            for t in times:
                if t >= T:
                    continue
                age_at_diag = age_offset + t
                ax1.axvline(x=age_at_diag, color='gray', linestyle=':', alpha=0.25, linewidth=0.8)
        
        # Add inset for average theta
        ax1_bar = ax1.inset_axes([0.84, 0.62, 0.14, 0.32])
        sorted_indices = np.argsort(avg_theta)[::-1]
        sorted_avg_theta = avg_theta[sorted_indices]
        sorted_colors = [sig_colors[i] for i in sorted_indices]
        
        bottom = 0
        for val, color in zip(sorted_avg_theta, sorted_colors):
            if val > 0.005:
                ax1_bar.barh(0, val, left=bottom, color=color, height=0.7, alpha=0.85, edgecolor='none')
                bottom += val
        
        ax1_bar.set_xlim([0, 1])
        ax1_bar.set_ylim([-0.5, 0.5])
        ax1_bar.set_xticks([0, 0.5, 1])
        ax1_bar.set_xticklabels(['0', '0.5', '1'], fontsize=8)
        ax1_bar.set_yticks([])
        ax1_bar.set_title('Avg θ', fontsize=9, fontweight='bold', pad=3)
        for spine in ax1_bar.spines.values():
            spine.set_visible(False)
        ax1_bar.tick_params(length=0)
        
        ax1.set_ylabel('Signature Loading (θ)', fontsize=13, fontweight='bold')
        ax1.set_title('Signature Trajectories Over Time', fontsize=15, fontweight='bold', pad=10)
        ncols = min(4, (K + 3) // 4)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                   ncol=ncols, columnspacing=1.0, handlelength=1.8, handletextpad=0.5,
                   framealpha=0.95, borderpad=0.4, borderaxespad=0.2)
        ax1.tick_params(labelsize=11)
        ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        ax1.set_xlim([age_offset, age_offset + T])
        ax1.set_ylim([0, max(theta.max() * 1.08, 0.5)])
        ax1.set_xlabel('Age (years)', fontsize=12)
        
        # Panel 2: Disease timeline
        ax2 = fig.add_subplot(gs[1, :2])
        
        if len(diagnosis_times) > 0:
            diag_order = sorted([(d, times[0]) for d, times in diagnosis_times.items()], 
                               key=lambda x: x[1])
            max_diseases_shown = min(30, len(diag_order))
            diag_order_shown = diag_order[:max_diseases_shown]
            
            for i, (d, t_diag) in enumerate(diag_order_shown):
                if t_diag >= T:
                    continue
                sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
                color = sig_colors[sig_for_disease] if 0 <= sig_for_disease < K else 'gray'
                age_at_diag = age_offset + t_diag
                y_pos = len(diag_order_shown) - i - 1
                
                ax2.plot([age_offset, age_at_diag], [y_pos, y_pos], color=color, linewidth=1, alpha=0.3)
                ax2.scatter(age_at_diag, y_pos, s=90, color=color, alpha=0.85, zorder=10, 
                           edgecolors='black', linewidths=1.2)
                ax2.text(age_offset - 1, y_pos, f'{i+1}', fontsize=8, fontweight='bold', 
                        verticalalignment='center', ha='right')
            
            ax2.set_yticks(range(len(diag_order_shown)))
            ax2.set_yticklabels([])
            ax2.set_ylim([-0.5, len(diag_order_shown) - 0.5])
            
            if len(diag_order) > max_diseases_shown:
                ax2.text(0.5, -0.02, f'(Showing first {max_diseases_shown} of {len(diag_order)} diseases)', 
                        transform=ax2.transAxes, ha='center', fontsize=9, style='italic', color='gray')
        else:
            ax2.text(0.5, 0.5, 'No diagnoses recorded', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=12, color='gray')
        
        ax2.set_ylabel('Disease Order\n(chronological)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Age (years)', fontsize=12)
        ax2.set_title('Disease Timeline', fontsize=14, fontweight='bold', pad=8)
        ax2.tick_params(labelsize=10)
        ax2.grid(True, alpha=0.15, axis='x', linestyle='-', linewidth=0.5)
        ax2.set_xlim([age_offset, age_offset + T])
        
        # Panel 2b: Disease details (two columns)
        ax2_legend1 = fig.add_subplot(gs[1, 2])
        ax2_legend2 = fig.add_subplot(gs[1, 3])
        ax2_legend1.axis('off')
        ax2_legend2.axis('off')
        
        if len(diagnosis_times) > 0:
            mid_point = (len(diag_order_shown) + 1) // 2
            first_column = diag_order_shown[:mid_point]
            second_column = diag_order_shown[mid_point:]
            
            legend_text1 = []
            for i, (d, t_diag) in enumerate(first_column):
                disease_name = self.disease_names[d] if d < len(self.disease_names) else f'Disease {d}'
                sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
                
                max_len = 24
                if len(disease_name) > max_len:
                    truncated = disease_name[:max_len-3]
                    last_space = truncated.rfind(' ')
                    if last_space > 12:
                        disease_name = truncated[:last_space] + '...'
                    else:
                        disease_name = truncated + '...'
                
                age_at_diag = age_offset + t_diag
                legend_text1.append(f'{i+1:2d}. {disease_name[:24]:<24s}\n    Sig {sig_for_disease:2d}, Age {age_at_diag:2d}')
            
            legend_text2 = []
            for i, (d, t_diag) in enumerate(second_column, start=mid_point):
                disease_name = self.disease_names[d] if d < len(self.disease_names) else f'Disease {d}'
                sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
                
                max_len = 24
                if len(disease_name) > max_len:
                    truncated = disease_name[:max_len-3]
                    last_space = truncated.rfind(' ')
                    if last_space > 12:
                        disease_name = truncated[:last_space] + '...'
                    else:
                        disease_name = truncated + '...'
                
                age_at_diag = age_offset + t_diag
                legend_text2.append(f'{i+1:2d}. {disease_name[:24]:<24s}\n    Sig {sig_for_disease:2d}, Age {age_at_diag:2d}')
            
            ax2_legend1.text(0.05, 0.98, 'Disease Details (1/2):', fontsize=10, fontweight='bold',
                            transform=ax2_legend1.transAxes, va='top')
            legend_str1 = '\n'.join(legend_text1)
            ax2_legend1.text(0.05, 0.93, legend_str1, fontsize=7,
                            transform=ax2_legend1.transAxes, va='top', 
                            fontfamily='monospace', linespacing=1.25)
            
            if second_column:
                ax2_legend2.text(0.05, 0.98, 'Disease Details (2/2):', fontsize=10, fontweight='bold',
                                transform=ax2_legend2.transAxes, va='top')
                legend_str2 = '\n'.join(legend_text2)
                ax2_legend2.text(0.05, 0.93, legend_str2, fontsize=7,
                                transform=ax2_legend2.transAxes, va='top', 
                                fontfamily='monospace', linespacing=1.25)
        
        # Panel 3: Disease probabilities (stopping at diagnosis)
        ax3 = fig.add_subplot(gs[2, :])
        
        if len(diagnosis_times) > 0:
            # Get diseases with events
            diseases_with_events = list(diagnosis_times.keys())
            
            # Smart filtering: prioritize diseases with events, then by max probability
            disease_scores = []
            for d in range(D):
                has_event = d in diagnosis_times
                if d in diagnosis_times:
                    first_diag_t = min(diagnosis_times[d])
                    if first_diag_t >= T:
                        first_diag_t = T - 1
                    max_prob = pi[:first_diag_t + 1, d].max()
                else:
                    max_prob = pi[:, d].max()
                
                # Score: events get priority, then by max probability
                score = (1000 if has_event else 0) + max_prob * 100
                
                if max_prob > 0.0001:  # Only consider diseases with some probability
                    disease_scores.append((d, score, has_event, max_prob))
            
            # Sort and take top N
            disease_scores.sort(key=lambda x: x[1], reverse=True)
            n_diseases_to_plot = min(20, len(disease_scores))
            top_diseases = [d for d, _, _, _ in disease_scores[:n_diseases_to_plot]]
            
            # Group by signature
            diseases_by_sig = {}
            for d in top_diseases:
                sig = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
                if sig not in diseases_by_sig:
                    diseases_by_sig[sig] = []
                diseases_by_sig[sig].append(d)
            
            plotted_count = 0
            for sig in sorted(diseases_by_sig.keys()):
                for d in diseases_by_sig[sig]:
                    # Get signature for this disease from cluster_assignments (already set up above)
                    sig_for_disease = int(cluster_assignments[d]) if d < len(cluster_assignments) else 0
                    color = sig_colors[sig_for_disease] if 0 <= sig_for_disease < K else 'gray'
                    
                    if d in diagnosis_times:
                        first_diag_t = min(diagnosis_times[d])
                        if first_diag_t >= T:
                            first_diag_t = T - 1
                        plot_ages = ages[:first_diag_t + 1]
                        plot_pi = pi[:first_diag_t + 1, d]
                    else:
                        plot_ages = ages
                        plot_pi = pi[:, d]
                    
                    ax3.plot(plot_ages, plot_pi, 
                             color=color, linewidth=1.8, alpha=0.7, linestyle='-')
                    
                    # Mark ONLY the first diagnosis time (not all diagnosis times)
                    if d in diagnosis_times:
                        first_diag_t = min(diagnosis_times[d])
                        if first_diag_t < T:
                            age_at_diag = age_offset + first_diag_t
                            ax3.scatter(age_at_diag, pi[first_diag_t, d], 
                                       color=color, s=80, zorder=10, marker='o', 
                                       edgecolors='black', linewidths=1.2, alpha=0.9)
                    
                    plotted_count += 1
            
            if len(disease_scores) > n_diseases_to_plot:
                ax3.text(0.98, 0.98, f'Top {n_diseases_to_plot} of {len(disease_scores)} diseases\n(prioritizing events + max probability)\nGrouped by signature, diagnoses marked with ●', 
                        transform=ax3.transAxes, ha='right', va='top', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            else:
                ax3.text(0.98, 0.98, f'Top {n_diseases_to_plot} diseases by max probability\nGrouped by signature, diagnoses marked with ●', 
                        transform=ax3.transAxes, ha='right', va='top', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Age (years)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Disease Probability (π)', fontsize=13, fontweight='bold')
        ax3.set_title('Disease Risk Trajectories (stopping at diagnosis)', fontsize=14, fontweight='bold', pad=8)
        ax3.tick_params(labelsize=11)
        ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        ax3.set_xlim([age_offset, age_offset + T])
        
        if len(diagnosis_times) > 0:
            y_max = min(0.1, ax3.get_ylim()[1] * 1.1)
            ax3.set_ylim([0, y_max])
        
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
        
        # Add title
        fig.suptitle('Comprehensive Disease Trajectory Analysis', 
                     fontsize=17, fontweight='bold', y=0.98)
        subtitle = f'Total diseases: {n_diseases} | Age range: {age_offset+time_range[0]}-{age_offset+time_range[1]} | Signatures: {K}'
        fig.text(0.5, 0.95, subtitle, ha='center', fontsize=11, style='italic', color='#666666')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    # ===== NEW FEATURE METHODS =====
    
    def plot_patient_comparison(self, person_idx1, person_idx2, time_window=None):
        """Side-by-side comparison of two patients."""
        if time_window is None:
            time_window = range(self.T)
        
        # Convert time_window to list for indexing
        time_window_list = list(time_window)
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 12))
        
        for i, (pidx, label) in enumerate([(person_idx1, "Patient 1"), (person_idx2, "Patient 2")]):
            # Signature trajectories
            ax = axes[0, i]
            for k in range(self.K):
                color = 'red' if k == 5 else sns.color_palette("tab20", self.K)[k]
                theta_window = np.array([self.theta[pidx, k, t] for t in time_window_list])
                ax.plot(time_window_list, theta_window, 
                       label=f'Sig {k}', linewidth=2, color=color, alpha=0.7)
            ax.set_title(f'{label}: Signature Trajectories', fontsize=12, fontweight='bold')
            ax.set_ylabel('Proportion')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Diagnosis timeline
            ax = axes[1, i]
            diagnosis_times = self.get_diagnosis_times(pidx)
            diag_order = sorted([(d, times[0]) for d, times in diagnosis_times.items() if len(times) > 0],
                               key=lambda x: x[1])
            for j, (d, t_diag) in enumerate(diag_order[:15]):
                if t_diag not in time_window_list:
                    continue
                sig_idx = np.argmax(self.psi[:, d])
                color = 'red' if sig_idx == 5 else sns.color_palette("tab20", self.K)[sig_idx]
                t_start = min(time_window_list[0], t_diag) if time_window_list else 0
                ax.hlines(j, t_start, t_diag, colors=color, linewidth=2, alpha=0.6)
                ax.scatter(t_diag, j, color=color, s=50, edgecolors='black', zorder=5)
            if len(diag_order) > 0:
                shown_diag = [d for d, t in diag_order[:15] if t in time_window_list]
                ax.set_yticks(range(min(15, len(shown_diag))))
                ax.set_yticklabels([f"Sig {np.argmax(self.psi[:, d])} | {self.disease_names[d][:20]}" 
                                   for d in shown_diag[:15]], fontsize=7)
            ax.set_title(f'{label}: Disease Timeline', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            if len(diag_order) > 0:
                ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            # Top disease risks
            ax = axes[2, i]
            pi_t = self.compute_disease_probabilities(pidx)
            top_diseases = np.argsort(np.max(pi_t, axis=0))[-5:][::-1]
            for d in top_diseases:
                pi_window = np.array([pi_t[t, d] for t in time_window_list])
                ax.plot(time_window_list, pi_window, label=self.disease_names[d], linewidth=2)
            ax.set_title(f'{label}: Top 5 Disease Risks', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Probability')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def forecast_disease_risks(self, person_idx, forecast_years=10, age_offset=30):
        """Show model-generated disease probabilities.
        
        NOTE: This only shows what the model has actually computed (π[t] for t in [0, T-1]).
        We cannot forecast beyond the observed time window because:
        - Future predictions would require future theta (signature trajectories)
        - The model needs theta[t] to compute π[t] via phi/psi relationships
        - We don't have future theta trajectories
        
        Only shows diseases that have NOT already been diagnosed.
        """
        pi_t = self.compute_disease_probabilities(person_idx)
        T, D = pi_t.shape
        
        # Get diseases that have already been diagnosed
        diagnosed_diseases = set()
        if self.Y is not None:
            diagnosis_times = self.get_diagnosis_times(person_idx)
            diagnosed_diseases = set(diagnosis_times.keys())
        
        # Filter to only diseases that haven't been diagnosed yet
        undiagnosed_diseases = [d for d in range(D) if d not in diagnosed_diseases]
        
        if len(undiagnosed_diseases) == 0:
            # All diseases already diagnosed
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, 'All diseases have already been diagnosed.\nNo forecast available.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Disease Risk Forecast', fontsize=14, fontweight='bold')
            ax.axis('off')
            return fig
        
        ages_observed = np.arange(age_offset, age_offset + T)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get top 10 diseases by current risk (only from undiagnosed diseases)
        undiagnosed_risks = pi_t[-1, undiagnosed_diseases]
        top_indices = np.argsort(undiagnosed_risks)[-10:][::-1]
        top_diseases = [undiagnosed_diseases[i] for i in top_indices]
        
        # Add notes
        note_text = []
        if len(diagnosed_diseases) > 0:
            note_text.append(f'{len(diagnosed_diseases)} already-diagnosed disease(s) excluded')
        note_text.append(f'Showing model predictions for t ∈ [0, {T-1}]')
        note_text.append('Cannot forecast beyond observed window without future theta trajectories')
        
        ax.text(0.02, 0.98, '\n'.join(note_text), 
               transform=ax.transAxes, ha='left', va='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        colors = sns.color_palette("tab10", len(top_diseases))
        
        # Plot only model-generated probabilities (no extrapolation)
        for i, d in enumerate(top_diseases):
            ax.plot(ages_observed, pi_t[:, d], color=colors[i], linewidth=2, 
                   label=self.disease_names[d], alpha=0.8)
            # Mark the end of observed data
            ax.axvline(x=age_offset + T - 1, color='gray', linestyle=':', linewidth=2, 
                      label='Last Observed Time' if i == 0 else '')
        
        ax.set_title(f'Model-Generated Disease Probabilities\n(Undiagnosed Diseases Only, t ∈ [0, {T-1}])', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Age (years)', fontsize=12)
        ax.set_ylabel('Disease Probability (π)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([ages_observed[0], ages_observed[-1]])
        ax.set_ylim([0, max(0.05, ax.get_ylim()[1])])  # Cap y-axis to reasonable range
        
        plt.tight_layout()
        return fig
    
    def plot_signature_transitions(self, person_idx, time_window=None, age_offset=30):
        """Plot signature transitions: when signatures become dominant/decline.
        
        Same as plot_signature_transitions in realfit_app_nolr.py:
        Shows signature dominance over time and disease events that trigger transitions.
        """
        if time_window is None:
            time_window = range(self.T)
        
        # Convert to list
        time_window_list = list(time_window)
        
        theta = self.theta[person_idx]  # [K, T]
        K, T = theta.shape
        ages = np.arange(age_offset, age_offset + T)
        
        # Find dominant signature at each timepoint
        dominant_sig = np.argmax(theta, axis=0)  # [T]
        
        # Find signature transitions (when dominant signature changes)
        transitions = []
        current_dom = dominant_sig[0]
        for t in range(1, T):
            if dominant_sig[t] != current_dom:
                transitions.append({
                    'time': t,
                    'age': age_offset + t,
                    'from_sig': int(current_dom),
                    'to_sig': int(dominant_sig[t])
                })
                current_dom = dominant_sig[t]
        
        # Find disease events
        diagnosis_times = {}
        if self.Y is not None:
            diagnosis_times = self.get_diagnosis_times(person_idx)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 10), height_ratios=[2, 1.5, 1])
        
        # Panel 1: Signature trajectories with transitions
        ax1 = axes[0]
        sig_colors = sns.color_palette("tab20", K)
        
        for k in range(K):
            ax1.plot(ages[time_window_list], theta[k, time_window_list], 
                    label=f'Sig {k}', linewidth=2, color=sig_colors[k], alpha=0.7)
        
        # Highlight transitions
        for trans in transitions:
            if trans['time'] in time_window_list:
                ax1.axvline(x=trans['age'], color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                ax1.text(trans['age'], ax1.get_ylim()[1] * 0.95, 
                        f"Sig {trans['from_sig']}→{trans['to_sig']}", 
                        rotation=90, va='top', ha='right', fontsize=8, fontweight='bold')
        
        # Mark disease events
        if self.clusters is not None:
            for d, times in diagnosis_times.items():
                sig = int(self.clusters[d]) if d < len(self.clusters) else 0
                color = sig_colors[sig] if 0 <= sig < K else 'gray'
                for t in times:
                    if t in time_window_list and t < T:
                        ax1.axvline(x=age_offset + t, color=color, linestyle=':', alpha=0.3, linewidth=0.8)
        
        ax1.set_ylabel('Signature Loading (θ)', fontsize=12, fontweight='bold')
        ax1.set_title('Signature Transitions Over Time', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim([ages[time_window_list[0]], ages[time_window_list[-1]]])
        
        # Panel 2: Dominant signature over time
        ax2 = axes[1]
        for k in range(K):
            mask = (dominant_sig[time_window_list] == k)
            if np.any(mask):
                mask_ages = np.array(ages)[np.array(time_window_list)[mask]]
                mask_indices = np.array(time_window_list)[mask]
                ax2.fill_between(mask_ages, k-0.4, k+0.4, color=sig_colors[k], alpha=0.7, label=f'Sig {k}')
        
        # Mark transitions
        for trans in transitions:
            if trans['time'] in time_window_list:
                ax2.axvline(x=trans['age'], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax2.set_ylabel('Dominant Signature', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Age (years)', fontsize=12)
        ax2.set_yticks(range(K))
        ax2.set_yticklabels([f'Sig {k}' for k in range(K)])
        ax2.set_ylim([-0.5, K-0.5])
        ax2.grid(True, alpha=0.2, axis='y')
        ax2.set_xlim([ages[time_window_list[0]], ages[time_window_list[-1]]])
        
        # Panel 3: Disease events timeline
        ax3 = axes[2]
        if diagnosis_times:
            y_pos = 0
            sorted_diseases = sorted(diagnosis_times.items(), key=lambda x: min(x[1]) if x[1] else T+1)
            for d, times in sorted_diseases:
                sig = int(self.clusters[d]) if self.clusters is not None and d < len(self.clusters) else 0
                color = sig_colors[sig] if 0 <= sig < K else 'gray'
                disease_name = self.disease_names[d] if d < len(self.disease_names) else f'Disease {d}'
                
                for t in times:
                    if t in time_window_list and t < T:
                        ax3.scatter(age_offset + t, y_pos, s=60, color=color, 
                                  alpha=0.7, edgecolors='black', linewidths=0.5)
                y_pos += 1
            
            ax3.set_ylabel('Disease Events', fontsize=10, fontweight='bold')
            ax3.set_ylim([-0.5, len(diagnosis_times) - 0.5])
            ax3.set_yticks([])
        
        ax3.set_xlabel('Age (years)', fontsize=12)
        ax3.grid(True, alpha=0.2, axis='x')
        ax3.set_xlim([ages[time_window_list[0]], ages[time_window_list[-1]]])
        
        plt.tight_layout()
        return fig
    
    
    def compute_population_percentiles(self, person_idx, age_offset=30):
        """Compare patient to population distribution."""
        if self.Y is None:
            return None
        
        # Compute population statistics
        pop_theta = np.mean(self.theta, axis=(0, 2))  # [K] - average across patients and time
        pop_pi = np.zeros(self.D)
        for t in range(self.T):
            for n in range(self.N):
                theta_t = self.theta[n, :, t]
                eta_t = expit(self.phi[:, :, t])
                pop_pi += np.dot(theta_t, eta_t)
        pop_pi /= (self.N * self.T)
        
        # Patient statistics
        patient_theta = np.mean(self.theta[person_idx], axis=1)  # [K]
        patient_pi = np.mean(self.compute_disease_probabilities(person_idx), axis=0)  # [D]
        
        # Compute percentiles
        theta_percentiles = []
        pi_percentiles = []
        
        for k in range(self.K):
            pop_dist = self.theta[:, k, :].flatten()
            patient_val = patient_theta[k]
            percentile = stats.percentileofscore(pop_dist, patient_val)
            theta_percentiles.append(percentile)
        
        for d in range(self.D):
            pop_dist = []
            for t in range(self.T):
                for n in range(self.N):
                    theta_t = self.theta[n, :, t]
                    eta_t = expit(self.phi[:, d, t])
                    pop_dist.append(np.dot(theta_t, eta_t))
            pop_dist = np.array(pop_dist)
            patient_val = patient_pi[d]
            percentile = stats.percentileofscore(pop_dist, patient_val) if len(pop_dist) > 0 else 50
            pi_percentiles.append(percentile)
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Signature percentiles
        sig_colors = ['red' if k == 5 else sns.color_palette("tab20", self.K)[k] 
                     for k in range(self.K)]
        bars1 = ax1.barh(range(self.K), theta_percentiles, color=sig_colors, alpha=0.7)
        ax1.axvline(x=50, color='gray', linestyle='--', linewidth=2, label='50th percentile')
        ax1.set_yticks(range(self.K))
        ax1.set_yticklabels([f'Sig {k}' for k in range(self.K)])
        ax1.set_xlabel('Population Percentile', fontsize=12)
        ax1.set_title('Signature Loadings: Patient vs Population', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Disease risk percentiles (top 20)
        top_diseases = np.argsort(pi_percentiles)[-20:][::-1]
        disease_labels = [self.disease_names[d][:25] for d in top_diseases]
        bars2 = ax2.barh(range(len(top_diseases)), [pi_percentiles[d] for d in top_diseases], 
                        alpha=0.7, color='steelblue')
        ax2.axvline(x=50, color='gray', linestyle='--', linewidth=2, label='50th percentile')
        ax2.set_yticks(range(len(top_diseases)))
        ax2.set_yticklabels(disease_labels, fontsize=9)
        ax2.set_xlabel('Population Percentile', fontsize=12)
        ax2.set_title('Disease Risks: Patient vs Population (Top 20)', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig, {'theta_percentiles': theta_percentiles, 'pi_percentiles': pi_percentiles}
    
    
    def what_if_scenario(self, person_idx, prs_adjustments, time_window=None):
        """What-if scenario: adjust PRS values and see impact."""
        if self.G is None or self.gamma is None:
            return None
        
        if time_window is None:
            time_window = range(self.T)
        
        time_window = list(time_window)
        T_window = len(time_window)
        
        # Original trajectory (need to handle indexing correctly)
        theta_orig = np.zeros((self.K, T_window))
        for i, t in enumerate(time_window):
            theta_orig[:, i] = self.theta[person_idx, :, t]
        
        # Modified G (adjust PRS)
        G_modified = self.G[person_idx].copy()
        for prs_idx, adjustment in prs_adjustments.items():
            if 0 <= prs_idx < len(G_modified):
                G_modified[prs_idx] *= (1 + adjustment)  # e.g., 0.2 = 20% increase
        
        # Compute modified lambda
        lambda_orig = np.zeros((self.K, T_window))
        for i, t in enumerate(time_window):
            lambda_orig[:, i] = self.lambda_[person_idx, :, t]
        
        genetic_effect_orig = np.dot(self.G[person_idx], self.gamma)  # [K]
        genetic_effect_mod = np.dot(G_modified, self.gamma)  # [K]
        
        lambda_modified = lambda_orig.copy()
        for k in range(self.K):
            delta = genetic_effect_mod[k] - genetic_effect_orig[k]
            lambda_modified[k, :] += delta
        
        # Compute modified theta
        theta_modified = np.zeros_like(theta_orig)
        for t_idx in range(T_window):
            theta_modified[:, t_idx] = softmax(lambda_modified[:, t_idx])
        
        # Compare
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Original vs modified signatures
        ax = axes[0, 0]
        for k in range(min(5, self.K)):  # Show top 5 signatures
            color = 'red' if k == 5 else sns.color_palette("tab20", self.K)[k]
            ax.plot(time_window, theta_orig[k, :], color=color, linewidth=2, 
                   linestyle='-', alpha=0.7, label=f'Sig {k} (original)')
            ax.plot(time_window, theta_modified[k, :], color=color, linewidth=2, 
                   linestyle='--', alpha=0.9, label=f'Sig {k} (modified)')
        ax.set_title('Signature Trajectories: Original vs Modified', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Change in signatures
        ax = axes[0, 1]
        delta_theta = theta_modified - theta_orig
        for k in range(min(5, self.K)):
            color = 'red' if k == 5 else sns.color_palette("tab20", self.K)[k]
            ax.plot(time_window, delta_theta[k, :], color=color, linewidth=2, label=f'Sig {k}')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_title('Change in Signature Loadings', fontsize=12, fontweight='bold')
        ax.set_ylabel('Delta (modified - original)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Original vs modified disease risks (only for time window)
        pi_orig_window = np.zeros((T_window, self.D))
        pi_modified_window = np.zeros((T_window, self.D))
        
        for i, t in enumerate(time_window):
            # Original
            theta_t_orig = self.theta[person_idx, :, t]
            eta_t = expit(self.phi[:, :, t])
            pi_orig_window[i] = np.dot(theta_t_orig, eta_t)
            
            # Modified
            theta_t_mod = theta_modified[:, i]
            pi_modified_window[i] = np.dot(theta_t_mod, eta_t)
        
        ax = axes[1, 0]
        top_diseases = np.argsort(pi_orig_window.max(axis=0))[-5:][::-1]
        colors = sns.color_palette("tab10", len(top_diseases))
        for i, d in enumerate(top_diseases):
            ax.plot(time_window, pi_orig_window[:, d], color=colors[i], 
                   linewidth=2, linestyle='-', label=f'{self.disease_names[d]} (orig)')
            ax.plot(time_window, pi_modified_window[:, d], color=colors[i], 
                   linewidth=2, linestyle='--', label=f'{self.disease_names[d]} (mod)')
        ax.set_title('Disease Risks: Original vs Modified', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Change in disease risks
        ax = axes[1, 1]
        delta_pi = pi_modified_window - pi_orig_window
        for i, d in enumerate(top_diseases):
            ax.plot(time_window, delta_pi[:, d], color=colors[i], 
                   linewidth=2, label=self.disease_names[d])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_title('Change in Disease Risks', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Delta (modified - original)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Add adjustment info to title
        fig.suptitle('What-If Scenario: Genetic PRS Adjustments', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_risk_radar_chart(self, person_idx, age_selected=None, age_offset=30):
        """Radar/spider chart of top disease risks at selected age."""
        if age_selected is None:
            age_selected = age_offset + self.T // 2
        
        t_idx = max(0, min(self.T - 1, age_selected - age_offset))
        pi_t = self.compute_disease_probabilities(person_idx)
        
        # Get top 8 diseases at this age
        top_diseases = np.argsort(pi_t[t_idx, :])[-8:][::-1]
        risks = pi_t[t_idx, top_diseases]
        labels = [self.disease_names[d][:20] for d in top_diseases]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(top_diseases), endpoint=False).tolist()
        risks = risks.tolist()
        
        # Complete the circle
        angles += angles[:1]
        risks += risks[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, risks, 'o-', linewidth=2, color='steelblue', label=f'Age {age_selected}')
        ax.fill(angles, risks, alpha=0.25, color='steelblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim([0, max(risks) * 1.2])
        ax.set_title(f'Disease Risk Profile at Age {age_selected}\n(Patient {person_idx})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_signature_disease_sankey(self, person_idx, top_n_diseases=15):
        """Sankey diagram: signature loadings → disease probabilities."""
        pi_t = self.compute_disease_probabilities(person_idx)
        theta = self.theta[person_idx]
        
        # Average over time
        avg_theta = np.mean(theta, axis=1)  # [K]
        avg_pi = np.mean(pi_t, axis=0)  # [D]
        
        # Get top diseases
        top_diseases = np.argsort(avg_pi)[-top_n_diseases:][::-1]
        
        # Compute signature contributions to each disease
        contributions = np.zeros((self.K, len(top_diseases)))
        for i, d in enumerate(top_diseases):
            for k in range(self.K):
                contributions[k, i] = np.mean(theta[k, :] * expit(self.phi[k, d, :]))
        
        # Normalize contributions to sum to disease risk
        for i, d in enumerate(top_diseases):
            total = contributions[:, i].sum()
            if total > 0:
                contributions[:, i] = contributions[:, i] / total * avg_pi[d]
        
        # Create plotly Sankey (if plotly available, otherwise use matplotlib)
        if HAS_PLOTLY:
            # Prepare data for Sankey
            source = []
            target = []
            value = []
            label = []
            
            # Signatures as sources
            for k in range(self.K):
                label.append(f'Sig {k}')
            
            # Diseases as targets
            for i, d in enumerate(top_diseases):
                label.append(self.disease_names[d][:25])
            
            # Add flows
            for k in range(self.K):
                for i, d in enumerate(top_diseases):
                    if contributions[k, i] > 1e-6:
                        source.append(k)
                        target.append(self.K + i)
                        value.append(contributions[k, i] * 100)  # Scale for visibility
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=label,
                    color=["red" if i == 5 else f"hsl({i*360//self.K}, 70%, 50%)" 
                          for i in range(len(label))]
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=[f"rgba({255 if s==5 else 100}, {100}, {255}, 0.4)" for s in source]
                )
            )])
            
            fig.update_layout(title_text=f"Signature → Disease Flow (Patient {person_idx})", 
                            font_size=12, height=800)
            return fig
        else:
            # Fallback to matplotlib heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(contributions, 
                       xticklabels=[self.disease_names[d][:20] for d in top_diseases],
                       yticklabels=[f'Sig {k}' for k in range(self.K)],
                       annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'Signature Contributions to Disease Risks\n(Patient {person_idx})',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Disease', fontsize=12)
            ax.set_ylabel('Signature', fontsize=12)
            plt.tight_layout()
            return fig
    
    def export_patient_data(self, person_idx, age_offset=30):
        """Export patient data as CSV."""
        pi_t = self.compute_disease_probabilities(person_idx)
        theta = self.theta[person_idx]
        
        ages = np.arange(age_offset, age_offset + self.T)
        
        data = {
            'Age': ages,
        }
        
        # Add signature loadings
        for k in range(self.K):
            data[f'Signature_{k}'] = theta[k, :]
        
        # Add disease probabilities (top 20)
        top_diseases = np.argsort(pi_t.max(axis=0))[-20:][::-1]
        for d in top_diseases:
            data[f'Risk_{self.disease_names[d]}'] = pi_t[:, d]
        
        # Add diagnosis indicators
        if self.Y is not None:
            for d in top_diseases:
                data[f'Diagnosed_{self.disease_names[d]}'] = self.Y[person_idx, d, :]
        
        df = pd.DataFrame(data)
        return df
    
    def find_similar_patients(self, person_idx, n_similar=5, metric='signature'):
        """Find patients with similar trajectories."""
        if metric == 'signature':
            # Compare signature trajectories using correlation
            target_theta = self.theta[person_idx].flatten()
            similarities = []
            for n in range(self.N):
                if n == person_idx:
                    continue
                other_theta = self.theta[n].flatten()
                corr = np.corrcoef(target_theta, other_theta)[0, 1]
                if not np.isnan(corr):
                    similarities.append((corr, n))
        else:  # disease
            # Compare disease patterns
            if self.Y is None:
                return []
            target_diseases = set(np.where(self.Y[person_idx].sum(axis=1) > 0)[0])
            similarities = []
            for n in range(self.N):
                if n == person_idx:
                    continue
                other_diseases = set(np.where(self.Y[n].sum(axis=1) > 0)[0])
                # Jaccard similarity
                intersection = len(target_diseases & other_diseases)
                union = len(target_diseases | other_diseases)
                if union > 0:
                    jaccard = intersection / union
                    similarities.append((jaccard, n))
        
        similarities.sort(reverse=True)
        return [n for _, n in similarities[:n_similar]]

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
    # Force clear ALL caching
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear()
    
    st.set_page_config(layout="wide")
    st.title("Patient Timeline Analysis")
    
    # Author/Institution information at the top
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
    <p style='margin: 0 0 10px 0; font-size: 1.0em; font-weight: bold; color: #000;'>
    <strong>Preprint:</strong> <a href='https://www.medrxiv.org/content/10.1101/2024.09.29.24314557v3' target='_blank' style='color: #1f77b4; font-weight: bold;'>medRxiv 2024.09.29.24314557v3</a>
    </p>
    <p style='margin: 5px 0 15px 0; font-size: 0.95em; color: #000; font-weight: 500;'>
    <strong>Authors:</strong> Sarah M. Urbut, Yi Ding, Tetsushi Nakao, Satoshi Koyama, Achyutha Harish, Xilin Jiang, Leslie Gaffney, Whitney Hornsby, Jordan W. Smoller, Alexander Gusev, Pradeep Natarajan, Giovanni Parmigiani
    </p>
    <p style='margin: 15px 0 5px 0; font-size: 0.85em; color: #666; border-top: 1px solid #ccc; padding-top: 10px;'><strong>Developed by:</strong> Sarah M. Urbut, M.D., Ph.D.</p>
    <p style='margin: 0 0 10px 0; font-size: 0.85em; color: #666;'>
    <strong>Affiliations:</strong> Massachusetts General Hospital, Harvard Medical School, Broad Institute of Harvard and MIT<br>
    <strong>Email:</strong> <a href='mailto:surbut@mgh.harvard.edu' style='color: #1f77b4;'>surbut@mgh.harvard.edu</a> | 
    <strong>Project:</strong> Aladynoulli Research
    </p>
    <p style='margin: 10px 0 0 0; font-size: 0.85em; font-style: italic; color: #555; border-top: 1px solid #ccc; padding-top: 10px;'>
    This work was funded in part by the Burroughs Wellcome Fund and by the American Heart Association.<br>
    For questions or feedback, please contact: <a href='mailto:surbut@mgh.harvard.edu' style='color: #1f77b4;'>surbut@mgh.harvard.edu</a>
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer about sample patients
    st.info("""
    **Sample Patient Visualization:** This app displays pre-selected sample patients for visualization purposes only. 
    Patient identifiers are not displayed and no personally identifiable information is shown. 
    These are illustrative examples from the dataset and are not intended to represent any specific individuals.
    """)
    
    # Load model data
    model_data = load_model()
    
    # === ADD DEBUG CODE HERE ===
    print(f"=== DEBUG INFO ===")
    print(f"Model file path: app_patients_compact_nolr.pt")
    print(f"Number of patients loaded: {len(model_data['model_state_dict']['lambda_'])}")
    print(f"Expected: 145 patients")
    print(f"Actual: {len(model_data['model_state_dict']['lambda_'])} patients")
    print(f"Model meta: {model_data.get('meta', 'No meta found')}")
    # === END DEBUG CODE ===
    
    prs_names = load_prs_names('prs_names.csv')
    visualizer = PatientTimelineVisualizer(model_data, prs_names=prs_names)
    
    # Find MI disease index
# With this more robust version:
    # Find MI disease index
    mi_idx = None
    for i, name in enumerate(visualizer.disease_names):
        # Handle DataFrame format - extract the actual name value
        if hasattr(name, 'item'):
            name_str = str(name.item())
        elif hasattr(name, 'values'):
            name_str = str(name.values[0])
        else:
            name_str = str(name)
        
        name_lower = name_str.lower()
        # More specific MI detection - avoid false positives
        if any(term in name_lower for term in ["myocardial infarction", "heart attack"]):
            mi_idx = i
            print(f"Found MI at index {i}: {name_str}")
            break
        # Only check "mi" if it's a standalone term, not part of other words
        elif " mi " in f" {name_lower} " or name_lower == "mi":
            mi_idx = i
            print(f"Found MI at index {i}: {name_str}")
            break

    # Debug: show what we found
    if mi_idx is not None:
        print(f"MI disease index: {mi_idx}")
        print(f"MI disease name: {visualizer.disease_names[mi_idx]}")
    else:
        print("WARNING: Could not find MI disease index!")
        print("Available disease names:")
        for i, name in enumerate(visualizer.disease_names):
            # Handle DataFrame format for display too
            if hasattr(name, 'item'):
                name_str = str(name.item())
            elif hasattr(name, 'values'):
                name_str = str(name.values[0])

    # Select default patient
    # Strategy: Prefer a patient with MI and high signature 5 loading at diagnosis time
    # Otherwise, default to first patient (index 0)
    good_patients = list(range(visualizer.N))  # All patients are available
    
    # Default to first patient (index 0)
    default_person_idx = 0
    
    # If MI disease is found, try to find a patient with MI and high sig5
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
        print(f"Default patient: {default_person_idx} (highest sig5 at MI diagnosis: {best_score:.3f})" if best_score > -np.inf else f"Default patient: {default_person_idx} (no MI patients found, using first)")
    
    # Sidebar controls
    st.sidebar.header("Sample Patient Selection")
    st.sidebar.caption("Select from pre-selected sample patients (no identifying information)")
    
    # Patient search/filter
    search_term = st.sidebar.text_input("🔍 Search by disease name", "", key="patient_search")
    filtered_patients = good_patients
    
    if search_term and visualizer.Y is not None:
        # Find patients with diseases matching search term
        search_lower = search_term.lower()
        matching_diseases = [i for i, name in enumerate(visualizer.disease_names) 
                           if search_lower in str(name).lower()]
        if matching_diseases:
            # Find patients with these diseases
            matching_patients = []
            for pidx in good_patients:
                if any(np.any(visualizer.Y[pidx, d, :] > 0.5) for d in matching_diseases):
                    matching_patients.append(pidx)
            if matching_patients:
                filtered_patients = matching_patients
                st.sidebar.success(f"Found {len(matching_patients)} patients with matching diseases")
            else:
                st.sidebar.warning("No patients found with matching diseases")
        else:
            st.sidebar.warning("No diseases match search term")
    
    # Similar patients quick access
    show_similar = st.sidebar.checkbox("Show similar patients", key="show_similar_sidebar")
    if show_similar and visualizer.Y is not None:
        similar_pids = visualizer.find_similar_patients(default_person_idx, 3, 'signature')
        if similar_pids:
            st.sidebar.write("**Similar patients:**")
            for pidx in similar_pids:
                st.sidebar.write(f"- Sample Patient {pidx}")
    
    person_idx = st.sidebar.selectbox(
        "Select Sample Patient",
        options=filtered_patients,
        index=filtered_patients.index(default_person_idx) if default_person_idx in filtered_patients else 0,
        format_func=lambda x: f"Sample Patient {x}"
    )
    
    # Time window selection
    st.sidebar.header("Time Window")
    time_start = st.sidebar.slider("Start Time", 0, visualizer.T-1, 0)
    time_end = st.sidebar.slider("End Time", time_start, visualizer.T-1, visualizer.T-1)
    time_window = range(time_start, time_end + 1)
    
    # Add tabs for all features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Plots", "Tables", "Genetics", "Risk Summary", 
        "Compare Patients", "Forecast", "Advanced", "Export"
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
            # Use the most common disease as default for highlighting
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
        person_idx_cf = st.number_input("Sample Patient Index", min_value=0, max_value=visualizer.N-1, value=1)
        signature_idx_cf = st.number_input("Signature Index", min_value=0, max_value=visualizer.K-1, value=5)
        fig_cf = visualizer.counterfactual_signature_trajectory(person_idx_cf, signature_idx_cf)
        if fig_cf:
            st.pyplot(fig_cf)
        st.markdown("---")
        st.header("Genetic Effect for Each Signature (Selected Sample Patient, Stacked by PRS Feature)")
        st.markdown("This stacked barplot shows the overall genetic effect for each signature for the currently selected sample patient, with each bar colored by the fractional contribution of each PRS feature (G*gamma).")
        fig_bar = visualizer.plot_patient_genetic_effect_barplot(person_idx)
        if fig_bar:
            st.pyplot(fig_bar)
        st.markdown("---")
        st.header("Genetic Impact on Signatures Over Time (Direct Weighting, Selected Sample Patient)")
        st.markdown("This plot shows the direct genetic impact on each signature over time for the currently selected sample patient, using their PRS and the model's gamma weights.")
        fig_impact = visualizer.plot_genetic_impact_on_signatures(person_idx, time_window)
        if fig_impact:
            st.pyplot(fig_impact)
        st.markdown("---")

    with tab4:
        st.markdown("### Comprehensive Disease Trajectory Analysis")
        st.markdown("**Multi-panel visualization showing signature evolution, disease timeline, and risk trajectories**")
        
        # Age offset input
        age_offset = st.sidebar.number_input("Age at Baseline (years)", min_value=0, max_value=100, value=30, key="age_offset_tab4")
        
        # Load cluster assignments if available
        cluster_assignments = None
        clusters_path = Path(__file__).parent / 'initial_clusters_400k.pt'
        if not clusters_path.exists():
            clusters_path = Path('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/initial_clusters_400k.pt')
        
        if clusters_path.exists():
            try:
                clusters = torch.load(clusters_path, map_location='cpu', weights_only=False)
                if isinstance(clusters, torch.Tensor):
                    cluster_assignments = clusters.cpu().numpy()
                elif isinstance(clusters, dict) and 'clusters' in clusters:
                    cluster_assignments = clusters['clusters']
                    if isinstance(cluster_assignments, torch.Tensor):
                        cluster_assignments = cluster_assignments.cpu().numpy()
            except:
                pass
        
        # Generate comprehensive timeline plot
        fig_comprehensive = visualizer.plot_comprehensive_timeline(
            person_idx, 
            age_offset=age_offset,
            cluster_assignments=cluster_assignments,
            figsize=(20, 14)
        )
        if fig_comprehensive:
            st.pyplot(fig_comprehensive)
            plt.close(fig_comprehensive)

    # Tab 5: Patient Comparison
    with tab5:
        st.header("Compare Two Patients")
        st.markdown("**Side-by-side comparison of two patients showing:**")
        st.markdown("- **Signature trajectories**: How signature loadings evolve over time")
        st.markdown("- **Disease timeline**: Chronological diagnoses, colored by primary signature")
        st.markdown("- **Top 5 disease risks**: Probability trajectories for highest-risk diseases")
        
        col1, col2 = st.columns(2)
        with col1:
            person_idx1 = st.selectbox("Patient 1", good_patients, 
                                       index=good_patients.index(person_idx),
                                       format_func=lambda x: f"Sample Patient {x}",
                                       key="compare_p1")
        with col2:
            person_idx2 = st.selectbox("Patient 2", good_patients,
                                       index=0 if person_idx == 0 else good_patients.index(person_idx) - 1,
                                       format_func=lambda x: f"Sample Patient {x}",
                                       key="compare_p2")
        
        # Show comparison automatically when patients are selected
        if person_idx1 is not None and person_idx2 is not None:
            try:
                compare_fig = visualizer.plot_patient_comparison(person_idx1, person_idx2, time_window)
                st.pyplot(compare_fig)
                plt.close(compare_fig)
            except Exception as e:
                st.error(f"Error generating comparison: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Tab 6: Forecast
    with tab6:
        st.header("Model-Generated Disease Probabilities")
        st.markdown("""
        **What this shows:**
        - **Only model-generated probabilities** (π[t] for t ∈ [0, T-1]) - no extrapolation
        - Shows **first-time disease risk** for diseases that have NOT yet been diagnosed
        - All predictions are from the actual model (using theta, phi, psi relationships)
        
        **Important:**
        - Only shows diseases that haven't occurred yet (already-diagnosed diseases are excluded)
        - Once a disease has been diagnosed, we can't predict "first-time" risk for it anymore
        - **Cannot forecast beyond observed time window** because:
          - Would require future theta (signature trajectories)
          - Model needs theta[t] to compute π[t] via phi/psi
          - We don't have future theta trajectories for these sample patients
        - This is for sample patients with pre-computed trajectories
        
        **Note:** True forecasting would require:
        1. Predicting future theta (signature loadings)
        2. Then computing π from those using phi/psi relationships
        Currently we only show what the model has computed for the observed timepoints.
        """)
        
        age_offset_forecast = st.number_input("Age at Baseline", 
                                              min_value=0, max_value=100, 
                                              value=30, key="age_forecast")
        
        # Show model predictions automatically
        try:
            forecast_fig = visualizer.forecast_disease_risks(person_idx, forecast_years=10, age_offset=age_offset_forecast)
            st.pyplot(forecast_fig)
            plt.close(forecast_fig)
            st.info("ℹ️ **Note:** This shows only model-generated probabilities (π[t] for observed timepoints). We cannot forecast beyond the observed window because future predictions would require future theta (signature trajectories), which we don't have for these sample patients.")
        except Exception as e:
            st.error(f"Error generating plot: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 7: Advanced Features
    with tab7:
        st.header("Signature Transitions")
        st.markdown("**Signature transitions over time showing when signatures become dominant or decline, and disease events that trigger transitions**")
        st.info("""
        **What to look for:**
        - **Red dashed lines**: Signature transitions (when dominant signature changes)
        - **Colored dots**: Disease events, colored by their primary signature
        - **Panel 2**: Shows which signature is dominant at each timepoint
        - **Panel 3**: Timeline of all disease events
        """)
        age_offset_transitions = st.number_input("Age at Baseline", 
                                                min_value=0, max_value=100, 
                                                value=30, key="age_transitions", 
                                                help="Age at time 0 (baseline)")
        
        # Show transitions plot automatically
        try:
            transition_fig = visualizer.plot_signature_transitions(person_idx, time_window, age_offset_transitions)
            st.pyplot(transition_fig)
            plt.close(transition_fig)
        except Exception as e:
            st.error(f"Error generating transition plot: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        st.markdown("---")
        st.header("Advanced Analysis Features")
        
        # Population Comparison
        st.subheader("Population Percentiles Comparison")
        age_offset_percentiles = st.sidebar.number_input("Age at Baseline (for percentiles)", 
                                                         min_value=0, max_value=100, 
                                                         value=30, key="age_percentiles")
        if st.button("Compare to Population", key="percentile_btn"):
            percentile_fig, percentile_data = visualizer.compute_population_percentiles(
                person_idx, age_offset_percentiles)
            st.pyplot(percentile_fig)
            plt.close(percentile_fig)
            
            with st.expander("Percentile Details"):
                st.write("**Signature Percentiles:**")
                for k in range(visualizer.K):
                    pct = percentile_data['theta_percentiles'][k]
                    st.write(f"  Sig {k}: {pct:.1f}th percentile")
                
                st.write("\n**Top Disease Risk Percentiles:**")
                top_diseases = np.argsort(percentile_data['pi_percentiles'])[-10:][::-1]
                for d in top_diseases:
                    pct = percentile_data['pi_percentiles'][d]
                    st.write(f"  {visualizer.disease_names[d][:40]}: {pct:.1f}th percentile")
        
        st.markdown("---")
        
        # What-If Scenarios
        st.subheader("What-If Scenario: Genetic PRS Adjustments")
        st.markdown("Adjust PRS values and see predicted impact on signatures and disease risks.")
        
        if visualizer.prs_names and visualizer.G is not None and visualizer.gamma is not None:
            num_adjustments = st.number_input("Number of PRS to adjust", 1, 5, 1, key="num_adj")
            prs_adjustments = {}
            
            for i in range(num_adjustments):
                col1, col2 = st.columns(2)
                with col1:
                    prs_idx = st.selectbox(f"PRS Feature {i+1}", 
                                          range(min(len(visualizer.prs_names), visualizer.G.shape[1])),
                                          format_func=lambda x: visualizer.prs_names[x] if x < len(visualizer.prs_names) else f"Feature {x}",
                                          key=f"prs_select_{i}")
                with col2:
                    adjustment = st.slider(f"Adjustment %", -50, 50, 0, 
                                          format="%d%%", key=f"prs_adj_{i}")
                    prs_adjustments[prs_idx] = adjustment / 100.0  # Convert to fraction
            
            if st.button("Run What-If Scenario", key="whatif_btn"):
                whatif_fig = visualizer.what_if_scenario(person_idx, prs_adjustments, time_window)
                if whatif_fig:
                    st.pyplot(whatif_fig)
                    plt.close(whatif_fig)
        else:
            st.warning("PRS data or genetic effects not available for what-if scenarios.")
        
        st.markdown("---")
        
        # Risk Radar Chart
        st.subheader("Disease Risk Radar Chart")
        age_offset_radar = st.sidebar.number_input("Age at Baseline (for radar)", 
                                                   min_value=0, max_value=100, 
                                                   value=30, key="age_radar")
        age_selected = st.slider("Select Age for Risk Profile", 
                                age_offset_radar, age_offset_radar + visualizer.T - 1,
                                age_offset_radar + visualizer.T // 2, key="age_radar_select")
        
        if st.button("Generate Radar Chart", key="radar_btn"):
            radar_fig = visualizer.plot_risk_radar_chart(person_idx, age_selected, age_offset_radar)
            st.pyplot(radar_fig)
            plt.close(radar_fig)
        
        st.markdown("---")
        
        # Signature-Disease Sankey
        st.subheader("Signature → Disease Flow (Sankey Diagram)")
        top_n_sankey = st.slider("Top N Diseases", 5, 20, 15, key="top_n_sankey")
        if st.button("Generate Sankey Diagram", key="sankey_btn"):
            try:
                sankey_fig = visualizer.plot_signature_disease_sankey(person_idx, top_n_sankey)
                # Check if it's a plotly figure or matplotlib figure
                if hasattr(sankey_fig, 'show'):
                    st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.pyplot(sankey_fig)
                    plt.close(sankey_fig)
            except Exception as e:
                st.error(f"Error generating Sankey diagram: {e}")
                st.info("Falling back to heatmap visualization.")
                sankey_fig = visualizer.plot_signature_disease_sankey(person_idx, top_n_sankey)
                if hasattr(sankey_fig, 'show'):
                    st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.pyplot(sankey_fig)
                    plt.close(sankey_fig)
        
        st.markdown("---")
        
        # Find Similar Patients
        st.subheader("Find Similar Patients")
        similarity_metric = st.radio("Similarity Metric", ["Signature trajectories", "Disease patterns"], key="sim_metric")
        n_similar = st.slider("Number of Similar Patients", 1, 10, 5, key="n_similar")
        
        if st.button("Find Similar Patients", key="similar_btn"):
            metric = 'signature' if similarity_metric == "Signature trajectories" else 'disease'
            similar_patients = visualizer.find_similar_patients(person_idx, n_similar, metric)
            
            if similar_patients:
                st.write(f"**Most similar patients to Patient {person_idx}:**")
                for i, pidx in enumerate(similar_patients):
                    st.write(f"{i+1}. Sample Patient {pidx}")
                    
                    # Quick comparison
                    if st.checkbox(f"Show comparison with Patient {pidx}", key=f"compare_sim_{pidx}"):
                        compare_fig = visualizer.plot_patient_comparison(person_idx, pidx, time_window)
                        st.pyplot(compare_fig)
                        plt.close(compare_fig)
            else:
                st.warning("No similar patients found.")

    # Tab 8: Export
    with tab8:
        st.header("Export Patient Data")
        
        export_format = st.radio("Export Format", ["CSV", "PDF Report (coming soon)"], key="export_format")
        
        if export_format == "CSV":
            age_offset_export = st.sidebar.number_input("Age at Baseline (for export)", 
                                                        min_value=0, max_value=100, 
                                                        value=30, key="age_export")
            
            if st.button("Export to CSV", key="export_csv_btn"):
                df = visualizer.export_patient_data(person_idx, age_offset_export)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"patient_{person_idx}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
                
                st.success("Data prepared! Click 'Download CSV' to save.")
                st.dataframe(df.head(20))
        
        st.markdown("---")
        
        st.header("Export Current Plot")
        st.markdown("Right-click on any plot above and select 'Save image as...' to export.")
        st.info("📊 For high-resolution exports, we recommend using the PDF export (coming soon) or taking screenshots at full resolution.")

if __name__ == "__main__":
    main() 