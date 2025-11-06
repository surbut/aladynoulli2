"""
Bayesian Pathway Transition Model (BPTM)

A Bayesian model for predicting disease transitions based on signature deviation trajectories.
Extends the aladynoulli framework to pathway-level transition analysis.

Model:
  π_i(t | d_precursor) = κ · sigmoid(η_i(t))
  
  η_i(t) = α_d + β_d · (t - t_precursor) + 
           Σ_k γ_k · δ_i,k,t + 
           Σ_k Σ_τ ω_k,τ · δ_i,k,t-τ +
           G_i^T · Γ_d

where δ_i,k,t = θ_i,k,t - μ_k,t (signature deviation from population reference)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class BayesianPathwayTransitionModel:
    """
    Bayesian model for disease transition prediction based on signature deviations.
    """
    
    def __init__(
        self,
        K: int,  # Number of signatures
        T: int,  # Number of timepoints
        P: int = 0,  # Number of genetic/demographic factors
        lookback_window: int = 10,  # Years before transition to consider
        n_pathways: int = 4,  # Number of pathways
        use_pathway_effects: bool = True,  # Whether to include pathway-specific effects
        device: str = 'cpu'
    ):
        """
        Initialize the Bayesian Pathway Transition Model.
        
        Parameters:
        -----------
        K : int
            Number of signatures
        T : int
            Number of timepoints
        P : int
            Number of genetic/demographic factors
        lookback_window : int
            Number of timepoints to look back for lagged effects
        n_pathways : int
            Number of pathways
        use_pathway_effects : bool
            Whether to include pathway-specific signature effects
        device : str
            Device to run on ('cpu' or 'cuda')
        """
        self.K = K
        self.T = T
        self.P = P
        self.lookback_window = lookback_window
        self.n_pathways = n_pathways
        self.use_pathway_effects = use_pathway_effects
        self.device = device
        
        # Initialize parameters (will be learned)
        self.alpha = {}  # Baseline log-odds per precursor disease
        self.beta = {}   # Time trend per precursor disease
        self.gamma = None  # Signature effects (K,)
        self.omega = None  # Lagged effects (K, lookback_window)
        self.Gamma = {}   # Genetic effects per precursor disease
        self.gamma_pathway = None  # Pathway-specific effects (K, n_pathways)
        self.kappa = 1.0  # Global calibration parameter
        
        # Population reference (computed from data)
        self.mu_k_t = None  # Population reference (K, T)
        
    def compute_signature_deviations(
        self, 
        thetas: np.ndarray, 
        population_reference: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute signature deviations from population reference.
        
        Parameters:
        -----------
        thetas : np.ndarray
            Signature loadings (N, K, T)
        population_reference : np.ndarray, optional
            Population reference (K, T). If None, computed from thetas.
            
        Returns:
        --------
        deviations : np.ndarray
            Signature deviations (N, K, T)
        """
        if population_reference is None:
            population_reference = np.mean(thetas, axis=0)  # (K, T)
        
        self.mu_k_t = population_reference
        
        # Compute deviations
        deviations = thetas - population_reference[np.newaxis, :, :]  # (N, K, T)
        
        return deviations
    
    def compute_transition_features(
        self,
        deviations: np.ndarray,
        t_precursor: int,
        genetic_factors: Optional[np.ndarray] = None,
        pathway_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute features for transition probability at time t.
        
        Parameters:
        -----------
        deviations : np.ndarray
            Signature deviations (K, T) for a single patient
        t_precursor : int
            Timepoint of precursor disease
        genetic_factors : np.ndarray, optional
            Genetic/demographic factors (P,)
        pathway_id : int, optional
            Pathway assignment for this patient
            
        Returns:
        --------
        features : np.ndarray
            Feature vector for transition prediction
        """
        K, T = deviations.shape
        
        # Current deviations (at time t_precursor)
        current_deviations = deviations[:, t_precursor]  # (K,)
        
        # Lagged deviations (lookback_window timepoints before)
        lagged_features = []
        for tau in range(1, min(self.lookback_window + 1, t_precursor + 1)):
            t_lag = t_precursor - tau
            if t_lag >= 0:
                lagged_deviations = deviations[:, t_lag]  # (K,)
                lagged_features.append(lagged_deviations)
            else:
                lagged_features.append(np.zeros(K))
        
        # Stack features
        features = np.concatenate([
            current_deviations,  # K features
            np.concatenate(lagged_features)  # K * lookback_window features
        ])
        
        # Add genetic factors if provided
        if genetic_factors is not None:
            features = np.concatenate([features, genetic_factors])
        
        return features
    
    def predict_transition_logit(
        self,
        deviations: np.ndarray,
        t_precursor: int,
        precursor_disease: str,
        genetic_factors: Optional[np.ndarray] = None,
        pathway_id: Optional[int] = None,
        t: Optional[int] = None
    ) -> float:
        """
        Predict transition logit η_i(t) for a patient.
        
        Parameters:
        -----------
        deviations : np.ndarray
            Signature deviations (K, T)
        t_precursor : int
            Timepoint of precursor disease
        precursor_disease : str
            Name of precursor disease
        genetic_factors : np.ndarray, optional
            Genetic factors (P,)
        pathway_id : int, optional
            Pathway assignment
        t : int, optional
            Timepoint to predict at. If None, uses t_precursor + 1.
            
        Returns:
        --------
        logit : float
            Transition logit η_i(t)
        """
        if t is None:
            t = t_precursor + 1
        
        # Baseline
        alpha_d = self.alpha.get(precursor_disease, 0.0)
        beta_d = self.beta.get(precursor_disease, 0.0)
        baseline = alpha_d + beta_d * (t - t_precursor)
        
        # Current signature effects
        current_deviations = deviations[:, t_precursor]  # (K,)
        if self.gamma is not None:
            signature_contribution = np.dot(self.gamma, current_deviations)
        else:
            signature_contribution = 0.0
        
        # Lagged effects
        lagged_contribution = 0.0
        if self.omega is not None:
            for tau in range(1, min(self.lookback_window + 1, t_precursor + 1)):
                t_lag = t_precursor - tau
                if t_lag >= 0:
                    lagged_deviations = deviations[:, t_lag]  # (K,)
                    lagged_contribution += np.dot(self.omega[:, tau-1], lagged_deviations)
        
        # Pathway-specific effects
        pathway_contribution = 0.0
        if self.use_pathway_effects and pathway_id is not None and self.gamma_pathway is not None:
            pathway_contribution = np.dot(self.gamma_pathway[:, pathway_id], current_deviations)
        
        # Genetic effects
        genetic_contribution = 0.0
        if genetic_factors is not None and precursor_disease in self.Gamma:
            genetic_contribution = np.dot(genetic_factors, self.Gamma[precursor_disease])
        
        # Total logit
        logit = baseline + signature_contribution + lagged_contribution + pathway_contribution + genetic_contribution
        
        return logit
    
    def predict_transition_probability(
        self,
        deviations: np.ndarray,
        t_precursor: int,
        precursor_disease: str,
        genetic_factors: Optional[np.ndarray] = None,
        pathway_id: Optional[int] = None,
        t: Optional[int] = None
    ) -> float:
        """
        Predict transition probability π_i(t).
        
        Parameters:
        -----------
        deviations : np.ndarray
            Signature deviations (K, T)
        t_precursor : int
            Timepoint of precursor disease
        precursor_disease : str
            Name of precursor disease
        genetic_factors : np.ndarray, optional
            Genetic factors (P,)
        pathway_id : int, optional
            Pathway assignment
        t : int, optional
            Timepoint to predict at
            
        Returns:
        --------
        probability : float
            Transition probability π_i(t)
        """
        logit = self.predict_transition_logit(
            deviations, t_precursor, precursor_disease, 
            genetic_factors, pathway_id, t
        )
        
        probability = self.kappa * (1.0 / (1.0 + np.exp(-logit)))
        
        return probability
    
    def fit_mcmc(
        self,
        Y: np.ndarray,
        thetas: np.ndarray,
        disease_names: List[str],
        precursor_disease: str,
        target_disease: str,
        pathway_labels: Optional[np.ndarray] = None,
        genetic_factors: Optional[np.ndarray] = None,
        n_iterations: int = 1000,
        burn_in: int = 200
    ):
        """
        Fit model using MCMC (simplified version - placeholder for full implementation).
        
        This is a placeholder that initializes parameters. Full MCMC implementation
        would require sampling from posterior distributions.
        
        Parameters:
        -----------
        Y : np.ndarray
            Disease outcomes (N, D, T)
        thetas : np.ndarray
            Signature loadings (N, K, T)
        disease_names : List[str]
            List of disease names
        precursor_disease : str
            Name of precursor disease
        target_disease : str
            Name of target disease
        pathway_labels : np.ndarray, optional
            Pathway assignments (N,)
        genetic_factors : np.ndarray, optional
            Genetic factors (N, P)
        n_iterations : int
            Number of MCMC iterations
        burn_in : int
            Number of burn-in iterations
        """
        print("="*80)
        print("FITTING BAYESIAN PATHWAY TRANSITION MODEL")
        print("="*80)
        
        N, K, T = thetas.shape
        assert K == self.K, f"Signature dimension mismatch: {K} != {self.K}"
        assert T == self.T, f"Time dimension mismatch: {T} != {self.T}"
        
        # Find disease indices
        precursor_idx = None
        target_idx = None
        for i, name in enumerate(disease_names):
            if precursor_disease.lower() in name.lower():
                precursor_idx = i
            if target_disease.lower() in name.lower():
                target_idx = i
        
        if precursor_idx is None or target_idx is None:
            raise ValueError(f"Could not find {precursor_disease} or {target_disease}")
        
        # Compute signature deviations
        print("\n1. Computing signature deviations...")
        deviations = self.compute_signature_deviations(thetas)  # (N, K, T)
        print(f"   Deviations shape: {deviations.shape}")
        
        # Find patients with precursor disease
        print("\n2. Identifying transition patients...")
        transition_patients = []
        non_transition_patients = []
        
        for i in range(N):
            # Check if patient has precursor disease
            if Y[i, precursor_idx, :].sum() > 0:
                precursor_times = np.where(Y[i, precursor_idx, :] > 0)[0]
                t_precursor = precursor_times[0]  # First occurrence
                
                # Check if they developed target disease after precursor
                if t_precursor < T - 1:
                    target_after = Y[i, target_idx, t_precursor+1:].sum() > 0
                    
                    patient_data = {
                        'patient_id': i,
                        't_precursor': t_precursor,
                        'deviations': deviations[i, :, :],  # (K, T)
                        'pathway_id': pathway_labels[i] if pathway_labels is not None else None,
                        'genetic': genetic_factors[i, :] if genetic_factors is not None else None
                    }
                    
                    if target_after:
                        target_times = np.where(Y[i, target_idx, t_precursor+1:] > 0)[0]
                        patient_data['t_target'] = t_precursor + 1 + target_times[0]
                        transition_patients.append(patient_data)
                    else:
                        non_transition_patients.append(patient_data)
        
        print(f"   Found {len(transition_patients)} transition patients")
        print(f"   Found {len(non_transition_patients)} non-transition patients")
        
        # Initialize parameters (simplified - would use proper priors in full MCMC)
        print("\n3. Initializing parameters...")
        
        # Baseline transition rate (empirical)
        if len(transition_patients) > 0:
            self.alpha[precursor_disease] = np.log(len(transition_patients) / len(non_transition_patients)) if len(non_transition_patients) > 0 else 0.0
        else:
            self.alpha[precursor_disease] = -2.0  # Low baseline
        
        self.beta[precursor_disease] = 0.0  # No time trend initially
        
        # Signature effects (initialize to small random values)
        self.gamma = np.random.normal(0, 0.1, size=K)
        
        # Lagged effects
        self.omega = np.random.normal(0, 0.05, size=(K, self.lookback_window))
        
        # Pathway-specific effects
        if self.use_pathway_effects and pathway_labels is not None:
            self.gamma_pathway = np.random.normal(0, 0.1, size=(K, self.n_pathways))
        
        # Genetic effects
        if genetic_factors is not None:
            self.Gamma[precursor_disease] = np.random.normal(0, 0.1, size=P)
        
        print("   ✅ Parameters initialized")
        print(f"   α_{precursor_disease} = {self.alpha[precursor_disease]:.3f}")
        print(f"   γ shape: {self.gamma.shape}")
        
        # TODO: Implement full MCMC sampling
        print("\n4. MCMC sampling (placeholder - not yet implemented)")
        print("   In full implementation, would sample from:")
        print("   - p(α, β | data)")
        print("   - p(γ | data)")
        print("   - p(ω | data)")
        print("   - p(γ_pathway | data)")
        print("   - p(Γ | data)")
        
        return {
            'transition_patients': transition_patients,
            'non_transition_patients': non_transition_patients,
            'n_iterations': n_iterations,
            'burn_in': burn_in
        }
    
    def predict_transition_trajectories(
        self,
        deviations: np.ndarray,
        t_precursor: int,
        precursor_disease: str,
        genetic_factors: Optional[np.ndarray] = None,
        pathway_id: Optional[int] = None,
        max_horizon: int = 10
    ) -> np.ndarray:
        """
        Predict transition probability trajectory for a patient.
        
        Parameters:
        -----------
        deviations : np.ndarray
            Signature deviations (K, T)
        t_precursor : int
            Timepoint of precursor disease
        precursor_disease : str
            Name of precursor disease
        genetic_factors : np.ndarray, optional
            Genetic factors (P,)
        pathway_id : int, optional
            Pathway assignment
        max_horizon : int
            Maximum timepoints ahead to predict
            
        Returns:
        --------
        probabilities : np.ndarray
            Transition probabilities (max_horizon,)
        """
        probabilities = []
        
        for t_offset in range(1, max_horizon + 1):
            t = t_precursor + t_offset
            if t < self.T:
                prob = self.predict_transition_probability(
                    deviations, t_precursor, precursor_disease,
                    genetic_factors, pathway_id, t
                )
                probabilities.append(prob)
            else:
                probabilities.append(0.0)
        
        return np.array(probabilities)


def example_usage():
    """
    Example of how to use the Bayesian Pathway Transition Model.
    """
    print("="*80)
    print("BAYESIAN PATHWAY TRANSITION MODEL - EXAMPLE USAGE")
    print("="*80)
    
    # Initialize model
    K = 21  # Number of signatures
    T = 50  # Number of timepoints
    P = 10  # Number of genetic factors
    
    model = BayesianPathwayTransitionModel(
        K=K,
        T=T,
        P=P,
        lookback_window=10,
        n_pathways=4,
        use_pathway_effects=True
    )
    
    print("\n✅ Model initialized")
    print(f"   Signatures: {K}")
    print(f"   Timepoints: {T}")
    print(f"   Lookback window: {model.lookback_window}")
    print(f"   Pathways: {model.n_pathways}")
    
    # Example: Create synthetic data
    print("\n" + "-"*80)
    print("SYNTHETIC DATA EXAMPLE")
    print("-"*80)
    
    N = 1000
    thetas = np.random.dirichlet([1.0] * K, size=(N, T)).transpose(0, 2, 1)  # (N, K, T)
    
    # Compute deviations
    deviations_example = model.compute_signature_deviations(thetas)
    print(f"\n✅ Computed deviations: {deviations_example.shape}")
    
    # Example prediction for a single patient
    patient_deviations = deviations_example[0, :, :]  # (K, T)
    t_precursor = 30
    
    logit = model.predict_transition_logit(
        patient_deviations,
        t_precursor,
        precursor_disease="Rheumatoid arthritis"
    )
    
    prob = model.predict_transition_probability(
        patient_deviations,
        t_precursor,
        precursor_disease="Rheumatoid arthritis"
    )
    
    print(f"\n✅ Example prediction:")
    print(f"   Logit: {logit:.3f}")
    print(f"   Probability: {prob:.3f}")
    
    # Predict trajectory
    trajectory = model.predict_transition_trajectories(
        patient_deviations,
        t_precursor,
        precursor_disease="Rheumatoid arthritis",
        max_horizon=10
    )
    
    print(f"\n✅ Transition probability trajectory (next 10 timepoints):")
    print(f"   {trajectory}")
    
    print("\n" + "="*80)
    print("NOTE: Full MCMC fitting not yet implemented")
    print("This is a framework for future development")
    print("="*80)


if __name__ == "__main__":
    example_usage()


