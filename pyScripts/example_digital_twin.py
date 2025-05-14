import torch
import numpy as np
from digital_twin import DigitalTwin

# Load your trained model
# Replace this with your actual model loading code
model_path = 'your_model_path.pt'
checkpoint = torch.load(model_path)

# Initialize model with saved parameters
model = AladynSurvivalFixedKernelsAvgLoss_clust_logitInit_psitest(
    N=checkpoint['hyperparameters']['N'],
    D=checkpoint['hyperparameters']['D'],
    T=checkpoint['hyperparameters']['T'],
    K=checkpoint['hyperparameters']['K'],
    P=checkpoint['hyperparameters']['P'],
    G=checkpoint['G'],
    Y=checkpoint['Y'],
    prevalence_t=checkpoint['prevalence_t'],
    disease_names=checkpoint['disease_names']
)

# Load the state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create digital twin manager
twin_manager = DigitalTwin(model, disease_names=checkpoint['disease_names'])

# Example 1: Create a digital twin for a new individual
# Generate a random genetic profile (replace with real data)
random_genetic_profile = np.random.randn(model.P)
twin_data = twin_manager.create_twin(random_genetic_profile)

# Plot the twin's trajectories
twin_manager.plot_twin_trajectories(twin_data, selected_diseases=[0, 1, 2, 3])

# Example 2: Simulate an intervention
# Let's say we want to reduce the effect of signature 0 by 30%
intervention_data = twin_manager.simulate_intervention(
    twin_data,
    intervention_type='reduce',
    target_signature=0,
    effect_size=0.3
)

# Plot original vs intervention trajectories
twin_manager.plot_twin_trajectories(
    twin_data,
    modified_data=intervention_data,
    selected_diseases=[0, 1, 2, 3]
)

# Example 3: Compare two different genetic profiles
# Create another twin with different genetic profile
another_genetic_profile = np.random.randn(model.P)
twin2_data = twin_manager.create_twin(another_genetic_profile)

# Compare the twins
comparison = twin_manager.compare_twins(twin_data, twin2_data, metric='pi')
print("\nMean differences in disease probabilities:")
for d, diff in enumerate(comparison['mean_difference']):
    print(f"Disease {d}: {diff:.4f}")

# Example 4: Simulate multiple interventions
# Create a twin with high cardiovascular risk
high_cv_risk_profile = np.random.randn(model.P)
high_cv_risk_profile[0:5] += 2  # Increase some genetic factors
cv_twin = twin_manager.create_twin(high_cv_risk_profile)

# Simulate lifestyle intervention (reducing multiple signatures)
lifestyle_intervention = twin_manager.simulate_intervention(
    cv_twin,
    intervention_type='reduce',
    target_signature=None,  # Affects all signatures
    effect_size=0.2,
    start_time=10,  # Start at time 10
    end_time=30     # End at time 30
)

# Plot the effect of lifestyle intervention
twin_manager.plot_twin_trajectories(
    cv_twin,
    modified_data=lifestyle_intervention,
    selected_diseases=[0, 1, 2, 3]
)

# Example: Plot diseases 111-115 and simulate an intervention reducing signature 5 by 20%
selected_diseases = [111, 112, 113, 114, 115]

# Create a digital twin for a random genetic profile
random_genetic_profile = np.random.randn(model.P)
twin_data = twin_manager.create_twin(random_genetic_profile)

# Simulate intervention: reduce signature 5 by 20%
modified_data = twin_manager.simulate_intervention(
    twin_data,
    intervention_type='reduce',
    target_signature=5,
    effect_size=0.2
)

# Plot the selected diseases before and after intervention
print("Plotting diseases 111-115 with intervention on signature 5 (20% reduction)...")
twin_manager.plot_twin_trajectories(
    twin_data,
    modified_data=modified_data,
    selected_diseases=selected_diseases
) 