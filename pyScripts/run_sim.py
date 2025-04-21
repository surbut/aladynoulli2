import numpy as np
from scipy.special import softmax, expit
from new_clust import generate_state_driven_data

# Use dimensions similar to real data
N = 1000  # number of individuals
D = 348   # number of diseases
T = 50    # time points
K = 20    # number of states
P = 10    # genetic components

print("Generating state-driven synthetic data...")
print(f"N={N}, D={D}, T={T}, K={K}, P={P}")

# Generate data
sim_data = generate_state_driven_data(N, D, T, K, P)

# Save the data
print("\nSaving data to state_driven_sim.npz...")
np.savez('state_driven_sim.npz', 
         Y=sim_data['Y'],
         G=sim_data['G'],
         event_times=sim_data['event_times'],
         theta=sim_data['theta'],
         lambda_ik=sim_data['lambda'],
         disease_state_weights=sim_data['disease_state_weights'],
         pi=sim_data['pi'],
         clusters=sim_data['clusters'],
         gamma=sim_data['gamma'])

# Print some basic statistics
print("\nSimulation Statistics:")
print(f"Number of events: {sim_data['Y'].sum()}")
print(f"Average events per person: {sim_data['Y'].sum() / N:.2f}")
print(f"Average events per disease: {sim_data['Y'].sum() / D:.2f}")

# Look at cluster sizes
unique_clusters, counts = np.unique(sim_data['clusters'], return_counts=True)
print("\nCluster Sizes:")
for k, count in zip(unique_clusters, counts):
    print(f"Cluster {k}: {count} diseases") 