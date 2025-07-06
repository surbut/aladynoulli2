import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from warpsim import *
# Assume you have loaded your simulation output as `sim`
#sim = generate_clustered_survival_data_warp(N=1000, K=10, D=10, T=100, rho=0.5, lambda_0=0.01, theta=0.01, pi=0.01, G=0.01, phi=0.01)
sim  = generate_clustered_survival_data_warp(N=1000, D=10, T=100, K=5, P=1, beta_warp_sd=0.1, warping=True)

st.title("Visualizing Warping in Survival Simulation")

# Sidebar controls
n = st.slider("Select individual (n)", 0, sim['Y'].shape[0]-1, 0)
k = st.slider("Select signature (k)", 0, sim['theta'].shape[1]-1, 0)
d = st.slider("Select disease (d)", 0, sim['Y'].shape[1]-1, 0)
show_warping = st.checkbox("Show warping", value=True)
show_genetics = st.checkbox("Show genetic effect", value=True)

T = sim['Y'].shape[2]
t = np.arange(T)
T1 = T-1

# Warping parameters
rho = sim['rho'][n, k] if sim['rho'] is not None else 1.0
t_warped = ((t / T1) ** (1.0 / rho)) * T1

# Plot t vs t_warped
fig, ax = plt.subplots()
ax.plot(t, t, label="Original time")
ax.plot(t, t_warped, label=f"Warped time (rho={rho:.2f})")
ax.set_xlabel("Original time")
ax.set_ylabel("Warped time")
ax.legend()
st.pyplot(fig)

# Plot hazard curve before and after warping
phi = sim['phi'][k, d, :]
eta_phi = 1 / (1 + np.exp(-phi))
hazard_orig = eta_phi
hazard_warped = np.interp(t, t_warped, eta_phi)

fig2, ax2 = plt.subplots()
ax2.plot(t, hazard_orig, label="Original hazard")
ax2.plot(t, hazard_warped, label="Warped hazard")
ax2.set_xlabel("Time")
ax2.set_ylabel("Hazard")
ax2.legend()
st.pyplot(fig2)

# Show effect of genetics on warping
if show_genetics:
    st.subheader("Genetic effect on warping (rho)")
    st.write(f"Genetic vector G[n]: {sim['G'][n]}")
    st.write(f"rho[n, k]: {rho:.3f}")

# Show pi for two individuals with same lambda but different rho
st.subheader("pi for two individuals with same lambda, different rho")
# Find two individuals with similar lambda but different rho
lambda_nk = sim['lambda'][n, k, :]
rho_all = sim['rho'][:, k]
lambda_all = sim['lambda'][:, k, :]
dists = np.linalg.norm(lambda_all - lambda_nk, axis=1)
idx_sorted = np.argsort(dists)
for idx in idx_sorted[1:]:
    if abs(rho_all[idx] - rho) > 0.1:
        n2 = idx
        break
pi1 = sim['pi'][n, d, :]
pi2 = sim['pi'][n2, d, :]
fig3, ax3 = plt.subplots()
ax3.plot(t, pi1, label=f"Individual {n} (rho={rho:.2f})")
ax3.plot(t, pi2, label=f"Individual {n2} (rho={rho_all[n2]:.2f})")
ax3.set_xlabel("Time")
ax3.set_ylabel("pi (event probability)")
ax3.legend()
st.pyplot(fig3)

st.write("Even with similar lambda, different warping (rho) leads to different disease timing and risk curves.")

# Optionally, add more interactive elements or explanations!