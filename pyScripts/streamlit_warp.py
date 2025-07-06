import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sim_with_warp import *
import pickle
import pandas as pd
# Assume you have loaded your simulation output as `sim`
#sim = generate_clustered_survival_data_warp(N=1000, K=10, D=10, T=100, rho=0.5, lambda_0=0.01, theta=0.01, pi=0.01, G=0.01, phi=0.01)
with open("sim_warp.pkl", "rb") as f:
    sim = pickle.load(f)

st.title("Visualizing Warping in Survival Simulation")

# Sidebar controls
k = st.slider("Select signature (k)", 0, sim['theta'].shape[1]-1, 0)

# Now you can use k to find lowest_rho_idx
lowest_rho_idx = int(np.argmin(sim['rho'][:, k]))
n = st.slider(
    "Select individual (n)",
    0, sim['Y'].shape[0]-1,
    value=lowest_rho_idx
)
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


# Optionally, add more interactive elements or explanations!

rho_vals = sim['rho'][:, k]
df = pd.DataFrame({'Individual': np.arange(len(rho_vals)), 'rho': rho_vals})
df_sorted = df.sort_values('rho')
st.write("Sample individuals with lowest rho:")
st.dataframe(df_sorted.head(5))
st.write("Sample individuals with highest rho:")
st.dataframe(df_sorted.tail(5))

T = 50
t = np.arange(T)
T1 = T-1
phi = np.exp(-0.5 * ((t-25)/5)**2)  # Example bell-shaped hazard

for rho in [0.5, 1.0, 2.0]:
    t_warped = ((t / T1) ** (1.0 / rho)) * T1
    hazard_warped = np.interp(t, t_warped, phi)
    plt.plot(t, hazard_warped, label=f"rho={rho}")

plt.plot(t, phi, 'k--', label="Original hazard")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Hazard")
plt.title("Effect of warping on hazard curve")
plt.show()

st.header("Compare warping vs. mixture effects")

# --- (A) Same topic mixture, different rho ---
# Pick two individuals with similar theta but different rho for the selected signature
theta_k = sim['theta'][:, k, :]
theta_k_mean = theta_k.mean(axis=1)
# Find two individuals with similar theta_k_mean but different rho
theta_diff = np.abs(theta_k_mean[:, None] - theta_k_mean[None, :])
rho_diff = np.abs(sim['rho'][:, k][:, None] - sim['rho'][:, k][None, :])
# Mask out self-comparisons
np.fill_diagonal(rho_diff, 0)
# Find pair with small theta_diff and large rho_diff
score = theta_diff - 0.5 * rho_diff  # tune weights as needed
i1, i2 = np.unravel_index(np.argmin(score), score.shape)

st.write(f"Same topic mixture, different rho: Individual {i1} (rho={sim['rho'][i1, k]:.2f}), Individual {i2} (rho={sim['rho'][i2, k]:.2f})")
pi1 = sim['pi'][i1, d, :]
pi2 = sim['pi'][i2, d, :]

figA, axA = plt.subplots()
axA.plot(t, pi1, label=f"Ind {i1} (rho={sim['rho'][i1, k]:.2f})")
axA.plot(t, pi2, label=f"Ind {i2} (rho={sim['rho'][i2, k]:.2f})")
axA.set_title("Same topic mixture, different warping")
axA.set_xlabel("Time")
axA.set_ylabel("pi (event probability)")
axA.legend()
st.pyplot(figA)

# --- (B) Different topic mixture, same rho ---
# Find two individuals with similar rho but different theta
rho_vals = sim['rho'][:, k]
rho_target = np.median(rho_vals)
rho_diff = np.abs(rho_vals - rho_target)
idx_rho = np.argsort(rho_diff)[:10]  # 10 closest to median
theta_k = sim['theta'][idx_rho, k, :]
theta_diff = np.abs(theta_k[:, None, :] - theta_k[None, :, :]).sum(axis=2)
i3, i4 = np.unravel_index(np.argmax(theta_diff), theta_diff.shape)
i3, i4 = idx_rho[i3], idx_rho[i4]

st.write(f"Different topic mixture, same rho: Individual {i3} (rho={sim['rho'][i3, k]:.2f}), Individual {i4} (rho={sim['rho'][i4, k]:.2f})")
pi3 = sim['pi'][i3, d, :]
pi4 = sim['pi'][i4, d, :]

figB, axB = plt.subplots()
axB.plot(t, pi3, label=f"Ind {i3} (theta diff)")
axB.plot(t, pi4, label=f"Ind {i4} (theta diff)")
axB.set_title("Different topic mixture, same warping")
axB.set_xlabel("Time")
axB.set_ylabel("pi (event probability)")
axB.legend()
st.pyplot(figB)

# --- (C) Overlay original hazard for reference ---
st.write("Original (unwarped) hazard for selected signature/disease:")
figC, axC = plt.subplots()
axC.plot(t, eta_phi, 'k--', label="Original hazard")
axC.set_xlabel("Time")
axC.set_ylabel("Hazard")
axC.legend()
st.pyplot(figC)