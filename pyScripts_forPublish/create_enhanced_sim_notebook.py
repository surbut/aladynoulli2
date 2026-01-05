#!/usr/bin/env python3
"""Script to create enhanced simulation notebook"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Enhanced Simulation Notebook for ALADYNOULLI\n",
            "\n",
            "This notebook provides an improved simulation framework with:\n",
            "- Better visualizations\n",
            "- Comprehensive parameter recovery analysis\n",
            "- Training progress tracking\n",
            "- Detailed trajectory comparisons"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import os\n",
            "\n",
            "# Get the parent directory of current directory\n",
            "parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))\n",
            "sys.path.append(parent_dir)\n",
            "\n",
            "sys.path.append('/Users/sarahurbut/aladynoulli2/pyScripts/')\n",
            "# Now you can import from pyScripts\n",
            "from oldmarch.cluster_g_logit_init_acceptpsi import *\n",
            "\n",
            "%load_ext autoreload"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%autoreload 2\n",
            "import numpy as np\n",
            "import seaborn as sns\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "from scipy.special import expit, softmax\n",
            "from scipy.stats import multivariate_normal\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score\n",
            "\n",
            "# Set style\n",
            "try:\n",
            "    plt.style.use('seaborn-v0_8-darkgrid')\n",
            "except:\n",
            "    plt.style.use('seaborn-darkgrid')\n",
            "sns.set_palette(\"husl\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def compute_smoothed_prevalence(Y, window_size=5):\n",
            "    \"\"\"Compute smoothed prevalence over time from binary disease indicators.\"\"\"\n",
            "    if isinstance(Y, torch.Tensor):\n",
            "        Y = Y.numpy()\n",
            "    \n",
            "    N, D, T = Y.shape\n",
            "    prevalence_t = np.zeros((D, T))\n",
            "    \n",
            "    for d in range(D):\n",
            "        for t in range(T):\n",
            "            start_idx = max(0, t - window_size // 2)\n",
            "            end_idx = min(T, t + window_size // 2 + 1)\n",
            "            prevalence_t[d, t] = Y[:, d, start_idx:end_idx].mean()\n",
            "    \n",
            "    epsilon = 1e-6\n",
            "    prevalence_t = np.clip(prevalence_t, epsilon, 1.0 - epsilon)\n",
            "    \n",
            "    return prevalence_t"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Generation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def generate_enhanced_simulation(N=10000, D=20, T=50, K=5, P=5, seed=42):\n",
            "    \"\"\"Generate synthetic data with clear structure for testing ALADYNOULLI\"\"\"\n",
            "    np.random.seed(seed)\n",
            "    torch.manual_seed(seed)\n",
            "    \n",
            "    print(f\"Generating simulation with:\")\n",
            "    print(f\"  N={N} individuals, D={D} diseases, T={T} timepoints\")\n",
            "    print(f\"  K={K} signatures, P={P} genetic features\")\n",
            "    \n",
            "    data = generate_clustered_survival_data(N=N, D=D, T=T, K=K, P=P)\n",
            "    \n",
            "    print(f\"\\nGenerated data shapes:\")\n",
            "    print(f\"  Y: {data['Y'].shape}\")\n",
            "    print(f\"  G: {data['G'].shape}\")\n",
            "    print(f\"  event_times: {data['event_times'].shape}\")\n",
            "    print(f\"  clusters: {data['clusters'].shape}\")\n",
            "    \n",
            "    return data\n",
            "\n",
            "# Generate data\n",
            "data = generate_enhanced_simulation(N=10000, D=20, T=50, K=5, P=5, seed=42)"
        ]
    }
]

# Create notebook structure
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write notebook
with open('/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/enhanced_simulation_showcase_v2.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created! You can now add more cells using the edit_notebook tool.")








