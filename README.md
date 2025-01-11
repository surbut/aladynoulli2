# aladynoulli
Code for Solving Aladynoulli! 

In this repo, you'll find the main script for running (solvinglogit.R) and the sampler (mcmc_sampler_softmax) in the Rscripts directory.

\begin{frame}{The Model}
\begin{align*}
\pi_{idt} &= \sum_{k=1}^K \theta_{ikt} \texit{expit}(\phi_{kdt}) \\
\theta_{ikt} &= \text{softmax}(\lambda_{ikt}) \\
\boldsymbol{\phi}_{kd} &\sim \mathcal{N}(\boldsymbol{\mu}_d + \psi_{kd}\mathbf{1}_T, K_\phi) \\
\boldsymbol{\lambda}_{ik} &\sim \mathcal{N}(\gamma_k g_i\mathbf{1}_T, K_\lambda)
\end{align*}

Additional scripts related to useful functions ('utils'), model specific functions, initialization and sampling methods (for example, elliptical) are all in the utils directory.

Please contact me at surbut@mgh.harvard.edu for qs!
