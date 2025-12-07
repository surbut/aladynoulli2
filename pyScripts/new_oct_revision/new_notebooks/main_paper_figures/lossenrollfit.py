============================================================
Running Aladyn batch: samples 0 to 10000
============================================================

Loading components...
Loaded all components successfully!
Subsetting data from 0 to 10000...
Loading covariates data...
G_with_sex shape: (10000, 47)
Covariates loaded: 407878 total samples
Loading reference trajectories...

Initializing model with K=20 clusters...
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.signature_refs = torch.tensor(signature_references, dtype=torch.float32)
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:88: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.Y = torch.tensor(Y, dtype=torch.float32)
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:91: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)

Cluster Sizes:
Cluster 0: 14 diseases
Cluster 1: 7 diseases
Cluster 2: 21 diseases
Cluster 3: 15 diseases
Cluster 4: 17 diseases
Cluster 5: 16 diseases
Cluster 6: 57 diseases
Cluster 7: 18 diseases
Cluster 8: 13 diseases
Cluster 9: 11 diseases
Cluster 10: 18 diseases
Cluster 11: 12 diseases
Cluster 12: 26 diseases
Cluster 13: 7 diseases
Cluster 14: 9 diseases
Cluster 15: 8 diseases
Cluster 16: 7 diseases
Cluster 17: 11 diseases
Cluster 18: 6 diseases
Cluster 19: 55 diseases

Calculating gamma for k=0:
Number of diseases in cluster: 14
Base value (first 5): tensor([-13.8155, -13.8155, -13.1095, -12.4036, -12.4036])
Base value centered (first 5): tensor([-0.3723, -0.3723,  0.3336,  1.0396,  1.0396])
Base value centered mean: 6.57081614008348e-07
Gamma init for k=0 (first 5): tensor([ 0.0026,  0.0050,  0.0103,  0.0132, -0.0116])

Calculating gamma for k=1:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.7043, -0.7043, -0.7043, -0.7043, -0.7043])
Base value centered mean: -2.2621155437718699e-07
Gamma init for k=1 (first 5): tensor([ 0.0200,  0.0102,  0.0029, -0.0021, -0.0044])

Calculating gamma for k=2:
Number of diseases in cluster: 21
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -10.9916, -13.3449])
Base value centered (first 5): tensor([-0.2393, -0.2393, -0.2393,  2.5846,  0.2313])
Base value centered mean: -3.040313742985745e-07
Gamma init for k=2 (first 5): tensor([ 0.0011, -0.0023,  0.0073,  0.0020, -0.0051])

Calculating gamma for k=3:
Number of diseases in cluster: 15
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1155, -0.1155, -0.1155, -0.1155, -0.1155])
Base value centered mean: 1.0881424117314964e-07
Gamma init for k=3 (first 5): tensor([ 0.0014, -0.0038,  0.0025,  0.0024, -0.0058])

Calculating gamma for k=4:
Number of diseases in cluster: 17
Base value (first 5): tensor([-13.2341, -13.8155, -13.2341, -13.2341, -11.4899])
Base value centered (first 5): tensor([-0.0441, -0.6255, -0.0441, -0.0441,  1.7001])
Base value centered mean: 1.0607719787003589e-06
Gamma init for k=4 (first 5): tensor([-0.0088, -0.0010, -0.0120,  0.0372, -0.0093])

Calculating gamma for k=5:
Number of diseases in cluster: 16
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -12.5800, -13.8155])
Base value centered (first 5): tensor([-0.1663, -0.1663, -0.1663,  1.0692, -0.1663])
Base value centered mean: -1.2866020142610068e-06
Gamma init for k=5 (first 5): tensor([ 0.0051,  0.0055, -0.0040,  0.0058,  0.0238])

Calculating gamma for k=6:
Number of diseases in cluster: 57
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.4687])
Base value centered (first 5): tensor([-0.1119, -0.1119, -0.1119, -0.1119,  0.2349])
Base value centered mean: -2.3013114969216986e-06
Gamma init for k=6 (first 5): tensor([ 0.0017,  0.0018,  0.0010, -0.0011, -0.0017])

Calculating gamma for k=7:
Number of diseases in cluster: 18
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -10.5209, -12.1682])
Base value centered (first 5): tensor([-0.4014, -0.4014, -0.4014,  2.8931,  1.2458])
Base value centered mean: -5.4836272056491e-07
Gamma init for k=7 (first 5): tensor([ 0.0075,  0.0045, -0.0065,  0.0055, -0.0043])

Calculating gamma for k=8:
Number of diseases in cluster: 13
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -11.5347, -13.0552])
Base value centered (first 5): tensor([-0.2446, -0.2446, -0.2446,  2.0363,  0.5157])
Base value centered mean: -1.083374030486084e-07
Gamma init for k=8 (first 5): tensor([ 0.0032, -0.0005,  0.0003,  0.0114, -0.0019])

Calculating gamma for k=9:
Number of diseases in cluster: 11
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -12.0185, -11.1200])
Base value centered (first 5): tensor([-0.2584, -0.2584, -0.2584,  1.5386,  2.4371])
Base value centered mean: -2.26306909212326e-07
Gamma init for k=9 (first 5): tensor([-0.0003,  0.0084,  0.0164, -0.0027,  0.0004])

Calculating gamma for k=10:
Number of diseases in cluster: 18
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1341, -0.1341, -0.1341, -0.1341, -0.1341])
Base value centered mean: -3.9024354236971703e-07
Gamma init for k=10 (first 5): tensor([ 0.0123, -0.0048,  0.0016, -0.0047,  0.0018])

Calculating gamma for k=11:
Number of diseases in cluster: 12
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -12.9919])
Base value centered (first 5): tensor([-0.1911, -0.1911, -0.1911, -0.1911,  0.6326])
Base value centered mean: -4.671096860420221e-07
Gamma init for k=11 (first 5): tensor([ 0.0037,  0.0055, -0.0066,  0.0036,  0.0014])

Calculating gamma for k=12:
Number of diseases in cluster: 26
Base value (first 5): tensor([-13.4354, -13.8155, -13.4354, -13.4354, -12.2949])
Base value centered (first 5): tensor([ 0.0574, -0.3228,  0.0574,  0.0574,  1.1978])
Base value centered mean: -1.0808944352902472e-06
Gamma init for k=12 (first 5): tensor([ 0.0068,  0.0008, -0.0014,  0.0007,  0.0027])

Calculating gamma for k=13:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1511, -0.1511, -0.1511, -0.1511, -0.1511])
Base value centered mean: -1.0584831215965096e-06
Gamma init for k=13 (first 5): tensor([ 0.0086,  0.0104, -0.0006,  0.0071, -0.0017])

Calculating gamma for k=14:
Number of diseases in cluster: 9
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1618, -0.1618, -0.1618, -0.1618, -0.1618])
Base value centered mean: -1.2922287169203628e-06
Gamma init for k=14 (first 5): tensor([-0.0111, -0.0044, -0.0045,  0.0119,  0.0053])

Calculating gamma for k=15:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1184, -0.1184, -0.1184, -0.1184, -0.1184])
Base value centered mean: -6.237029879230249e-07
Gamma init for k=15 (first 5): tensor([0.0026, 0.0020, 0.0032, 0.0007, 0.0098])

Calculating gamma for k=16:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -12.4036])
Base value centered (first 5): tensor([-0.2306, -0.2306, -0.2306, -0.2306,  1.1814])
Base value centered mean: -1.5748977375551476e-06
Gamma init for k=16 (first 5): tensor([-0.0015,  0.0126,  0.0149, -0.0003,  0.0101])

Calculating gamma for k=17:
Number of diseases in cluster: 11
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2706, -0.2706, -0.2706, -0.2706, -0.2706])
Base value centered mean: -7.493972589145415e-07
Gamma init for k=17 (first 5): tensor([-0.0084,  0.0188,  0.0010, -0.0041, -0.0089])

Calculating gamma for k=18:
Number of diseases in cluster: 6
Base value (first 5): tensor([-13.8155, -10.5209, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1006,  3.1939, -0.1006, -0.1006, -0.1006])
Base value centered mean: -1.0191916999247042e-06
Gamma init for k=18 (first 5): tensor([-0.0162, -0.0025,  0.0116,  0.0191, -0.0034])

Calculating gamma for k=19:
Number of diseases in cluster: 55
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.2764, -13.4561])
Base value centered (first 5): tensor([-0.1217, -0.1217, -0.1217,  0.4174,  0.2377])
Base value centered mean: -3.7849426917091478e-06
Gamma init for k=19 (first 5): tensor([ 0.0008, -0.0014,  0.0007,  0.0034,  0.0015])
Initializing with 20 disease states + 1 healthy state
Initialization complete!
Loading initial psi and clusters...

Calculating gamma for k=0:
Number of diseases in cluster: 16.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -11.9623, -13.8155])
Base value centered (first 5): tensor([-0.1879, -0.1879, -0.1879,  1.6653, -0.1879])
Base value centered mean: -3.345489574257954e-07
Gamma init for k=0 (first 5): tensor([ 0.0074,  0.0061, -0.0054,  0.0060,  0.0235])

Calculating gamma for k=1:
Number of diseases in cluster: 21.0
Base value (first 5): tensor([-13.3449, -13.8155, -13.3449, -13.3449, -12.4036])
Base value centered (first 5): tensor([ 0.1505, -0.3201,  0.1505,  0.1505,  1.0918])
Base value centered mean: -1.8495559288567165e-06
Gamma init for k=1 (first 5): tensor([0.0042, 0.0009, 0.0009, 0.0018, 0.0015])

Calculating gamma for k=2:
Number of diseases in cluster: 15.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.1566, -11.8388, -12.4977])
Base value centered (first 5): tensor([-0.3849, -0.3849,  0.2740,  1.5918,  0.9329])
Base value centered mean: 9.290695288655115e-07
Gamma init for k=2 (first 5): tensor([ 0.0016,  0.0073,  0.0100,  0.0140, -0.0118])

Calculating gamma for k=3:
Number of diseases in cluster: 82.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.6950, -13.5744])
Base value centered (first 5): tensor([-0.1026, -0.1026, -0.1026,  0.0179,  0.1384])
Base value centered mean: 4.7445297468584613e-07
Gamma init for k=3 (first 5): tensor([ 0.0012,  0.0003,  0.0016,  0.0018, -0.0006])

Calculating gamma for k=4:
Number of diseases in cluster: 5.0
Base value (first 5): tensor([-13.8155,  -9.8620, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1133,  3.8402, -0.1133, -0.1133, -0.1133])
Base value centered mean: -2.841758714566822e-06
Gamma init for k=4 (first 5): tensor([-0.0172, -0.0033,  0.0143,  0.0225, -0.0033])

Calculating gamma for k=5:
Number of diseases in cluster: 7.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.7043, -0.7043, -0.7043, -0.7043, -0.7043])
Base value centered mean: -2.2621155437718699e-07
Gamma init for k=5 (first 5): tensor([ 0.0200,  0.0102,  0.0029, -0.0021, -0.0044])

Calculating gamma for k=6:
Number of diseases in cluster: 8.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1523, -0.1523, -0.1523, -0.1523, -0.1523])
Base value centered mean: -1.0890006478803116e-06
Gamma init for k=6 (first 5): tensor([ 0.0090,  0.0121, -0.0006,  0.0042, -0.0027])

Calculating gamma for k=7:
Number of diseases in cluster: 22.0
Base value (first 5): tensor([-13.3663, -13.8155, -13.3663, -13.8155, -11.1200])
Base value centered (first 5): tensor([-0.1005, -0.5497, -0.1005, -0.5497,  2.1458])
Base value centered mean: 1.2460708376238472e-06
Gamma init for k=7 (first 5): tensor([ 0.0003, -0.0038, -0.0111,  0.0251, -0.0016])

Calculating gamma for k=8:
Number of diseases in cluster: 28.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1159, -0.1159, -0.1159, -0.1159, -0.1159])
Base value centered mean: -1.4129639112070436e-06
Gamma init for k=8 (first 5): tensor([ 0.0066, -0.0055,  0.0011, -0.0020, -0.0005])

Calculating gamma for k=9:
Number of diseases in cluster: 12.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -12.9919])
Base value centered (first 5): tensor([-0.1911, -0.1911, -0.1911, -0.1911,  0.6326])
Base value centered mean: -4.671096860420221e-07
Gamma init for k=9 (first 5): tensor([ 0.0037,  0.0055, -0.0066,  0.0036,  0.0014])

Calculating gamma for k=10:
Number of diseases in cluster: 11.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2706, -0.2706, -0.2706, -0.2706, -0.2706])
Base value centered mean: -7.493972589145415e-07
Gamma init for k=10 (first 5): tensor([-0.0084,  0.0188,  0.0010, -0.0041, -0.0089])

Calculating gamma for k=11:
Number of diseases in cluster: 8.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1184, -0.1184, -0.1184, -0.1184, -0.1184])
Base value centered mean: -6.237029879230249e-07
Gamma init for k=11 (first 5): tensor([0.0026, 0.0020, 0.0032, 0.0007, 0.0098])

Calculating gamma for k=12:
Number of diseases in cluster: 7.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.0860, -0.0860, -0.0860, -0.0860, -0.0860])
Base value centered mean: -1.034450519910024e-06
Gamma init for k=12 (first 5): tensor([ 0.0080, -0.0074,  0.0002, -0.0001, -0.0076])

Calculating gamma for k=13:
Number of diseases in cluster: 13.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -10.7744, -12.2949])
Base value centered (first 5): tensor([-0.1777, -0.1777, -0.1777,  2.8635,  1.3429])
Base value centered mean: -1.0064125035569305e-06
Gamma init for k=13 (first 5): tensor([-0.0002,  0.0060,  0.0127, -0.0051,  0.0007])

Calculating gamma for k=14:
Number of diseases in cluster: 10.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -11.8388, -12.8271])
Base value centered (first 5): tensor([-0.2562, -0.2562, -0.2562,  1.7205,  0.7322])
Base value centered mean: -1.5125274330785032e-06
Gamma init for k=14 (first 5): tensor([ 0.0038, -0.0052, -0.0089,  0.0162,  0.0005])

Calculating gamma for k=15:
Number of diseases in cluster: 5.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -11.8388])
Base value centered (first 5): tensor([-0.2850, -0.2850, -0.2850, -0.2850,  1.6917])
Base value centered mean: -1.2931823221151717e-06
Gamma init for k=15 (first 5): tensor([-0.0039,  0.0064,  0.0118, -0.0049,  0.0086])

Calculating gamma for k=16:
Number of diseases in cluster: 29.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -11.7706, -13.1339])
Base value centered (first 5): tensor([-0.2428, -0.2428, -0.2428,  1.8021,  0.4388])
Base value centered mean: -1.9989013253507437e-06
Gamma init for k=16 (first 5): tensor([ 0.0009,  0.0001,  0.0093,  0.0055, -0.0031])

Calculating gamma for k=17:
Number of diseases in cluster: 17.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -10.3272, -12.0713])
Base value centered (first 5): tensor([-0.4228, -0.4228, -0.4228,  3.0656,  1.3214])
Base value centered mean: -2.227592403869494e-06
Gamma init for k=17 (first 5): tensor([ 0.0081,  0.0052, -0.0070,  0.0058, -0.0042])

Calculating gamma for k=18:
Number of diseases in cluster: 9.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1618, -0.1618, -0.1618, -0.1618, -0.1618])
Base value centered mean: -1.2922287169203628e-06
Gamma init for k=18 (first 5): tensor([-0.0111, -0.0044, -0.0045,  0.0119,  0.0053])

Calculating gamma for k=19:
Number of diseases in cluster: 23.0
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.3858])
Base value centered (first 5): tensor([-0.1519, -0.1519, -0.1519, -0.1519,  0.2779])
Base value centered mean: -1.1179923831150518e-06
Gamma init for k=19 (first 5): tensor([-0.0027,  0.0053,  0.0013, -0.0016, -0.0018])
Initializing with 20 disease states + 1 healthy state
Initialization complete!
Clusters match exactly: True
Model clusters shape: (348,)
Model clusters type: <class 'numpy.ndarray'>
Model clusters range: 0 to 19
Model psi shape: torch.Size([21, 348])
✓ Clusters are set and model has been initialized

Training model for 200 epochs...
Learning rate: 0.1, Lambda: 0.01

Epoch 0
Loss: 4.1965

Monitoring signature responses:

Disease 161 (signature 7, LR=30.91):
  Theta for diagnosed: 0.150 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 76 (signature 7, LR=30.73):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 347 (signature 3, LR=29.25):
  Theta for diagnosed: 0.149 ± 0.070
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 1
Loss: 651.7099

Monitoring signature responses:

Disease 76 (signature 7, LR=30.68):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 161 (signature 7, LR=30.65):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 260 (signature 8, LR=30.47):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 347 (signature 3, LR=29.27):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 2
Loss: 55.6754

Monitoring signature responses:

Disease 76 (signature 7, LR=30.66):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.51):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 161 (signature 7, LR=30.40):
  Theta for diagnosed: 0.151 ± 0.036
  Theta for others: 0.148
  Proportion difference: 0.003

Disease 347 (signature 3, LR=29.30):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 3
Loss: 150.9390

Monitoring signature responses:

Disease 76 (signature 7, LR=30.64):
  Theta for diagnosed: 0.154 ± 0.039
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.52):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 161 (signature 7, LR=30.16):
  Theta for diagnosed: 0.151 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.003

Disease 347 (signature 3, LR=29.29):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 4
Loss: 366.3895

Monitoring signature responses:

Disease 76 (signature 7, LR=30.60):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.51):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 161 (signature 7, LR=29.93):
  Theta for diagnosed: 0.151 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.003

Disease 347 (signature 3, LR=29.26):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 5
Loss: 273.9113

Monitoring signature responses:

Disease 76 (signature 7, LR=30.55):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.48):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 161 (signature 7, LR=29.71):
  Theta for diagnosed: 0.151 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 347 (signature 3, LR=29.23):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 6
Loss: 88.1933

Monitoring signature responses:

Disease 76 (signature 7, LR=30.51):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.45):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 161 (signature 7, LR=29.49):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 347 (signature 3, LR=29.20):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 7
Loss: 18.8929

Monitoring signature responses:

Disease 76 (signature 7, LR=30.46):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.43):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 161 (signature 7, LR=29.29):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 347 (signature 3, LR=29.18):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 8
Loss: 89.7783

Monitoring signature responses:

Disease 76 (signature 7, LR=30.43):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 347 (signature 3, LR=29.17):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 161 (signature 7, LR=29.11):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 9
Loss: 171.4495

Monitoring signature responses:

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.40):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.17):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 161 (signature 7, LR=28.94):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 10
Loss: 162.0545

Monitoring signature responses:

Disease 260 (signature 8, LR=30.43):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.38):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.18):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 161 (signature 7, LR=28.78):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Epoch 11
Loss: 85.3971

Monitoring signature responses:

Disease 260 (signature 8, LR=30.44):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.37):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.18):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 161 (signature 7, LR=28.65):
  Theta for diagnosed: 0.150 ± 0.036
  Theta for others: 0.147
  Proportion difference: 0.003

Epoch 12
Loss: 21.4236

Monitoring signature responses:

Disease 260 (signature 8, LR=30.45):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.35):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.17):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 13
Loss: 19.6957

Monitoring signature responses:

Disease 260 (signature 8, LR=30.46):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.33):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.16):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 14
Loss: 62.0562

Monitoring signature responses:

Disease 260 (signature 8, LR=30.46):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.32):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.15):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 15
Loss: 93.1343

Monitoring signature responses:

Disease 260 (signature 8, LR=30.45):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.30):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.14):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 16
Loss: 81.0244

Monitoring signature responses:

Disease 260 (signature 8, LR=30.44):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.29):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.13):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 17
Loss: 41.7120

Monitoring signature responses:

Disease 260 (signature 8, LR=30.43):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.28):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.12):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 18
Loss: 12.5310

Monitoring signature responses:

Disease 260 (signature 8, LR=30.41):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.27):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.11):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 19
Loss: 14.4156

Monitoring signature responses:

Disease 260 (signature 8, LR=30.41):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.26):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.10):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 20
Loss: 36.1967

Monitoring signature responses:

Disease 260 (signature 8, LR=30.41):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.25):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.09):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 21
Loss: 50.4683

Monitoring signature responses:

Disease 260 (signature 8, LR=30.41):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.23):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.09):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 22
Loss: 42.5900

Monitoring signature responses:

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.21):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.08):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 23
Loss: 21.9501

Monitoring signature responses:

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.20):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.07):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 24
Loss: 8.2479

Monitoring signature responses:

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.18):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.07):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 25
Loss: 11.1587

Monitoring signature responses:

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.16):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.06):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 26
Loss: 22.8604

Monitoring signature responses:

Disease 260 (signature 8, LR=30.42):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.14):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.05):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 27
Loss: 28.5152

Monitoring signature responses:

Disease 260 (signature 8, LR=30.41):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.13):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.05):
  Theta for diagnosed: 0.149 ± 0.069
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 28
Loss: 22.1122

Monitoring signature responses:

Disease 260 (signature 8, LR=30.40):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.11):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.04):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 29
Loss: 10.7932

Monitoring signature responses:

Disease 260 (signature 8, LR=30.40):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.10):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.03):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 30
Loss: 5.4043

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.09):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.02):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 31
Loss: 9.0704

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.08):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.01):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 32
Loss: 15.2966

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.06):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.00):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 33
Loss: 16.2017

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.05):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=29.00):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 34
Loss: 10.9167

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.04):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.99):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 35
Loss: 5.2597

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.03):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.98):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 36
Loss: 4.4863

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.02):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.98):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 37
Loss: 7.8213

Monitoring signature responses:

Disease 260 (signature 8, LR=30.39):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.01):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.97):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 38
Loss: 10.2825

Monitoring signature responses:

Disease 260 (signature 8, LR=30.38):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.96):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.87):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 39
Loss: 8.7877

Monitoring signature responses:

Disease 260 (signature 8, LR=30.38):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.96):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 40
Loss: 5.1888

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.96):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 41
Loss: 3.3754

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.95):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 42
Loss: 4.5982

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.95):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 43
Loss: 6.4958

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.94):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.57):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 44
Loss: 6.3928

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.94):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 45
Loss: 4.4306

Monitoring signature responses:

Disease 260 (signature 8, LR=30.38):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.93):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 46
Loss: 2.9187

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.93):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 47
Loss: 3.2640

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.93):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 48
Loss: 4.4596

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.92):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 49
Loss: 4.6874

Monitoring signature responses:

Disease 260 (signature 8, LR=30.37):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.92):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 50
Loss: 3.6379

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.91):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 51
Loss: 2.6237

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.91):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 52
Loss: 2.6978

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.90):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 53
Loss: 3.3998

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.90):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.88):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.58):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 54
Loss: 3.5853

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.89):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 55
Loss: 2.9759

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 347 (signature 3, LR=28.89):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 56
Loss: 2.3603

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.89):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 57
Loss: 2.4064

Monitoring signature responses:

Disease 260 (signature 8, LR=30.36):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.88):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 58
Loss: 2.8248

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.88):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 59
Loss: 2.9073

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.87):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 60
Loss: 2.5212

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.148
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.87):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 61
Loss: 2.1824

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.86):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 62
Loss: 2.2595

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.86):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.59):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 63
Loss: 2.5066

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.85):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 64
Loss: 2.4990

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.85):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 65
Loss: 2.2374

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.85):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 66
Loss: 2.0733

Monitoring signature responses:

Disease 260 (signature 8, LR=30.35):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.84):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 67
Loss: 2.1659

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.84):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 68
Loss: 2.2963

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.84):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.027
  Theta for others: 0.026
  Proportion difference: 0.000

Epoch 69
Loss: 2.2369

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.83):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.60):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 70
Loss: 2.0708

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.83):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.61):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 71
Loss: 2.0200

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.89):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.82):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.61):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 72
Loss: 2.1061

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.82):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.61):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 73
Loss: 2.1541

Monitoring signature responses:

Disease 260 (signature 8, LR=30.34):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.82):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.61):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 74
Loss: 2.0774

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.81):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.61):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 75
Loss: 1.9872

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.81):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.61):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 76
Loss: 1.9960

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.80):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.62):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 77
Loss: 2.0531

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.80):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.62):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 78
Loss: 2.0476

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.79):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.62):
  Theta for diagnosed: 0.026 ± 0.028
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 79
Loss: 1.9829

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.79):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.62):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 80
Loss: 1.9515

Monitoring signature responses:

Disease 260 (signature 8, LR=30.33):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.79):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.62):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 81
Loss: 1.9808

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.78):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.62):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 82
Loss: 2.0032

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.78):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.63):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 83
Loss: 1.9748

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.77):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.63):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 84
Loss: 1.9379

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.77):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.63):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 85
Loss: 1.9408

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.76):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.63):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 86
Loss: 1.9626

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.90):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.76):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.63):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 87
Loss: 1.9576

Monitoring signature responses:

Disease 260 (signature 8, LR=30.32):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.76):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.63):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 88
Loss: 1.9306

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.75):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.64):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 89
Loss: 1.9207

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.75):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.64):
  Theta for diagnosed: 0.026 ± 0.029
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 90
Loss: 1.9336

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.74):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.64):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 91
Loss: 1.9385

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.74):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.64):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 92
Loss: 1.9235

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.73):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.64):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 93
Loss: 1.9109

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.73):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.65):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 94
Loss: 1.9155

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.73):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.65):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 95
Loss: 1.9220

Monitoring signature responses:

Disease 260 (signature 8, LR=30.31):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.72):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.65):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 96
Loss: 1.9151

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.72):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.65):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 97
Loss: 1.9042

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.037
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.71):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.65):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 98
Loss: 1.9038

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.71):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.66):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 99
Loss: 1.9088

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.94):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.70):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.66):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 100
Loss: 1.9064

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.91):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.70):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.66):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 101
Loss: 1.8985

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.69):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.66):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 102
Loss: 1.8962

Monitoring signature responses:

Disease 260 (signature 8, LR=30.30):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.69):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.66):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 103
Loss: 1.8992

Monitoring signature responses:

Disease 260 (signature 8, LR=30.29):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.68):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.67):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 104
Loss: 1.8986

Monitoring signature responses:

Disease 260 (signature 8, LR=30.29):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.68):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.67):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 105
Loss: 1.8933

Monitoring signature responses:

Disease 260 (signature 8, LR=30.29):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 347 (signature 3, LR=28.67):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Disease 84 (signature 10, LR=28.67):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 106
Loss: 1.8904

Monitoring signature responses:

Disease 260 (signature 8, LR=30.29):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.67):
  Theta for diagnosed: 0.026 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.67):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 107
Loss: 1.8917

Monitoring signature responses:

Disease 260 (signature 8, LR=30.29):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.67):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.67):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 108
Loss: 1.8917

Monitoring signature responses:

Disease 260 (signature 8, LR=30.29):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.68):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.66):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 109
Loss: 1.8881

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.95):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.68):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.66):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 110
Loss: 1.8854

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.68):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.65):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 111
Loss: 1.8858

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.68):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.65):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 112
Loss: 1.8858

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.92):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.69):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 347 (signature 3, LR=28.64):
  Theta for diagnosed: 0.149 ± 0.068
  Theta for others: 0.150
  Proportion difference: -0.001

Epoch 113
Loss: 1.8834

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.69):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 336 (signature 9, LR=28.68):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Epoch 114
Loss: 1.8811

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 336 (signature 9, LR=28.71):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 84 (signature 10, LR=28.69):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 115
Loss: 1.8809

Monitoring signature responses:

Disease 260 (signature 8, LR=30.28):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 336 (signature 9, LR=28.75):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 84 (signature 10, LR=28.69):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 116
Loss: 1.8807

Monitoring signature responses:

Disease 260 (signature 8, LR=30.27):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 336 (signature 9, LR=28.79):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 84 (signature 10, LR=28.69):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 117
Loss: 1.8788

Monitoring signature responses:

Disease 260 (signature 8, LR=30.27):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.96):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 336 (signature 9, LR=28.83):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 84 (signature 10, LR=28.70):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 118
Loss: 1.8769

Monitoring signature responses:

Disease 260 (signature 8, LR=30.27):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 336 (signature 9, LR=28.86):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 84 (signature 10, LR=28.70):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 119
Loss: 1.8764

Monitoring signature responses:

Disease 260 (signature 8, LR=30.27):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 336 (signature 9, LR=28.90):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 84 (signature 10, LR=28.70):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 120
Loss: 1.8760

Monitoring signature responses:

Disease 260 (signature 8, LR=30.27):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=28.94):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 84 (signature 10, LR=28.70):
  Theta for diagnosed: 0.026 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 121
Loss: 1.8746

Monitoring signature responses:

Disease 260 (signature 8, LR=30.27):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=28.98):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 56 (signature 16, LR=28.71):
  Theta for diagnosed: 0.084 ± 0.043
  Theta for others: 0.082
  Proportion difference: 0.002

Epoch 122
Loss: 1.8730

Monitoring signature responses:

Disease 260 (signature 8, LR=30.26):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.02):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 56 (signature 16, LR=28.77):
  Theta for diagnosed: 0.084 ± 0.043
  Theta for others: 0.082
  Proportion difference: 0.002

Epoch 123
Loss: 1.8724

Monitoring signature responses:

Disease 260 (signature 8, LR=30.26):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.97):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.06):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 50 (signature 15, LR=28.93):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 56 (signature 16, LR=28.82):
  Theta for diagnosed: 0.084 ± 0.043
  Theta for others: 0.082
  Proportion difference: 0.002

Epoch 124
Loss: 1.8718

Monitoring signature responses:

Disease 260 (signature 8, LR=30.26):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.10):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 56 (signature 16, LR=28.87):
  Theta for diagnosed: 0.084 ± 0.043
  Theta for others: 0.082
  Proportion difference: 0.002

Epoch 125
Loss: 1.8706

Monitoring signature responses:

Disease 260 (signature 8, LR=30.26):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.14):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 56 (signature 16, LR=28.93):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Epoch 126
Loss: 1.8692

Monitoring signature responses:

Disease 260 (signature 8, LR=30.26):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.19):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=28.98):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 127
Loss: 1.8685

Monitoring signature responses:

Disease 260 (signature 8, LR=30.26):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.23):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.04):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 128
Loss: 1.8679

Monitoring signature responses:

Disease 260 (signature 8, LR=30.25):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.27):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.09):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 129
Loss: 1.8667

Monitoring signature responses:

Disease 260 (signature 8, LR=30.25):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.98):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.31):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.15):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 130
Loss: 1.8656

Monitoring signature responses:

Disease 260 (signature 8, LR=30.25):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.36):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.20):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 131
Loss: 1.8648

Monitoring signature responses:

Disease 260 (signature 8, LR=30.25):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.40):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.26):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 50 (signature 15, LR=28.94):
  Theta for diagnosed: 0.016 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.002

Epoch 132
Loss: 1.8641

Monitoring signature responses:

Disease 260 (signature 8, LR=30.25):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.45):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.32):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=28.99):
  Theta for diagnosed: 0.027 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 133
Loss: 1.8630

Monitoring signature responses:

Disease 260 (signature 8, LR=30.25):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.49):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.37):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.06):
  Theta for diagnosed: 0.027 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 134
Loss: 1.8620

Monitoring signature responses:

Disease 260 (signature 8, LR=30.24):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=29.99):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.54):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.43):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.13):
  Theta for diagnosed: 0.027 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 135
Loss: 1.8613

Monitoring signature responses:

Disease 260 (signature 8, LR=30.24):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.58):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.49):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.19):
  Theta for diagnosed: 0.027 ± 0.030
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 136
Loss: 1.8605

Monitoring signature responses:

Disease 260 (signature 8, LR=30.24):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.63):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.55):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.26):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 137
Loss: 1.8596

Monitoring signature responses:

Disease 260 (signature 8, LR=30.24):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.68):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.61):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.33):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 138
Loss: 1.8587

Monitoring signature responses:

Disease 260 (signature 8, LR=30.24):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.72):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.67):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.40):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 139
Loss: 1.8579

Monitoring signature responses:

Disease 260 (signature 8, LR=30.24):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.00):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.77):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.73):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.47):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 140
Loss: 1.8571

Monitoring signature responses:

Disease 260 (signature 8, LR=30.23):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.01):
  Theta for diagnosed: 0.153 ± 0.038
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.82):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.79):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.54):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 141
Loss: 1.8562

Monitoring signature responses:

Disease 260 (signature 8, LR=30.23):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.01):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.87):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.85):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.61):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 142
Loss: 1.8553

Monitoring signature responses:

Disease 260 (signature 8, LR=30.23):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.01):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 336 (signature 9, LR=29.92):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 56 (signature 16, LR=29.91):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=29.68):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 143
Loss: 1.8545

Monitoring signature responses:

Disease 260 (signature 8, LR=30.23):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 76 (signature 7, LR=30.01):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 56 (signature 16, LR=29.97):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=29.97):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 85 (signature 10, LR=29.76):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 144
Loss: 1.8537

Monitoring signature responses:

Disease 260 (signature 8, LR=30.23):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 56 (signature 16, LR=30.03):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.02):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 76 (signature 7, LR=30.01):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 85 (signature 10, LR=29.83):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 145
Loss: 1.8528

Monitoring signature responses:

Disease 260 (signature 8, LR=30.23):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 56 (signature 16, LR=30.09):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.07):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 76 (signature 7, LR=30.02):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 85 (signature 10, LR=29.90):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 146
Loss: 1.8519

Monitoring signature responses:

Disease 260 (signature 8, LR=30.22):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 56 (signature 16, LR=30.16):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.12):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 76 (signature 7, LR=30.02):
  Theta for diagnosed: 0.153 ± 0.039
  Theta for others: 0.147
  Proportion difference: 0.006

Disease 90 (signature 10, LR=30.01):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Epoch 147
Loss: 1.8512

Monitoring signature responses:

Disease 260 (signature 8, LR=30.22):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 56 (signature 16, LR=30.22):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.17):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 90 (signature 10, LR=30.16):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=30.05):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 148
Loss: 1.8504

Monitoring signature responses:

Disease 90 (signature 10, LR=30.31):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.28):
  Theta for diagnosed: 0.084 ± 0.044
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.23):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 260 (signature 8, LR=30.22):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 85 (signature 10, LR=30.13):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 149
Loss: 1.8495

Monitoring signature responses:

Disease 90 (signature 10, LR=30.46):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.35):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.28):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 260 (signature 8, LR=30.22):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Disease 85 (signature 10, LR=30.20):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Epoch 150
Loss: 1.8487

Monitoring signature responses:

Disease 90 (signature 10, LR=30.61):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.41):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.33):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 85 (signature 10, LR=30.28):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 260 (signature 8, LR=30.22):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Epoch 151
Loss: 1.8479

Monitoring signature responses:

Disease 90 (signature 10, LR=30.76):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.48):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.39):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 85 (signature 10, LR=30.35):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 260 (signature 8, LR=30.22):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Epoch 152
Loss: 1.8472

Monitoring signature responses:

Disease 90 (signature 10, LR=30.91):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.54):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.44):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 85 (signature 10, LR=30.43):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 260 (signature 8, LR=30.21):
  Theta for diagnosed: 0.096 ± 0.080
  Theta for others: 0.087
  Proportion difference: 0.009

Epoch 153
Loss: 1.8464

Monitoring signature responses:

Disease 90 (signature 10, LR=31.07):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.61):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.51):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.49):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.26):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 154
Loss: 1.8457

Monitoring signature responses:

Disease 90 (signature 10, LR=31.22):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.67):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.59):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.55):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.33):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 155
Loss: 1.8450

Monitoring signature responses:

Disease 90 (signature 10, LR=31.38):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.74):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.67):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.61):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.40):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 156
Loss: 1.8442

Monitoring signature responses:

Disease 90 (signature 10, LR=31.54):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.81):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.75):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.66):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.47):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 157
Loss: 1.8434

Monitoring signature responses:

Disease 90 (signature 10, LR=31.70):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.88):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.83):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.72):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.54):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 158
Loss: 1.8427

Monitoring signature responses:

Disease 90 (signature 10, LR=31.85):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=30.95):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.91):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.78):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.61):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 159
Loss: 1.8419

Monitoring signature responses:

Disease 90 (signature 10, LR=32.01):
  Theta for diagnosed: 0.027 ± 0.031
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=31.01):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=30.99):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.84):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.68):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 160
Loss: 1.8411

Monitoring signature responses:

Disease 90 (signature 10, LR=32.18):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=31.08):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=31.07):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.89):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.75):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 161
Loss: 1.8403

Monitoring signature responses:

Disease 90 (signature 10, LR=32.34):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 56 (signature 16, LR=31.15):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 85 (signature 10, LR=31.15):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 336 (signature 9, LR=30.95):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.83):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 162
Loss: 1.8396

Monitoring signature responses:

Disease 90 (signature 10, LR=32.50):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.24):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.22):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.01):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.90):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 163
Loss: 1.8389

Monitoring signature responses:

Disease 90 (signature 10, LR=32.66):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.32):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.29):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.07):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=30.97):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 164
Loss: 1.8382

Monitoring signature responses:

Disease 90 (signature 10, LR=32.83):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.40):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.37):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.13):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=31.05):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 165
Loss: 1.8376

Monitoring signature responses:

Disease 90 (signature 10, LR=33.00):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.49):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.44):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.20):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=31.12):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 166
Loss: 1.8369

Monitoring signature responses:

Disease 90 (signature 10, LR=33.16):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.57):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.51):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.26):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=31.20):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 167
Loss: 1.8360

Monitoring signature responses:

Disease 90 (signature 10, LR=33.33):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.66):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.58):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.32):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 131 (signature 0, LR=31.27):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 168
Loss: 1.8353

Monitoring signature responses:

Disease 90 (signature 10, LR=33.50):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.75):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.65):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 336 (signature 9, LR=31.38):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Disease 130 (signature 0, LR=31.37):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 169
Loss: 1.8347

Monitoring signature responses:

Disease 90 (signature 10, LR=33.67):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.84):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.73):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=31.48):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 336 (signature 9, LR=31.45):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Epoch 170
Loss: 1.8339

Monitoring signature responses:

Disease 90 (signature 10, LR=33.84):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=31.92):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.80):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=31.59):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 336 (signature 9, LR=31.51):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.030
  Proportion difference: 0.000

Epoch 171
Loss: 1.8332

Monitoring signature responses:

Disease 90 (signature 10, LR=34.01):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.01):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.88):
  Theta for diagnosed: 0.084 ± 0.045
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=31.71):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 131 (signature 0, LR=31.58):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 172
Loss: 1.8327

Monitoring signature responses:

Disease 90 (signature 10, LR=34.18):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.10):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=31.95):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=31.82):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 131 (signature 0, LR=31.66):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 173
Loss: 1.8320

Monitoring signature responses:

Disease 90 (signature 10, LR=34.36):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.19):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=32.03):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=31.93):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 131 (signature 0, LR=31.74):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 174
Loss: 1.8312

Monitoring signature responses:

Disease 90 (signature 10, LR=34.53):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.28):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=32.10):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.05):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 131 (signature 0, LR=31.81):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 175
Loss: 1.8306

Monitoring signature responses:

Disease 90 (signature 10, LR=34.71):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.37):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=32.18):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.16):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 131 (signature 0, LR=31.89):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 176
Loss: 1.8299

Monitoring signature responses:

Disease 90 (signature 10, LR=34.88):
  Theta for diagnosed: 0.027 ± 0.032
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.46):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.28):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.26):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=31.98):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 177
Loss: 1.8291

Monitoring signature responses:

Disease 90 (signature 10, LR=35.06):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.56):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.40):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.33):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.06):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 178
Loss: 1.8285

Monitoring signature responses:

Disease 90 (signature 10, LR=35.24):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.65):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.52):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.41):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.14):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 179
Loss: 1.8278

Monitoring signature responses:

Disease 90 (signature 10, LR=35.42):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.74):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.64):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.49):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.22):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 180
Loss: 1.8271

Monitoring signature responses:

Disease 90 (signature 10, LR=35.60):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.84):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.76):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.57):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.30):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 181
Loss: 1.8263

Monitoring signature responses:

Disease 90 (signature 10, LR=35.78):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=32.93):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=32.88):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.65):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.39):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 182
Loss: 1.8255

Monitoring signature responses:

Disease 90 (signature 10, LR=35.96):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 85 (signature 10, LR=33.03):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 130 (signature 0, LR=33.00):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 56 (signature 16, LR=32.73):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.47):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 183
Loss: 1.8247

Monitoring signature responses:

Disease 90 (signature 10, LR=36.14):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.13):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 85 (signature 10, LR=33.12):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=32.81):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.55):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 184
Loss: 1.8239

Monitoring signature responses:

Disease 90 (signature 10, LR=36.33):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.25):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 85 (signature 10, LR=33.22):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=32.89):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.64):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 185
Loss: 1.8232

Monitoring signature responses:

Disease 90 (signature 10, LR=36.51):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.38):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 85 (signature 10, LR=33.32):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=32.97):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.72):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 186
Loss: 1.8225

Monitoring signature responses:

Disease 90 (signature 10, LR=36.70):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.50):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=33.42):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.06):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.81):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 187
Loss: 1.8217

Monitoring signature responses:

Disease 90 (signature 10, LR=36.88):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.63):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=33.52):
  Theta for diagnosed: 0.028 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.14):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.90):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 188
Loss: 1.8209

Monitoring signature responses:

Disease 90 (signature 10, LR=37.07):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.76):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=33.62):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.22):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.002

Disease 131 (signature 0, LR=32.98):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 189
Loss: 1.8202

Monitoring signature responses:

Disease 90 (signature 10, LR=37.26):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=33.89):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=33.72):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.31):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.07):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 190
Loss: 1.8195

Monitoring signature responses:

Disease 90 (signature 10, LR=37.45):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.02):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=33.82):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.39):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.16):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 191
Loss: 1.8187

Monitoring signature responses:

Disease 90 (signature 10, LR=37.64):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.15):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=33.92):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.48):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.25):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 192
Loss: 1.8180

Monitoring signature responses:

Disease 90 (signature 10, LR=37.83):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.28):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.02):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.56):
  Theta for diagnosed: 0.084 ± 0.046
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.34):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 193
Loss: 1.8172

Monitoring signature responses:

Disease 90 (signature 10, LR=38.02):
  Theta for diagnosed: 0.027 ± 0.033
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.41):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.12):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.65):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.43):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 194
Loss: 1.8164

Monitoring signature responses:

Disease 90 (signature 10, LR=38.21):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.55):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.23):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.73):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.52):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 195
Loss: 1.8156

Monitoring signature responses:

Disease 90 (signature 10, LR=38.41):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.68):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.33):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.82):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.61):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 196
Loss: 1.8148

Monitoring signature responses:

Disease 90 (signature 10, LR=38.60):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.82):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.44):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=33.91):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.70):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 197
Loss: 1.8140

Monitoring signature responses:

Disease 90 (signature 10, LR=38.80):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=34.96):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.54):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=34.00):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.80):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 198
Loss: 1.8133

Monitoring signature responses:

Disease 90 (signature 10, LR=38.99):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=35.09):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.65):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=34.08):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.89):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Epoch 199
Loss: 1.8126

Monitoring signature responses:

Disease 90 (signature 10, LR=39.19):
  Theta for diagnosed: 0.027 ± 0.034
  Theta for others: 0.026
  Proportion difference: 0.001

Disease 130 (signature 0, LR=35.23):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 85 (signature 10, LR=34.75):
  Theta for diagnosed: 0.028 ± 0.035
  Theta for others: 0.026
  Proportion difference: 0.002

Disease 56 (signature 16, LR=34.17):
  Theta for diagnosed: 0.084 ± 0.047
  Theta for others: 0.082
  Proportion difference: 0.003

Disease 131 (signature 0, LR=33.98):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.002

Saving model to /Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt...

============================================================
Training complete! Model saved to:
/Users/sarahurbut/Library/CloudStorage/Dropbox/censor_e_batchrun_vectorized/enrollment_model_W0.0001_batch_0_10000.pt
============================================================