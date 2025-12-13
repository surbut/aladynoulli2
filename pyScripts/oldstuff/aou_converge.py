
============================================================
AOU: Initializing model...
============================================================
AOU: K=20 signatures
/var/folders/fl/ng5crz0x0fnb6c6x8dk7tfth0000gn/T/ipykernel_50600/3010181512.py:9: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  aou_checkpoint_old = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/model_with_kappa_bigam_AOU.pt', map_location='cpu')
/var/folders/fl/ng5crz0x0fnb6c6x8dk7tfth0000gn/T/ipykernel_50600/3010181512.py:27: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  prevalence_t_aou = torch.load('/Users/sarahurbut/aladynoulli2/aou_prevalence_corrected_E.pt')
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:76: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.signature_refs = torch.tensor(signature_references, dtype=torch.float32)
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:83: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.G = torch.tensor(G, dtype=torch.float32)
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:86: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.G = torch.tensor(G_scaled, dtype=torch.float32)
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:88: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.Y = torch.tensor(Y, dtype=torch.float32)
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:91: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  self.prevalence_t = torch.tensor(prevalence_t, dtype=torch.float32)

Cluster Sizes:
Cluster 0: 4 diseases
Cluster 1: 38 diseases
Cluster 2: 10 diseases
Cluster 3: 18 diseases
Cluster 4: 8 diseases
Cluster 5: 67 diseases
Cluster 6: 20 diseases
Cluster 7: 23 diseases
Cluster 8: 19 diseases
Cluster 9: 17 diseases
Cluster 10: 6 diseases
Cluster 11: 8 diseases
Cluster 12: 36 diseases
Cluster 13: 13 diseases
Cluster 14: 13 diseases
Cluster 15: 8 diseases
Cluster 16: 21 diseases
Cluster 17: 4 diseases
Cluster 18: 3 diseases
Cluster 19: 12 diseases

Calculating gamma for k=0:
Number of diseases in cluster: 4
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.5078, -0.5078, -0.5078, -0.5078, -0.5078])
Base value centered mean: -2.5932311018550536e-06
Gamma init for k=0 (first 5): tensor([0.0042, 0.0134])

Calculating gamma for k=1:
Number of diseases in cluster: 38
Base value (first 5): tensor([-10.6881, -13.8155, -13.2943, -13.0337, -13.2943])
Base value centered (first 5): tensor([ 2.2550, -0.8724, -0.3512, -0.0906, -0.3512])
Base value centered mean: -2.0380019805088523e-07
Gamma init for k=1 (first 5): tensor([ 0.0226, -0.0054])

Calculating gamma for k=2:
Number of diseases in cluster: 10
Base value (first 5): tensor([ -8.8638, -12.8252, -13.8155, -12.8252, -13.8155])
Base value centered (first 5): tensor([ 4.0851,  0.1237, -0.8667,  0.1237, -0.8667])
Base value centered mean: -8.888244451554783e-07
Gamma init for k=2 (first 5): tensor([ 0.0031, -0.0233])

Calculating gamma for k=3:
Number of diseases in cluster: 18
Base value (first 5): tensor([ -9.4140, -13.2653, -13.8155, -10.5143, -12.7151])
Base value centered (first 5): tensor([ 3.2547, -0.5966, -1.1468,  2.1543, -0.0464])
Base value centered mean: -2.056121815030565e-07
Gamma init for k=3 (first 5): tensor([ 0.0003, -0.0294])

Calculating gamma for k=4:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.4285, -0.4285, -0.4285, -0.4285, -0.4285])
Base value centered mean: -2.1224975625955267e-06
Gamma init for k=4 (first 5): tensor([ 0.0184, -0.0181])

Calculating gamma for k=5:
Number of diseases in cluster: 67
Base value (first 5): tensor([-12.3374, -13.5199, -13.6677, -13.3721, -13.8155])
Base value centered (first 5): tensor([ 1.2251,  0.0426, -0.1052,  0.1904, -0.2530])
Base value centered mean: -1.4557838312612148e-06
Gamma init for k=5 (first 5): tensor([ 0.0030, -0.0013])

Calculating gamma for k=6:
Number of diseases in cluster: 20
Base value (first 5): tensor([-10.3493, -13.8155, -12.3300, -12.3300, -13.3203])
Base value centered (first 5): tensor([ 1.7654, -1.7008, -0.2153, -0.2153, -1.2056])
Base value centered mean: -1.6130447875184473e-06
Gamma init for k=6 (first 5): tensor([-0.0102, -0.0430])

Calculating gamma for k=7:
Number of diseases in cluster: 23
Base value (first 5): tensor([-10.3708, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 3.0431, -0.4016, -0.4016, -0.4016, -0.4016])
Base value centered mean: -9.796142421691911e-07
Gamma init for k=7 (first 5): tensor([ 0.0020, -0.0131])

Calculating gamma for k=8:
Number of diseases in cluster: 19
Base value (first 5): tensor([-12.7730, -13.8155, -13.8155, -12.7730, -13.8155])
Base value centered (first 5): tensor([ 0.4233, -0.6191, -0.6191,  0.4233, -0.6191])
Base value centered mean: -2.1749497136624996e-06
Gamma init for k=8 (first 5): tensor([ 0.0122, -0.0192])

Calculating gamma for k=9:
Number of diseases in cluster: 17
Base value (first 5): tensor([-11.4853, -12.6504, -12.0678, -12.6504, -13.2330])
Base value centered (first 5): tensor([ 1.1775,  0.0124,  0.5950,  0.0124, -0.5702])
Base value centered mean: -4.834175229007087e-07
Gamma init for k=9 (first 5): tensor([ 0.0242, -0.0119])

Calculating gamma for k=10:
Number of diseases in cluster: 6
Base value (first 5): tensor([-10.5143, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 3.0584, -0.2428, -0.2428, -0.2428, -0.2428])
Base value centered mean: -2.1996497707732487e-06
Gamma init for k=10 (first 5): tensor([0.0036, 0.0025])

Calculating gamma for k=11:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1330, -0.1330, -0.1330, -0.1330, -0.1330])
Base value centered mean: -2.2764206732972525e-06
Gamma init for k=11 (first 5): tensor([ 0.0102, -0.0048])

Calculating gamma for k=12:
Number of diseases in cluster: 36
Base value (first 5): tensor([-11.0645, -13.8155, -13.5404, -13.2653, -13.5404])
Base value centered (first 5): tensor([ 2.0529, -0.6981, -0.4230, -0.1479, -0.4230])
Base value centered mean: 1.319885285511191e-07
Gamma init for k=12 (first 5): tensor([ 0.0107, -0.0053])

Calculating gamma for k=13:
Number of diseases in cluster: 13
Base value (first 5): tensor([-11.5301, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 1.3428, -0.9427, -0.9427, -0.9427, -0.9427])
Base value centered mean: -5.043029887019657e-07
Gamma init for k=13 (first 5): tensor([-0.0135, -0.0103])

Calculating gamma for k=14:
Number of diseases in cluster: 13
Base value (first 5): tensor([-11.5301, -12.2919, -11.5301, -12.2919, -13.8155])
Base value centered (first 5): tensor([ 0.5149, -0.2469,  0.5149, -0.2469, -1.7705])
Base value centered mean: -8.068084866863501e-07
Gamma init for k=14 (first 5): tensor([ 0.0202, -0.0069])

Calculating gamma for k=15:
Number of diseases in cluster: 8
Base value (first 5): tensor([-10.1017, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 3.2464, -0.4674, -0.4674, -0.4674, -0.4674])
Base value centered mean: 7.803917014825856e-07
Gamma init for k=15 (first 5): tensor([ 0.0003, -0.0101])

Calculating gamma for k=16:
Number of diseases in cluster: 21
Base value (first 5): tensor([-12.8723, -11.9291, -11.4575, -10.9859, -13.8155])
Base value centered (first 5): tensor([ 0.4690,  1.4121,  1.8837,  2.3553, -0.4742])
Base value centered mean: -1.3469696114043472e-06
Gamma init for k=16 (first 5): tensor([0.0129, 0.0041])

Calculating gamma for k=17:
Number of diseases in cluster: 4
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.3484, -0.3484, -0.3484, -0.3484, -0.3484])
Base value centered mean: -1.0765076012830832e-06
Gamma init for k=17 (first 5): tensor([-0.0015, -0.0104])

Calculating gamma for k=18:
Number of diseases in cluster: 3
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1915, -0.1915, -0.1915, -0.1915, -0.1915])
Base value centered mean: -3.6956787425879156e-06
Gamma init for k=18 (first 5): tensor([-0.0018,  0.0013])

Calculating gamma for k=19:
Number of diseases in cluster: 12
Base value (first 5): tensor([-12.9902, -13.8155, -13.8155, -12.1649, -13.8155])
Base value centered (first 5): tensor([ 0.3380, -0.4873, -0.4873,  1.1632, -0.4873])
Base value centered mean: 2.7132034574606223e-07
Gamma init for k=19 (first 5): tensor([ 0.0178, -0.0092])
Initializing with 20 disease states + 1 healthy state
Initialization complete!

Calculating gamma for k=0:
Number of diseases in cluster: 4
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.5078, -0.5078, -0.5078, -0.5078, -0.5078])
Base value centered mean: -2.5932311018550536e-06
Gamma init for k=0 (first 5): tensor([0.0042, 0.0134])

Calculating gamma for k=1:
Number of diseases in cluster: 38
Base value (first 5): tensor([-10.6881, -13.8155, -13.2943, -13.0337, -13.2943])
Base value centered (first 5): tensor([ 2.2550, -0.8724, -0.3512, -0.0906, -0.3512])
Base value centered mean: -2.0380019805088523e-07
Gamma init for k=1 (first 5): tensor([ 0.0226, -0.0054])

Calculating gamma for k=2:
Number of diseases in cluster: 10
Base value (first 5): tensor([ -8.8638, -12.8252, -13.8155, -12.8252, -13.8155])
Base value centered (first 5): tensor([ 4.0851,  0.1237, -0.8667,  0.1237, -0.8667])
Base value centered mean: -8.888244451554783e-07
Gamma init for k=2 (first 5): tensor([ 0.0031, -0.0233])

Calculating gamma for k=3:
Number of diseases in cluster: 18
Base value (first 5): tensor([ -9.4140, -13.2653, -13.8155, -10.5143, -12.7151])
Base value centered (first 5): tensor([ 3.2547, -0.5966, -1.1468,  2.1543, -0.0464])
Base value centered mean: -2.056121815030565e-07
Gamma init for k=3 (first 5): tensor([ 0.0003, -0.0294])

Calculating gamma for k=4:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.4285, -0.4285, -0.4285, -0.4285, -0.4285])
Base value centered mean: -2.1224975625955267e-06
Gamma init for k=4 (first 5): tensor([ 0.0184, -0.0181])

Calculating gamma for k=5:
Number of diseases in cluster: 67
Base value (first 5): tensor([-12.3374, -13.5199, -13.6677, -13.3721, -13.8155])
Base value centered (first 5): tensor([ 1.2251,  0.0426, -0.1052,  0.1904, -0.2530])
Base value centered mean: -1.4557838312612148e-06
Gamma init for k=5 (first 5): tensor([ 0.0030, -0.0013])

Calculating gamma for k=6:
Number of diseases in cluster: 20
Base value (first 5): tensor([-10.3493, -13.8155, -12.3300, -12.3300, -13.3203])
Base value centered (first 5): tensor([ 1.7654, -1.7008, -0.2153, -0.2153, -1.2056])
Base value centered mean: -1.6130447875184473e-06
Gamma init for k=6 (first 5): tensor([-0.0102, -0.0430])

Calculating gamma for k=7:
Number of diseases in cluster: 23
Base value (first 5): tensor([-10.3708, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 3.0431, -0.4016, -0.4016, -0.4016, -0.4016])
Base value centered mean: -9.796142421691911e-07
Gamma init for k=7 (first 5): tensor([ 0.0020, -0.0131])

Calculating gamma for k=8:
Number of diseases in cluster: 19
Base value (first 5): tensor([-12.7730, -13.8155, -13.8155, -12.7730, -13.8155])
Base value centered (first 5): tensor([ 0.4233, -0.6191, -0.6191,  0.4233, -0.6191])
Base value centered mean: -2.1749497136624996e-06
Gamma init for k=8 (first 5): tensor([ 0.0122, -0.0192])

Calculating gamma for k=9:
Number of diseases in cluster: 17
Base value (first 5): tensor([-11.4853, -12.6504, -12.0678, -12.6504, -13.2330])
Base value centered (first 5): tensor([ 1.1775,  0.0124,  0.5950,  0.0124, -0.5702])
Base value centered mean: -4.834175229007087e-07
Gamma init for k=9 (first 5): tensor([ 0.0242, -0.0119])

Calculating gamma for k=10:
Number of diseases in cluster: 6
Base value (first 5): tensor([-10.5143, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 3.0584, -0.2428, -0.2428, -0.2428, -0.2428])
Base value centered mean: -2.1996497707732487e-06
Gamma init for k=10 (first 5): tensor([0.0036, 0.0025])

Calculating gamma for k=11:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1330, -0.1330, -0.1330, -0.1330, -0.1330])
Base value centered mean: -2.2764206732972525e-06
Gamma init for k=11 (first 5): tensor([ 0.0102, -0.0048])

Calculating gamma for k=12:
Number of diseases in cluster: 36
Base value (first 5): tensor([-11.0645, -13.8155, -13.5404, -13.2653, -13.5404])
Base value centered (first 5): tensor([ 2.0529, -0.6981, -0.4230, -0.1479, -0.4230])
Base value centered mean: 1.319885285511191e-07
Gamma init for k=12 (first 5): tensor([ 0.0107, -0.0053])

Calculating gamma for k=13:
Number of diseases in cluster: 13
Base value (first 5): tensor([-11.5301, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 1.3428, -0.9427, -0.9427, -0.9427, -0.9427])
Base value centered mean: -5.043029887019657e-07
Gamma init for k=13 (first 5): tensor([-0.0135, -0.0103])

Calculating gamma for k=14:
Number of diseases in cluster: 13
Base value (first 5): tensor([-11.5301, -12.2919, -11.5301, -12.2919, -13.8155])
Base value centered (first 5): tensor([ 0.5149, -0.2469,  0.5149, -0.2469, -1.7705])
Base value centered mean: -8.068084866863501e-07
Gamma init for k=14 (first 5): tensor([ 0.0202, -0.0069])

Calculating gamma for k=15:
Number of diseases in cluster: 8
Base value (first 5): tensor([-10.1017, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([ 3.2464, -0.4674, -0.4674, -0.4674, -0.4674])
Base value centered mean: 7.803917014825856e-07
Gamma init for k=15 (first 5): tensor([ 0.0003, -0.0101])

Calculating gamma for k=16:
Number of diseases in cluster: 21
Base value (first 5): tensor([-12.8723, -11.9291, -11.4575, -10.9859, -13.8155])
Base value centered (first 5): tensor([ 0.4690,  1.4121,  1.8837,  2.3553, -0.4742])
Base value centered mean: -1.3469696114043472e-06
Gamma init for k=16 (first 5): tensor([0.0129, 0.0041])

Calculating gamma for k=17:
Number of diseases in cluster: 4
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.3484, -0.3484, -0.3484, -0.3484, -0.3484])
Base value centered mean: -1.0765076012830832e-06
Gamma init for k=17 (first 5): tensor([-0.0015, -0.0104])

Calculating gamma for k=18:
Number of diseases in cluster: 3
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.1915, -0.1915, -0.1915, -0.1915, -0.1915])
Base value centered mean: -3.6956787425879156e-06
Gamma init for k=18 (first 5): tensor([-0.0018,  0.0013])

Calculating gamma for k=19:
Number of diseases in cluster: 12
Base value (first 5): tensor([-12.9902, -13.8155, -13.8155, -12.1649, -13.8155])
Base value centered (first 5): tensor([ 0.3380, -0.4873, -0.4873,  1.1632, -0.4873])
Base value centered mean: 2.7132034574606223e-07
Gamma init for k=19 (first 5): tensor([ 0.0178, -0.0092])
Initializing with 20 disease states + 1 healthy state
Initialization complete!
✓ Clusters match: True
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:238: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  event_times_tensor = torch.tensor(event_times, dtype=torch.long)

Epoch 0
Loss: 162.5917

Monitoring signature responses:

Disease 31 (signature 13, LR=29.44):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=29.37):
  Theta for diagnosed: 0.100 ± 0.015
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=28.84):
  Theta for diagnosed: 0.067 ± 0.016
  Theta for others: 0.067
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.49):
  Theta for diagnosed: 0.067 ± 0.016
  Theta for others: 0.067
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.46):
  Theta for diagnosed: 0.067 ± 0.016
  Theta for others: 0.067
  Proportion difference: 0.000

Epoch 1
Loss: 783.3713

Monitoring signature responses:

Disease 31 (signature 13, LR=29.63):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=29.57):
  Theta for diagnosed: 0.100 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=28.89):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.44):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.067
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.44):
  Theta for diagnosed: 0.067 ± 0.014
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 2
Loss: 214.2152

Monitoring signature responses:

Disease 31 (signature 13, LR=29.81):
  Theta for diagnosed: 0.048 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=29.76):
  Theta for diagnosed: 0.100 ± 0.013
  Theta for others: 0.100
  Proportion difference: -0.000

Disease 36 (signature 5, LR=28.93):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 80 (signature 5, LR=28.41):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.40):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Epoch 3
Loss: 302.9288

Monitoring signature responses:

Disease 31 (signature 13, LR=29.97):
  Theta for diagnosed: 0.048 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=29.93):
  Theta for diagnosed: 0.100 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=28.98):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 80 (signature 5, LR=28.42):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.42):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Epoch 4
Loss: 502.2620

Monitoring signature responses:

Disease 31 (signature 13, LR=30.13):
  Theta for diagnosed: 0.048 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.07):
  Theta for diagnosed: 0.100 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=29.05):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.46):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.45):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 5
Loss: 413.4489

Monitoring signature responses:

Disease 31 (signature 13, LR=30.29):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.21):
  Theta for diagnosed: 0.100 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=29.12):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.51):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.48):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 6
Loss: 240.0372

Monitoring signature responses:

Disease 31 (signature 13, LR=30.46):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.35):
  Theta for diagnosed: 0.101 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=29.19):
  Theta for diagnosed: 0.067 ± 0.014
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.56):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.52):
  Theta for diagnosed: 0.067 ± 0.014
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 7
Loss: 176.0219

Monitoring signature responses:

Disease 31 (signature 13, LR=30.64):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.48):
  Theta for diagnosed: 0.101 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=29.27):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.59):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.067
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.55):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Epoch 8
Loss: 240.8304

Monitoring signature responses:

Disease 31 (signature 13, LR=30.82):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.61):
  Theta for diagnosed: 0.101 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.000

Disease 36 (signature 5, LR=29.34):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.61):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.067
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.57):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Epoch 9
Loss: 314.1893

Monitoring signature responses:

Disease 31 (signature 13, LR=31.00):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.73):
  Theta for diagnosed: 0.101 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.42):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.61):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.067
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.59):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Epoch 10
Loss: 303.4488

Monitoring signature responses:

Disease 31 (signature 13, LR=31.18):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.84):
  Theta for diagnosed: 0.101 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.49):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.62):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.067
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.60):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.000

Epoch 11
Loss: 231.9668

Monitoring signature responses:

Disease 31 (signature 13, LR=31.36):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=30.94):
  Theta for diagnosed: 0.101 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.56):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.63):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.62):
  Theta for diagnosed: 0.067 ± 0.014
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 12
Loss: 173.6573

Monitoring signature responses:

Disease 31 (signature 13, LR=31.53):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=31.02):
  Theta for diagnosed: 0.101 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.63):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.65):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.63):
  Theta for diagnosed: 0.067 ± 0.014
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 13
Loss: 172.1789

Monitoring signature responses:

Disease 31 (signature 13, LR=31.71):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=31.09):
  Theta for diagnosed: 0.101 ± 0.013
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.70):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.68):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.65):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 14
Loss: 210.2905

Monitoring signature responses:

Disease 31 (signature 13, LR=31.87):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=31.13):
  Theta for diagnosed: 0.101 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.76):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.71):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.68):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 15
Loss: 237.6143

Monitoring signature responses:

Disease 31 (signature 13, LR=32.03):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 225 (signature 12, LR=31.16):
  Theta for diagnosed: 0.101 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.82):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.75):
  Theta for diagnosed: 0.066 ± 0.015
  Theta for others: 0.066
  Proportion difference: -0.000

Disease 80 (signature 5, LR=28.70):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 16
Loss: 225.7730

Monitoring signature responses:

Disease 31 (signature 13, LR=32.19):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=31.17):
  Theta for diagnosed: 0.101 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.87):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.001

Disease 289 (signature 13, LR=28.84):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.000

Disease 11 (signature 5, LR=28.78):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.000

Epoch 17
Loss: 189.6857

Monitoring signature responses:

Disease 31 (signature 13, LR=32.34):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=31.17):
  Theta for diagnosed: 0.101 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.92):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.066
  Proportion difference: 0.001

Disease 289 (signature 13, LR=29.06):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=28.81):
  Theta for diagnosed: 0.045 ± 0.020
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 18
Loss: 163.0923

Monitoring signature responses:

Disease 31 (signature 13, LR=32.49):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=31.16):
  Theta for diagnosed: 0.102 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=29.97):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 289 (signature 13, LR=29.28):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=28.97):
  Theta for diagnosed: 0.045 ± 0.020
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 19
Loss: 164.5646

Monitoring signature responses:

Disease 31 (signature 13, LR=32.64):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=31.12):
  Theta for diagnosed: 0.102 ± 0.014
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=30.02):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 289 (signature 13, LR=29.50):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.12):
  Theta for diagnosed: 0.045 ± 0.020
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 20
Loss: 183.7986

Monitoring signature responses:

Disease 31 (signature 13, LR=32.79):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=31.08):
  Theta for diagnosed: 0.102 ± 0.015
  Theta for others: 0.100
  Proportion difference: 0.001

Disease 36 (signature 5, LR=30.07):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 289 (signature 13, LR=29.71):
  Theta for diagnosed: 0.049 ± 0.009
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.27):
  Theta for diagnosed: 0.045 ± 0.020
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 21
Loss: 196.2072

Monitoring signature responses:

Disease 31 (signature 13, LR=32.93):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=31.01):
  Theta for diagnosed: 0.102 ± 0.015
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.13):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 289 (signature 13, LR=29.93):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.42):
  Theta for diagnosed: 0.046 ± 0.020
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 22
Loss: 188.8418

Monitoring signature responses:

Disease 31 (signature 13, LR=33.07):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.94):
  Theta for diagnosed: 0.102 ± 0.015
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.17):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 289 (signature 13, LR=30.14):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.56):
  Theta for diagnosed: 0.046 ± 0.020
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 23
Loss: 170.1958

Monitoring signature responses:

Disease 31 (signature 13, LR=33.20):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.85):
  Theta for diagnosed: 0.102 ± 0.015
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 289 (signature 13, LR=30.34):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 36 (signature 5, LR=30.22):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.69):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 24
Loss: 157.8088

Monitoring signature responses:

Disease 31 (signature 13, LR=33.33):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.74):
  Theta for diagnosed: 0.102 ± 0.015
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 289 (signature 13, LR=30.54):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 36 (signature 5, LR=30.25):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.82):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 25
Loss: 160.1911

Monitoring signature responses:

Disease 31 (signature 13, LR=33.46):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=30.73):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.63):
  Theta for diagnosed: 0.102 ± 0.016
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.28):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=29.94):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 26
Loss: 170.3321

Monitoring signature responses:

Disease 31 (signature 13, LR=33.57):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=30.91):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.51):
  Theta for diagnosed: 0.102 ± 0.016
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.31):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.05):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 27
Loss: 175.1040

Monitoring signature responses:

Disease 31 (signature 13, LR=33.68):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=31.09):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.37):
  Theta for diagnosed: 0.102 ± 0.016
  Theta for others: 0.100
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.33):
  Theta for diagnosed: 0.067 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.15):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.001

Epoch 28
Loss: 169.2509

Monitoring signature responses:

Disease 31 (signature 13, LR=33.79):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=31.26):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 36 (signature 5, LR=30.35):
  Theta for diagnosed: 0.068 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.25):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.001

Disease 225 (signature 12, LR=30.23):
  Theta for diagnosed: 0.103 ± 0.016
  Theta for others: 0.100
  Proportion difference: 0.002

Epoch 29
Loss: 159.1125

Monitoring signature responses:

Disease 31 (signature 13, LR=33.89):
  Theta for diagnosed: 0.049 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=31.42):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 36 (signature 5, LR=30.37):
  Theta for diagnosed: 0.068 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.34):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.002

Disease 225 (signature 12, LR=30.08):
  Theta for diagnosed: 0.103 ± 0.017
  Theta for others: 0.100
  Proportion difference: 0.002

Epoch 30
Loss: 154.2333

Monitoring signature responses:

Disease 31 (signature 13, LR=33.98):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=31.58):
  Theta for diagnosed: 0.049 ± 0.010
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.43):
  Theta for diagnosed: 0.046 ± 0.021
  Theta for others: 0.044
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.39):
  Theta for diagnosed: 0.068 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 236 (signature 2, LR=30.14):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Epoch 31
Loss: 157.3187

Monitoring signature responses:

Disease 31 (signature 13, LR=34.07):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=31.74):
  Theta for diagnosed: 0.049 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.51):
  Theta for diagnosed: 0.046 ± 0.022
  Theta for others: 0.044
  Proportion difference: 0.002

Disease 36 (signature 5, LR=30.41):
  Theta for diagnosed: 0.068 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Disease 236 (signature 2, LR=30.35):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Epoch 32
Loss: 162.6310

Monitoring signature responses:

Disease 31 (signature 13, LR=34.15):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=31.89):
  Theta for diagnosed: 0.049 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.59):
  Theta for diagnosed: 0.046 ± 0.022
  Theta for others: 0.044
  Proportion difference: 0.002

Disease 236 (signature 2, LR=30.55):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 36 (signature 5, LR=30.43):
  Theta for diagnosed: 0.068 ± 0.015
  Theta for others: 0.067
  Proportion difference: 0.001

Epoch 33
Loss: 163.2608

Monitoring signature responses:

Disease 31 (signature 13, LR=34.23):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=32.04):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 236 (signature 2, LR=30.75):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 84 (signature 8, LR=30.66):
  Theta for diagnosed: 0.046 ± 0.022
  Theta for others: 0.044
  Proportion difference: 0.002

Disease 248 (signature 2, LR=30.53):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Epoch 34
Loss: 158.4718

Monitoring signature responses:

Disease 31 (signature 13, LR=34.31):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=32.18):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 236 (signature 2, LR=30.95):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=30.74):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 84 (signature 8, LR=30.72):
  Theta for diagnosed: 0.046 ± 0.022
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 35
Loss: 153.3723

Monitoring signature responses:

Disease 31 (signature 13, LR=34.38):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=32.32):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 236 (signature 2, LR=31.15):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=30.96):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 84 (signature 8, LR=30.78):
  Theta for diagnosed: 0.046 ± 0.022
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 36
Loss: 152.5517

Monitoring signature responses:

Disease 31 (signature 13, LR=34.45):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 289 (signature 13, LR=32.45):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 236 (signature 2, LR=31.35):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=31.17):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.84):
  Theta for diagnosed: 0.046 ± 0.022
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 37
Loss: 155.3154

Monitoring signature responses:

Disease 31 (signature 13, LR=34.52):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=32.58):
  Theta for diagnosed: 0.050 ± 0.011
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 236 (signature 2, LR=31.54):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=31.38):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.89):
  Theta for diagnosed: 0.047 ± 0.023
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 38
Loss: 157.3082

Monitoring signature responses:

Disease 31 (signature 13, LR=34.58):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=32.70):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.001

Disease 236 (signature 2, LR=31.74):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=31.59):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.94):
  Theta for diagnosed: 0.047 ± 0.023
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 39
Loss: 155.8530

Monitoring signature responses:

Disease 31 (signature 13, LR=34.64):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=32.82):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=31.93):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=31.80):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=30.98):
  Theta for diagnosed: 0.047 ± 0.023
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 40
Loss: 152.5815

Monitoring signature responses:

Disease 31 (signature 13, LR=34.69):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=32.94):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=32.13):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=32.01):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.02):
  Theta for diagnosed: 0.047 ± 0.023
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 41
Loss: 150.8810

Monitoring signature responses:

Disease 31 (signature 13, LR=34.75):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.05):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=32.32):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=32.22):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.06):
  Theta for diagnosed: 0.047 ± 0.023
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 42
Loss: 151.8246

Monitoring signature responses:

Disease 31 (signature 13, LR=34.80):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.17):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=32.52):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=32.43):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.10):
  Theta for diagnosed: 0.047 ± 0.024
  Theta for others: 0.044
  Proportion difference: 0.002

Epoch 43
Loss: 153.3547

Monitoring signature responses:

Disease 31 (signature 13, LR=34.85):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.28):
  Theta for diagnosed: 0.050 ± 0.012
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=32.71):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=32.65):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.14):
  Theta for diagnosed: 0.047 ± 0.024
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 44
Loss: 153.1522

Monitoring signature responses:

Disease 31 (signature 13, LR=34.90):
  Theta for diagnosed: 0.050 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.38):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=32.91):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=32.87):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.17):
  Theta for diagnosed: 0.047 ± 0.024
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 45
Loss: 151.3410

Monitoring signature responses:

Disease 31 (signature 13, LR=34.95):
  Theta for diagnosed: 0.050 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.49):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=33.11):
  Theta for diagnosed: 0.034 ± 0.005
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=33.08):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.20):
  Theta for diagnosed: 0.047 ± 0.024
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 46
Loss: 149.9183

Monitoring signature responses:

Disease 31 (signature 13, LR=34.99):
  Theta for diagnosed: 0.051 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.60):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=33.30):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 248 (signature 2, LR=33.30):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 84 (signature 8, LR=31.24):
  Theta for diagnosed: 0.047 ± 0.024
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 47
Loss: 150.0993

Monitoring signature responses:

Disease 31 (signature 13, LR=35.04):
  Theta for diagnosed: 0.051 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.70):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 248 (signature 2, LR=33.52):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=33.50):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 84 (signature 8, LR=31.27):
  Theta for diagnosed: 0.047 ± 0.025
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 48
Loss: 151.0166

Monitoring signature responses:

Disease 31 (signature 13, LR=35.09):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 289 (signature 13, LR=33.80):
  Theta for diagnosed: 0.050 ± 0.013
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 248 (signature 2, LR=33.74):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=33.70):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 84 (signature 8, LR=31.30):
  Theta for diagnosed: 0.047 ± 0.025
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 49
Loss: 151.1053

Monitoring signature responses:

Disease 31 (signature 13, LR=35.13):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 248 (signature 2, LR=33.97):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=33.90):
  Theta for diagnosed: 0.050 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 236 (signature 2, LR=33.90):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 84 (signature 8, LR=31.33):
  Theta for diagnosed: 0.048 ± 0.025
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 50
Loss: 150.0990

Monitoring signature responses:

Disease 31 (signature 13, LR=35.18):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 248 (signature 2, LR=34.19):
  Theta for diagnosed: 0.035 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=34.10):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 289 (signature 13, LR=34.01):
  Theta for diagnosed: 0.051 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 84 (signature 8, LR=31.36):
  Theta for diagnosed: 0.048 ± 0.025
  Theta for others: 0.044
  Proportion difference: 0.003

Epoch 51
Loss: 149.1261

Monitoring signature responses:

Disease 31 (signature 13, LR=35.23):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 248 (signature 2, LR=34.42):
  Theta for diagnosed: 0.035 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=34.30):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.000

Disease 289 (signature 13, LR=34.11):
  Theta for diagnosed: 0.051 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 291 (signature 13, LR=31.43):
  Theta for diagnosed: 0.052 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 52
Loss: 149.0907

Monitoring signature responses:

Disease 31 (signature 13, LR=35.27):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 248 (signature 2, LR=34.64):
  Theta for diagnosed: 0.035 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=34.50):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=34.21):
  Theta for diagnosed: 0.051 ± 0.014
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 291 (signature 13, LR=31.57):
  Theta for diagnosed: 0.052 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 53
Loss: 149.5984

Monitoring signature responses:

Disease 31 (signature 13, LR=35.32):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 248 (signature 2, LR=34.87):
  Theta for diagnosed: 0.035 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=34.70):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=34.31):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.002

Disease 291 (signature 13, LR=31.71):
  Theta for diagnosed: 0.052 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 54
Loss: 149.6679

Monitoring signature responses:

Disease 31 (signature 13, LR=35.37):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 248 (signature 2, LR=35.11):
  Theta for diagnosed: 0.035 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=34.91):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=34.42):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=31.85):
  Theta for diagnosed: 0.052 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 55
Loss: 149.0588

Monitoring signature responses:

Disease 31 (signature 13, LR=35.42):
  Theta for diagnosed: 0.051 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 248 (signature 2, LR=35.34):
  Theta for diagnosed: 0.035 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=35.11):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=34.52):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.00):
  Theta for diagnosed: 0.052 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 56
Loss: 148.4439

Monitoring signature responses:

Disease 248 (signature 2, LR=35.58):
  Theta for diagnosed: 0.035 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.47):
  Theta for diagnosed: 0.051 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 236 (signature 2, LR=35.32):
  Theta for diagnosed: 0.034 ± 0.006
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=34.63):
  Theta for diagnosed: 0.051 ± 0.015
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.14):
  Theta for diagnosed: 0.053 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 57
Loss: 148.3937

Monitoring signature responses:

Disease 248 (signature 2, LR=35.82):
  Theta for diagnosed: 0.035 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=35.53):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.52):
  Theta for diagnosed: 0.051 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 289 (signature 13, LR=34.73):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.29):
  Theta for diagnosed: 0.053 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 58
Loss: 148.6625

Monitoring signature responses:

Disease 248 (signature 2, LR=36.06):
  Theta for diagnosed: 0.035 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=35.74):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.57):
  Theta for diagnosed: 0.052 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 289 (signature 13, LR=34.84):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.44):
  Theta for diagnosed: 0.053 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.004

Epoch 59
Loss: 148.6472

Monitoring signature responses:

Disease 248 (signature 2, LR=36.30):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=35.95):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.62):
  Theta for diagnosed: 0.052 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 289 (signature 13, LR=34.94):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.59):
  Theta for diagnosed: 0.053 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 60
Loss: 148.2339

Monitoring signature responses:

Disease 248 (signature 2, LR=36.54):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=36.16):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.68):
  Theta for diagnosed: 0.052 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 289 (signature 13, LR=35.05):
  Theta for diagnosed: 0.051 ± 0.016
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.74):
  Theta for diagnosed: 0.053 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 61
Loss: 147.8606

Monitoring signature responses:

Disease 248 (signature 2, LR=36.79):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=36.37):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.73):
  Theta for diagnosed: 0.052 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 289 (signature 13, LR=35.16):
  Theta for diagnosed: 0.051 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=32.90):
  Theta for diagnosed: 0.053 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 62
Loss: 147.8410

Monitoring signature responses:

Disease 248 (signature 2, LR=37.04):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=36.58):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.79):
  Theta for diagnosed: 0.052 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 289 (signature 13, LR=35.27):
  Theta for diagnosed: 0.051 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=33.06):
  Theta for diagnosed: 0.053 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 63
Loss: 147.9681

Monitoring signature responses:

Disease 248 (signature 2, LR=37.29):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=36.80):
  Theta for diagnosed: 0.034 ± 0.007
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.85):
  Theta for diagnosed: 0.052 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=35.38):
  Theta for diagnosed: 0.052 ± 0.017
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=33.21):
  Theta for diagnosed: 0.053 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 64
Loss: 147.8821

Monitoring signature responses:

Disease 248 (signature 2, LR=37.54):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=37.01):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.91):
  Theta for diagnosed: 0.052 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=35.49):
  Theta for diagnosed: 0.052 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=33.37):
  Theta for diagnosed: 0.054 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 65
Loss: 147.5825

Monitoring signature responses:

Disease 248 (signature 2, LR=37.80):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=37.23):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=35.97):
  Theta for diagnosed: 0.052 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=35.60):
  Theta for diagnosed: 0.052 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.003

Disease 291 (signature 13, LR=33.53):
  Theta for diagnosed: 0.054 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.005

Epoch 66
Loss: 147.3651

Monitoring signature responses:

Disease 248 (signature 2, LR=38.06):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=37.44):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.03):
  Theta for diagnosed: 0.052 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=35.71):
  Theta for diagnosed: 0.052 ± 0.018
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=33.69):
  Theta for diagnosed: 0.054 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 67
Loss: 147.3644

Monitoring signature responses:

Disease 248 (signature 2, LR=38.31):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=37.66):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.09):
  Theta for diagnosed: 0.052 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=35.82):
  Theta for diagnosed: 0.052 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=33.86):
  Theta for diagnosed: 0.054 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 68
Loss: 147.3963

Monitoring signature responses:

Disease 248 (signature 2, LR=38.57):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=37.88):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.15):
  Theta for diagnosed: 0.053 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=35.93):
  Theta for diagnosed: 0.052 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=34.02):
  Theta for diagnosed: 0.054 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 69
Loss: 147.2692

Monitoring signature responses:

Disease 248 (signature 2, LR=38.83):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=38.10):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.21):
  Theta for diagnosed: 0.053 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=36.05):
  Theta for diagnosed: 0.052 ± 0.019
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=34.19):
  Theta for diagnosed: 0.054 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 70
Loss: 147.0526

Monitoring signature responses:

Disease 248 (signature 2, LR=39.09):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=38.31):
  Theta for diagnosed: 0.034 ± 0.008
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.27):
  Theta for diagnosed: 0.053 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=36.16):
  Theta for diagnosed: 0.052 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=34.35):
  Theta for diagnosed: 0.054 ± 0.023
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 71
Loss: 146.9329

Monitoring signature responses:

Disease 248 (signature 2, LR=39.36):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=38.53):
  Theta for diagnosed: 0.034 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.34):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 289 (signature 13, LR=36.27):
  Theta for diagnosed: 0.052 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=34.52):
  Theta for diagnosed: 0.055 ± 0.023
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 72
Loss: 146.9287

Monitoring signature responses:

Disease 248 (signature 2, LR=39.62):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=38.75):
  Theta for diagnosed: 0.034 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 31 (signature 13, LR=36.40):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.049
  Proportion difference: 0.004

Disease 289 (signature 13, LR=36.38):
  Theta for diagnosed: 0.052 ± 0.020
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 291 (signature 13, LR=34.68):
  Theta for diagnosed: 0.055 ± 0.023
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 73
Loss: 146.8937

Monitoring signature responses:

Disease 248 (signature 2, LR=39.89):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=38.97):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=36.49):
  Theta for diagnosed: 0.052 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 31 (signature 13, LR=36.46):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 291 (signature 13, LR=34.85):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 74
Loss: 146.7552

Monitoring signature responses:

Disease 248 (signature 2, LR=40.15):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 236 (signature 2, LR=39.18):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=36.60):
  Theta for diagnosed: 0.053 ± 0.021
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 31 (signature 13, LR=36.53):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 291 (signature 13, LR=35.02):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 75
Loss: 146.6061

Monitoring signature responses:

Disease 248 (signature 2, LR=40.42):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=39.40):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=36.71):
  Theta for diagnosed: 0.053 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.004

Disease 31 (signature 13, LR=36.59):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 291 (signature 13, LR=35.19):
  Theta for diagnosed: 0.055 ± 0.025
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 76
Loss: 146.5400

Monitoring signature responses:

Disease 248 (signature 2, LR=40.69):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=39.62):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=36.81):
  Theta for diagnosed: 0.053 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=36.66):
  Theta for diagnosed: 0.053 ± 0.025
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 291 (signature 13, LR=35.36):
  Theta for diagnosed: 0.055 ± 0.025
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 77
Loss: 146.5147

Monitoring signature responses:

Disease 248 (signature 2, LR=40.96):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=39.83):
  Theta for diagnosed: 0.035 ± 0.009
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=36.92):
  Theta for diagnosed: 0.053 ± 0.022
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=36.72):
  Theta for diagnosed: 0.053 ± 0.025
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 232 (signature 15, LR=35.56):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Epoch 78
Loss: 146.4378

Monitoring signature responses:

Disease 248 (signature 2, LR=41.23):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=40.04):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.02):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=36.78):
  Theta for diagnosed: 0.054 ± 0.026
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 232 (signature 15, LR=35.83):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Epoch 79
Loss: 146.3120

Monitoring signature responses:

Disease 248 (signature 2, LR=41.50):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=40.26):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.13):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=36.84):
  Theta for diagnosed: 0.054 ± 0.026
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 232 (signature 15, LR=36.10):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Epoch 80
Loss: 146.2142

Monitoring signature responses:

Disease 248 (signature 2, LR=41.77):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=40.47):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.23):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=36.91):
  Theta for diagnosed: 0.054 ± 0.027
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 232 (signature 15, LR=36.38):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Epoch 81
Loss: 146.1672

Monitoring signature responses:

Disease 248 (signature 2, LR=42.04):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=40.68):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.33):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=36.97):
  Theta for diagnosed: 0.054 ± 0.027
  Theta for others: 0.049
  Proportion difference: 0.005

Disease 232 (signature 15, LR=36.66):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Epoch 82
Loss: 146.1149

Monitoring signature responses:

Disease 248 (signature 2, LR=42.31):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=40.89):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.42):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 31 (signature 13, LR=37.03):
  Theta for diagnosed: 0.054 ± 0.028
  Theta for others: 0.049
  Proportion difference: 0.006

Disease 232 (signature 15, LR=36.94):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Epoch 83
Loss: 146.0211

Monitoring signature responses:

Disease 248 (signature 2, LR=42.58):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=41.10):
  Theta for diagnosed: 0.035 ± 0.010
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.52):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 232 (signature 15, LR=37.23):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 31 (signature 13, LR=37.09):
  Theta for diagnosed: 0.054 ± 0.028
  Theta for others: 0.049
  Proportion difference: 0.006

Epoch 84
Loss: 145.9208

Monitoring signature responses:

Disease 248 (signature 2, LR=42.85):
  Theta for diagnosed: 0.036 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=41.30):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 289 (signature 13, LR=37.61):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.048
  Proportion difference: 0.005

Disease 232 (signature 15, LR=37.51):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 31 (signature 13, LR=37.15):
  Theta for diagnosed: 0.054 ± 0.028
  Theta for others: 0.049
  Proportion difference: 0.006

Epoch 85
Loss: 145.8540

Monitoring signature responses:

Disease 248 (signature 2, LR=43.12):
  Theta for diagnosed: 0.036 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=41.51):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=37.80):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 289 (signature 13, LR=37.70):
  Theta for diagnosed: 0.054 ± 0.026
  Theta for others: 0.048
  Proportion difference: 0.006

Disease 31 (signature 13, LR=37.21):
  Theta for diagnosed: 0.054 ± 0.029
  Theta for others: 0.049
  Proportion difference: 0.006

Epoch 86
Loss: 145.8032

Monitoring signature responses:

Disease 248 (signature 2, LR=43.39):
  Theta for diagnosed: 0.036 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=41.71):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=38.09):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 289 (signature 13, LR=37.79):
  Theta for diagnosed: 0.054 ± 0.026
  Theta for others: 0.048
  Proportion difference: 0.006

Disease 31 (signature 13, LR=37.26):
  Theta for diagnosed: 0.055 ± 0.029
  Theta for others: 0.049
  Proportion difference: 0.006

Epoch 87
Loss: 145.7307

Monitoring signature responses:

Disease 248 (signature 2, LR=43.66):
  Theta for diagnosed: 0.036 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=41.91):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=38.38):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 289 (signature 13, LR=37.88):
  Theta for diagnosed: 0.054 ± 0.027
  Theta for others: 0.048
  Proportion difference: 0.006

Disease 169 (signature 19, LR=37.49):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Epoch 88
Loss: 145.6405

Monitoring signature responses:

Disease 248 (signature 2, LR=43.93):
  Theta for diagnosed: 0.036 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=42.11):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=38.68):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 289 (signature 13, LR=37.96):
  Theta for diagnosed: 0.054 ± 0.027
  Theta for others: 0.048
  Proportion difference: 0.006

Disease 169 (signature 19, LR=37.76):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Epoch 89
Loss: 145.5648

Monitoring signature responses:

Disease 248 (signature 2, LR=44.19):
  Theta for diagnosed: 0.036 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=42.30):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=38.97):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 289 (signature 13, LR=38.04):
  Theta for diagnosed: 0.054 ± 0.028
  Theta for others: 0.048
  Proportion difference: 0.006

Disease 169 (signature 19, LR=38.04):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Epoch 90
Loss: 145.5081

Monitoring signature responses:

Disease 248 (signature 2, LR=44.46):
  Theta for diagnosed: 0.036 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=42.49):
  Theta for diagnosed: 0.035 ± 0.011
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=39.27):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=38.31):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 289 (signature 13, LR=38.12):
  Theta for diagnosed: 0.054 ± 0.028
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 91
Loss: 145.4453

Monitoring signature responses:

Disease 248 (signature 2, LR=44.73):
  Theta for diagnosed: 0.036 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=42.68):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=39.57):
  Theta for diagnosed: 0.015 ± 0.004
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=38.59):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 289 (signature 13, LR=38.20):
  Theta for diagnosed: 0.054 ± 0.028
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 92
Loss: 145.3664

Monitoring signature responses:

Disease 248 (signature 2, LR=45.00):
  Theta for diagnosed: 0.036 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=42.87):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=39.87):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=38.86):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 289 (signature 13, LR=38.27):
  Theta for diagnosed: 0.055 ± 0.029
  Theta for others: 0.048
  Proportion difference: 0.006

Epoch 93
Loss: 145.2900

Monitoring signature responses:

Disease 248 (signature 2, LR=45.26):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=43.06):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=40.18):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=39.14):
  Theta for diagnosed: 0.023 ± 0.009
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 289 (signature 13, LR=38.35):
  Theta for diagnosed: 0.055 ± 0.029
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 94
Loss: 145.2281

Monitoring signature responses:

Disease 248 (signature 2, LR=45.53):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=43.25):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=40.48):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=39.42):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 289 (signature 13, LR=38.41):
  Theta for diagnosed: 0.055 ± 0.030
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 95
Loss: 145.1681

Monitoring signature responses:

Disease 248 (signature 2, LR=45.79):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=43.43):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=40.79):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=39.70):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 289 (signature 13, LR=38.48):
  Theta for diagnosed: 0.055 ± 0.030
  Theta for others: 0.048
  Proportion difference: 0.007

Epoch 96
Loss: 145.0975

Monitoring signature responses:

Disease 248 (signature 2, LR=46.06):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=43.61):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=41.10):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=39.99):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=38.57):
  Theta for diagnosed: 0.059 ± 0.036
  Theta for others: 0.048
  Proportion difference: 0.010

Epoch 97
Loss: 145.0240

Monitoring signature responses:

Disease 248 (signature 2, LR=46.32):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 236 (signature 2, LR=43.79):
  Theta for diagnosed: 0.035 ± 0.012
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=41.41):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=40.27):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=38.72):
  Theta for diagnosed: 0.059 ± 0.036
  Theta for others: 0.048
  Proportion difference: 0.011

Epoch 98
Loss: 144.9595

Monitoring signature responses:

Disease 248 (signature 2, LR=46.58):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=43.97):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=41.73):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=40.56):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=38.87):
  Theta for diagnosed: 0.059 ± 0.037
  Theta for others: 0.048
  Proportion difference: 0.011

Epoch 99
Loss: 144.8997

Monitoring signature responses:

Disease 248 (signature 2, LR=46.84):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=44.14):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=42.05):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=40.84):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.02):
  Theta for diagnosed: 0.059 ± 0.038
  Theta for others: 0.048
  Proportion difference: 0.011

Epoch 100
Loss: 144.8341

Monitoring signature responses:

Disease 248 (signature 2, LR=47.10):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=44.31):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=42.36):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=41.13):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.16):
  Theta for diagnosed: 0.059 ± 0.038
  Theta for others: 0.048
  Proportion difference: 0.011

Epoch 101
Loss: 144.7644

Monitoring signature responses:

Disease 248 (signature 2, LR=47.36):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=44.49):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=42.68):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=41.42):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.30):
  Theta for diagnosed: 0.060 ± 0.039
  Theta for others: 0.048
  Proportion difference: 0.011

Epoch 102
Loss: 144.6996

Monitoring signature responses:

Disease 248 (signature 2, LR=47.62):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=44.65):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.001

Disease 232 (signature 15, LR=43.01):
  Theta for diagnosed: 0.015 ± 0.005
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=41.71):
  Theta for diagnosed: 0.023 ± 0.010
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.45):
  Theta for diagnosed: 0.060 ± 0.039
  Theta for others: 0.048
  Proportion difference: 0.012

Epoch 103
Loss: 144.6395

Monitoring signature responses:

Disease 248 (signature 2, LR=47.88):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=44.82):
  Theta for diagnosed: 0.035 ± 0.013
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=43.33):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=42.01):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.59):
  Theta for diagnosed: 0.060 ± 0.040
  Theta for others: 0.048
  Proportion difference: 0.012

Epoch 104
Loss: 144.5767

Monitoring signature responses:

Disease 248 (signature 2, LR=48.14):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=44.99):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=43.66):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=42.30):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.73):
  Theta for diagnosed: 0.060 ± 0.041
  Theta for others: 0.048
  Proportion difference: 0.012

Epoch 105
Loss: 144.5104

Monitoring signature responses:

Disease 248 (signature 2, LR=48.39):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=45.15):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=43.99):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=42.60):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=39.87):
  Theta for diagnosed: 0.060 ± 0.041
  Theta for others: 0.048
  Proportion difference: 0.012

Epoch 106
Loss: 144.4465

Monitoring signature responses:

Disease 248 (signature 2, LR=48.65):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=45.31):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=44.32):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=42.89):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.01):
  Theta for diagnosed: 0.061 ± 0.042
  Theta for others: 0.048
  Proportion difference: 0.012

Epoch 107
Loss: 144.3864

Monitoring signature responses:

Disease 248 (signature 2, LR=48.90):
  Theta for diagnosed: 0.037 ± 0.018
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=45.47):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=44.65):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=43.19):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.15):
  Theta for diagnosed: 0.061 ± 0.042
  Theta for others: 0.048
  Proportion difference: 0.013

Epoch 108
Loss: 144.3254

Monitoring signature responses:

Disease 248 (signature 2, LR=49.16):
  Theta for diagnosed: 0.037 ± 0.019
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=45.63):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=44.99):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=43.49):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.28):
  Theta for diagnosed: 0.061 ± 0.043
  Theta for others: 0.048
  Proportion difference: 0.013

Epoch 109
Loss: 144.2619

Monitoring signature responses:

Disease 248 (signature 2, LR=49.41):
  Theta for diagnosed: 0.037 ± 0.019
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=45.79):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=45.33):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=43.79):
  Theta for diagnosed: 0.023 ± 0.011
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.42):
  Theta for diagnosed: 0.061 ± 0.044
  Theta for others: 0.048
  Proportion difference: 0.013

Epoch 110
Loss: 144.1994

Monitoring signature responses:

Disease 248 (signature 2, LR=49.66):
  Theta for diagnosed: 0.037 ± 0.019
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=45.95):
  Theta for diagnosed: 0.035 ± 0.014
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=45.67):
  Theta for diagnosed: 0.015 ± 0.006
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=44.10):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.55):
  Theta for diagnosed: 0.061 ± 0.044
  Theta for others: 0.048
  Proportion difference: 0.013

Epoch 111
Loss: 144.1398

Monitoring signature responses:

Disease 248 (signature 2, LR=49.91):
  Theta for diagnosed: 0.037 ± 0.019
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 236 (signature 2, LR=46.10):
  Theta for diagnosed: 0.035 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 232 (signature 15, LR=46.01):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=44.40):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.69):
  Theta for diagnosed: 0.062 ± 0.045
  Theta for others: 0.048
  Proportion difference: 0.013

Epoch 112
Loss: 144.0800

Monitoring signature responses:

Disease 248 (signature 2, LR=50.16):
  Theta for diagnosed: 0.037 ± 0.019
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 232 (signature 15, LR=46.35):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=46.25):
  Theta for diagnosed: 0.035 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 169 (signature 19, LR=44.70):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.82):
  Theta for diagnosed: 0.062 ± 0.045
  Theta for others: 0.048
  Proportion difference: 0.014

Epoch 113
Loss: 144.0186

Monitoring signature responses:

Disease 248 (signature 2, LR=50.41):
  Theta for diagnosed: 0.037 ± 0.020
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 232 (signature 15, LR=46.70):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=46.41):
  Theta for diagnosed: 0.035 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 169 (signature 19, LR=45.01):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=40.96):
  Theta for diagnosed: 0.062 ± 0.046
  Theta for others: 0.048
  Proportion difference: 0.014

Epoch 114
Loss: 143.9577

Monitoring signature responses:

Disease 248 (signature 2, LR=50.66):
  Theta for diagnosed: 0.037 ± 0.020
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 232 (signature 15, LR=47.05):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=46.56):
  Theta for diagnosed: 0.035 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 169 (signature 19, LR=45.32):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=41.09):
  Theta for diagnosed: 0.062 ± 0.047
  Theta for others: 0.048
  Proportion difference: 0.014

Epoch 115
Loss: 143.8988

Monitoring signature responses:

Disease 248 (signature 2, LR=50.90):
  Theta for diagnosed: 0.037 ± 0.020
  Theta for others: 0.034
  Proportion difference: 0.003

Disease 232 (signature 15, LR=47.40):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=46.71):
  Theta for diagnosed: 0.035 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 169 (signature 19, LR=45.63):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=41.22):
  Theta for diagnosed: 0.062 ± 0.047
  Theta for others: 0.048
  Proportion difference: 0.014

Epoch 116
Loss: 143.8402

Monitoring signature responses:

Disease 248 (signature 2, LR=51.15):
  Theta for diagnosed: 0.037 ± 0.020
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=47.75):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=46.85):
  Theta for diagnosed: 0.035 ± 0.015
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 169 (signature 19, LR=45.94):
  Theta for diagnosed: 0.023 ± 0.012
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=41.36):
  Theta for diagnosed: 0.063 ± 0.048
  Theta for others: 0.048
  Proportion difference: 0.014

Epoch 117
Loss: 143.7805

Monitoring signature responses:

Disease 248 (signature 2, LR=51.39):
  Theta for diagnosed: 0.037 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=48.11):
  Theta for diagnosed: 0.015 ± 0.007
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=47.00):
  Theta for diagnosed: 0.035 ± 0.016
  Theta for others: 0.034
  Proportion difference: 0.002

Disease 169 (signature 19, LR=46.25):
  Theta for diagnosed: 0.023 ± 0.013
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=41.49):
  Theta for diagnosed: 0.063 ± 0.049
  Theta for others: 0.048
  Proportion difference: 0.015

Epoch 118
Loss: 143.7211

Monitoring signature responses:

Disease 248 (signature 2, LR=51.64):
  Theta for diagnosed: 0.037 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=48.46):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=47.15):
  Theta for diagnosed: 0.035 ± 0.016
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 169 (signature 19, LR=46.56):
  Theta for diagnosed: 0.023 ± 0.013
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=41.62):
  Theta for diagnosed: 0.063 ± 0.049
  Theta for others: 0.048
  Proportion difference: 0.015

Epoch 119
Loss: 143.6632

Monitoring signature responses:

Disease 248 (signature 2, LR=51.88):
  Theta for diagnosed: 0.037 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=48.82):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=47.29):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 169 (signature 19, LR=46.88):
  Theta for diagnosed: 0.023 ± 0.013
  Theta for others: 0.022
  Proportion difference: 0.001

Disease 291 (signature 13, LR=41.75):
  Theta for diagnosed: 0.063 ± 0.050
  Theta for others: 0.048
  Proportion difference: 0.015

Epoch 120
Loss: 143.6056

Monitoring signature responses:

Disease 248 (signature 2, LR=52.12):
  Theta for diagnosed: 0.037 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=49.18):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=47.44):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 169 (signature 19, LR=47.19):
  Theta for diagnosed: 0.023 ± 0.013
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 291 (signature 13, LR=41.88):
  Theta for diagnosed: 0.063 ± 0.050
  Theta for others: 0.048
  Proportion difference: 0.015

Epoch 121
Loss: 143.5474

Monitoring signature responses:

Disease 248 (signature 2, LR=52.37):
  Theta for diagnosed: 0.037 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=49.55):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 236 (signature 2, LR=47.58):
  Theta for diagnosed: 0.036 ± 0.016
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 169 (signature 19, LR=47.51):
  Theta for diagnosed: 0.023 ± 0.013
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.01):
  Theta for diagnosed: 0.064 ± 0.051
  Theta for others: 0.048
  Proportion difference: 0.015

Epoch 122
Loss: 143.4895

Monitoring signature responses:

Disease 248 (signature 2, LR=52.61):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=49.91):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=47.83):
  Theta for diagnosed: 0.023 ± 0.013
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=47.72):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.14):
  Theta for diagnosed: 0.064 ± 0.052
  Theta for others: 0.048
  Proportion difference: 0.016

Epoch 123
Loss: 143.4326

Monitoring signature responses:

Disease 248 (signature 2, LR=52.85):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=50.28):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=48.15):
  Theta for diagnosed: 0.024 ± 0.014
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=47.86):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.28):
  Theta for diagnosed: 0.064 ± 0.052
  Theta for others: 0.048
  Proportion difference: 0.016

Epoch 124
Loss: 143.3761

Monitoring signature responses:

Disease 248 (signature 2, LR=53.08):
  Theta for diagnosed: 0.038 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=50.65):
  Theta for diagnosed: 0.015 ± 0.008
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=48.47):
  Theta for diagnosed: 0.024 ± 0.014
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.00):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.41):
  Theta for diagnosed: 0.064 ± 0.053
  Theta for others: 0.048
  Proportion difference: 0.016

Epoch 125
Loss: 143.3192

Monitoring signature responses:

Disease 248 (signature 2, LR=53.32):
  Theta for diagnosed: 0.038 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=51.02):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=48.79):
  Theta for diagnosed: 0.024 ± 0.014
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.14):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.54):
  Theta for diagnosed: 0.064 ± 0.053
  Theta for others: 0.048
  Proportion difference: 0.016

Epoch 126
Loss: 143.2626

Monitoring signature responses:

Disease 248 (signature 2, LR=53.56):
  Theta for diagnosed: 0.038 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=51.40):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=49.11):
  Theta for diagnosed: 0.024 ± 0.014
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.28):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.67):
  Theta for diagnosed: 0.065 ± 0.054
  Theta for others: 0.048
  Proportion difference: 0.017

Epoch 127
Loss: 143.2068

Monitoring signature responses:

Disease 248 (signature 2, LR=53.80):
  Theta for diagnosed: 0.038 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=51.77):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.000

Disease 169 (signature 19, LR=49.43):
  Theta for diagnosed: 0.024 ± 0.014
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.42):
  Theta for diagnosed: 0.036 ± 0.017
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.80):
  Theta for diagnosed: 0.065 ± 0.055
  Theta for others: 0.048
  Proportion difference: 0.017

Epoch 128
Loss: 143.1512

Monitoring signature responses:

Disease 248 (signature 2, LR=54.03):
  Theta for diagnosed: 0.038 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=52.15):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=49.76):
  Theta for diagnosed: 0.024 ± 0.014
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.56):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=42.93):
  Theta for diagnosed: 0.065 ± 0.055
  Theta for others: 0.048
  Proportion difference: 0.017

Epoch 129
Loss: 143.0956

Monitoring signature responses:

Disease 248 (signature 2, LR=54.26):
  Theta for diagnosed: 0.038 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 232 (signature 15, LR=52.53):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=50.08):
  Theta for diagnosed: 0.024 ± 0.015
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.69):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=43.06):
  Theta for diagnosed: 0.065 ± 0.056
  Theta for others: 0.048
  Proportion difference: 0.017

Epoch 130
Loss: 143.0402

Monitoring signature responses:

Disease 248 (signature 2, LR=54.50):
  Theta for diagnosed: 0.038 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=52.92):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=50.41):
  Theta for diagnosed: 0.024 ± 0.015
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.83):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=43.19):
  Theta for diagnosed: 0.065 ± 0.056
  Theta for others: 0.048
  Proportion difference: 0.017

Epoch 131
Loss: 142.9855

Monitoring signature responses:

Disease 248 (signature 2, LR=54.73):
  Theta for diagnosed: 0.038 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=53.30):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=50.73):
  Theta for diagnosed: 0.024 ± 0.015
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=48.97):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.033
  Proportion difference: 0.002

Disease 291 (signature 13, LR=43.32):
  Theta for diagnosed: 0.066 ± 0.057
  Theta for others: 0.048
  Proportion difference: 0.018

Epoch 132
Loss: 142.9310

Monitoring signature responses:

Disease 248 (signature 2, LR=54.96):
  Theta for diagnosed: 0.038 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=53.69):
  Theta for diagnosed: 0.015 ± 0.009
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=51.06):
  Theta for diagnosed: 0.024 ± 0.015
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.10):
  Theta for diagnosed: 0.036 ± 0.018
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 291 (signature 13, LR=43.45):
  Theta for diagnosed: 0.066 ± 0.058
  Theta for others: 0.048
  Proportion difference: 0.018

Epoch 133
Loss: 142.8764

Monitoring signature responses:

Disease 248 (signature 2, LR=55.19):
  Theta for diagnosed: 0.038 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=54.08):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=51.39):
  Theta for diagnosed: 0.024 ± 0.015
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.24):
  Theta for diagnosed: 0.036 ± 0.019
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 291 (signature 13, LR=43.58):
  Theta for diagnosed: 0.066 ± 0.058
  Theta for others: 0.048
  Proportion difference: 0.018

Epoch 134
Loss: 142.8222

Monitoring signature responses:

Disease 248 (signature 2, LR=55.42):
  Theta for diagnosed: 0.038 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=54.47):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=51.72):
  Theta for diagnosed: 0.024 ± 0.016
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.37):
  Theta for diagnosed: 0.036 ± 0.019
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 291 (signature 13, LR=43.71):
  Theta for diagnosed: 0.066 ± 0.059
  Theta for others: 0.048
  Proportion difference: 0.018

Epoch 135
Loss: 142.7686

Monitoring signature responses:

Disease 248 (signature 2, LR=55.65):
  Theta for diagnosed: 0.038 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=54.86):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=52.05):
  Theta for diagnosed: 0.024 ± 0.016
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.51):
  Theta for diagnosed: 0.036 ± 0.019
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 291 (signature 13, LR=43.84):
  Theta for diagnosed: 0.066 ± 0.059
  Theta for others: 0.048
  Proportion difference: 0.018

Epoch 136
Loss: 142.7150

Monitoring signature responses:

Disease 248 (signature 2, LR=55.87):
  Theta for diagnosed: 0.038 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=55.26):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=52.38):
  Theta for diagnosed: 0.024 ± 0.016
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.64):
  Theta for diagnosed: 0.036 ± 0.019
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 291 (signature 13, LR=43.97):
  Theta for diagnosed: 0.067 ± 0.060
  Theta for others: 0.048
  Proportion difference: 0.019

Epoch 137
Loss: 142.6615

Monitoring signature responses:

Disease 248 (signature 2, LR=56.10):
  Theta for diagnosed: 0.038 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=55.65):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=52.71):
  Theta for diagnosed: 0.024 ± 0.016
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.77):
  Theta for diagnosed: 0.036 ± 0.019
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 237 (signature 2, LR=44.19):
  Theta for diagnosed: 0.038 ± 0.025
  Theta for others: 0.034
  Proportion difference: 0.004

Epoch 138
Loss: 142.6084

Monitoring signature responses:

Disease 248 (signature 2, LR=56.32):
  Theta for diagnosed: 0.039 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=56.05):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=53.04):
  Theta for diagnosed: 0.024 ± 0.016
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=49.91):
  Theta for diagnosed: 0.036 ± 0.020
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 237 (signature 2, LR=44.40):
  Theta for diagnosed: 0.038 ± 0.025
  Theta for others: 0.034
  Proportion difference: 0.005

Epoch 139
Loss: 142.5558

Monitoring signature responses:

Disease 248 (signature 2, LR=56.55):
  Theta for diagnosed: 0.039 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 232 (signature 15, LR=56.45):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=53.37):
  Theta for diagnosed: 0.024 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=50.04):
  Theta for diagnosed: 0.036 ± 0.020
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 237 (signature 2, LR=44.62):
  Theta for diagnosed: 0.038 ± 0.026
  Theta for others: 0.034
  Proportion difference: 0.005

Epoch 140
Loss: 142.5033

Monitoring signature responses:

Disease 232 (signature 15, LR=56.86):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=56.77):
  Theta for diagnosed: 0.039 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 169 (signature 19, LR=53.70):
  Theta for diagnosed: 0.024 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=50.17):
  Theta for diagnosed: 0.036 ± 0.020
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 237 (signature 2, LR=44.83):
  Theta for diagnosed: 0.038 ± 0.026
  Theta for others: 0.034
  Proportion difference: 0.005

Epoch 141
Loss: 142.4509

Monitoring signature responses:

Disease 232 (signature 15, LR=57.26):
  Theta for diagnosed: 0.015 ± 0.010
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=56.99):
  Theta for diagnosed: 0.039 ± 0.027
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 169 (signature 19, LR=54.04):
  Theta for diagnosed: 0.024 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=50.31):
  Theta for diagnosed: 0.036 ± 0.020
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 237 (signature 2, LR=45.05):
  Theta for diagnosed: 0.038 ± 0.026
  Theta for others: 0.034
  Proportion difference: 0.005

Epoch 142
Loss: 142.3988

Monitoring signature responses:

Disease 232 (signature 15, LR=57.67):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=57.21):
  Theta for diagnosed: 0.039 ± 0.027
  Theta for others: 0.033
  Proportion difference: 0.005

Disease 169 (signature 19, LR=54.37):
  Theta for diagnosed: 0.024 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=50.44):
  Theta for diagnosed: 0.036 ± 0.020
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=45.32):
  Theta for diagnosed: 0.024 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 143
Loss: 142.3471

Monitoring signature responses:

Disease 232 (signature 15, LR=58.08):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=57.43):
  Theta for diagnosed: 0.039 ± 0.027
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=54.70):
  Theta for diagnosed: 0.024 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.002

Disease 236 (signature 2, LR=50.57):
  Theta for diagnosed: 0.036 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=45.62):
  Theta for diagnosed: 0.025 ± 0.017
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 144
Loss: 142.2956

Monitoring signature responses:

Disease 232 (signature 15, LR=58.49):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=57.65):
  Theta for diagnosed: 0.039 ± 0.027
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=55.04):
  Theta for diagnosed: 0.024 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=50.70):
  Theta for diagnosed: 0.036 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=45.92):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 145
Loss: 142.2443

Monitoring signature responses:

Disease 232 (signature 15, LR=58.90):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=57.86):
  Theta for diagnosed: 0.039 ± 0.028
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=55.37):
  Theta for diagnosed: 0.024 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=50.83):
  Theta for diagnosed: 0.036 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=46.22):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 146
Loss: 142.1931

Monitoring signature responses:

Disease 232 (signature 15, LR=59.32):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=58.08):
  Theta for diagnosed: 0.039 ± 0.028
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=55.70):
  Theta for diagnosed: 0.024 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=50.97):
  Theta for diagnosed: 0.036 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=46.52):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 147
Loss: 142.1423

Monitoring signature responses:

Disease 232 (signature 15, LR=59.73):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=58.30):
  Theta for diagnosed: 0.039 ± 0.028
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=56.04):
  Theta for diagnosed: 0.024 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.10):
  Theta for diagnosed: 0.037 ± 0.021
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=46.83):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 148
Loss: 142.0918

Monitoring signature responses:

Disease 232 (signature 15, LR=60.15):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=58.51):
  Theta for diagnosed: 0.039 ± 0.029
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=56.37):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.23):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=47.14):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 149
Loss: 142.0414

Monitoring signature responses:

Disease 232 (signature 15, LR=60.57):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=58.72):
  Theta for diagnosed: 0.039 ± 0.029
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=56.71):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.36):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=47.44):
  Theta for diagnosed: 0.025 ± 0.018
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 150
Loss: 141.9913

Monitoring signature responses:

Disease 232 (signature 15, LR=60.99):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=58.93):
  Theta for diagnosed: 0.039 ± 0.029
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=57.04):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.50):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=47.75):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 151
Loss: 141.9414

Monitoring signature responses:

Disease 232 (signature 15, LR=61.41):
  Theta for diagnosed: 0.015 ± 0.011
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=59.15):
  Theta for diagnosed: 0.040 ± 0.030
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=57.37):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.63):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=48.06):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 152
Loss: 141.8918

Monitoring signature responses:

Disease 232 (signature 15, LR=61.84):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=59.36):
  Theta for diagnosed: 0.040 ± 0.030
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=57.71):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.76):
  Theta for diagnosed: 0.037 ± 0.022
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=48.38):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 153
Loss: 141.8423

Monitoring signature responses:

Disease 232 (signature 15, LR=62.27):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=59.57):
  Theta for diagnosed: 0.040 ± 0.030
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=58.04):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=51.89):
  Theta for diagnosed: 0.037 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=48.69):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 154
Loss: 141.7932

Monitoring signature responses:

Disease 232 (signature 15, LR=62.69):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=59.77):
  Theta for diagnosed: 0.040 ± 0.030
  Theta for others: 0.033
  Proportion difference: 0.006

Disease 169 (signature 19, LR=58.38):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.03):
  Theta for diagnosed: 0.037 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.003

Disease 162 (signature 19, LR=49.01):
  Theta for diagnosed: 0.025 ± 0.019
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 155
Loss: 141.7443

Monitoring signature responses:

Disease 232 (signature 15, LR=63.12):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=59.98):
  Theta for diagnosed: 0.040 ± 0.031
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=58.71):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.16):
  Theta for diagnosed: 0.037 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=49.33):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.003

Epoch 156
Loss: 141.6955

Monitoring signature responses:

Disease 232 (signature 15, LR=63.56):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=60.19):
  Theta for diagnosed: 0.040 ± 0.031
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=59.04):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.29):
  Theta for diagnosed: 0.037 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=49.65):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 157
Loss: 141.6470

Monitoring signature responses:

Disease 232 (signature 15, LR=63.99):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=60.39):
  Theta for diagnosed: 0.040 ± 0.031
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=59.37):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.42):
  Theta for diagnosed: 0.037 ± 0.023
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=49.97):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 158
Loss: 141.5987

Monitoring signature responses:

Disease 232 (signature 15, LR=64.42):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=60.60):
  Theta for diagnosed: 0.040 ± 0.032
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=59.71):
  Theta for diagnosed: 0.025 ± 0.021
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.56):
  Theta for diagnosed: 0.037 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=50.29):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 159
Loss: 141.5506

Monitoring signature responses:

Disease 232 (signature 15, LR=64.86):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=60.80):
  Theta for diagnosed: 0.040 ± 0.032
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=60.04):
  Theta for diagnosed: 0.025 ± 0.021
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.69):
  Theta for diagnosed: 0.037 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=50.61):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 160
Loss: 141.5027

Monitoring signature responses:

Disease 232 (signature 15, LR=65.30):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=61.00):
  Theta for diagnosed: 0.040 ± 0.032
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=60.37):
  Theta for diagnosed: 0.025 ± 0.021
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.83):
  Theta for diagnosed: 0.037 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=50.94):
  Theta for diagnosed: 0.025 ± 0.020
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 161
Loss: 141.4551

Monitoring signature responses:

Disease 232 (signature 15, LR=65.74):
  Theta for diagnosed: 0.015 ± 0.012
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=61.21):
  Theta for diagnosed: 0.040 ± 0.033
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=60.70):
  Theta for diagnosed: 0.025 ± 0.021
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=52.96):
  Theta for diagnosed: 0.037 ± 0.024
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 162 (signature 19, LR=51.27):
  Theta for diagnosed: 0.025 ± 0.021
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 162
Loss: 141.4076

Monitoring signature responses:

Disease 232 (signature 15, LR=66.18):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=61.41):
  Theta for diagnosed: 0.040 ± 0.033
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=61.03):
  Theta for diagnosed: 0.025 ± 0.021
  Theta for others: 0.022
  Proportion difference: 0.003

Disease 236 (signature 2, LR=53.10):
  Theta for diagnosed: 0.037 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 132 (signature 4, LR=51.69):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 163
Loss: 141.3604

Monitoring signature responses:

Disease 232 (signature 15, LR=66.62):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=61.61):
  Theta for diagnosed: 0.041 ± 0.033
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=61.36):
  Theta for diagnosed: 0.025 ± 0.022
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 236 (signature 2, LR=53.23):
  Theta for diagnosed: 0.037 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 132 (signature 4, LR=52.12):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 164
Loss: 141.3134

Monitoring signature responses:

Disease 232 (signature 15, LR=67.07):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 248 (signature 2, LR=61.81):
  Theta for diagnosed: 0.041 ± 0.033
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 169 (signature 19, LR=61.68):
  Theta for diagnosed: 0.025 ± 0.022
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 236 (signature 2, LR=53.37):
  Theta for diagnosed: 0.037 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 132 (signature 4, LR=52.56):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 165
Loss: 141.2666

Monitoring signature responses:

Disease 232 (signature 15, LR=67.51):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=62.01):
  Theta for diagnosed: 0.025 ± 0.022
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=62.01):
  Theta for diagnosed: 0.041 ± 0.034
  Theta for others: 0.033
  Proportion difference: 0.007

Disease 236 (signature 2, LR=53.50):
  Theta for diagnosed: 0.037 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 132 (signature 4, LR=53.00):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 166
Loss: 141.2200

Monitoring signature responses:

Disease 232 (signature 15, LR=67.96):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=62.34):
  Theta for diagnosed: 0.025 ± 0.022
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=62.21):
  Theta for diagnosed: 0.041 ± 0.034
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 236 (signature 2, LR=53.64):
  Theta for diagnosed: 0.037 ± 0.025
  Theta for others: 0.033
  Proportion difference: 0.004

Disease 132 (signature 4, LR=53.44):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 167
Loss: 141.1736

Monitoring signature responses:

Disease 232 (signature 15, LR=68.41):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=62.66):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=62.41):
  Theta for diagnosed: 0.041 ± 0.034
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=53.89):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 236 (signature 2, LR=53.78):
  Theta for diagnosed: 0.037 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.004

Epoch 168
Loss: 141.1274

Monitoring signature responses:

Disease 232 (signature 15, LR=68.86):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=62.99):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=62.60):
  Theta for diagnosed: 0.041 ± 0.035
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=54.34):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 236 (signature 2, LR=53.91):
  Theta for diagnosed: 0.037 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.004

Epoch 169
Loss: 141.0813

Monitoring signature responses:

Disease 232 (signature 15, LR=69.31):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=63.31):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=62.80):
  Theta for diagnosed: 0.041 ± 0.035
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=54.79):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 236 (signature 2, LR=54.05):
  Theta for diagnosed: 0.037 ± 0.026
  Theta for others: 0.033
  Proportion difference: 0.004

Epoch 170
Loss: 141.0355

Monitoring signature responses:

Disease 232 (signature 15, LR=69.77):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=63.63):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=63.00):
  Theta for diagnosed: 0.041 ± 0.035
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=55.25):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=54.27):
  Theta for diagnosed: 0.026 ± 0.022
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 171
Loss: 140.9899

Monitoring signature responses:

Disease 232 (signature 15, LR=70.22):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=63.96):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=63.19):
  Theta for diagnosed: 0.041 ± 0.036
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=55.72):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=54.61):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 172
Loss: 140.9445

Monitoring signature responses:

Disease 232 (signature 15, LR=70.68):
  Theta for diagnosed: 0.015 ± 0.013
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=64.28):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=63.39):
  Theta for diagnosed: 0.041 ± 0.036
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=56.18):
  Theta for diagnosed: 0.015 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=54.96):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 173
Loss: 140.8991

Monitoring signature responses:

Disease 232 (signature 15, LR=71.14):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=64.60):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=63.58):
  Theta for diagnosed: 0.041 ± 0.036
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=56.65):
  Theta for diagnosed: 0.015 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=55.30):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.004

Epoch 174
Loss: 140.8539

Monitoring signature responses:

Disease 232 (signature 15, LR=71.60):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=64.91):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=63.78):
  Theta for diagnosed: 0.042 ± 0.036
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=57.12):
  Theta for diagnosed: 0.015 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=55.64):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.005

Epoch 175
Loss: 140.8091

Monitoring signature responses:

Disease 232 (signature 15, LR=72.06):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=65.23):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=63.97):
  Theta for diagnosed: 0.042 ± 0.037
  Theta for others: 0.033
  Proportion difference: 0.008

Disease 132 (signature 4, LR=57.60):
  Theta for diagnosed: 0.015 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=55.99):
  Theta for diagnosed: 0.026 ± 0.023
  Theta for others: 0.022
  Proportion difference: 0.005

Epoch 176
Loss: 140.7643

Monitoring signature responses:

Disease 232 (signature 15, LR=72.52):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=65.55):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=64.17):
  Theta for diagnosed: 0.042 ± 0.037
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=58.08):
  Theta for diagnosed: 0.015 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=56.34):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 177
Loss: 140.7197

Monitoring signature responses:

Disease 232 (signature 15, LR=72.98):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=65.86):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=64.36):
  Theta for diagnosed: 0.042 ± 0.037
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=58.56):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=56.68):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 178
Loss: 140.6754

Monitoring signature responses:

Disease 232 (signature 15, LR=73.45):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=66.17):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.022
  Proportion difference: 0.004

Disease 248 (signature 2, LR=64.55):
  Theta for diagnosed: 0.042 ± 0.038
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=59.05):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=57.03):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 179
Loss: 140.6312

Monitoring signature responses:

Disease 232 (signature 15, LR=73.91):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=66.49):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=64.75):
  Theta for diagnosed: 0.042 ± 0.038
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=59.54):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=57.38):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 180
Loss: 140.5871

Monitoring signature responses:

Disease 232 (signature 15, LR=74.38):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=66.80):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=64.94):
  Theta for diagnosed: 0.042 ± 0.038
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=60.04):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=57.73):
  Theta for diagnosed: 0.026 ± 0.024
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 181
Loss: 140.5432

Monitoring signature responses:

Disease 232 (signature 15, LR=74.85):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=67.10):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=65.13):
  Theta for diagnosed: 0.042 ± 0.039
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=60.54):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=58.09):
  Theta for diagnosed: 0.026 ± 0.025
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 182
Loss: 140.4995

Monitoring signature responses:

Disease 232 (signature 15, LR=75.32):
  Theta for diagnosed: 0.015 ± 0.014
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=67.41):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=65.32):
  Theta for diagnosed: 0.042 ± 0.039
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=61.04):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 162 (signature 19, LR=58.44):
  Theta for diagnosed: 0.027 ± 0.025
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 183
Loss: 140.4559

Monitoring signature responses:

Disease 232 (signature 15, LR=75.79):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=67.72):
  Theta for diagnosed: 0.026 ± 0.026
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=65.52):
  Theta for diagnosed: 0.042 ± 0.039
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=61.54):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 162 (signature 19, LR=58.79):
  Theta for diagnosed: 0.027 ± 0.025
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 184
Loss: 140.4125

Monitoring signature responses:

Disease 232 (signature 15, LR=76.26):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=68.02):
  Theta for diagnosed: 0.027 ± 0.026
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=65.71):
  Theta for diagnosed: 0.042 ± 0.040
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=62.05):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 162 (signature 19, LR=59.15):
  Theta for diagnosed: 0.027 ± 0.025
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 185
Loss: 140.3694

Monitoring signature responses:

Disease 232 (signature 15, LR=76.74):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=68.32):
  Theta for diagnosed: 0.027 ± 0.027
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=65.90):
  Theta for diagnosed: 0.043 ± 0.040
  Theta for others: 0.033
  Proportion difference: 0.009

Disease 132 (signature 4, LR=62.56):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 162 (signature 19, LR=59.50):
  Theta for diagnosed: 0.027 ± 0.025
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 186
Loss: 140.3264

Monitoring signature responses:

Disease 232 (signature 15, LR=77.21):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=68.63):
  Theta for diagnosed: 0.027 ± 0.027
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=66.09):
  Theta for diagnosed: 0.043 ± 0.040
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=63.08):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 162 (signature 19, LR=59.86):
  Theta for diagnosed: 0.027 ± 0.026
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 187
Loss: 140.2833

Monitoring signature responses:

Disease 232 (signature 15, LR=77.69):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=68.92):
  Theta for diagnosed: 0.027 ± 0.027
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=66.28):
  Theta for diagnosed: 0.043 ± 0.040
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=63.60):
  Theta for diagnosed: 0.016 ± 0.028
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 162 (signature 19, LR=60.22):
  Theta for diagnosed: 0.027 ± 0.026
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 188
Loss: 140.2405

Monitoring signature responses:

Disease 232 (signature 15, LR=78.16):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=69.22):
  Theta for diagnosed: 0.027 ± 0.027
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=66.47):
  Theta for diagnosed: 0.043 ± 0.041
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=64.12):
  Theta for diagnosed: 0.016 ± 0.028
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 162 (signature 19, LR=60.57):
  Theta for diagnosed: 0.027 ± 0.026
  Theta for others: 0.021
  Proportion difference: 0.005

Epoch 189
Loss: 140.1979

Monitoring signature responses:

Disease 232 (signature 15, LR=78.64):
  Theta for diagnosed: 0.015 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=69.52):
  Theta for diagnosed: 0.027 ± 0.028
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=66.66):
  Theta for diagnosed: 0.043 ± 0.041
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=64.64):
  Theta for diagnosed: 0.016 ± 0.028
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 137 (signature 4, LR=60.95):
  Theta for diagnosed: 0.019 ± 0.038
  Theta for others: 0.012
  Proportion difference: 0.006

Epoch 190
Loss: 140.1556

Monitoring signature responses:

Disease 232 (signature 15, LR=79.12):
  Theta for diagnosed: 0.016 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.001

Disease 169 (signature 19, LR=69.81):
  Theta for diagnosed: 0.027 ± 0.028
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=66.85):
  Theta for diagnosed: 0.043 ± 0.041
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=65.17):
  Theta for diagnosed: 0.016 ± 0.028
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 137 (signature 4, LR=61.36):
  Theta for diagnosed: 0.019 ± 0.038
  Theta for others: 0.012
  Proportion difference: 0.006

Epoch 191
Loss: 140.1134

Monitoring signature responses:

Disease 232 (signature 15, LR=79.60):
  Theta for diagnosed: 0.016 ± 0.015
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=70.11):
  Theta for diagnosed: 0.027 ± 0.028
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=67.04):
  Theta for diagnosed: 0.043 ± 0.042
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=65.70):
  Theta for diagnosed: 0.016 ± 0.029
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 137 (signature 4, LR=61.78):
  Theta for diagnosed: 0.019 ± 0.039
  Theta for others: 0.012
  Proportion difference: 0.006

Epoch 192
Loss: 140.0712

Monitoring signature responses:

Disease 232 (signature 15, LR=80.09):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=70.40):
  Theta for diagnosed: 0.027 ± 0.028
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=67.23):
  Theta for diagnosed: 0.043 ± 0.042
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=66.24):
  Theta for diagnosed: 0.016 ± 0.029
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 137 (signature 4, LR=62.20):
  Theta for diagnosed: 0.019 ± 0.039
  Theta for others: 0.012
  Proportion difference: 0.006

Epoch 193
Loss: 140.0293

Monitoring signature responses:

Disease 232 (signature 15, LR=80.57):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=70.68):
  Theta for diagnosed: 0.027 ± 0.028
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=67.43):
  Theta for diagnosed: 0.043 ± 0.042
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=66.78):
  Theta for diagnosed: 0.016 ± 0.029
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 137 (signature 4, LR=62.63):
  Theta for diagnosed: 0.019 ± 0.039
  Theta for others: 0.012
  Proportion difference: 0.007

Epoch 194
Loss: 139.9877

Monitoring signature responses:

Disease 232 (signature 15, LR=81.05):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=70.97):
  Theta for diagnosed: 0.027 ± 0.029
  Theta for others: 0.022
  Proportion difference: 0.005

Disease 248 (signature 2, LR=67.62):
  Theta for diagnosed: 0.043 ± 0.043
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 132 (signature 4, LR=67.32):
  Theta for diagnosed: 0.016 ± 0.029
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 137 (signature 4, LR=63.05):
  Theta for diagnosed: 0.019 ± 0.039
  Theta for others: 0.012
  Proportion difference: 0.007

Epoch 195
Loss: 139.9461

Monitoring signature responses:

Disease 232 (signature 15, LR=81.54):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=71.26):
  Theta for diagnosed: 0.027 ± 0.029
  Theta for others: 0.022
  Proportion difference: 0.006

Disease 132 (signature 4, LR=67.87):
  Theta for diagnosed: 0.016 ± 0.030
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 248 (signature 2, LR=67.81):
  Theta for diagnosed: 0.044 ± 0.043
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 137 (signature 4, LR=63.48):
  Theta for diagnosed: 0.019 ± 0.040
  Theta for others: 0.012
  Proportion difference: 0.007

Epoch 196
Loss: 139.9045

Monitoring signature responses:

Disease 232 (signature 15, LR=82.03):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=71.54):
  Theta for diagnosed: 0.027 ± 0.029
  Theta for others: 0.022
  Proportion difference: 0.006

Disease 132 (signature 4, LR=68.41):
  Theta for diagnosed: 0.016 ± 0.030
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 248 (signature 2, LR=68.00):
  Theta for diagnosed: 0.044 ± 0.043
  Theta for others: 0.033
  Proportion difference: 0.010

Disease 137 (signature 4, LR=63.92):
  Theta for diagnosed: 0.019 ± 0.040
  Theta for others: 0.012
  Proportion difference: 0.007

Epoch 197
Loss: 139.8631

Monitoring signature responses:

Disease 232 (signature 15, LR=82.51):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=71.82):
  Theta for diagnosed: 0.027 ± 0.029
  Theta for others: 0.022
  Proportion difference: 0.006

Disease 132 (signature 4, LR=68.97):
  Theta for diagnosed: 0.017 ± 0.030
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 248 (signature 2, LR=68.19):
  Theta for diagnosed: 0.044 ± 0.043
  Theta for others: 0.033
  Proportion difference: 0.011

Disease 137 (signature 4, LR=64.35):
  Theta for diagnosed: 0.019 ± 0.040
  Theta for others: 0.012
  Proportion difference: 0.007

Epoch 198
Loss: 139.8220

Monitoring signature responses:

Disease 232 (signature 15, LR=83.00):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=72.10):
  Theta for diagnosed: 0.027 ± 0.030
  Theta for others: 0.022
  Proportion difference: 0.006

Disease 132 (signature 4, LR=69.52):
  Theta for diagnosed: 0.017 ± 0.030
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 248 (signature 2, LR=68.38):
  Theta for diagnosed: 0.044 ± 0.044
  Theta for others: 0.033
  Proportion difference: 0.011

Disease 137 (signature 4, LR=64.79):
  Theta for diagnosed: 0.020 ± 0.041
  Theta for others: 0.012
  Proportion difference: 0.007

Epoch 199
Loss: 139.7810

Monitoring signature responses:

Disease 232 (signature 15, LR=83.49):
  Theta for diagnosed: 0.016 ± 0.016
  Theta for others: 0.014
  Proportion difference: 0.002

Disease 169 (signature 19, LR=72.38):
  Theta for diagnosed: 0.027 ± 0.030
  Theta for others: 0.022
  Proportion difference: 0.006

Disease 132 (signature 4, LR=70.08):
  Theta for diagnosed: 0.017 ± 0.031
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 248 (signature 2, LR=68.57):
  Theta for diagnosed: 0.044 ± 0.044
  Theta for others: 0.033
  Proportion difference: 0.011

Disease 137 (signature 4, LR=65.23):
  Theta for diagnosed: 0.020 ± 0.041
  Theta for others: 0.012
  Proportion difference: 0.007
✓ Saved AOU initialized model to: aou_model_initialized.pt