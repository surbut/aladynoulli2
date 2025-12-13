============================================================
MGB: Initializing model...
============================================================
/var/folders/fl/ng5crz0x0fnb6c6x8dk7tfth0000gn/T/ipykernel_50600/510016609.py:9: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  mgb_checkpoint_old = torch.load('/Users/sarahurbut/Dropbox-Personal/model_with_kappa_bigam_MGB.pt', map_location='cpu')
MGB: K=20 signatures
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
Cluster 0: 6 diseases
Cluster 1: 28 diseases
Cluster 2: 24 diseases
Cluster 3: 11 diseases
Cluster 4: 24 diseases
Cluster 5: 20 diseases
Cluster 6: 15 diseases
Cluster 7: 16 diseases
Cluster 8: 19 diseases
Cluster 9: 13 diseases
Cluster 10: 7 diseases
Cluster 11: 7 diseases
Cluster 12: 68 diseases
Cluster 13: 7 diseases
Cluster 14: 13 diseases
Cluster 15: 12 diseases
Cluster 16: 16 diseases
Cluster 17: 8 diseases
Cluster 18: 12 diseases
Cluster 19: 20 diseases

Calculating gamma for k=0:
Number of diseases in cluster: 6
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.7714, -0.7714, -0.7714, -0.7714, -0.7714])
Base value centered mean: -1.1301173117317376e-06
Gamma init for k=0 (first 5): tensor([ 0.0043,  0.0104, -0.0070,  0.0494,  0.0009])

Calculating gamma for k=1:
Number of diseases in cluster: 28
Base value (first 5): tensor([-13.8155, -12.7544, -13.8155, -13.4618, -13.8155])
Base value centered (first 5): tensor([-1.2812, -0.2201, -1.2812, -0.9275, -1.2812])
Base value centered mean: -1.3286156672620564e-06
Gamma init for k=1 (first 5): tensor([-0.0143,  0.0273, -0.0095,  0.0241, -0.0131])

Calculating gamma for k=2:
Number of diseases in cluster: 24
Base value (first 5): tensor([-13.8155, -11.7523, -13.8155, -10.1017, -11.7523])
Base value centered (first 5): tensor([-1.2360,  0.8273, -1.2360,  2.4779,  0.8273])
Base value centered mean: -7.869356863920984e-07
Gamma init for k=2 (first 5): tensor([-0.0032,  0.0213, -0.0101, -0.0018,  0.0155])

Calculating gamma for k=3:
Number of diseases in cluster: 11
Base value (first 5): tensor([-13.8155, -12.9152, -13.8155, -12.9152, -12.9152])
Base value centered (first 5): tensor([-0.6787,  0.2216, -0.6787,  0.2216,  0.2216])
Base value centered mean: -6.845987741144199e-07
Gamma init for k=3 (first 5): tensor([ 0.0035,  0.0213, -0.0016, -0.0081, -0.0042])

Calculating gamma for k=4:
Number of diseases in cluster: 24
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.6051, -0.6051, -0.6051, -0.6051, -0.6051])
Base value centered mean: -1.4362458387040533e-06
Gamma init for k=4 (first 5): tensor([-0.0050,  0.0126,  0.0104,  0.0023,  0.0076])

Calculating gamma for k=5:
Number of diseases in cluster: 20
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.6844, -0.6844, -0.6844, -0.6844, -0.6844])
Base value centered mean: -5.399155043050996e-07
Gamma init for k=5 (first 5): tensor([0.0030, 0.0173, 0.0048, 0.0028, 0.0403])

Calculating gamma for k=6:
Number of diseases in cluster: 15
Base value (first 5): tensor([-13.8155, -13.1553, -13.1553, -13.1553, -13.1553])
Base value centered (first 5): tensor([-1.3616, -0.7014, -0.7014, -0.7014, -0.7014])
Base value centered mean: -9.492632671026513e-07
Gamma init for k=6 (first 5): tensor([-0.0052,  0.0253, -0.0008,  0.0025, -0.0036])

Calculating gamma for k=7:
Number of diseases in cluster: 16
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -12.5776, -13.8155])
Base value centered (first 5): tensor([-0.3747, -0.3747, -0.3747,  0.8633, -0.3747])
Base value centered mean: -7.957578418427147e-07
Gamma init for k=7 (first 5): tensor([ 0.0013,  0.0091, -0.0016, -0.0023, -0.0096])

Calculating gamma for k=8:
Number of diseases in cluster: 19
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -12.7730])
Base value centered (first 5): tensor([-0.7036, -0.7036, -0.7036, -0.7036,  0.3389])
Base value centered mean: -3.5112154250782623e-07
Gamma init for k=8 (first 5): tensor([-0.0031,  0.0140, -0.0068,  0.0039,  0.0094])

Calculating gamma for k=9:
Number of diseases in cluster: 13
Base value (first 5): tensor([-13.0537, -10.7683, -13.8155,  -9.2447, -13.8155])
Base value centered (first 5): tensor([-0.3931,  1.8923, -1.1549,  3.4159, -1.1549])
Base value centered mean: -5.681463903783879e-07
Gamma init for k=9 (first 5): tensor([-0.0037,  0.0158, -0.0254,  0.0131, -0.0033])

Calculating gamma for k=10:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.4089, -0.4089, -0.4089, -0.4089, -0.4089])
Base value centered mean: -1.2051056046402664e-06
Gamma init for k=10 (first 5): tensor([-0.0047,  0.0179,  0.0159, -0.0035, -0.0040])

Calculating gamma for k=11:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2849, -0.2849, -0.2849, -0.2849, -0.2849])
Base value centered mean: -2.8795493562938645e-06
Gamma init for k=11 (first 5): tensor([-2.2637e-03, -1.1999e-02, -9.5493e-05, -1.2029e-02,  1.0361e-02])

Calculating gamma for k=12:
Number of diseases in cluster: 68
Base value (first 5): tensor([-13.8155, -13.6699, -13.8155, -13.6699, -13.8155])
Base value centered (first 5): tensor([-0.2191, -0.0734, -0.2191, -0.0734, -0.2191])
Base value centered mean: 9.179445896734251e-07
Gamma init for k=12 (first 5): tensor([-0.0006,  0.0037,  0.0004, -0.0008, -0.0010])

Calculating gamma for k=13:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2384, -0.2384, -0.2384, -0.2384, -0.2384])
Base value centered mean: -2.2937588255445007e-06
Gamma init for k=13 (first 5): tensor([-0.0045,  0.0135,  0.0033,  0.0050,  0.0014])

Calculating gamma for k=14:
Number of diseases in cluster: 13
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.6261, -0.6261, -0.6261, -0.6261, -0.6261])
Base value centered mean: 2.5937117698049406e-07
Gamma init for k=14 (first 5): tensor([0.0008, 0.0170, 0.0058, 0.0021, 0.0185])

Calculating gamma for k=15:
Number of diseases in cluster: 12
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-1.3508, -1.3508, -1.3508, -1.3508, -1.3508])
Base value centered mean: 8.504551942678518e-07
Gamma init for k=15 (first 5): tensor([-0.0062,  0.0417,  0.0011, -0.0039,  0.0200])

Calculating gamma for k=16:
Number of diseases in cluster: 16
Base value (first 5): tensor([-13.8155, -13.8155, -10.7207,  -9.4827, -13.8155])
Base value centered (first 5): tensor([-0.4649, -0.4649,  2.6299,  3.8678, -0.4649])
Base value centered mean: -1.7370811065120506e-06
Gamma init for k=16 (first 5): tensor([-0.0037,  0.0145, -0.0071, -0.0041, -0.0005])

Calculating gamma for k=17:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -12.5776, -12.5776])
Base value centered (first 5): tensor([-0.5247, -0.5247, -0.5247,  0.7132,  0.7132])
Base value centered mean: -7.498826448681939e-07
Gamma init for k=17 (first 5): tensor([ 0.0006,  0.0083, -0.0014,  0.0046, -0.0021])

Calculating gamma for k=18:
Number of diseases in cluster: 12
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2925, -0.2925, -0.2925, -0.2925, -0.2925])
Base value centered mean: -7.260628649419232e-07
Gamma init for k=18 (first 5): tensor([ 0.0081, -0.0050,  0.0013,  0.0012,  0.0003])

Calculating gamma for k=19:
Number of diseases in cluster: 20
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2563, -0.2563, -0.2563, -0.2563, -0.2563])
Base value centered mean: -9.964617220248329e-07
Gamma init for k=19 (first 5): tensor([ 0.0053,  0.0035,  0.0019, -0.0023,  0.0023])
Initializing with 20 disease states + 1 healthy state
Initialization complete!

Calculating gamma for k=0:
Number of diseases in cluster: 6
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.7714, -0.7714, -0.7714, -0.7714, -0.7714])
Base value centered mean: -1.1301173117317376e-06
Gamma init for k=0 (first 5): tensor([ 0.0043,  0.0104, -0.0070,  0.0494,  0.0009])

Calculating gamma for k=1:
Number of diseases in cluster: 28
Base value (first 5): tensor([-13.8155, -12.7544, -13.8155, -13.4618, -13.8155])
Base value centered (first 5): tensor([-1.2812, -0.2201, -1.2812, -0.9275, -1.2812])
Base value centered mean: -1.3286156672620564e-06
Gamma init for k=1 (first 5): tensor([-0.0143,  0.0273, -0.0095,  0.0241, -0.0131])

Calculating gamma for k=2:
Number of diseases in cluster: 24
Base value (first 5): tensor([-13.8155, -11.7523, -13.8155, -10.1017, -11.7523])
Base value centered (first 5): tensor([-1.2360,  0.8273, -1.2360,  2.4779,  0.8273])
Base value centered mean: -7.869356863920984e-07
Gamma init for k=2 (first 5): tensor([-0.0032,  0.0213, -0.0101, -0.0018,  0.0155])

Calculating gamma for k=3:
Number of diseases in cluster: 11
Base value (first 5): tensor([-13.8155, -12.9152, -13.8155, -12.9152, -12.9152])
Base value centered (first 5): tensor([-0.6787,  0.2216, -0.6787,  0.2216,  0.2216])
Base value centered mean: -6.845987741144199e-07
Gamma init for k=3 (first 5): tensor([ 0.0035,  0.0213, -0.0016, -0.0081, -0.0042])

Calculating gamma for k=4:
Number of diseases in cluster: 24
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.6051, -0.6051, -0.6051, -0.6051, -0.6051])
Base value centered mean: -1.4362458387040533e-06
Gamma init for k=4 (first 5): tensor([-0.0050,  0.0126,  0.0104,  0.0023,  0.0076])

Calculating gamma for k=5:
Number of diseases in cluster: 20
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.6844, -0.6844, -0.6844, -0.6844, -0.6844])
Base value centered mean: -5.399155043050996e-07
Gamma init for k=5 (first 5): tensor([0.0030, 0.0173, 0.0048, 0.0028, 0.0403])

Calculating gamma for k=6:
Number of diseases in cluster: 15
Base value (first 5): tensor([-13.8155, -13.1553, -13.1553, -13.1553, -13.1553])
Base value centered (first 5): tensor([-1.3616, -0.7014, -0.7014, -0.7014, -0.7014])
Base value centered mean: -9.492632671026513e-07
Gamma init for k=6 (first 5): tensor([-0.0052,  0.0253, -0.0008,  0.0025, -0.0036])

Calculating gamma for k=7:
Number of diseases in cluster: 16
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -12.5776, -13.8155])
Base value centered (first 5): tensor([-0.3747, -0.3747, -0.3747,  0.8633, -0.3747])
Base value centered mean: -7.957578418427147e-07
Gamma init for k=7 (first 5): tensor([ 0.0013,  0.0091, -0.0016, -0.0023, -0.0096])

Calculating gamma for k=8:
Number of diseases in cluster: 19
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -12.7730])
Base value centered (first 5): tensor([-0.7036, -0.7036, -0.7036, -0.7036,  0.3389])
Base value centered mean: -3.5112154250782623e-07
Gamma init for k=8 (first 5): tensor([-0.0031,  0.0140, -0.0068,  0.0039,  0.0094])

Calculating gamma for k=9:
Number of diseases in cluster: 13
Base value (first 5): tensor([-13.0537, -10.7683, -13.8155,  -9.2447, -13.8155])
Base value centered (first 5): tensor([-0.3931,  1.8923, -1.1549,  3.4159, -1.1549])
Base value centered mean: -5.681463903783879e-07
Gamma init for k=9 (first 5): tensor([-0.0037,  0.0158, -0.0254,  0.0131, -0.0033])

Calculating gamma for k=10:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.4089, -0.4089, -0.4089, -0.4089, -0.4089])
Base value centered mean: -1.2051056046402664e-06
Gamma init for k=10 (first 5): tensor([-0.0047,  0.0179,  0.0159, -0.0035, -0.0040])

Calculating gamma for k=11:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2849, -0.2849, -0.2849, -0.2849, -0.2849])
Base value centered mean: -2.8795493562938645e-06
Gamma init for k=11 (first 5): tensor([-2.2637e-03, -1.1999e-02, -9.5493e-05, -1.2029e-02,  1.0361e-02])

Calculating gamma for k=12:
Number of diseases in cluster: 68
Base value (first 5): tensor([-13.8155, -13.6699, -13.8155, -13.6699, -13.8155])
Base value centered (first 5): tensor([-0.2191, -0.0734, -0.2191, -0.0734, -0.2191])
Base value centered mean: 9.179445896734251e-07
Gamma init for k=12 (first 5): tensor([-0.0006,  0.0037,  0.0004, -0.0008, -0.0010])

Calculating gamma for k=13:
Number of diseases in cluster: 7
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2384, -0.2384, -0.2384, -0.2384, -0.2384])
Base value centered mean: -2.2937588255445007e-06
Gamma init for k=13 (first 5): tensor([-0.0045,  0.0135,  0.0033,  0.0050,  0.0014])

Calculating gamma for k=14:
Number of diseases in cluster: 13
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.6261, -0.6261, -0.6261, -0.6261, -0.6261])
Base value centered mean: 2.5937117698049406e-07
Gamma init for k=14 (first 5): tensor([0.0008, 0.0170, 0.0058, 0.0021, 0.0185])

Calculating gamma for k=15:
Number of diseases in cluster: 12
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-1.3508, -1.3508, -1.3508, -1.3508, -1.3508])
Base value centered mean: 8.504551942678518e-07
Gamma init for k=15 (first 5): tensor([-0.0062,  0.0417,  0.0011, -0.0039,  0.0200])

Calculating gamma for k=16:
Number of diseases in cluster: 16
Base value (first 5): tensor([-13.8155, -13.8155, -10.7207,  -9.4827, -13.8155])
Base value centered (first 5): tensor([-0.4649, -0.4649,  2.6299,  3.8678, -0.4649])
Base value centered mean: -1.7370811065120506e-06
Gamma init for k=16 (first 5): tensor([-0.0037,  0.0145, -0.0071, -0.0041, -0.0005])

Calculating gamma for k=17:
Number of diseases in cluster: 8
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -12.5776, -12.5776])
Base value centered (first 5): tensor([-0.5247, -0.5247, -0.5247,  0.7132,  0.7132])
Base value centered mean: -7.498826448681939e-07
Gamma init for k=17 (first 5): tensor([ 0.0006,  0.0083, -0.0014,  0.0046, -0.0021])

Calculating gamma for k=18:
Number of diseases in cluster: 12
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2925, -0.2925, -0.2925, -0.2925, -0.2925])
Base value centered mean: -7.260628649419232e-07
Gamma init for k=18 (first 5): tensor([ 0.0081, -0.0050,  0.0013,  0.0012,  0.0003])

Calculating gamma for k=19:
Number of diseases in cluster: 20
Base value (first 5): tensor([-13.8155, -13.8155, -13.8155, -13.8155, -13.8155])
Base value centered (first 5): tensor([-0.2563, -0.2563, -0.2563, -0.2563, -0.2563])
Base value centered mean: -9.964617220248329e-07
Gamma init for k=19 (first 5): tensor([ 0.0053,  0.0035,  0.0019, -0.0023,  0.0023])
Initializing with 20 disease states + 1 healthy state
Initialization complete!
✓ Clusters match: True
/Users/sarahurbut/aladynoulli2/pyScripts_forPublish/clust_huge_amp_vectorized.py:238: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  event_times_tensor = torch.tensor(event_times, dtype=torch.long)

Epoch 0
Loss: 151.1897

Monitoring signature responses:

Disease 216 (signature 13, LR=27.40):
  Theta for diagnosed: 0.007 ± 0.001
  Theta for others: 0.007
  Proportion difference: 0.000

Disease 192 (signature 18, LR=27.34):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 55 (signature 4, LR=27.13):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.000

Disease 238 (signature 3, LR=26.73):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 107 (signature 5, LR=26.70):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.001

Epoch 1
Loss: 763.6578

Monitoring signature responses:

Disease 55 (signature 4, LR=27.38):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.000

Disease 216 (signature 13, LR=27.36):
  Theta for diagnosed: 0.007 ± 0.001
  Theta for others: 0.007
  Proportion difference: 0.000

Disease 192 (signature 18, LR=27.34):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 107 (signature 5, LR=26.93):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.001

Disease 238 (signature 3, LR=26.71):
  Theta for diagnosed: 0.029 ± 0.006
  Theta for others: 0.029
  Proportion difference: 0.000

Epoch 2
Loss: 201.3125

Monitoring signature responses:

Disease 55 (signature 4, LR=27.64):
  Theta for diagnosed: 0.062 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.000

Disease 192 (signature 18, LR=27.32):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 216 (signature 13, LR=27.31):
  Theta for diagnosed: 0.007 ± 0.001
  Theta for others: 0.007
  Proportion difference: 0.000

Disease 107 (signature 5, LR=27.16):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.001

Disease 325 (signature 2, LR=26.78):
  Theta for diagnosed: 0.133 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.001

Epoch 3
Loss: 294.4911

Monitoring signature responses:

Disease 55 (signature 4, LR=27.88):
  Theta for diagnosed: 0.062 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.000

Disease 107 (signature 5, LR=27.39):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.001

Disease 192 (signature 18, LR=27.33):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 216 (signature 13, LR=27.30):
  Theta for diagnosed: 0.007 ± 0.001
  Theta for others: 0.007
  Proportion difference: -0.000

Disease 325 (signature 2, LR=26.95):
  Theta for diagnosed: 0.134 ± 0.022
  Theta for others: 0.133
  Proportion difference: 0.001

Epoch 4
Loss: 488.1272

Monitoring signature responses:

Disease 55 (signature 4, LR=28.12):
  Theta for diagnosed: 0.062 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.000

Disease 107 (signature 5, LR=27.61):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.001

Disease 192 (signature 18, LR=27.36):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 216 (signature 13, LR=27.31):
  Theta for diagnosed: 0.007 ± 0.001
  Theta for others: 0.007
  Proportion difference: 0.000

Disease 325 (signature 2, LR=27.11):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.001

Epoch 5
Loss: 396.1184

Monitoring signature responses:

Disease 55 (signature 4, LR=28.35):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.000

Disease 107 (signature 5, LR=27.82):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.001

Disease 192 (signature 18, LR=27.41):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 216 (signature 13, LR=27.33):
  Theta for diagnosed: 0.007 ± 0.001
  Theta for others: 0.007
  Proportion difference: 0.000

Disease 115 (signature 5, LR=27.32):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.003

Epoch 6
Loss: 225.5216

Monitoring signature responses:

Disease 55 (signature 4, LR=28.58):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=28.03):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.001

Disease 115 (signature 5, LR=27.54):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 192 (signature 18, LR=27.47):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 325 (signature 2, LR=27.42):
  Theta for diagnosed: 0.134 ± 0.020
  Theta for others: 0.133
  Proportion difference: 0.001

Epoch 7
Loss: 166.1804

Monitoring signature responses:

Disease 55 (signature 4, LR=28.81):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=28.24):
  Theta for diagnosed: 0.053 ± 0.023
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=27.77):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 325 (signature 2, LR=27.56):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.132
  Proportion difference: 0.001

Disease 192 (signature 18, LR=27.53):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 8
Loss: 232.2523

Monitoring signature responses:

Disease 55 (signature 4, LR=29.05):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=28.45):
  Theta for diagnosed: 0.053 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=28.00):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=27.69):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.132
  Proportion difference: 0.002

Disease 192 (signature 18, LR=27.58):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 9
Loss: 302.5570

Monitoring signature responses:

Disease 55 (signature 4, LR=29.28):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=28.66):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=28.23):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=27.82):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.132
  Proportion difference: 0.002

Disease 192 (signature 18, LR=27.62):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 10
Loss: 288.7993

Monitoring signature responses:

Disease 55 (signature 4, LR=29.52):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=28.86):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=28.46):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=27.92):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Disease 192 (signature 18, LR=27.65):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 11
Loss: 217.5327

Monitoring signature responses:

Disease 55 (signature 4, LR=29.76):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=29.07):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=28.69):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=28.00):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Disease 192 (signature 18, LR=27.69):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 12
Loss: 161.9603

Monitoring signature responses:

Disease 55 (signature 4, LR=29.99):
  Theta for diagnosed: 0.063 ± 0.010
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=29.27):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=28.91):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=28.06):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Disease 192 (signature 18, LR=27.72):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 13
Loss: 162.8460

Monitoring signature responses:

Disease 55 (signature 4, LR=30.22):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=29.46):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=29.13):
  Theta for diagnosed: 0.055 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=28.10):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Disease 192 (signature 18, LR=27.76):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 14
Loss: 200.7433

Monitoring signature responses:

Disease 55 (signature 4, LR=30.44):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=29.64):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=29.34):
  Theta for diagnosed: 0.055 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=28.11):
  Theta for diagnosed: 0.134 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Disease 282 (signature 4, LR=27.80):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Epoch 15
Loss: 225.8946

Monitoring signature responses:

Disease 55 (signature 4, LR=30.65):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=29.81):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=29.54):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 325 (signature 2, LR=28.10):
  Theta for diagnosed: 0.135 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Disease 282 (signature 4, LR=27.97):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Epoch 16
Loss: 212.4059

Monitoring signature responses:

Disease 55 (signature 4, LR=30.85):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=29.97):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=29.73):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=28.13):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 325 (signature 2, LR=28.06):
  Theta for diagnosed: 0.135 ± 0.021
  Theta for others: 0.133
  Proportion difference: 0.002

Epoch 17
Loss: 176.7088

Monitoring signature responses:

Disease 55 (signature 4, LR=31.04):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.12):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=29.92):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=28.28):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 325 (signature 2, LR=28.00):
  Theta for diagnosed: 0.135 ± 0.021
  Theta for others: 0.132
  Proportion difference: 0.002

Epoch 18
Loss: 151.8980

Monitoring signature responses:

Disease 55 (signature 4, LR=31.23):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.26):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=30.09):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=28.44):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 192 (signature 18, LR=27.99):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 19
Loss: 154.7126

Monitoring signature responses:

Disease 55 (signature 4, LR=31.41):
  Theta for diagnosed: 0.063 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.39):
  Theta for diagnosed: 0.054 ± 0.024
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=30.26):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=28.59):
  Theta for diagnosed: 0.064 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 232 (signature 18, LR=28.06):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 20
Loss: 173.6564

Monitoring signature responses:

Disease 55 (signature 4, LR=31.58):
  Theta for diagnosed: 0.063 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.52):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=30.42):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=28.73):
  Theta for diagnosed: 0.064 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 330 (signature 3, LR=28.22):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 21
Loss: 184.6888

Monitoring signature responses:

Disease 55 (signature 4, LR=31.75):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.63):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=30.58):
  Theta for diagnosed: 0.056 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=28.88):
  Theta for diagnosed: 0.064 ± 0.011
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 330 (signature 3, LR=28.43):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 22
Loss: 176.3878

Monitoring signature responses:

Disease 55 (signature 4, LR=31.90):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.73):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 115 (signature 5, LR=30.72):
  Theta for diagnosed: 0.056 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 282 (signature 4, LR=29.01):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 330 (signature 3, LR=28.63):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 23
Loss: 158.0980

Monitoring signature responses:

Disease 55 (signature 4, LR=32.05):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.001

Disease 115 (signature 5, LR=30.85):
  Theta for diagnosed: 0.056 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.004

Disease 107 (signature 5, LR=30.82):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 282 (signature 4, LR=29.15):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 330 (signature 3, LR=28.83):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 24
Loss: 146.8411

Monitoring signature responses:

Disease 55 (signature 4, LR=32.19):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=30.97):
  Theta for diagnosed: 0.056 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=30.90):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 282 (signature 4, LR=29.27):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 330 (signature 3, LR=29.03):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 25
Loss: 149.9390

Monitoring signature responses:

Disease 55 (signature 4, LR=32.31):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.08):
  Theta for diagnosed: 0.056 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=30.96):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.002

Disease 282 (signature 4, LR=29.38):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 330 (signature 3, LR=29.23):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 26
Loss: 159.7568

Monitoring signature responses:

Disease 55 (signature 4, LR=32.42):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.17):
  Theta for diagnosed: 0.056 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.01):
  Theta for diagnosed: 0.054 ± 0.025
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 282 (signature 4, LR=29.49):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 330 (signature 3, LR=29.42):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 27
Loss: 163.6684

Monitoring signature responses:

Disease 55 (signature 4, LR=32.51):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.25):
  Theta for diagnosed: 0.057 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.04):
  Theta for diagnosed: 0.055 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=29.61):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 282 (signature 4, LR=29.60):
  Theta for diagnosed: 0.064 ± 0.012
  Theta for others: 0.062
  Proportion difference: 0.002

Epoch 28
Loss: 157.3876

Monitoring signature responses:

Disease 55 (signature 4, LR=32.60):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.31):
  Theta for diagnosed: 0.057 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.07):
  Theta for diagnosed: 0.055 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=29.80):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 282 (signature 4, LR=29.69):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Epoch 29
Loss: 147.6385

Monitoring signature responses:

Disease 55 (signature 4, LR=32.67):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.37):
  Theta for diagnosed: 0.057 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.08):
  Theta for diagnosed: 0.055 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=29.99):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 282 (signature 4, LR=29.78):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Epoch 30
Loss: 143.4690

Monitoring signature responses:

Disease 55 (signature 4, LR=32.74):
  Theta for diagnosed: 0.064 ± 0.014
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.42):
  Theta for diagnosed: 0.057 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.08):
  Theta for diagnosed: 0.055 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=30.17):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 282 (signature 4, LR=29.86):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Epoch 31
Loss: 146.8314

Monitoring signature responses:

Disease 55 (signature 4, LR=32.79):
  Theta for diagnosed: 0.064 ± 0.014
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.45):
  Theta for diagnosed: 0.057 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.07):
  Theta for diagnosed: 0.055 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=30.36):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 282 (signature 4, LR=29.94):
  Theta for diagnosed: 0.064 ± 0.013
  Theta for others: 0.062
  Proportion difference: 0.002

Epoch 32
Loss: 151.7875

Monitoring signature responses:

Disease 55 (signature 4, LR=32.84):
  Theta for diagnosed: 0.064 ± 0.014
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.48):
  Theta for diagnosed: 0.057 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.06):
  Theta for diagnosed: 0.055 ± 0.026
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=30.55):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=30.06):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Epoch 33
Loss: 151.9212

Monitoring signature responses:

Disease 55 (signature 4, LR=32.88):
  Theta for diagnosed: 0.064 ± 0.014
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.50):
  Theta for diagnosed: 0.057 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.03):
  Theta for diagnosed: 0.055 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=30.73):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=30.23):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Epoch 34
Loss: 147.0641

Monitoring signature responses:

Disease 55 (signature 4, LR=32.91):
  Theta for diagnosed: 0.065 ± 0.014
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.50):
  Theta for diagnosed: 0.057 ± 0.028
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 107 (signature 5, LR=31.00):
  Theta for diagnosed: 0.055 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 330 (signature 3, LR=30.92):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=30.42):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 35
Loss: 142.3517

Monitoring signature responses:

Disease 55 (signature 4, LR=32.93):
  Theta for diagnosed: 0.065 ± 0.015
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.50):
  Theta for diagnosed: 0.057 ± 0.028
  Theta for others: 0.052
  Proportion difference: 0.005

Disease 330 (signature 3, LR=31.10):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.96):
  Theta for diagnosed: 0.055 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 247 (signature 18, LR=30.62):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 36
Loss: 141.9096

Monitoring signature responses:

Disease 55 (signature 4, LR=32.94):
  Theta for diagnosed: 0.065 ± 0.015
  Theta for others: 0.062
  Proportion difference: 0.002

Disease 115 (signature 5, LR=31.49):
  Theta for diagnosed: 0.057 ± 0.028
  Theta for others: 0.052
  Proportion difference: 0.006

Disease 330 (signature 3, LR=31.28):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 107 (signature 5, LR=30.91):
  Theta for diagnosed: 0.055 ± 0.027
  Theta for others: 0.052
  Proportion difference: 0.003

Disease 247 (signature 18, LR=30.82):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Epoch 37
Loss: 144.6551

Monitoring signature responses:

Disease 55 (signature 4, LR=32.94):
  Theta for diagnosed: 0.065 ± 0.015
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 115 (signature 5, LR=31.47):
  Theta for diagnosed: 0.057 ± 0.028
  Theta for others: 0.052
  Proportion difference: 0.006

Disease 330 (signature 3, LR=31.46):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=31.03):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=30.90):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Epoch 38
Loss: 146.3362

Monitoring signature responses:

Disease 55 (signature 4, LR=32.94):
  Theta for diagnosed: 0.065 ± 0.016
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=31.64):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 115 (signature 5, LR=31.44):
  Theta for diagnosed: 0.058 ± 0.028
  Theta for others: 0.052
  Proportion difference: 0.006

Disease 247 (signature 18, LR=31.23):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=31.07):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Epoch 39
Loss: 144.6888

Monitoring signature responses:

Disease 55 (signature 4, LR=32.93):
  Theta for diagnosed: 0.065 ± 0.016
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=31.82):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=31.43):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 115 (signature 5, LR=31.41):
  Theta for diagnosed: 0.058 ± 0.029
  Theta for others: 0.052
  Proportion difference: 0.006

Disease 255 (signature 3, LR=31.23):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Epoch 40
Loss: 141.5606

Monitoring signature responses:

Disease 55 (signature 4, LR=32.91):
  Theta for diagnosed: 0.065 ± 0.016
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=31.99):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=31.63):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=31.40):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 115 (signature 5, LR=31.37):
  Theta for diagnosed: 0.058 ± 0.029
  Theta for others: 0.052
  Proportion difference: 0.006

Epoch 41
Loss: 140.1445

Monitoring signature responses:

Disease 55 (signature 4, LR=32.89):
  Theta for diagnosed: 0.065 ± 0.016
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=32.17):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=31.83):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=31.56):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 115 (signature 5, LR=31.33):
  Theta for diagnosed: 0.058 ± 0.029
  Theta for others: 0.052
  Proportion difference: 0.006

Epoch 42
Loss: 141.1873

Monitoring signature responses:

Disease 55 (signature 4, LR=32.87):
  Theta for diagnosed: 0.065 ± 0.017
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=32.35):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=32.04):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=31.73):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 115 (signature 5, LR=31.28):
  Theta for diagnosed: 0.058 ± 0.029
  Theta for others: 0.052
  Proportion difference: 0.006

Epoch 43
Loss: 142.5651

Monitoring signature responses:

Disease 55 (signature 4, LR=32.85):
  Theta for diagnosed: 0.065 ± 0.017
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=32.52):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=32.25):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=31.90):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=31.32):
  Theta for diagnosed: 0.029 ± 0.010
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 44
Loss: 142.1940

Monitoring signature responses:

Disease 55 (signature 4, LR=32.82):
  Theta for diagnosed: 0.066 ± 0.017
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 330 (signature 3, LR=32.70):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=32.46):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=32.07):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=31.52):
  Theta for diagnosed: 0.029 ± 0.010
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 45
Loss: 140.4157

Monitoring signature responses:

Disease 330 (signature 3, LR=32.88):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 55 (signature 4, LR=32.79):
  Theta for diagnosed: 0.066 ± 0.018
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 247 (signature 18, LR=32.67):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=32.24):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=31.71):
  Theta for diagnosed: 0.029 ± 0.010
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 46
Loss: 139.1740

Monitoring signature responses:

Disease 330 (signature 3, LR=33.05):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=32.88):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 55 (signature 4, LR=32.76):
  Theta for diagnosed: 0.066 ± 0.018
  Theta for others: 0.062
  Proportion difference: 0.003

Disease 255 (signature 3, LR=32.41):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=31.91):
  Theta for diagnosed: 0.029 ± 0.010
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 47
Loss: 139.4621

Monitoring signature responses:

Disease 330 (signature 3, LR=33.23):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=33.09):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 55 (signature 4, LR=32.73):
  Theta for diagnosed: 0.066 ± 0.018
  Theta for others: 0.062
  Proportion difference: 0.004

Disease 255 (signature 3, LR=32.58):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=32.11):
  Theta for diagnosed: 0.029 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 48
Loss: 140.3117

Monitoring signature responses:

Disease 330 (signature 3, LR=33.40):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=33.30):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=32.75):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 55 (signature 4, LR=32.69):
  Theta for diagnosed: 0.066 ± 0.018
  Theta for others: 0.062
  Proportion difference: 0.004

Disease 90 (signature 16, LR=32.30):
  Theta for diagnosed: 0.029 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 49
Loss: 140.2833

Monitoring signature responses:

Disease 330 (signature 3, LR=33.58):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=33.52):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=32.92):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 55 (signature 4, LR=32.66):
  Theta for diagnosed: 0.066 ± 0.019
  Theta for others: 0.062
  Proportion difference: 0.004

Disease 90 (signature 16, LR=32.50):
  Theta for diagnosed: 0.029 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Epoch 50
Loss: 139.2766

Monitoring signature responses:

Disease 330 (signature 3, LR=33.76):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 247 (signature 18, LR=33.73):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 255 (signature 3, LR=33.09):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=32.70):
  Theta for diagnosed: 0.029 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 55 (signature 4, LR=32.62):
  Theta for diagnosed: 0.066 ± 0.019
  Theta for others: 0.062
  Proportion difference: 0.004

Epoch 51
Loss: 138.4165

Monitoring signature responses:

Disease 247 (signature 18, LR=33.95):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=33.93):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=33.27):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=32.90):
  Theta for diagnosed: 0.029 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=32.70):
  Theta for diagnosed: 0.024 ± 0.005
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 52
Loss: 138.4637

Monitoring signature responses:

Disease 247 (signature 18, LR=34.17):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=34.11):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=33.44):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=33.11):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=32.88):
  Theta for diagnosed: 0.024 ± 0.005
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 53
Loss: 138.9395

Monitoring signature responses:

Disease 247 (signature 18, LR=34.39):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=34.29):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=33.62):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=33.31):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=33.06):
  Theta for diagnosed: 0.024 ± 0.005
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 54
Loss: 138.9353

Monitoring signature responses:

Disease 247 (signature 18, LR=34.62):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=34.47):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=33.80):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=33.52):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=33.25):
  Theta for diagnosed: 0.024 ± 0.005
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 55
Loss: 138.3249

Monitoring signature responses:

Disease 247 (signature 18, LR=34.84):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=34.65):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=33.98):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=33.73):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=33.44):
  Theta for diagnosed: 0.024 ± 0.005
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 56
Loss: 137.7843

Monitoring signature responses:

Disease 247 (signature 18, LR=35.07):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=34.83):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=34.16):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=33.94):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=33.63):
  Theta for diagnosed: 0.024 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 57
Loss: 137.7892

Monitoring signature responses:

Disease 247 (signature 18, LR=35.30):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=35.01):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=34.34):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=34.15):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=33.82):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 58
Loss: 138.0387

Monitoring signature responses:

Disease 247 (signature 18, LR=35.53):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=35.19):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=34.53):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=34.36):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=34.01):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 59
Loss: 137.9805

Monitoring signature responses:

Disease 247 (signature 18, LR=35.77):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=35.38):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=34.71):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=34.57):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=34.20):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 60
Loss: 137.5747

Monitoring signature responses:

Disease 247 (signature 18, LR=36.00):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=35.56):
  Theta for diagnosed: 0.030 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=34.90):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=34.79):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.028
  Proportion difference: 0.001

Disease 224 (signature 7, LR=34.40):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 61
Loss: 137.2543

Monitoring signature responses:

Disease 247 (signature 18, LR=36.24):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=35.74):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=35.09):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=35.00):
  Theta for diagnosed: 0.030 ± 0.013
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 224 (signature 7, LR=34.60):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 62
Loss: 137.2680

Monitoring signature responses:

Disease 247 (signature 18, LR=36.48):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=35.93):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=35.28):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=35.22):
  Theta for diagnosed: 0.030 ± 0.013
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 224 (signature 7, LR=34.80):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.000

Epoch 63
Loss: 137.3805

Monitoring signature responses:

Disease 247 (signature 18, LR=36.72):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=36.12):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 255 (signature 3, LR=35.47):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 90 (signature 16, LR=35.44):
  Theta for diagnosed: 0.030 ± 0.013
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 224 (signature 7, LR=35.01):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 64
Loss: 137.2732

Monitoring signature responses:

Disease 247 (signature 18, LR=36.96):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.000

Disease 330 (signature 3, LR=36.31):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 90 (signature 16, LR=35.67):
  Theta for diagnosed: 0.030 ± 0.013
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=35.66):
  Theta for diagnosed: 0.029 ± 0.007
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 224 (signature 7, LR=35.22):
  Theta for diagnosed: 0.025 ± 0.006
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 65
Loss: 136.9895

Monitoring signature responses:

Disease 247 (signature 18, LR=37.21):
  Theta for diagnosed: 0.012 ± 0.007
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=36.49):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 90 (signature 16, LR=35.89):
  Theta for diagnosed: 0.030 ± 0.013
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=35.86):
  Theta for diagnosed: 0.029 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 224 (signature 7, LR=35.43):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 66
Loss: 136.8107

Monitoring signature responses:

Disease 247 (signature 18, LR=37.46):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=36.68):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 90 (signature 16, LR=36.12):
  Theta for diagnosed: 0.030 ± 0.014
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=36.06):
  Theta for diagnosed: 0.029 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.000

Disease 224 (signature 7, LR=35.64):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 67
Loss: 136.8277

Monitoring signature responses:

Disease 247 (signature 18, LR=37.71):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=36.87):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=36.34):
  Theta for diagnosed: 0.030 ± 0.014
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=36.26):
  Theta for diagnosed: 0.029 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=35.85):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 68
Loss: 136.8489

Monitoring signature responses:

Disease 247 (signature 18, LR=37.96):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=37.06):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=36.57):
  Theta for diagnosed: 0.030 ± 0.014
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=36.45):
  Theta for diagnosed: 0.029 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=36.07):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 69
Loss: 136.7163

Monitoring signature responses:

Disease 247 (signature 18, LR=38.21):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=37.25):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=36.80):
  Theta for diagnosed: 0.030 ± 0.014
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=36.66):
  Theta for diagnosed: 0.029 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=36.28):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 70
Loss: 136.5202

Monitoring signature responses:

Disease 247 (signature 18, LR=38.47):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=37.44):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=37.03):
  Theta for diagnosed: 0.030 ± 0.014
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=36.86):
  Theta for diagnosed: 0.029 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=36.50):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 71
Loss: 136.4274

Monitoring signature responses:

Disease 247 (signature 18, LR=38.73):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=37.63):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=37.26):
  Theta for diagnosed: 0.030 ± 0.014
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=37.06):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=36.73):
  Theta for diagnosed: 0.025 ± 0.007
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 72
Loss: 136.4308

Monitoring signature responses:

Disease 247 (signature 18, LR=38.98):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=37.83):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=37.50):
  Theta for diagnosed: 0.030 ± 0.015
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=37.27):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=36.95):
  Theta for diagnosed: 0.025 ± 0.008
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 73
Loss: 136.3903

Monitoring signature responses:

Disease 247 (signature 18, LR=39.25):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=38.02):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=37.73):
  Theta for diagnosed: 0.030 ± 0.015
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=37.47):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=37.18):
  Theta for diagnosed: 0.025 ± 0.008
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 74
Loss: 136.2574

Monitoring signature responses:

Disease 247 (signature 18, LR=39.51):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=38.21):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=37.97):
  Theta for diagnosed: 0.030 ± 0.015
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=37.68):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=37.41):
  Theta for diagnosed: 0.025 ± 0.008
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 75
Loss: 136.1288

Monitoring signature responses:

Disease 247 (signature 18, LR=39.77):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=38.41):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=38.21):
  Theta for diagnosed: 0.030 ± 0.015
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=37.89):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=37.64):
  Theta for diagnosed: 0.025 ± 0.008
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 76
Loss: 136.0796

Monitoring signature responses:

Disease 247 (signature 18, LR=40.04):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=38.60):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=38.45):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=38.10):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=37.87):
  Theta for diagnosed: 0.025 ± 0.008
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 77
Loss: 136.0570

Monitoring signature responses:

Disease 247 (signature 18, LR=40.31):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=38.79):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=38.68):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=38.31):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=38.11):
  Theta for diagnosed: 0.025 ± 0.009
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 78
Loss: 135.9809

Monitoring signature responses:

Disease 247 (signature 18, LR=40.58):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=38.99):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=38.92):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=38.52):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=38.35):
  Theta for diagnosed: 0.025 ± 0.009
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 79
Loss: 135.8670

Monitoring signature responses:

Disease 247 (signature 18, LR=40.85):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 330 (signature 3, LR=39.18):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 90 (signature 16, LR=39.16):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 255 (signature 3, LR=38.73):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=38.58):
  Theta for diagnosed: 0.025 ± 0.009
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 80
Loss: 135.7863

Monitoring signature responses:

Disease 247 (signature 18, LR=41.12):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=39.40):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 330 (signature 3, LR=39.37):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=38.94):
  Theta for diagnosed: 0.030 ± 0.008
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=38.83):
  Theta for diagnosed: 0.025 ± 0.009
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 81
Loss: 135.7486

Monitoring signature responses:

Disease 247 (signature 18, LR=41.40):
  Theta for diagnosed: 0.012 ± 0.008
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=39.65):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 330 (signature 3, LR=39.57):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=39.15):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=39.07):
  Theta for diagnosed: 0.025 ± 0.009
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 82
Loss: 135.6988

Monitoring signature responses:

Disease 247 (signature 18, LR=41.67):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=39.89):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.028
  Proportion difference: 0.002

Disease 330 (signature 3, LR=39.76):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=39.37):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=39.31):
  Theta for diagnosed: 0.025 ± 0.009
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 83
Loss: 135.6116

Monitoring signature responses:

Disease 247 (signature 18, LR=41.95):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=40.13):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=39.95):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=39.58):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 224 (signature 7, LR=39.56):
  Theta for diagnosed: 0.025 ± 0.010
  Theta for others: 0.024
  Proportion difference: 0.001

Epoch 84
Loss: 135.5249

Monitoring signature responses:

Disease 247 (signature 18, LR=42.23):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=40.37):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=40.14):
  Theta for diagnosed: 0.031 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 224 (signature 7, LR=39.81):
  Theta for diagnosed: 0.025 ± 0.010
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=39.79):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 85
Loss: 135.4701

Monitoring signature responses:

Disease 247 (signature 18, LR=42.51):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=40.62):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=40.34):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 224 (signature 7, LR=40.06):
  Theta for diagnosed: 0.025 ± 0.010
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=40.01):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 86
Loss: 135.4249

Monitoring signature responses:

Disease 247 (signature 18, LR=42.79):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=40.86):
  Theta for diagnosed: 0.031 ± 0.018
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=40.53):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 224 (signature 7, LR=40.31):
  Theta for diagnosed: 0.025 ± 0.010
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=40.22):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 87
Loss: 135.3574

Monitoring signature responses:

Disease 247 (signature 18, LR=43.08):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=41.10):
  Theta for diagnosed: 0.031 ± 0.018
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=40.72):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 224 (signature 7, LR=40.56):
  Theta for diagnosed: 0.025 ± 0.010
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=40.43):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 88
Loss: 135.2774

Monitoring signature responses:

Disease 247 (signature 18, LR=43.36):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=41.34):
  Theta for diagnosed: 0.031 ± 0.018
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=40.91):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 224 (signature 7, LR=40.82):
  Theta for diagnosed: 0.025 ± 0.010
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=40.65):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 89
Loss: 135.2134

Monitoring signature responses:

Disease 247 (signature 18, LR=43.65):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=41.59):
  Theta for diagnosed: 0.031 ± 0.018
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 330 (signature 3, LR=41.10):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 224 (signature 7, LR=41.07):
  Theta for diagnosed: 0.025 ± 0.011
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=40.86):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 90
Loss: 135.1646

Monitoring signature responses:

Disease 247 (signature 18, LR=43.93):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=41.83):
  Theta for diagnosed: 0.031 ± 0.018
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=41.33):
  Theta for diagnosed: 0.025 ± 0.011
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=41.29):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=41.07):
  Theta for diagnosed: 0.030 ± 0.009
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 91
Loss: 135.1071

Monitoring signature responses:

Disease 247 (signature 18, LR=44.22):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=42.07):
  Theta for diagnosed: 0.031 ± 0.019
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=41.59):
  Theta for diagnosed: 0.025 ± 0.011
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=41.47):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=41.29):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 92
Loss: 135.0363

Monitoring signature responses:

Disease 247 (signature 18, LR=44.51):
  Theta for diagnosed: 0.012 ± 0.009
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=42.31):
  Theta for diagnosed: 0.031 ± 0.019
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=41.85):
  Theta for diagnosed: 0.025 ± 0.011
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=41.66):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=41.50):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 93
Loss: 134.9705

Monitoring signature responses:

Disease 247 (signature 18, LR=44.80):
  Theta for diagnosed: 0.012 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=42.55):
  Theta for diagnosed: 0.031 ± 0.019
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=42.11):
  Theta for diagnosed: 0.025 ± 0.011
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=41.85):
  Theta for diagnosed: 0.031 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=41.71):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 94
Loss: 134.9174

Monitoring signature responses:

Disease 247 (signature 18, LR=45.09):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=42.80):
  Theta for diagnosed: 0.031 ± 0.019
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=42.38):
  Theta for diagnosed: 0.025 ± 0.011
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=42.03):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=41.92):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 95
Loss: 134.8637

Monitoring signature responses:

Disease 247 (signature 18, LR=45.39):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=43.04):
  Theta for diagnosed: 0.032 ± 0.020
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=42.64):
  Theta for diagnosed: 0.026 ± 0.012
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=42.22):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=42.13):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 96
Loss: 134.8001

Monitoring signature responses:

Disease 247 (signature 18, LR=45.68):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=43.28):
  Theta for diagnosed: 0.032 ± 0.020
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=42.91):
  Theta for diagnosed: 0.026 ± 0.012
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=42.40):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=42.34):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 97
Loss: 134.7360

Monitoring signature responses:

Disease 247 (signature 18, LR=45.98):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=43.52):
  Theta for diagnosed: 0.032 ± 0.020
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=43.18):
  Theta for diagnosed: 0.026 ± 0.012
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=42.58):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=42.55):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 98
Loss: 134.6805

Monitoring signature responses:

Disease 247 (signature 18, LR=46.27):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=43.76):
  Theta for diagnosed: 0.032 ± 0.020
  Theta for others: 0.028
  Proportion difference: 0.003

Disease 224 (signature 7, LR=43.45):
  Theta for diagnosed: 0.026 ± 0.012
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 330 (signature 3, LR=42.76):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Disease 255 (signature 3, LR=42.76):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 99
Loss: 134.6276

Monitoring signature responses:

Disease 247 (signature 18, LR=46.57):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=43.99):
  Theta for diagnosed: 0.032 ± 0.021
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=43.72):
  Theta for diagnosed: 0.026 ± 0.012
  Theta for others: 0.024
  Proportion difference: 0.001

Disease 255 (signature 3, LR=42.97):
  Theta for diagnosed: 0.030 ± 0.010
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=42.94):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 100
Loss: 134.5688

Monitoring signature responses:

Disease 247 (signature 18, LR=46.87):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=44.23):
  Theta for diagnosed: 0.032 ± 0.021
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=43.99):
  Theta for diagnosed: 0.026 ± 0.013
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=43.18):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=43.12):
  Theta for diagnosed: 0.031 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 101
Loss: 134.5075

Monitoring signature responses:

Disease 247 (signature 18, LR=47.17):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=44.47):
  Theta for diagnosed: 0.032 ± 0.021
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=44.26):
  Theta for diagnosed: 0.026 ± 0.013
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=43.38):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=43.30):
  Theta for diagnosed: 0.031 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 102
Loss: 134.4513

Monitoring signature responses:

Disease 247 (signature 18, LR=47.47):
  Theta for diagnosed: 0.013 ± 0.010
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=44.71):
  Theta for diagnosed: 0.032 ± 0.021
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=44.53):
  Theta for diagnosed: 0.026 ± 0.013
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=43.59):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=43.48):
  Theta for diagnosed: 0.031 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 103
Loss: 134.3984

Monitoring signature responses:

Disease 247 (signature 18, LR=47.77):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=44.94):
  Theta for diagnosed: 0.032 ± 0.022
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=44.81):
  Theta for diagnosed: 0.026 ± 0.013
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=43.79):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=43.66):
  Theta for diagnosed: 0.031 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 104
Loss: 134.3424

Monitoring signature responses:

Disease 247 (signature 18, LR=48.08):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=45.18):
  Theta for diagnosed: 0.032 ± 0.022
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=45.08):
  Theta for diagnosed: 0.026 ± 0.013
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=43.99):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=43.83):
  Theta for diagnosed: 0.031 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 105
Loss: 134.2839

Monitoring signature responses:

Disease 247 (signature 18, LR=48.38):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=45.41):
  Theta for diagnosed: 0.032 ± 0.022
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=45.36):
  Theta for diagnosed: 0.026 ± 0.013
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=44.20):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=44.01):
  Theta for diagnosed: 0.031 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 106
Loss: 134.2282

Monitoring signature responses:

Disease 247 (signature 18, LR=48.69):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 90 (signature 16, LR=45.65):
  Theta for diagnosed: 0.032 ± 0.023
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 224 (signature 7, LR=45.64):
  Theta for diagnosed: 0.026 ± 0.014
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 255 (signature 3, LR=44.40):
  Theta for diagnosed: 0.030 ± 0.011
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 330 (signature 3, LR=44.18):
  Theta for diagnosed: 0.031 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 107
Loss: 134.1754

Monitoring signature responses:

Disease 247 (signature 18, LR=48.99):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=45.91):
  Theta for diagnosed: 0.026 ± 0.014
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=45.88):
  Theta for diagnosed: 0.032 ± 0.023
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 255 (signature 3, LR=44.60):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 21 (signature 18, LR=44.38):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 108
Loss: 134.1211

Monitoring signature responses:

Disease 247 (signature 18, LR=49.30):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=46.19):
  Theta for diagnosed: 0.026 ± 0.014
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=46.11):
  Theta for diagnosed: 0.032 ± 0.023
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 255 (signature 3, LR=44.80):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Disease 21 (signature 18, LR=44.74):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 109
Loss: 134.0649

Monitoring signature responses:

Disease 247 (signature 18, LR=49.61):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=46.47):
  Theta for diagnosed: 0.026 ± 0.014
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=46.34):
  Theta for diagnosed: 0.032 ± 0.023
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 21 (signature 18, LR=45.10):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=45.00):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 110
Loss: 134.0102

Monitoring signature responses:

Disease 247 (signature 18, LR=49.92):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=46.75):
  Theta for diagnosed: 0.026 ± 0.014
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=46.57):
  Theta for diagnosed: 0.033 ± 0.024
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 21 (signature 18, LR=45.46):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=45.20):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 111
Loss: 133.9577

Monitoring signature responses:

Disease 247 (signature 18, LR=50.23):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=47.04):
  Theta for diagnosed: 0.026 ± 0.015
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=46.80):
  Theta for diagnosed: 0.033 ± 0.024
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 21 (signature 18, LR=45.83):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=45.39):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 112
Loss: 133.9047

Monitoring signature responses:

Disease 247 (signature 18, LR=50.54):
  Theta for diagnosed: 0.013 ± 0.011
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=47.32):
  Theta for diagnosed: 0.026 ± 0.015
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=47.03):
  Theta for diagnosed: 0.033 ± 0.024
  Theta for others: 0.028
  Proportion difference: 0.004

Disease 21 (signature 18, LR=46.20):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=45.59):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 113
Loss: 133.8503

Monitoring signature responses:

Disease 247 (signature 18, LR=50.86):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=47.60):
  Theta for diagnosed: 0.026 ± 0.015
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=47.26):
  Theta for diagnosed: 0.033 ± 0.024
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 21 (signature 18, LR=46.58):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=45.78):
  Theta for diagnosed: 0.030 ± 0.012
  Theta for others: 0.029
  Proportion difference: 0.001

Epoch 114
Loss: 133.7968

Monitoring signature responses:

Disease 247 (signature 18, LR=51.17):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=47.89):
  Theta for diagnosed: 0.026 ± 0.015
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=47.49):
  Theta for diagnosed: 0.033 ± 0.025
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 21 (signature 18, LR=46.95):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=45.98):
  Theta for diagnosed: 0.030 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 115
Loss: 133.7449

Monitoring signature responses:

Disease 247 (signature 18, LR=51.48):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=48.17):
  Theta for diagnosed: 0.026 ± 0.015
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=47.71):
  Theta for diagnosed: 0.033 ± 0.025
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 21 (signature 18, LR=47.33):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=46.17):
  Theta for diagnosed: 0.031 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 116
Loss: 133.6929

Monitoring signature responses:

Disease 247 (signature 18, LR=51.80):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=48.46):
  Theta for diagnosed: 0.026 ± 0.015
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=47.94):
  Theta for diagnosed: 0.033 ± 0.025
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 21 (signature 18, LR=47.72):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=46.36):
  Theta for diagnosed: 0.031 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 117
Loss: 133.6400

Monitoring signature responses:

Disease 247 (signature 18, LR=52.12):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=48.74):
  Theta for diagnosed: 0.026 ± 0.016
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=48.16):
  Theta for diagnosed: 0.033 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 21 (signature 18, LR=48.11):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 255 (signature 3, LR=46.55):
  Theta for diagnosed: 0.031 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 118
Loss: 133.5876

Monitoring signature responses:

Disease 247 (signature 18, LR=52.44):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=49.03):
  Theta for diagnosed: 0.026 ± 0.016
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 21 (signature 18, LR=48.50):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 90 (signature 16, LR=48.39):
  Theta for diagnosed: 0.033 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=46.74):
  Theta for diagnosed: 0.031 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 119
Loss: 133.5364

Monitoring signature responses:

Disease 247 (signature 18, LR=52.76):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=49.32):
  Theta for diagnosed: 0.026 ± 0.016
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 21 (signature 18, LR=48.89):
  Theta for diagnosed: 0.014 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 90 (signature 16, LR=48.61):
  Theta for diagnosed: 0.033 ± 0.026
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=46.93):
  Theta for diagnosed: 0.031 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 120
Loss: 133.4853

Monitoring signature responses:

Disease 247 (signature 18, LR=53.08):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=49.60):
  Theta for diagnosed: 0.027 ± 0.016
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 21 (signature 18, LR=49.29):
  Theta for diagnosed: 0.014 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 90 (signature 16, LR=48.83):
  Theta for diagnosed: 0.033 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=47.12):
  Theta for diagnosed: 0.031 ± 0.013
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 121
Loss: 133.4337

Monitoring signature responses:

Disease 247 (signature 18, LR=53.40):
  Theta for diagnosed: 0.013 ± 0.012
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=49.89):
  Theta for diagnosed: 0.027 ± 0.016
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 21 (signature 18, LR=49.69):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 90 (signature 16, LR=49.06):
  Theta for diagnosed: 0.033 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=47.31):
  Theta for diagnosed: 0.031 ± 0.014
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 122
Loss: 133.3825

Monitoring signature responses:

Disease 247 (signature 18, LR=53.72):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 224 (signature 7, LR=50.18):
  Theta for diagnosed: 0.027 ± 0.017
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 21 (signature 18, LR=50.09):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 90 (signature 16, LR=49.28):
  Theta for diagnosed: 0.033 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=47.50):
  Theta for diagnosed: 0.031 ± 0.014
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 123
Loss: 133.3321

Monitoring signature responses:

Disease 247 (signature 18, LR=54.05):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=50.50):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=50.47):
  Theta for diagnosed: 0.027 ± 0.017
  Theta for others: 0.024
  Proportion difference: 0.002

Disease 90 (signature 16, LR=49.50):
  Theta for diagnosed: 0.034 ± 0.027
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=47.68):
  Theta for diagnosed: 0.031 ± 0.014
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 124
Loss: 133.2819

Monitoring signature responses:

Disease 247 (signature 18, LR=54.37):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=50.91):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=50.76):
  Theta for diagnosed: 0.027 ± 0.017
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=49.72):
  Theta for diagnosed: 0.034 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=47.87):
  Theta for diagnosed: 0.031 ± 0.014
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 125
Loss: 133.2314

Monitoring signature responses:

Disease 247 (signature 18, LR=54.70):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=51.33):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=51.05):
  Theta for diagnosed: 0.027 ± 0.017
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=49.94):
  Theta for diagnosed: 0.034 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.005

Disease 255 (signature 3, LR=48.05):
  Theta for diagnosed: 0.031 ± 0.014
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 126
Loss: 133.1812

Monitoring signature responses:

Disease 247 (signature 18, LR=55.03):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=51.74):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=51.34):
  Theta for diagnosed: 0.027 ± 0.018
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=50.15):
  Theta for diagnosed: 0.034 ± 0.028
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=48.24):
  Theta for diagnosed: 0.031 ± 0.014
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 127
Loss: 133.1317

Monitoring signature responses:

Disease 247 (signature 18, LR=55.35):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=52.16):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=51.63):
  Theta for diagnosed: 0.027 ± 0.018
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=50.37):
  Theta for diagnosed: 0.034 ± 0.029
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=48.42):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 128
Loss: 133.0823

Monitoring signature responses:

Disease 247 (signature 18, LR=55.68):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=52.59):
  Theta for diagnosed: 0.014 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=51.92):
  Theta for diagnosed: 0.027 ± 0.018
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=50.59):
  Theta for diagnosed: 0.034 ± 0.029
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=48.60):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 129
Loss: 133.0329

Monitoring signature responses:

Disease 247 (signature 18, LR=56.01):
  Theta for diagnosed: 0.013 ± 0.013
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=53.01):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=52.21):
  Theta for diagnosed: 0.027 ± 0.018
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=50.80):
  Theta for diagnosed: 0.034 ± 0.029
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=48.78):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 130
Loss: 132.9837

Monitoring signature responses:

Disease 247 (signature 18, LR=56.35):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=53.44):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=52.50):
  Theta for diagnosed: 0.027 ± 0.018
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=51.02):
  Theta for diagnosed: 0.034 ± 0.030
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=48.96):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 131
Loss: 132.9350

Monitoring signature responses:

Disease 247 (signature 18, LR=56.68):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=53.87):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=52.79):
  Theta for diagnosed: 0.027 ± 0.019
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=51.23):
  Theta for diagnosed: 0.034 ± 0.030
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=49.14):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 132
Loss: 132.8865

Monitoring signature responses:

Disease 247 (signature 18, LR=57.01):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=54.31):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=53.07):
  Theta for diagnosed: 0.027 ± 0.019
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=51.45):
  Theta for diagnosed: 0.034 ± 0.030
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=49.32):
  Theta for diagnosed: 0.031 ± 0.015
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 133
Loss: 132.8380

Monitoring signature responses:

Disease 247 (signature 18, LR=57.35):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=54.75):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=53.36):
  Theta for diagnosed: 0.027 ± 0.019
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=51.66):
  Theta for diagnosed: 0.034 ± 0.031
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=49.50):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 134
Loss: 132.7898

Monitoring signature responses:

Disease 247 (signature 18, LR=57.68):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=55.19):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=53.65):
  Theta for diagnosed: 0.027 ± 0.019
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=51.87):
  Theta for diagnosed: 0.034 ± 0.031
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=49.67):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 135
Loss: 132.7420

Monitoring signature responses:

Disease 247 (signature 18, LR=58.02):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=55.63):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=53.94):
  Theta for diagnosed: 0.027 ± 0.020
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=52.09):
  Theta for diagnosed: 0.035 ± 0.031
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=49.85):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 136
Loss: 132.6943

Monitoring signature responses:

Disease 247 (signature 18, LR=58.35):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.001

Disease 21 (signature 18, LR=56.08):
  Theta for diagnosed: 0.014 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=54.23):
  Theta for diagnosed: 0.027 ± 0.020
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=52.30):
  Theta for diagnosed: 0.035 ± 0.032
  Theta for others: 0.028
  Proportion difference: 0.006

Disease 255 (signature 3, LR=50.02):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 137
Loss: 132.6467

Monitoring signature responses:

Disease 247 (signature 18, LR=58.69):
  Theta for diagnosed: 0.013 ± 0.014
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=56.53):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=54.52):
  Theta for diagnosed: 0.027 ± 0.020
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=52.51):
  Theta for diagnosed: 0.035 ± 0.032
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 255 (signature 3, LR=50.20):
  Theta for diagnosed: 0.031 ± 0.016
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 138
Loss: 132.5994

Monitoring signature responses:

Disease 247 (signature 18, LR=59.03):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=56.98):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=54.80):
  Theta for diagnosed: 0.028 ± 0.020
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=52.72):
  Theta for diagnosed: 0.035 ± 0.032
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 255 (signature 3, LR=50.37):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 139
Loss: 132.5525

Monitoring signature responses:

Disease 247 (signature 18, LR=59.37):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=57.44):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=55.09):
  Theta for diagnosed: 0.028 ± 0.021
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=52.93):
  Theta for diagnosed: 0.035 ± 0.033
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 255 (signature 3, LR=50.54):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 140
Loss: 132.5056

Monitoring signature responses:

Disease 247 (signature 18, LR=59.71):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=57.89):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=55.37):
  Theta for diagnosed: 0.028 ± 0.021
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=53.14):
  Theta for diagnosed: 0.035 ± 0.033
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 255 (signature 3, LR=50.72):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 141
Loss: 132.4589

Monitoring signature responses:

Disease 247 (signature 18, LR=60.05):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=58.35):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=55.66):
  Theta for diagnosed: 0.028 ± 0.021
  Theta for others: 0.024
  Proportion difference: 0.003

Disease 90 (signature 16, LR=53.34):
  Theta for diagnosed: 0.035 ± 0.033
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 255 (signature 3, LR=50.89):
  Theta for diagnosed: 0.031 ± 0.017
  Theta for others: 0.029
  Proportion difference: 0.002

Epoch 142
Loss: 132.4124

Monitoring signature responses:

Disease 247 (signature 18, LR=60.39):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=58.82):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=55.94):
  Theta for diagnosed: 0.028 ± 0.021
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=53.55):
  Theta for diagnosed: 0.035 ± 0.034
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 232 (signature 18, LR=51.19):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 143
Loss: 132.3663

Monitoring signature responses:

Disease 247 (signature 18, LR=60.74):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=59.28):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=56.23):
  Theta for diagnosed: 0.028 ± 0.021
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=53.76):
  Theta for diagnosed: 0.035 ± 0.034
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 232 (signature 18, LR=51.53):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 144
Loss: 132.3202

Monitoring signature responses:

Disease 247 (signature 18, LR=61.08):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=59.75):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=56.51):
  Theta for diagnosed: 0.028 ± 0.022
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=53.96):
  Theta for diagnosed: 0.035 ± 0.034
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 232 (signature 18, LR=51.87):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 145
Loss: 132.2744

Monitoring signature responses:

Disease 247 (signature 18, LR=61.43):
  Theta for diagnosed: 0.013 ± 0.015
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=60.22):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=56.79):
  Theta for diagnosed: 0.028 ± 0.022
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=54.17):
  Theta for diagnosed: 0.035 ± 0.035
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 232 (signature 18, LR=52.22):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 146
Loss: 132.2288

Monitoring signature responses:

Disease 247 (signature 18, LR=61.77):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=60.69):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=57.07):
  Theta for diagnosed: 0.028 ± 0.022
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=54.37):
  Theta for diagnosed: 0.036 ± 0.035
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 232 (signature 18, LR=52.57):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 147
Loss: 132.1834

Monitoring signature responses:

Disease 247 (signature 18, LR=62.12):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=61.16):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=57.35):
  Theta for diagnosed: 0.028 ± 0.022
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=54.58):
  Theta for diagnosed: 0.036 ± 0.035
  Theta for others: 0.028
  Proportion difference: 0.007

Disease 232 (signature 18, LR=52.92):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 148
Loss: 132.1382

Monitoring signature responses:

Disease 247 (signature 18, LR=62.46):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=61.64):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=57.63):
  Theta for diagnosed: 0.028 ± 0.023
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=54.78):
  Theta for diagnosed: 0.036 ± 0.036
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=53.27):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 149
Loss: 132.0931

Monitoring signature responses:

Disease 247 (signature 18, LR=62.81):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=62.12):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=57.90):
  Theta for diagnosed: 0.028 ± 0.023
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=54.98):
  Theta for diagnosed: 0.036 ± 0.036
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=53.62):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 150
Loss: 132.0483

Monitoring signature responses:

Disease 247 (signature 18, LR=63.16):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=62.60):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=58.18):
  Theta for diagnosed: 0.028 ± 0.023
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=55.18):
  Theta for diagnosed: 0.036 ± 0.036
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=53.98):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 151
Loss: 132.0037

Monitoring signature responses:

Disease 247 (signature 18, LR=63.51):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=63.08):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=58.45):
  Theta for diagnosed: 0.028 ± 0.023
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=55.39):
  Theta for diagnosed: 0.036 ± 0.037
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=54.34):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 152
Loss: 131.9592

Monitoring signature responses:

Disease 247 (signature 18, LR=63.86):
  Theta for diagnosed: 0.013 ± 0.016
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=63.57):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=58.72):
  Theta for diagnosed: 0.028 ± 0.024
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=55.59):
  Theta for diagnosed: 0.036 ± 0.037
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=54.70):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 153
Loss: 131.9149

Monitoring signature responses:

Disease 247 (signature 18, LR=64.21):
  Theta for diagnosed: 0.013 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=64.05):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=58.99):
  Theta for diagnosed: 0.028 ± 0.024
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=55.79):
  Theta for diagnosed: 0.036 ± 0.037
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=55.07):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Epoch 154
Loss: 131.8709

Monitoring signature responses:

Disease 247 (signature 18, LR=64.56):
  Theta for diagnosed: 0.013 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 21 (signature 18, LR=64.54):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=59.26):
  Theta for diagnosed: 0.028 ± 0.024
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=55.99):
  Theta for diagnosed: 0.036 ± 0.038
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=55.44):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 155
Loss: 131.8270

Monitoring signature responses:

Disease 21 (signature 18, LR=65.03):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=64.91):
  Theta for diagnosed: 0.013 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=59.53):
  Theta for diagnosed: 0.029 ± 0.024
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=56.18):
  Theta for diagnosed: 0.036 ± 0.038
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=55.81):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 156
Loss: 131.7833

Monitoring signature responses:

Disease 21 (signature 18, LR=65.52):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=65.26):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=59.79):
  Theta for diagnosed: 0.029 ± 0.025
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=56.38):
  Theta for diagnosed: 0.037 ± 0.038
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=56.18):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 157
Loss: 131.7398

Monitoring signature responses:

Disease 21 (signature 18, LR=66.01):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=65.61):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=60.06):
  Theta for diagnosed: 0.029 ± 0.025
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 90 (signature 16, LR=56.58):
  Theta for diagnosed: 0.037 ± 0.039
  Theta for others: 0.028
  Proportion difference: 0.008

Disease 232 (signature 18, LR=56.56):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Epoch 158
Loss: 131.6965

Monitoring signature responses:

Disease 21 (signature 18, LR=66.50):
  Theta for diagnosed: 0.015 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=65.96):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=60.32):
  Theta for diagnosed: 0.029 ± 0.025
  Theta for others: 0.024
  Proportion difference: 0.004

Disease 232 (signature 18, LR=56.94):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=56.78):
  Theta for diagnosed: 0.037 ± 0.039
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 159
Loss: 131.6534

Monitoring signature responses:

Disease 21 (signature 18, LR=67.00):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=66.31):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=60.58):
  Theta for diagnosed: 0.029 ± 0.025
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=57.32):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=56.98):
  Theta for diagnosed: 0.037 ± 0.039
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 160
Loss: 131.6104

Monitoring signature responses:

Disease 21 (signature 18, LR=67.49):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=66.67):
  Theta for diagnosed: 0.014 ± 0.017
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=60.84):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=57.70):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=57.17):
  Theta for diagnosed: 0.037 ± 0.040
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 161
Loss: 131.5676

Monitoring signature responses:

Disease 21 (signature 18, LR=67.99):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=67.02):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=61.09):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=58.08):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=57.37):
  Theta for diagnosed: 0.037 ± 0.040
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 162
Loss: 131.5251

Monitoring signature responses:

Disease 21 (signature 18, LR=68.49):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=67.37):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=61.35):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=58.47):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=57.56):
  Theta for diagnosed: 0.037 ± 0.040
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 163
Loss: 131.4826

Monitoring signature responses:

Disease 21 (signature 18, LR=68.99):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=67.73):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=61.60):
  Theta for diagnosed: 0.029 ± 0.026
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=58.86):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=57.76):
  Theta for diagnosed: 0.037 ± 0.041
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 164
Loss: 131.4403

Monitoring signature responses:

Disease 21 (signature 18, LR=69.49):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=68.08):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=61.85):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=59.25):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=57.95):
  Theta for diagnosed: 0.037 ± 0.041
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 165
Loss: 131.3983

Monitoring signature responses:

Disease 21 (signature 18, LR=69.99):
  Theta for diagnosed: 0.015 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=68.44):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=62.09):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=59.65):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=58.15):
  Theta for diagnosed: 0.037 ± 0.041
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 166
Loss: 131.3565

Monitoring signature responses:

Disease 21 (signature 18, LR=70.49):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=68.79):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=62.34):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=60.04):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=58.34):
  Theta for diagnosed: 0.038 ± 0.042
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 167
Loss: 131.3147

Monitoring signature responses:

Disease 21 (signature 18, LR=70.99):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=69.15):
  Theta for diagnosed: 0.014 ± 0.018
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=62.58):
  Theta for diagnosed: 0.029 ± 0.027
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=60.44):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 90 (signature 16, LR=58.54):
  Theta for diagnosed: 0.038 ± 0.042
  Theta for others: 0.028
  Proportion difference: 0.009

Epoch 168
Loss: 131.2731

Monitoring signature responses:

Disease 21 (signature 18, LR=71.49):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=69.51):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=62.82):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=60.84):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=58.94):
  Theta for diagnosed: 0.030 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 169
Loss: 131.2318

Monitoring signature responses:

Disease 21 (signature 18, LR=72.00):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=69.86):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=63.06):
  Theta for diagnosed: 0.029 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=61.25):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=59.37):
  Theta for diagnosed: 0.030 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 170
Loss: 131.1907

Monitoring signature responses:

Disease 21 (signature 18, LR=72.50):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=70.22):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=63.30):
  Theta for diagnosed: 0.030 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=61.65):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=59.81):
  Theta for diagnosed: 0.030 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 171
Loss: 131.1496

Monitoring signature responses:

Disease 21 (signature 18, LR=73.00):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=70.58):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=63.53):
  Theta for diagnosed: 0.030 ± 0.028
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=62.06):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=60.24):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 172
Loss: 131.1087

Monitoring signature responses:

Disease 21 (signature 18, LR=73.51):
  Theta for diagnosed: 0.015 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=70.93):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=63.77):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=62.46):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=60.68):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 173
Loss: 131.0679

Monitoring signature responses:

Disease 21 (signature 18, LR=74.01):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=71.29):
  Theta for diagnosed: 0.014 ± 0.019
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=64.00):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=62.87):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=61.11):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 174
Loss: 131.0273

Monitoring signature responses:

Disease 21 (signature 18, LR=74.51):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 247 (signature 18, LR=71.65):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.002

Disease 224 (signature 7, LR=64.22):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.005

Disease 232 (signature 18, LR=63.29):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=61.55):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 175
Loss: 130.9869

Monitoring signature responses:

Disease 21 (signature 18, LR=75.01):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=72.01):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=64.45):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 232 (signature 18, LR=63.70):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=61.98):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 176
Loss: 130.9466

Monitoring signature responses:

Disease 21 (signature 18, LR=75.52):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=72.37):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=64.67):
  Theta for diagnosed: 0.030 ± 0.029
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 232 (signature 18, LR=64.11):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 190 (signature 7, LR=62.42):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 177
Loss: 130.9065

Monitoring signature responses:

Disease 21 (signature 18, LR=76.02):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=72.73):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=64.89):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 232 (signature 18, LR=64.53):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=62.85):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 178
Loss: 130.8665

Monitoring signature responses:

Disease 21 (signature 18, LR=76.52):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=73.09):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 224 (signature 7, LR=65.11):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 232 (signature 18, LR=64.95):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=63.29):
  Theta for diagnosed: 0.030 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 179
Loss: 130.8267

Monitoring signature responses:

Disease 21 (signature 18, LR=77.02):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=73.45):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 232 (signature 18, LR=65.36):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=65.33):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=63.73):
  Theta for diagnosed: 0.030 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 180
Loss: 130.7870

Monitoring signature responses:

Disease 21 (signature 18, LR=77.52):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=73.81):
  Theta for diagnosed: 0.014 ± 0.020
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 232 (signature 18, LR=65.78):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=65.54):
  Theta for diagnosed: 0.030 ± 0.030
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=64.16):
  Theta for diagnosed: 0.031 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 181
Loss: 130.7475

Monitoring signature responses:

Disease 21 (signature 18, LR=78.02):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=74.17):
  Theta for diagnosed: 0.014 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 232 (signature 18, LR=66.21):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=65.75):
  Theta for diagnosed: 0.030 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=64.60):
  Theta for diagnosed: 0.031 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 182
Loss: 130.7082

Monitoring signature responses:

Disease 21 (signature 18, LR=78.52):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=74.53):
  Theta for diagnosed: 0.014 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 232 (signature 18, LR=66.63):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=65.96):
  Theta for diagnosed: 0.030 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=65.03):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 183
Loss: 130.6689

Monitoring signature responses:

Disease 21 (signature 18, LR=79.02):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=74.89):
  Theta for diagnosed: 0.014 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 232 (signature 18, LR=67.05):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=66.17):
  Theta for diagnosed: 0.030 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=65.47):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 184
Loss: 130.6298

Monitoring signature responses:

Disease 21 (signature 18, LR=79.52):
  Theta for diagnosed: 0.016 ± 0.023
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=75.25):
  Theta for diagnosed: 0.014 ± 0.021
  Theta for others: 0.012
  Proportion difference: 0.003

Disease 232 (signature 18, LR=67.47):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=66.37):
  Theta for diagnosed: 0.030 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=65.90):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 185
Loss: 130.5909

Monitoring signature responses:

Disease 21 (signature 18, LR=80.01):
  Theta for diagnosed: 0.016 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=75.62):
  Theta for diagnosed: 0.014 ± 0.021
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=67.90):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=66.57):
  Theta for diagnosed: 0.031 ± 0.031
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=66.33):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 186
Loss: 130.5521

Monitoring signature responses:

Disease 21 (signature 18, LR=80.51):
  Theta for diagnosed: 0.016 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=75.98):
  Theta for diagnosed: 0.014 ± 0.021
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=68.32):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 224 (signature 7, LR=66.77):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.006

Disease 190 (signature 7, LR=66.76):
  Theta for diagnosed: 0.031 ± 0.033
  Theta for others: 0.024
  Proportion difference: 0.007

Epoch 187
Loss: 130.5134

Monitoring signature responses:

Disease 21 (signature 18, LR=81.00):
  Theta for diagnosed: 0.016 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=76.34):
  Theta for diagnosed: 0.014 ± 0.022
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=68.75):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=67.20):
  Theta for diagnosed: 0.031 ± 0.033
  Theta for others: 0.024
  Proportion difference: 0.007

Disease 224 (signature 7, LR=66.97):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 188
Loss: 130.4749

Monitoring signature responses:

Disease 21 (signature 18, LR=81.50):
  Theta for diagnosed: 0.016 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=76.70):
  Theta for diagnosed: 0.014 ± 0.022
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=69.18):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=67.63):
  Theta for diagnosed: 0.031 ± 0.033
  Theta for others: 0.024
  Proportion difference: 0.007

Disease 224 (signature 7, LR=67.16):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 189
Loss: 130.4364

Monitoring signature responses:

Disease 21 (signature 18, LR=81.99):
  Theta for diagnosed: 0.016 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=77.07):
  Theta for diagnosed: 0.014 ± 0.022
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=69.61):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=68.06):
  Theta for diagnosed: 0.031 ± 0.033
  Theta for others: 0.024
  Proportion difference: 0.007

Disease 224 (signature 7, LR=67.36):
  Theta for diagnosed: 0.031 ± 0.032
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 190
Loss: 130.3981

Monitoring signature responses:

Disease 21 (signature 18, LR=82.48):
  Theta for diagnosed: 0.016 ± 0.024
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=77.43):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=70.03):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=68.49):
  Theta for diagnosed: 0.031 ± 0.034
  Theta for others: 0.024
  Proportion difference: 0.007

Disease 224 (signature 7, LR=67.55):
  Theta for diagnosed: 0.031 ± 0.033
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 191
Loss: 130.3600

Monitoring signature responses:

Disease 21 (signature 18, LR=82.96):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=77.79):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=70.46):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=68.92):
  Theta for diagnosed: 0.031 ± 0.034
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 224 (signature 7, LR=67.74):
  Theta for diagnosed: 0.031 ± 0.033
  Theta for others: 0.024
  Proportion difference: 0.006

Epoch 192
Loss: 130.3220

Monitoring signature responses:

Disease 21 (signature 18, LR=83.45):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=78.16):
  Theta for diagnosed: 0.015 ± 0.022
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=70.89):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=69.34):
  Theta for diagnosed: 0.031 ± 0.034
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=68.11):
  Theta for diagnosed: 0.019 ± 0.021
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 193
Loss: 130.2841

Monitoring signature responses:

Disease 21 (signature 18, LR=83.93):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=78.52):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=71.32):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=69.77):
  Theta for diagnosed: 0.032 ± 0.034
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=68.52):
  Theta for diagnosed: 0.019 ± 0.021
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 194
Loss: 130.2464

Monitoring signature responses:

Disease 21 (signature 18, LR=84.42):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=78.88):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=71.75):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=70.20):
  Theta for diagnosed: 0.032 ± 0.035
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=68.93):
  Theta for diagnosed: 0.019 ± 0.021
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 195
Loss: 130.2088

Monitoring signature responses:

Disease 21 (signature 18, LR=84.90):
  Theta for diagnosed: 0.016 ± 0.025
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=79.25):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=72.18):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 190 (signature 7, LR=70.62):
  Theta for diagnosed: 0.032 ± 0.035
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=69.34):
  Theta for diagnosed: 0.019 ± 0.021
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 196
Loss: 130.1712

Monitoring signature responses:

Disease 21 (signature 18, LR=85.38):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.004

Disease 247 (signature 18, LR=79.61):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=72.61):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 190 (signature 7, LR=71.04):
  Theta for diagnosed: 0.032 ± 0.035
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=69.75):
  Theta for diagnosed: 0.019 ± 0.021
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 197
Loss: 130.1339

Monitoring signature responses:

Disease 21 (signature 18, LR=85.85):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 247 (signature 18, LR=79.98):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=73.03):
  Theta for diagnosed: 0.016 ± 0.027
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 190 (signature 7, LR=71.47):
  Theta for diagnosed: 0.032 ± 0.035
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=70.16):
  Theta for diagnosed: 0.019 ± 0.022
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 198
Loss: 130.0967

Monitoring signature responses:

Disease 21 (signature 18, LR=86.33):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 247 (signature 18, LR=80.34):
  Theta for diagnosed: 0.015 ± 0.023
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=73.46):
  Theta for diagnosed: 0.016 ± 0.028
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 190 (signature 7, LR=71.89):
  Theta for diagnosed: 0.032 ± 0.036
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=70.58):
  Theta for diagnosed: 0.019 ± 0.022
  Theta for others: 0.016
  Proportion difference: 0.003

Epoch 199
Loss: 130.0596

Monitoring signature responses:

Disease 21 (signature 18, LR=86.80):
  Theta for diagnosed: 0.016 ± 0.026
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 247 (signature 18, LR=80.70):
  Theta for diagnosed: 0.015 ± 0.024
  Theta for others: 0.011
  Proportion difference: 0.003

Disease 232 (signature 18, LR=73.89):
  Theta for diagnosed: 0.016 ± 0.028
  Theta for others: 0.012
  Proportion difference: 0.005

Disease 190 (signature 7, LR=72.30):
  Theta for diagnosed: 0.032 ± 0.036
  Theta for others: 0.024
  Proportion difference: 0.008

Disease 246 (signature 17, LR=70.99):
  Theta for diagnosed: 0.019 ± 0.022
  Theta for others: 0.016
  Proportion difference: 0.003
✓ Saved MGB initialized model to: mgb_model_initialized.pt

============================================================
✓ All initialization complete!
============================================================