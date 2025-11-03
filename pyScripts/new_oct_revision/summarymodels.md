Index(['eid', 'Sex', 'Dm_Any', 'Dm_censor_age', 'Ht_Any', 'Ht_censor_age',
       'Cad_Any', 'Cad_censor_age', 'HyperLip_Any', 'HyperLip_censor_age',
       'DmT1_Any', 'DmT1_censor_age', 'Enrollment_Date', 'age_enrolled',
       'birth_year', 'Birthdate', 'SBP', 'tchol', 'hdl', 'SmokingStatusv2',
       'pce_goff', 'race', 'antihtnbase', 'enrollment', 'age_at_enroll',
       'prev_dm', 'prev_dm1', 'prev_ht', 'prev_hl', 'prev_cad', 'bmi', 'CAD',
       'LDL_SF', 'BMI', 'T2D', 'pce', 'prevent_base_ascvd_risk',
       'enrollment_year'],
      dtype='object')
================================================================================
STATIN EFFECT REPRODUCTION TESTING
================================================================================
Loading data...
Loading FIXED treatment patterns...
/Users/sarahurbut/dtwin_noulli/test_statin_effects.py:83: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  Y = torch.load('/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/data_for_running/Y_tensor.pt')
...
  95% CI: 1.448 - 2.171
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 80.9% increase in risk

============================================================
TESTING: Quad sig [4, 7, 11, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 7, 11, 18] + High-risk:
  Hazard Ratio: 1.719
  95% CI: 1.375 - 2.063
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 71.9% increase in risk

============================================================
TESTING: Quad sig [4, 7, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 7, 13, 15] + High-risk:
  Hazard Ratio: 1.752
  95% CI: 1.402 - 2.103
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 75.2% increase in risk

============================================================
TESTING: Quad sig [4, 7, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 7, 15, 16] + High-risk:
  Hazard Ratio: 1.803
  95% CI: 1.443 - 2.164
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 80.3% increase in risk

============================================================
TESTING: Quad sig [4, 8, 9, 10] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 8, 9, 10] + High-risk:
  Hazard Ratio: 1.516
  95% CI: 1.213 - 1.819
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 51.6% increase in risk

============================================================
TESTING: Quad sig [4, 8, 10, 11] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 8, 10, 11] + High-risk:
  Hazard Ratio: 1.568
  95% CI: 1.254 - 1.881
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 56.8% increase in risk

============================================================
TESTING: Quad sig [4, 8, 11, 13] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 8, 11, 13] + High-risk:
  Hazard Ratio: 1.624
  95% CI: 1.299 - 1.948
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 62.4% increase in risk

============================================================
TESTING: Quad sig [4, 8, 12, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 8, 12, 16] + High-risk:
  Hazard Ratio: 1.648
  95% CI: 1.319 - 1.978
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 64.8% increase in risk

============================================================
TESTING: Quad sig [4, 8, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 8, 14, 15] + High-risk:
  Hazard Ratio: 1.528
  95% CI: 1.223 - 1.834
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 52.8% increase in risk

============================================================
TESTING: Quad sig [4, 8, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 8, 16, 18] + High-risk:
  Hazard Ratio: 1.803
  95% CI: 1.442 - 2.163
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 80.3% increase in risk

============================================================
TESTING: Quad sig [4, 9, 10, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 9, 10, 16] + High-risk:
  Hazard Ratio: 1.702
  95% CI: 1.362 - 2.043
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 70.2% increase in risk

============================================================
TESTING: Quad sig [4, 9, 11, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 9, 11, 18] + High-risk:
  Hazard Ratio: 1.633
  95% CI: 1.307 - 1.960
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.3% increase in risk

============================================================
TESTING: Quad sig [4, 9, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 9, 13, 15] + High-risk:
  Hazard Ratio: 1.678
  95% CI: 1.342 - 2.014
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 67.8% increase in risk

============================================================
TESTING: Quad sig [4, 9, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 9, 15, 16] + High-risk:
  Hazard Ratio: 1.770
  95% CI: 1.416 - 2.124
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.0% increase in risk

============================================================
TESTING: Quad sig [4, 10, 11, 12] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 10, 11, 12] + High-risk:
  Hazard Ratio: 1.692
  95% CI: 1.354 - 2.030
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 69.2% increase in risk

============================================================
TESTING: Quad sig [4, 10, 12, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 10, 12, 15] + High-risk:
  Hazard Ratio: 1.559
  95% CI: 1.247 - 1.871
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 55.9% increase in risk

============================================================
TESTING: Quad sig [4, 10, 13, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 10, 13, 19] + High-risk:
  Hazard Ratio: 1.809
  95% CI: 1.447 - 2.171
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 80.9% increase in risk

============================================================
TESTING: Quad sig [4, 10, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 10, 16, 17] + High-risk:
  Hazard Ratio: 1.830
  95% CI: 1.464 - 2.196
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 83.0% increase in risk

============================================================
TESTING: Quad sig [4, 11, 12, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 11, 12, 17] + High-risk:
  Hazard Ratio: 1.710
  95% CI: 1.368 - 2.052
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 71.0% increase in risk

============================================================
TESTING: Quad sig [4, 11, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 11, 14, 16] + High-risk:
  Hazard Ratio: 1.746
  95% CI: 1.397 - 2.095
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 74.6% increase in risk

============================================================
TESTING: Quad sig [4, 11, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 11, 16, 19] + High-risk:
  Hazard Ratio: 1.843
  95% CI: 1.474 - 2.212
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 84.3% increase in risk

============================================================
TESTING: Quad sig [4, 12, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 12, 14, 15] + High-risk:
  Hazard Ratio: 1.659
  95% CI: 1.327 - 1.991
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 65.9% increase in risk

============================================================
TESTING: Quad sig [4, 12, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 12, 16, 18] + High-risk:
  Hazard Ratio: 1.680
  95% CI: 1.344 - 2.016
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 68.0% increase in risk

============================================================
TESTING: Quad sig [4, 13, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 13, 15, 16] + High-risk:
  Hazard Ratio: 1.823
  95% CI: 1.458 - 2.187
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 82.3% increase in risk

============================================================
TESTING: Quad sig [4, 14, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 14, 15, 16] + High-risk:
  Hazard Ratio: 1.743
  95% CI: 1.394 - 2.092
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 74.3% increase in risk

============================================================
TESTING: Quad sig [4, 15, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [4, 15, 16, 17] + High-risk:
  Hazard Ratio: 1.767
  95% CI: 1.413 - 2.120
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 76.7% increase in risk

============================================================
TESTING: Quad sig [5, 6, 7, 8] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 7, 8] + High-risk:
  Hazard Ratio: 1.615
  95% CI: 1.292 - 1.938
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 61.5% increase in risk

============================================================
TESTING: Quad sig [5, 6, 7, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 7, 18] + High-risk:
  Hazard Ratio: 1.637
  95% CI: 1.309 - 1.964
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.7% increase in risk

============================================================
TESTING: Quad sig [5, 6, 8, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 8, 17] + High-risk:
  Hazard Ratio: 1.534
  95% CI: 1.228 - 1.841
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 53.4% increase in risk

============================================================
TESTING: Quad sig [5, 6, 9, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 9, 17] + High-risk:
  Hazard Ratio: 1.551
  95% CI: 1.241 - 1.862
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 55.1% increase in risk

============================================================
TESTING: Quad sig [5, 6, 10, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 10, 18] + High-risk:
  Hazard Ratio: 1.544
  95% CI: 1.235 - 1.853
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 54.4% increase in risk

============================================================
TESTING: Quad sig [5, 6, 12, 13] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 12, 13] + High-risk:
  Hazard Ratio: 1.585
  95% CI: 1.268 - 1.902
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 58.5% increase in risk

============================================================
TESTING: Quad sig [5, 6, 13, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 13, 17] + High-risk:
  Hazard Ratio: 1.631
  95% CI: 1.305 - 1.957
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.1% increase in risk

============================================================
TESTING: Quad sig [5, 6, 15, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 6, 15, 18] + High-risk:
  Hazard Ratio: 1.494
  95% CI: 1.196 - 1.793
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 49.4% increase in risk

============================================================
TESTING: Quad sig [5, 7, 8, 11] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 8, 11] + High-risk:
  Hazard Ratio: 1.607
  95% CI: 1.285 - 1.928
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 60.7% increase in risk

============================================================
TESTING: Quad sig [5, 7, 9, 11] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 9, 11] + High-risk:
  Hazard Ratio: 1.632
  95% CI: 1.306 - 1.959
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.2% increase in risk

============================================================
TESTING: Quad sig [5, 7, 10, 12] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 10, 12] + High-risk:
  Hazard Ratio: 1.759
  95% CI: 1.407 - 2.111
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 75.9% increase in risk

============================================================
TESTING: Quad sig [5, 7, 11, 14] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 11, 14] + High-risk:
  Hazard Ratio: 1.694
  95% CI: 1.355 - 2.033
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 69.4% increase in risk

============================================================
TESTING: Quad sig [5, 7, 12, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 12, 17] + High-risk:
  Hazard Ratio: 1.621
  95% CI: 1.297 - 1.946
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 62.1% increase in risk

============================================================
TESTING: Quad sig [5, 7, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 14, 16] + High-risk:
  Hazard Ratio: 1.702
  95% CI: 1.362 - 2.042
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 70.2% increase in risk

============================================================
TESTING: Quad sig [5, 7, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 7, 16, 19] + High-risk:
  Hazard Ratio: 1.673
  95% CI: 1.338 - 2.007
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 67.3% increase in risk

============================================================
TESTING: Quad sig [5, 8, 9, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 8, 9, 16] + High-risk:
  Hazard Ratio: 1.466
  95% CI: 1.173 - 1.760
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 46.6% increase in risk

============================================================
TESTING: Quad sig [5, 8, 10, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 8, 10, 17] + High-risk:
  Hazard Ratio: 1.555
  95% CI: 1.244 - 1.867
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 55.5% increase in risk

============================================================
TESTING: Quad sig [5, 8, 11, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 8, 11, 19] + High-risk:
  Hazard Ratio: 1.549
  95% CI: 1.239 - 1.859
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 54.9% increase in risk

============================================================
TESTING: Quad sig [5, 8, 13, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 8, 13, 16] + High-risk:
  Hazard Ratio: 1.557
  95% CI: 1.246 - 1.868
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 55.7% increase in risk

============================================================
TESTING: Quad sig [5, 8, 15, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 8, 15, 17] + High-risk:
  Hazard Ratio: 1.506
  95% CI: 1.204 - 1.807
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 50.6% increase in risk

============================================================
TESTING: Quad sig [5, 9, 10, 12] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 9, 10, 12] + High-risk:
  Hazard Ratio: 1.585
  95% CI: 1.268 - 1.901
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 58.5% increase in risk

============================================================
TESTING: Quad sig [5, 9, 11, 14] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 9, 11, 14] + High-risk:
  Hazard Ratio: 1.573
  95% CI: 1.258 - 1.888
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 57.3% increase in risk

============================================================
TESTING: Quad sig [5, 9, 12, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 9, 12, 17] + High-risk:
  Hazard Ratio: 1.623
  95% CI: 1.299 - 1.948
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 62.3% increase in risk

============================================================
TESTING: Quad sig [5, 9, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 9, 14, 16] + High-risk:
  Hazard Ratio: 1.611
  95% CI: 1.289 - 1.933
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 61.1% increase in risk

============================================================
TESTING: Quad sig [5, 9, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 9, 16, 19] + High-risk:
  Hazard Ratio: 1.518
  95% CI: 1.214 - 1.821
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 51.8% increase in risk

============================================================
TESTING: Quad sig [5, 10, 11, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 10, 11, 18] + High-risk:
  Hazard Ratio: 1.517
  95% CI: 1.214 - 1.820
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 51.7% increase in risk

============================================================
TESTING: Quad sig [5, 10, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 10, 13, 15] + High-risk:
  Hazard Ratio: 1.504
  95% CI: 1.203 - 1.805
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 50.4% increase in risk

============================================================
TESTING: Quad sig [5, 10, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 10, 15, 16] + High-risk:
  Hazard Ratio: 1.555
  95% CI: 1.244 - 1.866
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 55.5% increase in risk

============================================================
TESTING: Quad sig [5, 11, 12, 13] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 11, 12, 13] + High-risk:
  Hazard Ratio: 1.612
  95% CI: 1.289 - 1.934
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 61.2% increase in risk

============================================================
TESTING: Quad sig [5, 11, 13, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 11, 13, 17] + High-risk:
  Hazard Ratio: 1.646
  95% CI: 1.317 - 1.975
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 64.6% increase in risk

============================================================
TESTING: Quad sig [5, 11, 15, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 11, 15, 18] + High-risk:
  Hazard Ratio: 1.531
  95% CI: 1.225 - 1.837
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 53.1% increase in risk

============================================================
TESTING: Quad sig [5, 12, 13, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 12, 13, 16] + High-risk:
  Hazard Ratio: 1.631
  95% CI: 1.305 - 1.957
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.1% increase in risk

============================================================
TESTING: Quad sig [5, 12, 15, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 12, 15, 17] + High-risk:
  Hazard Ratio: 1.606
  95% CI: 1.285 - 1.927
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 60.6% increase in risk

============================================================
TESTING: Quad sig [5, 13, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 13, 14, 16] + High-risk:
  Hazard Ratio: 1.707
  95% CI: 1.365 - 2.048
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 70.7% increase in risk

============================================================
TESTING: Quad sig [5, 13, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 13, 16, 19] + High-risk:
  Hazard Ratio: 1.640
  95% CI: 1.312 - 1.968
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 64.0% increase in risk

============================================================
TESTING: Quad sig [5, 14, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 14, 16, 19] + High-risk:
  Hazard Ratio: 1.622
  95% CI: 1.297 - 1.946
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 62.2% increase in risk

============================================================
TESTING: Quad sig [5, 16, 17, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [5, 16, 17, 18] + High-risk:
  Hazard Ratio: 1.585
  95% CI: 1.268 - 1.902
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 58.5% increase in risk

============================================================
TESTING: Quad sig [6, 7, 8, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 7, 8, 15] + High-risk:
  Hazard Ratio: 1.682
  95% CI: 1.346 - 2.018
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 68.2% increase in risk

============================================================
TESTING: Quad sig [6, 7, 9, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 7, 9, 15] + High-risk:
  Hazard Ratio: 1.804
  95% CI: 1.443 - 2.165
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 80.4% increase in risk

============================================================
TESTING: Quad sig [6, 7, 10, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 7, 10, 16] + High-risk:
  Hazard Ratio: 1.769
  95% CI: 1.415 - 2.122
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 76.9% increase in risk

============================================================
TESTING: Quad sig [6, 7, 11, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 7, 11, 18] + High-risk:
  Hazard Ratio: 1.665
  95% CI: 1.332 - 1.997
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 66.5% increase in risk

============================================================
TESTING: Quad sig [6, 7, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 7, 13, 15] + High-risk:
  Hazard Ratio: 1.788
  95% CI: 1.430 - 2.146
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 78.8% increase in risk

============================================================
TESTING: Quad sig [6, 7, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 7, 15, 16] + High-risk:
  Hazard Ratio: 1.743
  95% CI: 1.395 - 2.092
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 74.3% increase in risk

============================================================
TESTING: Quad sig [6, 8, 9, 10] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 8, 9, 10] + High-risk:
  Hazard Ratio: 1.529
  95% CI: 1.223 - 1.835
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 52.9% increase in risk

============================================================
TESTING: Quad sig [6, 8, 10, 11] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 8, 10, 11] + High-risk:
  Hazard Ratio: 1.572
  95% CI: 1.258 - 1.887
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 57.2% increase in risk

============================================================
TESTING: Quad sig [6, 8, 11, 13] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 8, 11, 13] + High-risk:
  Hazard Ratio: 1.709
  95% CI: 1.367 - 2.051
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 70.9% increase in risk

============================================================
TESTING: Quad sig [6, 8, 12, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 8, 12, 16] + High-risk:
  Hazard Ratio: 1.630
  95% CI: 1.304 - 1.956
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.0% increase in risk

============================================================
TESTING: Quad sig [6, 8, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 8, 14, 15] + High-risk:
  Hazard Ratio: 1.542
  95% CI: 1.234 - 1.850
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 54.2% increase in risk

============================================================
TESTING: Quad sig [6, 8, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 8, 16, 18] + High-risk:
  Hazard Ratio: 1.636
  95% CI: 1.309 - 1.964
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.6% increase in risk

============================================================
TESTING: Quad sig [6, 9, 10, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 9, 10, 16] + High-risk:
  Hazard Ratio: 1.621
  95% CI: 1.297 - 1.946
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 62.1% increase in risk

============================================================
TESTING: Quad sig [6, 9, 11, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 9, 11, 18] + High-risk:
  Hazard Ratio: 1.549
  95% CI: 1.239 - 1.858
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 54.9% increase in risk

============================================================
TESTING: Quad sig [6, 9, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 9, 13, 15] + High-risk:
  Hazard Ratio: 1.618
  95% CI: 1.295 - 1.942
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 61.8% increase in risk

============================================================
TESTING: Quad sig [6, 9, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 9, 15, 16] + High-risk:
  Hazard Ratio: 1.615
  95% CI: 1.292 - 1.938
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 61.5% increase in risk

============================================================
TESTING: Quad sig [6, 10, 11, 12] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 10, 11, 12] + High-risk:
  Hazard Ratio: 1.544
  95% CI: 1.235 - 1.853
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 54.4% increase in risk

============================================================
TESTING: Quad sig [6, 10, 12, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 10, 12, 15] + High-risk:
  Hazard Ratio: 1.482
  95% CI: 1.186 - 1.778
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 48.2% increase in risk

============================================================
TESTING: Quad sig [6, 10, 13, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 10, 13, 19] + High-risk:
  Hazard Ratio: 1.638
  95% CI: 1.310 - 1.965
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 63.8% increase in risk

============================================================
TESTING: Quad sig [6, 10, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 10, 16, 17] + High-risk:
  Hazard Ratio: 1.839
  95% CI: 1.472 - 2.207
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 83.9% increase in risk

============================================================
TESTING: Quad sig [6, 11, 12, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 11, 12, 17] + High-risk:
  Hazard Ratio: 1.686
  95% CI: 1.348 - 2.023
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 68.6% increase in risk

============================================================
TESTING: Quad sig [6, 11, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 11, 14, 16] + High-risk:
  Hazard Ratio: 1.744
  95% CI: 1.395 - 2.093
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 74.4% increase in risk

============================================================
TESTING: Quad sig [6, 11, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 11, 16, 19] + High-risk:
  Hazard Ratio: 1.697
  95% CI: 1.357 - 2.036
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 69.7% increase in risk

============================================================
TESTING: Quad sig [6, 12, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 12, 14, 15] + High-risk:
  Hazard Ratio: 1.550
  95% CI: 1.240 - 1.860
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 55.0% increase in risk

============================================================
TESTING: Quad sig [6, 12, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 12, 16, 18] + High-risk:
  Hazard Ratio: 1.722
  95% CI: 1.378 - 2.067
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 72.2% increase in risk

============================================================
TESTING: Quad sig [6, 13, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 13, 15, 16] + High-risk:
  Hazard Ratio: 1.731
  95% CI: 1.384 - 2.077
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 73.1% increase in risk

============================================================
TESTING: Quad sig [6, 14, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 14, 15, 16] + High-risk:
  Hazard Ratio: 1.731
  95% CI: 1.385 - 2.077
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 73.1% increase in risk

============================================================
TESTING: Quad sig [6, 15, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [6, 15, 16, 17] + High-risk:
  Hazard Ratio: 1.678
  95% CI: 1.343 - 2.014
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 67.8% increase in risk

============================================================
TESTING: Quad sig [7, 8, 9, 10] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 8, 9, 10] + High-risk:
  Hazard Ratio: 1.684
  95% CI: 1.347 - 2.021
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 68.4% increase in risk

============================================================
TESTING: Quad sig [7, 8, 10, 11] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 8, 10, 11] + High-risk:
  Hazard Ratio: 1.591
  95% CI: 1.273 - 1.910
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 59.1% increase in risk

============================================================
TESTING: Quad sig [7, 8, 11, 13] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 8, 11, 13] + High-risk:
  Hazard Ratio: 1.757
  95% CI: 1.405 - 2.108
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 75.7% increase in risk

============================================================
TESTING: Quad sig [7, 8, 12, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 8, 12, 16] + High-risk:
  Hazard Ratio: 1.820
  95% CI: 1.456 - 2.184
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 82.0% increase in risk

============================================================
TESTING: Quad sig [7, 8, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 8, 14, 15] + High-risk:
  Hazard Ratio: 1.845
  95% CI: 1.476 - 2.214
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 84.5% increase in risk

============================================================
TESTING: Quad sig [7, 8, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 8, 16, 18] + High-risk:
  Hazard Ratio: 1.886
  95% CI: 1.509 - 2.263
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 88.6% increase in risk

============================================================
TESTING: Quad sig [7, 9, 10, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 9, 10, 16] + High-risk:
  Hazard Ratio: 1.893
  95% CI: 1.514 - 2.272
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 89.3% increase in risk

============================================================
TESTING: Quad sig [7, 9, 11, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 9, 11, 18] + High-risk:
  Hazard Ratio: 1.778
  95% CI: 1.423 - 2.134
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.8% increase in risk

============================================================
TESTING: Quad sig [7, 9, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 9, 13, 15] + High-risk:
  Hazard Ratio: 1.779
  95% CI: 1.423 - 2.134
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.9% increase in risk

============================================================
TESTING: Quad sig [7, 9, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 9, 15, 16] + High-risk:
  Hazard Ratio: 1.883
  95% CI: 1.507 - 2.260
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 88.3% increase in risk

============================================================
TESTING: Quad sig [7, 10, 11, 12] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 10, 11, 12] + High-risk:
  Hazard Ratio: 1.758
  95% CI: 1.407 - 2.110
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 75.8% increase in risk

============================================================
TESTING: Quad sig [7, 10, 12, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 10, 12, 15] + High-risk:
  Hazard Ratio: 1.770
  95% CI: 1.416 - 2.124
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.0% increase in risk

============================================================
TESTING: Quad sig [7, 10, 13, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 10, 13, 19] + High-risk:
  Hazard Ratio: 2.028
  95% CI: 1.622 - 2.434
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 102.8% increase in risk

============================================================
TESTING: Quad sig [7, 10, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 10, 16, 17] + High-risk:
  Hazard Ratio: 1.832
  95% CI: 1.466 - 2.198
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 83.2% increase in risk

============================================================
TESTING: Quad sig [7, 11, 12, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 11, 12, 17] + High-risk:
  Hazard Ratio: 1.786
  95% CI: 1.429 - 2.144
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 78.6% increase in risk

============================================================
TESTING: Quad sig [7, 11, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 11, 14, 16] + High-risk:
  Hazard Ratio: 1.896
  95% CI: 1.517 - 2.276
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 89.6% increase in risk

============================================================
TESTING: Quad sig [7, 11, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 11, 16, 19] + High-risk:
  Hazard Ratio: 1.933
  95% CI: 1.546 - 2.319
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 93.3% increase in risk

============================================================
TESTING: Quad sig [7, 12, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 12, 14, 15] + High-risk:
  Hazard Ratio: 1.804
  95% CI: 1.443 - 2.164
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 80.4% increase in risk

============================================================
TESTING: Quad sig [7, 12, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 12, 16, 18] + High-risk:
  Hazard Ratio: 1.962
  95% CI: 1.570 - 2.354
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 96.2% increase in risk

============================================================
TESTING: Quad sig [7, 13, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 13, 15, 16] + High-risk:
  Hazard Ratio: 1.888
  95% CI: 1.511 - 2.266
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 88.8% increase in risk

============================================================
TESTING: Quad sig [7, 14, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 14, 15, 16] + High-risk:
  Hazard Ratio: 1.903
  95% CI: 1.522 - 2.284
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 90.3% increase in risk

============================================================
TESTING: Quad sig [7, 15, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [7, 15, 16, 17] + High-risk:
  Hazard Ratio: 1.728
  95% CI: 1.382 - 2.073
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 72.8% increase in risk

============================================================
TESTING: Quad sig [8, 9, 10, 11] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 9, 10, 11] + High-risk:
  Hazard Ratio: 1.609
  95% CI: 1.287 - 1.931
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 60.9% increase in risk

============================================================
TESTING: Quad sig [8, 9, 11, 13] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 9, 11, 13] + High-risk:
  Hazard Ratio: 1.618
  95% CI: 1.294 - 1.941
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 61.8% increase in risk

============================================================
TESTING: Quad sig [8, 9, 12, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 9, 12, 16] + High-risk:
  Hazard Ratio: 1.664
  95% CI: 1.331 - 1.997
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 66.4% increase in risk

============================================================
TESTING: Quad sig [8, 9, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 9, 14, 15] + High-risk:
  Hazard Ratio: 1.542
  95% CI: 1.233 - 1.850
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 54.2% increase in risk

============================================================
TESTING: Quad sig [8, 9, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 9, 16, 18] + High-risk:
  Hazard Ratio: 1.770
  95% CI: 1.416 - 2.124
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.0% increase in risk

============================================================
TESTING: Quad sig [8, 10, 11, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 10, 11, 17] + High-risk:
  Hazard Ratio: 1.603
  95% CI: 1.282 - 1.923
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 60.3% increase in risk

============================================================
TESTING: Quad sig [8, 10, 13, 14] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 10, 13, 14] + High-risk:
  Hazard Ratio: 1.596
  95% CI: 1.277 - 1.915
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 59.6% increase in risk

============================================================
TESTING: Quad sig [8, 10, 14, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 10, 14, 19] + High-risk:
  Hazard Ratio: 1.775
  95% CI: 1.420 - 2.130
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.5% increase in risk

============================================================
TESTING: Quad sig [8, 10, 18, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 10, 18, 19] + High-risk:
  Hazard Ratio: 1.678
  95% CI: 1.342 - 2.013
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 67.8% increase in risk

============================================================
TESTING: Quad sig [8, 11, 13, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 11, 13, 16] + High-risk:
  Hazard Ratio: 1.840
  95% CI: 1.472 - 2.208
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 84.0% increase in risk

============================================================
TESTING: Quad sig [8, 11, 15, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 11, 15, 17] + High-risk:
  Hazard Ratio: 1.679
  95% CI: 1.343 - 2.014
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 67.9% increase in risk

============================================================
TESTING: Quad sig [8, 12, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 12, 13, 15] + High-risk:
  Hazard Ratio: 1.668
  95% CI: 1.335 - 2.002
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 66.8% increase in risk

============================================================
TESTING: Quad sig [8, 12, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 12, 15, 16] + High-risk:
  Hazard Ratio: 1.750
  95% CI: 1.400 - 2.100
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 75.0% increase in risk

============================================================
TESTING: Quad sig [8, 13, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 13, 14, 15] + High-risk:
  Hazard Ratio: 1.586
  95% CI: 1.269 - 1.904
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 58.6% increase in risk

============================================================
TESTING: Quad sig [8, 13, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 13, 16, 18] + High-risk:
  Hazard Ratio: 1.709
  95% CI: 1.368 - 2.051
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 70.9% increase in risk

============================================================
TESTING: Quad sig [8, 14, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 14, 16, 18] + High-risk:
  Hazard Ratio: 1.850
  95% CI: 1.480 - 2.220
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 85.0% increase in risk

============================================================
TESTING: Quad sig [8, 15, 18, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [8, 15, 18, 19] + High-risk:
  Hazard Ratio: 1.784
  95% CI: 1.427 - 2.141
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 78.4% increase in risk

============================================================
TESTING: Quad sig [9, 10, 11, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 10, 11, 17] + High-risk:
  Hazard Ratio: 1.660
  95% CI: 1.328 - 1.992
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 66.0% increase in risk

============================================================
TESTING: Quad sig [9, 10, 13, 14] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 10, 13, 14] + High-risk:
  Hazard Ratio: 1.650
  95% CI: 1.320 - 1.980
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 65.0% increase in risk

============================================================
TESTING: Quad sig [9, 10, 14, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 10, 14, 19] + High-risk:
  Hazard Ratio: 1.693
  95% CI: 1.354 - 2.031
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 69.3% increase in risk

============================================================
TESTING: Quad sig [9, 10, 18, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 10, 18, 19] + High-risk:
  Hazard Ratio: 1.768
  95% CI: 1.414 - 2.121
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 76.8% increase in risk

============================================================
TESTING: Quad sig [9, 11, 13, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 11, 13, 16] + High-risk:
  Hazard Ratio: 1.720
  95% CI: 1.376 - 2.064
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 72.0% increase in risk

============================================================
TESTING: Quad sig [9, 11, 15, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 11, 15, 17] + High-risk:
  Hazard Ratio: 1.756
  95% CI: 1.405 - 2.107
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 75.6% increase in risk

============================================================
TESTING: Quad sig [9, 12, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 12, 13, 15] + High-risk:
  Hazard Ratio: 1.669
  95% CI: 1.335 - 2.002
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 66.9% increase in risk

============================================================
TESTING: Quad sig [9, 12, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 12, 15, 16] + High-risk:
  Hazard Ratio: 1.713
  95% CI: 1.370 - 2.056
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 71.3% increase in risk

============================================================
TESTING: Quad sig [9, 13, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 13, 14, 15] + High-risk:
  Hazard Ratio: 1.663
  95% CI: 1.330 - 1.995
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 66.3% increase in risk

============================================================
TESTING: Quad sig [9, 13, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 13, 16, 18] + High-risk:
  Hazard Ratio: 1.894
  95% CI: 1.515 - 2.273
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 89.4% increase in risk

============================================================
TESTING: Quad sig [9, 14, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 14, 16, 18] + High-risk:
  Hazard Ratio: 1.899
  95% CI: 1.519 - 2.279
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 89.9% increase in risk

============================================================
TESTING: Quad sig [9, 15, 18, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [9, 15, 18, 19] + High-risk:
  Hazard Ratio: 1.935
  95% CI: 1.548 - 2.322
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 93.5% increase in risk

============================================================
TESTING: Quad sig [10, 11, 12, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 11, 12, 18] + High-risk:
  Hazard Ratio: 1.579
  95% CI: 1.263 - 1.894
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 57.9% increase in risk

============================================================
TESTING: Quad sig [10, 11, 14, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 11, 14, 17] + High-risk:
  Hazard Ratio: 1.785
  95% CI: 1.428 - 2.142
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 78.5% increase in risk

============================================================
TESTING: Quad sig [10, 11, 17, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 11, 17, 18] + High-risk:
  Hazard Ratio: 1.680
  95% CI: 1.344 - 2.016
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 68.0% increase in risk

============================================================
TESTING: Quad sig [10, 12, 14, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 12, 14, 16] + High-risk:
  Hazard Ratio: 1.797
  95% CI: 1.437 - 2.156
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 79.7% increase in risk

============================================================
TESTING: Quad sig [10, 12, 16, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 12, 16, 19] + High-risk:
  Hazard Ratio: 1.736
  95% CI: 1.389 - 2.083
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 73.6% increase in risk

============================================================
TESTING: Quad sig [10, 13, 15, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 13, 15, 17] + High-risk:
  Hazard Ratio: 1.642
  95% CI: 1.314 - 1.971
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 64.2% increase in risk

============================================================
TESTING: Quad sig [10, 14, 15, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 14, 15, 17] + High-risk:
  Hazard Ratio: 1.644
  95% CI: 1.315 - 1.972
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 64.4% increase in risk

============================================================
TESTING: Quad sig [10, 15, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [10, 15, 16, 18] + High-risk:
  Hazard Ratio: 1.760
  95% CI: 1.408 - 2.112
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 76.0% increase in risk

============================================================
TESTING: Quad sig [11, 12, 13, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [11, 12, 13, 15] + High-risk:
  Hazard Ratio: 1.646
  95% CI: 1.317 - 1.975
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 64.6% increase in risk

============================================================
TESTING: Quad sig [11, 12, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [11, 12, 15, 16] + High-risk:
  Hazard Ratio: 1.726
  95% CI: 1.381 - 2.072
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 72.6% increase in risk

============================================================
TESTING: Quad sig [11, 13, 14, 15] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [11, 13, 14, 15] + High-risk:
  Hazard Ratio: 1.793
  95% CI: 1.434 - 2.151
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 79.3% increase in risk

============================================================
TESTING: Quad sig [11, 13, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [11, 13, 16, 18] + High-risk:
  Hazard Ratio: 1.788
  95% CI: 1.430 - 2.145
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 78.8% increase in risk

============================================================
TESTING: Quad sig [11, 14, 16, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [11, 14, 16, 18] + High-risk:
  Hazard Ratio: 1.848
  95% CI: 1.478 - 2.217
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 84.8% increase in risk

============================================================
TESTING: Quad sig [11, 15, 18, 19] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [11, 15, 18, 19] + High-risk:
  Hazard Ratio: 1.845
  95% CI: 1.476 - 2.214
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 84.5% increase in risk

============================================================
TESTING: Quad sig [12, 13, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [12, 13, 15, 16] + High-risk:
  Hazard Ratio: 1.777
  95% CI: 1.422 - 2.133
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.7% increase in risk

============================================================
TESTING: Quad sig [12, 14, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [12, 14, 15, 16] + High-risk:
  Hazard Ratio: 1.784
  95% CI: 1.427 - 2.140
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 78.4% increase in risk

============================================================
TESTING: Quad sig [12, 15, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [12, 15, 16, 17] + High-risk:
  Hazard Ratio: 1.814
  95% CI: 1.452 - 2.177
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 81.4% increase in risk

============================================================
TESTING: Quad sig [13, 14, 15, 16] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [13, 14, 15, 16] + High-risk:
  Hazard Ratio: 1.878
  95% CI: 1.503 - 2.254
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 87.8% increase in risk

============================================================
TESTING: Quad sig [13, 15, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [13, 15, 16, 17] + High-risk:
  Hazard Ratio: 1.889
  95% CI: 1.511 - 2.267
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 88.9% increase in risk

============================================================
TESTING: Quad sig [14, 15, 16, 17] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [14, 15, 16, 17] + High-risk:
  Hazard Ratio: 1.773
  95% CI: 1.418 - 2.127
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 77.3% increase in risk

============================================================
TESTING: Quad sig [15, 16, 17, 18] + High-risk
============================================================

Building control cohort...
Control patients with 10-year follow-up: 116,129
Building treated features...
After PCE-Goff exclusion: 35875/48420 patients remaining
Treated patients after exclusions: 29,706
Building control features...
After PCE-Goff exclusion: 91203/116129 patients remaining
Control patients after exclusions: 89,636
After additional exclusions - Treated: 19750, Control: 28064
Using signatures + clinical matching: 51 features
Performing nearest_neighbor matching...
Matched pairs: 19750

RESULTS for Quad sig [15, 16, 17, 18] + High-risk:
  Hazard Ratio: 1.856
  95% CI: 1.485 - 2.227
  P-value: 0.0000
  Sample size: 19750 matched pairs
  ❌ HARMFUL: 85.6% increase in risk

================================================================================
SUMMARY OF ALL RESULTS
================================================================================
Baseline (PCE-Goff exclusion)       | HR: 1.425 | 42.5% increase | ❌ HARMFUL
Clinical-only matching              | HR: 1.549 | 54.9% increase | ❌ HARMFUL
Single signature [5] + NN           | HR: 1.383 | 38.3% increase | ❌ HARMFUL
Single signature [5] + PS           | HR: 1.390 | 39.0% increase | ❌ HARMFUL (PS)
Signature [5] AUC + NN              | HR: 1.597 | 59.7% increase | ❌ HARMFUL (AUC)
Signature [5] AUC + PS              | HR: 1.608 | 60.8% increase | ❌ HARMFUL (AUC) (PS)
Age 40-75 restriction               | HR: 1.525 | 52.5% increase | ❌ HARMFUL
High-risk only (PCE > 7.5%)         | HR: 1.368 | 36.8% increase | ❌ HARMFUL
Primary prevention (no diabetes)    | HR: 1.710 | 71.0% increase | ❌ HARMFUL
Combined: Age 40-75 + High-risk     | HR: 1.368 | 36.8% increase | ❌ HARMFUL
Combined: Age 40-75 + Primary prevention | HR: 1.710 | 71.0% increase | ❌ HARMFUL
Single sig [5] + Age 40-75 + PS     | HR: 1.390 | 39.0% increase | ❌ HARMFUL (PS)
Single sig [5] + High-risk + PS     | HR: 1.497 | 49.7% increase | ❌ HARMFUL (PS)
Very high-risk only (PCE > 10%)     | HR: 1.358 | 35.8% increase | ❌ HARMFUL
Age 50-70 + High-risk               | HR: 1.395 | 39.5% increase | ❌ HARMFUL
Clinical-only + Age 50-70 + High-risk | HR: 1.445 | 44.5% increase | ❌ HARMFUL
No signatures + Age 50-70 + High-risk | HR: 1.445 | 44.5% increase | ❌ HARMFUL
Single sig [0] + High-risk          | HR: 1.613 | 61.3% increase | ❌ HARMFUL
Single sig [1] + High-risk          | HR: 1.518 | 51.8% increase | ❌ HARMFUL
Single sig [2] + High-risk          | HR: 1.478 | 47.8% increase | ❌ HARMFUL
Single sig [3] + High-risk          | HR: 1.550 | 55.0% increase | ❌ HARMFUL
Single sig [4] + High-risk          | HR: 1.489 | 48.9% increase | ❌ HARMFUL
Single sig [5] + High-risk          | HR: 1.274 | 27.4% increase | ❌ HARMFUL
Single sig [6] + High-risk          | HR: 1.462 | 46.2% increase | ❌ HARMFUL
Single sig [7] + High-risk          | HR: 1.528 | 52.8% increase | ❌ HARMFUL
Single sig [8] + High-risk          | HR: 1.441 | 44.1% increase | ❌ HARMFUL
Single sig [9] + High-risk          | HR: 1.477 | 47.7% increase | ❌ HARMFUL
Single sig [10] + High-risk         | HR: 1.368 | 36.8% increase | ❌ HARMFUL
Single sig [15] + High-risk         | HR: 1.353 | 35.3% increase | ❌ HARMFUL
Single sig [20] + High-risk         | HR: 1.493 | 49.3% increase | ❌ HARMFUL
PCE Moderate Risk (5-10%)           | HR: 1.753 | 75.3% increase | ❌ HARMFUL
PCE High Risk (10-20%)              | HR: 1.574 | 57.4% increase | ❌ HARMFUL
PCE Very High Risk (>20%)           | HR: 1.279 | 27.9% increase | ❌ HARMFUL
Sig[5] + PCE High Risk (10-20%)     | HR: 1.454 | 45.4% increase | ❌ HARMFUL
Clinical-only + PCE High Risk       | HR: 1.762 | 76.2% increase | ❌ HARMFUL
High Adherence (10+ scripts) + High Risk | HR: 1.347 | 34.7% increase | ❌ HARMFUL
Very High Adherence (20+ scripts) + High Risk | HR: 1.316 | 31.6% increase | ❌ HARMFUL
Sig[5] + High Adherence (10+ scripts) + PCE High | HR: 1.442 | 44.2% increase | ❌ HARMFUL
Sig[5] + Very High Adherence (20+ scripts) + PCE High | HR: 1.406 | 40.6% increase | ❌ HARMFUL
Propensity Score + High Risk        | HR: 1.388 | 38.8% increase | ❌ HARMFUL (PS)
Propensity Score + PCE High Risk    | HR: 1.622 | 62.2% increase | ❌ HARMFUL (PS)
Sig[5] + Propensity Score + High Risk | HR: 1.497 | 49.7% increase | ❌ HARMFUL (PS)
Non-Restrictive Follow-up + High Risk | HR: 1.635 | 63.5% increase | ❌ HARMFUL
Sig[5] + Non-Restrictive Follow-up + PCE High | HR: 1.421 | 42.1% increase | ❌ HARMFUL
Best Combo: Sig[5] + 10+ scripts + PS + PCE Very High | HR: 1.048 | 4.8% increase | ❌ HARMFUL (PS)
AUC Sig[5] + High Adherence (10+) + PCE Very High | HR: 1.230 | 23.0% increase | ❌ HARMFUL (AUC)
Double sig [0,1] + High-risk        | HR: 1.734 | 73.4% increase | ❌ HARMFUL
Double sig [0,2] + High-risk        | HR: 1.686 | 68.6% increase | ❌ HARMFUL
Double sig [0,3] + High-risk        | HR: 1.740 | 74.0% increase | ❌ HARMFUL
Double sig [0,4] + High-risk        | HR: 1.704 | 70.4% increase | ❌ HARMFUL
Double sig [0,5] + High-risk        | HR: 1.414 | 41.4% increase | ❌ HARMFUL
Double sig [0,6] + High-risk        | HR: 1.639 | 63.9% increase | ❌ HARMFUL
Double sig [0,7] + High-risk        | HR: 1.871 | 87.1% increase | ❌ HARMFUL
Double sig [0,8] + High-risk        | HR: 1.589 | 58.9% increase | ❌ HARMFUL
Double sig [0,9] + High-risk        | HR: 1.692 | 69.2% increase | ❌ HARMFUL
Double sig [0,10] + High-risk       | HR: 1.607 | 60.7% increase | ❌ HARMFUL
Double sig [0,11] + High-risk       | HR: 1.544 | 54.4% increase | ❌ HARMFUL
Double sig [0,12] + High-risk       | HR: 1.655 | 65.5% increase | ❌ HARMFUL
Double sig [0,13] + High-risk       | HR: 1.761 | 76.1% increase | ❌ HARMFUL
Double sig [0,14] + High-risk       | HR: 1.731 | 73.1% increase | ❌ HARMFUL
Double sig [0,15] + High-risk       | HR: 1.546 | 54.6% increase | ❌ HARMFUL
Double sig [0,16] + High-risk       | HR: 1.758 | 75.8% increase | ❌ HARMFUL
Double sig [0,17] + High-risk       | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Double sig [0,18] + High-risk       | HR: 1.673 | 67.3% increase | ❌ HARMFUL
Double sig [0,19] + High-risk       | HR: 1.751 | 75.1% increase | ❌ HARMFUL
Double sig [1,2] + High-risk        | HR: 1.701 | 70.1% increase | ❌ HARMFUL
Double sig [1,3] + High-risk        | HR: 1.552 | 55.2% increase | ❌ HARMFUL
Double sig [1,4] + High-risk        | HR: 1.608 | 60.8% increase | ❌ HARMFUL
Double sig [1,5] + High-risk        | HR: 1.457 | 45.7% increase | ❌ HARMFUL
Double sig [1,6] + High-risk        | HR: 1.480 | 48.0% increase | ❌ HARMFUL
Double sig [1,7] + High-risk        | HR: 1.588 | 58.8% increase | ❌ HARMFUL
Double sig [1,8] + High-risk        | HR: 1.533 | 53.3% increase | ❌ HARMFUL
Double sig [1,9] + High-risk        | HR: 1.513 | 51.3% increase | ❌ HARMFUL
Double sig [1,10] + High-risk       | HR: 1.481 | 48.1% increase | ❌ HARMFUL
Double sig [1,11] + High-risk       | HR: 1.576 | 57.6% increase | ❌ HARMFUL
Double sig [1,12] + High-risk       | HR: 1.620 | 62.0% increase | ❌ HARMFUL
Double sig [1,13] + High-risk       | HR: 1.581 | 58.1% increase | ❌ HARMFUL
Double sig [1,14] + High-risk       | HR: 1.661 | 66.1% increase | ❌ HARMFUL
Double sig [1,15] + High-risk       | HR: 1.485 | 48.5% increase | ❌ HARMFUL
Double sig [1,16] + High-risk       | HR: 1.830 | 83.0% increase | ❌ HARMFUL
Double sig [1,17] + High-risk       | HR: 1.625 | 62.5% increase | ❌ HARMFUL
Double sig [1,18] + High-risk       | HR: 1.631 | 63.1% increase | ❌ HARMFUL
Double sig [1,19] + High-risk       | HR: 1.535 | 53.5% increase | ❌ HARMFUL
Double sig [2,3] + High-risk        | HR: 1.718 | 71.8% increase | ❌ HARMFUL
Double sig [2,4] + High-risk        | HR: 1.629 | 62.9% increase | ❌ HARMFUL
Double sig [2,5] + High-risk        | HR: 1.333 | 33.3% increase | ❌ HARMFUL
Double sig [2,6] + High-risk        | HR: 1.469 | 46.9% increase | ❌ HARMFUL
Double sig [2,7] + High-risk        | HR: 1.568 | 56.8% increase | ❌ HARMFUL
Double sig [2,8] + High-risk        | HR: 1.519 | 51.9% increase | ❌ HARMFUL
Double sig [2,9] + High-risk        | HR: 1.439 | 43.9% increase | ❌ HARMFUL
Double sig [2,10] + High-risk       | HR: 1.549 | 54.9% increase | ❌ HARMFUL
Double sig [2,11] + High-risk       | HR: 1.539 | 53.9% increase | ❌ HARMFUL
Double sig [2,12] + High-risk       | HR: 1.620 | 62.0% increase | ❌ HARMFUL
Double sig [2,13] + High-risk       | HR: 1.563 | 56.3% increase | ❌ HARMFUL
Double sig [2,14] + High-risk       | HR: 1.523 | 52.3% increase | ❌ HARMFUL
Double sig [2,15] + High-risk       | HR: 1.445 | 44.5% increase | ❌ HARMFUL
Double sig [2,16] + High-risk       | HR: 1.676 | 67.6% increase | ❌ HARMFUL
Double sig [2,17] + High-risk       | HR: 1.600 | 60.0% increase | ❌ HARMFUL
Double sig [2,18] + High-risk       | HR: 1.507 | 50.7% increase | ❌ HARMFUL
Double sig [2,19] + High-risk       | HR: 1.701 | 70.1% increase | ❌ HARMFUL
Double sig [3,4] + High-risk        | HR: 1.659 | 65.9% increase | ❌ HARMFUL
Double sig [3,5] + High-risk        | HR: 1.446 | 44.6% increase | ❌ HARMFUL
Double sig [3,6] + High-risk        | HR: 1.538 | 53.8% increase | ❌ HARMFUL
Double sig [3,7] + High-risk        | HR: 1.711 | 71.1% increase | ❌ HARMFUL
Double sig [3,8] + High-risk        | HR: 1.540 | 54.0% increase | ❌ HARMFUL
Double sig [3,9] + High-risk        | HR: 1.552 | 55.2% increase | ❌ HARMFUL
Double sig [3,10] + High-risk       | HR: 1.606 | 60.6% increase | ❌ HARMFUL
Double sig [3,11] + High-risk       | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Double sig [3,12] + High-risk       | HR: 1.641 | 64.1% increase | ❌ HARMFUL
Double sig [3,13] + High-risk       | HR: 1.613 | 61.3% increase | ❌ HARMFUL
Double sig [3,14] + High-risk       | HR: 1.650 | 65.0% increase | ❌ HARMFUL
Double sig [3,15] + High-risk       | HR: 1.581 | 58.1% increase | ❌ HARMFUL
Double sig [3,16] + High-risk       | HR: 1.816 | 81.6% increase | ❌ HARMFUL
Double sig [3,17] + High-risk       | HR: 1.593 | 59.3% increase | ❌ HARMFUL
Double sig [3,18] + High-risk       | HR: 1.708 | 70.8% increase | ❌ HARMFUL
Double sig [3,19] + High-risk       | HR: 1.604 | 60.4% increase | ❌ HARMFUL
Double sig [4,5] + High-risk        | HR: 1.364 | 36.4% increase | ❌ HARMFUL
Double sig [4,6] + High-risk        | HR: 1.530 | 53.0% increase | ❌ HARMFUL
Double sig [4,7] + High-risk        | HR: 1.577 | 57.7% increase | ❌ HARMFUL
Double sig [4,8] + High-risk        | HR: 1.495 | 49.5% increase | ❌ HARMFUL
Double sig [4,9] + High-risk        | HR: 1.533 | 53.3% increase | ❌ HARMFUL
Double sig [4,10] + High-risk       | HR: 1.413 | 41.3% increase | ❌ HARMFUL
Double sig [4,11] + High-risk       | HR: 1.539 | 53.9% increase | ❌ HARMFUL
Double sig [4,12] + High-risk       | HR: 1.509 | 50.9% increase | ❌ HARMFUL
Double sig [4,13] + High-risk       | HR: 1.542 | 54.2% increase | ❌ HARMFUL
Double sig [4,14] + High-risk       | HR: 1.638 | 63.8% increase | ❌ HARMFUL
Double sig [4,15] + High-risk       | HR: 1.355 | 35.5% increase | ❌ HARMFUL
Double sig [4,16] + High-risk       | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Double sig [4,17] + High-risk       | HR: 1.634 | 63.4% increase | ❌ HARMFUL
Double sig [4,18] + High-risk       | HR: 1.608 | 60.8% increase | ❌ HARMFUL
Double sig [4,19] + High-risk       | HR: 1.566 | 56.6% increase | ❌ HARMFUL
Double sig [5,6] + High-risk        | HR: 1.344 | 34.4% increase | ❌ HARMFUL
Double sig [5,7] + High-risk        | HR: 1.503 | 50.3% increase | ❌ HARMFUL
Double sig [5,8] + High-risk        | HR: 1.414 | 41.4% increase | ❌ HARMFUL
Double sig [5,9] + High-risk        | HR: 1.413 | 41.3% increase | ❌ HARMFUL
Double sig [5,10] + High-risk       | HR: 1.358 | 35.8% increase | ❌ HARMFUL
Double sig [5,11] + High-risk       | HR: 1.369 | 36.9% increase | ❌ HARMFUL
Double sig [5,12] + High-risk       | HR: 1.376 | 37.6% increase | ❌ HARMFUL
Double sig [5,13] + High-risk       | HR: 1.435 | 43.5% increase | ❌ HARMFUL
Double sig [5,14] + High-risk       | HR: 1.439 | 43.9% increase | ❌ HARMFUL
Double sig [5,15] + High-risk       | HR: 1.284 | 28.4% increase | ❌ HARMFUL
Double sig [5,16] + High-risk       | HR: 1.349 | 34.9% increase | ❌ HARMFUL
Double sig [5,17] + High-risk       | HR: 1.421 | 42.1% increase | ❌ HARMFUL
Double sig [5,18] + High-risk       | HR: 1.389 | 38.9% increase | ❌ HARMFUL
Double sig [5,19] + High-risk       | HR: 1.470 | 47.0% increase | ❌ HARMFUL
Double sig [6,7] + High-risk        | HR: 1.601 | 60.1% increase | ❌ HARMFUL
Double sig [6,8] + High-risk        | HR: 1.573 | 57.3% increase | ❌ HARMFUL
Double sig [6,9] + High-risk        | HR: 1.479 | 47.9% increase | ❌ HARMFUL
Double sig [6,10] + High-risk       | HR: 1.428 | 42.8% increase | ❌ HARMFUL
Double sig [6,11] + High-risk       | HR: 1.427 | 42.7% increase | ❌ HARMFUL
Double sig [6,12] + High-risk       | HR: 1.415 | 41.5% increase | ❌ HARMFUL
Double sig [6,13] + High-risk       | HR: 1.479 | 47.9% increase | ❌ HARMFUL
Double sig [6,14] + High-risk       | HR: 1.524 | 52.4% increase | ❌ HARMFUL
Double sig [6,15] + High-risk       | HR: 1.258 | 25.8% increase | ❌ HARMFUL
Double sig [6,16] + High-risk       | HR: 1.588 | 58.8% increase | ❌ HARMFUL
Double sig [6,17] + High-risk       | HR: 1.582 | 58.2% increase | ❌ HARMFUL
Double sig [6,18] + High-risk       | HR: 1.359 | 35.9% increase | ❌ HARMFUL
Double sig [6,19] + High-risk       | HR: 1.548 | 54.8% increase | ❌ HARMFUL
Double sig [7,8] + High-risk        | HR: 1.579 | 57.9% increase | ❌ HARMFUL
Double sig [7,9] + High-risk        | HR: 1.615 | 61.5% increase | ❌ HARMFUL
Double sig [7,10] + High-risk       | HR: 1.679 | 67.9% increase | ❌ HARMFUL
Double sig [7,11] + High-risk       | HR: 1.680 | 68.0% increase | ❌ HARMFUL
Double sig [7,12] + High-risk       | HR: 1.582 | 58.2% increase | ❌ HARMFUL
Double sig [7,13] + High-risk       | HR: 1.604 | 60.4% increase | ❌ HARMFUL
Double sig [7,14] + High-risk       | HR: 1.674 | 67.4% increase | ❌ HARMFUL
Double sig [7,15] + High-risk       | HR: 1.556 | 55.6% increase | ❌ HARMFUL
Double sig [7,16] + High-risk       | HR: 1.708 | 70.8% increase | ❌ HARMFUL
Double sig [7,17] + High-risk       | HR: 1.577 | 57.7% increase | ❌ HARMFUL
Double sig [7,18] + High-risk       | HR: 1.652 | 65.2% increase | ❌ HARMFUL
Double sig [7,19] + High-risk       | HR: 1.712 | 71.2% increase | ❌ HARMFUL
Double sig [8,9] + High-risk        | HR: 1.518 | 51.8% increase | ❌ HARMFUL
Double sig [8,10] + High-risk       | HR: 1.473 | 47.3% increase | ❌ HARMFUL
Double sig [8,11] + High-risk       | HR: 1.517 | 51.7% increase | ❌ HARMFUL
Double sig [8,12] + High-risk       | HR: 1.533 | 53.3% increase | ❌ HARMFUL
Double sig [8,13] + High-risk       | HR: 1.511 | 51.1% increase | ❌ HARMFUL
Double sig [8,14] + High-risk       | HR: 1.534 | 53.4% increase | ❌ HARMFUL
Double sig [8,15] + High-risk       | HR: 1.373 | 37.3% increase | ❌ HARMFUL
Double sig [8,16] + High-risk       | HR: 1.526 | 52.6% increase | ❌ HARMFUL
Double sig [8,17] + High-risk       | HR: 1.543 | 54.3% increase | ❌ HARMFUL
Double sig [8,18] + High-risk       | HR: 1.476 | 47.6% increase | ❌ HARMFUL
Double sig [8,19] + High-risk       | HR: 1.549 | 54.9% increase | ❌ HARMFUL
Double sig [9,10] + High-risk       | HR: 1.462 | 46.2% increase | ❌ HARMFUL
Double sig [9,11] + High-risk       | HR: 1.462 | 46.2% increase | ❌ HARMFUL
Double sig [9,12] + High-risk       | HR: 1.691 | 69.1% increase | ❌ HARMFUL
Double sig [9,13] + High-risk       | HR: 1.498 | 49.8% increase | ❌ HARMFUL
Double sig [9,14] + High-risk       | HR: 1.481 | 48.1% increase | ❌ HARMFUL
Double sig [9,15] + High-risk       | HR: 1.418 | 41.8% increase | ❌ HARMFUL
Double sig [9,16] + High-risk       | HR: 1.603 | 60.3% increase | ❌ HARMFUL
Double sig [9,17] + High-risk       | HR: 1.603 | 60.3% increase | ❌ HARMFUL
Double sig [9,18] + High-risk       | HR: 1.603 | 60.3% increase | ❌ HARMFUL
Double sig [9,19] + High-risk       | HR: 1.505 | 50.5% increase | ❌ HARMFUL
Double sig [10,11] + High-risk      | HR: 1.493 | 49.3% increase | ❌ HARMFUL
Double sig [10,12] + High-risk      | HR: 1.504 | 50.4% increase | ❌ HARMFUL
Double sig [10,13] + High-risk      | HR: 1.504 | 50.4% increase | ❌ HARMFUL
Double sig [10,14] + High-risk      | HR: 1.567 | 56.7% increase | ❌ HARMFUL
Double sig [10,15] + High-risk      | HR: 1.302 | 30.2% increase | ❌ HARMFUL
Double sig [10,16] + High-risk      | HR: 1.655 | 65.5% increase | ❌ HARMFUL
Double sig [10,17] + High-risk      | HR: 1.523 | 52.3% increase | ❌ HARMFUL
Double sig [10,18] + High-risk      | HR: 1.458 | 45.8% increase | ❌ HARMFUL
Double sig [10,19] + High-risk      | HR: 1.599 | 59.9% increase | ❌ HARMFUL
Double sig [11,12] + High-risk      | HR: 1.474 | 47.4% increase | ❌ HARMFUL
Double sig [11,13] + High-risk      | HR: 1.569 | 56.9% increase | ❌ HARMFUL
Double sig [11,14] + High-risk      | HR: 1.559 | 55.9% increase | ❌ HARMFUL
Double sig [11,15] + High-risk      | HR: 1.407 | 40.7% increase | ❌ HARMFUL
Double sig [11,16] + High-risk      | HR: 1.608 | 60.8% increase | ❌ HARMFUL
Double sig [11,17] + High-risk      | HR: 1.535 | 53.5% increase | ❌ HARMFUL
Double sig [11,18] + High-risk      | HR: 1.600 | 60.0% increase | ❌ HARMFUL
Double sig [11,19] + High-risk      | HR: 1.692 | 69.2% increase | ❌ HARMFUL
Double sig [12,13] + High-risk      | HR: 1.639 | 63.9% increase | ❌ HARMFUL
Double sig [12,14] + High-risk      | HR: 1.714 | 71.4% increase | ❌ HARMFUL
Double sig [12,15] + High-risk      | HR: 1.482 | 48.2% increase | ❌ HARMFUL
Double sig [12,16] + High-risk      | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Double sig [12,17] + High-risk      | HR: 1.566 | 56.6% increase | ❌ HARMFUL
Double sig [12,18] + High-risk      | HR: 1.526 | 52.6% increase | ❌ HARMFUL
Double sig [12,19] + High-risk      | HR: 1.700 | 70.0% increase | ❌ HARMFUL
Double sig [13,14] + High-risk      | HR: 1.638 | 63.8% increase | ❌ HARMFUL
Double sig [13,15] + High-risk      | HR: 1.510 | 51.0% increase | ❌ HARMFUL
Double sig [13,16] + High-risk      | HR: 1.755 | 75.5% increase | ❌ HARMFUL
Double sig [13,17] + High-risk      | HR: 1.623 | 62.3% increase | ❌ HARMFUL
Double sig [13,18] + High-risk      | HR: 1.616 | 61.6% increase | ❌ HARMFUL
Double sig [13,19] + High-risk      | HR: 1.613 | 61.3% increase | ❌ HARMFUL
Double sig [14,15] + High-risk      | HR: 1.517 | 51.7% increase | ❌ HARMFUL
Double sig [14,16] + High-risk      | HR: 1.782 | 78.2% increase | ❌ HARMFUL
Double sig [14,17] + High-risk      | HR: 1.675 | 67.5% increase | ❌ HARMFUL
Double sig [14,18] + High-risk      | HR: 1.608 | 60.8% increase | ❌ HARMFUL
Double sig [14,19] + High-risk      | HR: 1.676 | 67.6% increase | ❌ HARMFUL
Double sig [15,16] + High-risk      | HR: 1.612 | 61.2% increase | ❌ HARMFUL
Double sig [15,17] + High-risk      | HR: 1.502 | 50.2% increase | ❌ HARMFUL
Double sig [15,18] + High-risk      | HR: 1.452 | 45.2% increase | ❌ HARMFUL
Double sig [15,19] + High-risk      | HR: 1.626 | 62.6% increase | ❌ HARMFUL
Double sig [16,17] + High-risk      | HR: 1.714 | 71.4% increase | ❌ HARMFUL
Double sig [16,18] + High-risk      | HR: 1.716 | 71.6% increase | ❌ HARMFUL
Double sig [16,19] + High-risk      | HR: 1.728 | 72.8% increase | ❌ HARMFUL
Double sig [17,18] + High-risk      | HR: 1.655 | 65.5% increase | ❌ HARMFUL
Double sig [17,19] + High-risk      | HR: 1.698 | 69.8% increase | ❌ HARMFUL
Double sig [18,19] + High-risk      | HR: 1.692 | 69.2% increase | ❌ HARMFUL
Triple sig [0, 1, 2] + High-risk    | HR: 1.716 | 71.6% increase | ❌ HARMFUL
Triple sig [0, 1, 5] + High-risk    | HR: 1.582 | 58.2% increase | ❌ HARMFUL
Triple sig [0, 1, 8] + High-risk    | HR: 1.748 | 74.8% increase | ❌ HARMFUL
Triple sig [0, 1, 11] + High-risk   | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Triple sig [0, 1, 14] + High-risk   | HR: 1.787 | 78.7% increase | ❌ HARMFUL
Triple sig [0, 1, 17] + High-risk   | HR: 1.790 | 79.0% increase | ❌ HARMFUL
Triple sig [0, 2, 3] + High-risk    | HR: 2.089 | 108.9% increase | ❌ HARMFUL
Triple sig [0, 2, 6] + High-risk    | HR: 1.673 | 67.3% increase | ❌ HARMFUL
Triple sig [0, 2, 9] + High-risk    | HR: 1.649 | 64.9% increase | ❌ HARMFUL
Triple sig [0, 2, 12] + High-risk   | HR: 1.813 | 81.3% increase | ❌ HARMFUL
Triple sig [0, 2, 15] + High-risk   | HR: 1.727 | 72.7% increase | ❌ HARMFUL
Triple sig [0, 2, 18] + High-risk   | HR: 1.727 | 72.7% increase | ❌ HARMFUL
Triple sig [0, 3, 5] + High-risk    | HR: 1.592 | 59.2% increase | ❌ HARMFUL
Triple sig [0, 3, 8] + High-risk    | HR: 1.734 | 73.4% increase | ❌ HARMFUL
Triple sig [0, 3, 11] + High-risk   | HR: 1.807 | 80.7% increase | ❌ HARMFUL
Triple sig [0, 3, 14] + High-risk   | HR: 1.891 | 89.1% increase | ❌ HARMFUL
Triple sig [0, 3, 17] + High-risk   | HR: 1.846 | 84.6% increase | ❌ HARMFUL
Triple sig [0, 4, 5] + High-risk    | HR: 1.501 | 50.1% increase | ❌ HARMFUL
Triple sig [0, 4, 8] + High-risk    | HR: 1.622 | 62.2% increase | ❌ HARMFUL
Triple sig [0, 4, 11] + High-risk   | HR: 1.621 | 62.1% increase | ❌ HARMFUL
Triple sig [0, 4, 14] + High-risk   | HR: 1.761 | 76.1% increase | ❌ HARMFUL
Triple sig [0, 4, 17] + High-risk   | HR: 1.731 | 73.1% increase | ❌ HARMFUL
Triple sig [0, 5, 6] + High-risk    | HR: 1.466 | 46.6% increase | ❌ HARMFUL
Triple sig [0, 5, 9] + High-risk    | HR: 1.485 | 48.5% increase | ❌ HARMFUL
Triple sig [0, 5, 12] + High-risk   | HR: 1.538 | 53.8% increase | ❌ HARMFUL
Triple sig [0, 5, 15] + High-risk   | HR: 1.368 | 36.8% increase | ❌ HARMFUL
Triple sig [0, 5, 18] + High-risk   | HR: 1.513 | 51.3% increase | ❌ HARMFUL
Triple sig [0, 6, 8] + High-risk    | HR: 1.680 | 68.0% increase | ❌ HARMFUL
Triple sig [0, 6, 11] + High-risk   | HR: 1.539 | 53.9% increase | ❌ HARMFUL
Triple sig [0, 6, 14] + High-risk   | HR: 1.798 | 79.8% increase | ❌ HARMFUL
Triple sig [0, 6, 17] + High-risk   | HR: 1.718 | 71.8% increase | ❌ HARMFUL
Triple sig [0, 7, 8] + High-risk    | HR: 1.859 | 85.9% increase | ❌ HARMFUL
Triple sig [0, 7, 11] + High-risk   | HR: 1.767 | 76.7% increase | ❌ HARMFUL
Triple sig [0, 7, 14] + High-risk   | HR: 1.930 | 93.0% increase | ❌ HARMFUL
Triple sig [0, 7, 17] + High-risk   | HR: 1.768 | 76.8% increase | ❌ HARMFUL
Triple sig [0, 8, 9] + High-risk    | HR: 1.615 | 61.5% increase | ❌ HARMFUL
Triple sig [0, 8, 12] + High-risk   | HR: 1.668 | 66.8% increase | ❌ HARMFUL
Triple sig [0, 8, 15] + High-risk   | HR: 1.548 | 54.8% increase | ❌ HARMFUL
Triple sig [0, 8, 18] + High-risk   | HR: 1.707 | 70.7% increase | ❌ HARMFUL
Triple sig [0, 9, 11] + High-risk   | HR: 1.586 | 58.6% increase | ❌ HARMFUL
Triple sig [0, 9, 14] + High-risk   | HR: 1.833 | 83.3% increase | ❌ HARMFUL
Triple sig [0, 9, 17] + High-risk   | HR: 1.709 | 70.9% increase | ❌ HARMFUL
Triple sig [0, 10, 11] + High-risk  | HR: 1.503 | 50.3% increase | ❌ HARMFUL
Triple sig [0, 10, 14] + High-risk  | HR: 1.741 | 74.1% increase | ❌ HARMFUL
Triple sig [0, 10, 17] + High-risk  | HR: 1.722 | 72.2% increase | ❌ HARMFUL
Triple sig [0, 11, 12] + High-risk  | HR: 1.618 | 61.8% increase | ❌ HARMFUL
Triple sig [0, 11, 15] + High-risk  | HR: 1.561 | 56.1% increase | ❌ HARMFUL
Triple sig [0, 11, 18] + High-risk  | HR: 1.716 | 71.6% increase | ❌ HARMFUL
Triple sig [0, 12, 14] + High-risk  | HR: 1.940 | 94.0% increase | ❌ HARMFUL
Triple sig [0, 12, 17] + High-risk  | HR: 1.696 | 69.6% increase | ❌ HARMFUL
Triple sig [0, 13, 14] + High-risk  | HR: 1.891 | 89.1% increase | ❌ HARMFUL
Triple sig [0, 13, 17] + High-risk  | HR: 1.815 | 81.5% increase | ❌ HARMFUL
Triple sig [0, 14, 15] + High-risk  | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Triple sig [0, 14, 18] + High-risk  | HR: 1.880 | 88.0% increase | ❌ HARMFUL
Triple sig [0, 15, 17] + High-risk  | HR: 1.731 | 73.1% increase | ❌ HARMFUL
Triple sig [0, 16, 17] + High-risk  | HR: 1.876 | 87.6% increase | ❌ HARMFUL
Triple sig [0, 17, 18] + High-risk  | HR: 1.719 | 71.9% increase | ❌ HARMFUL
Triple sig [1, 2, 3] + High-risk    | HR: 1.763 | 76.3% increase | ❌ HARMFUL
Triple sig [1, 2, 6] + High-risk    | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Triple sig [1, 2, 9] + High-risk    | HR: 1.661 | 66.1% increase | ❌ HARMFUL
Triple sig [1, 2, 12] + High-risk   | HR: 1.686 | 68.6% increase | ❌ HARMFUL
Triple sig [1, 2, 15] + High-risk   | HR: 1.589 | 58.9% increase | ❌ HARMFUL
Triple sig [1, 2, 18] + High-risk   | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Triple sig [1, 3, 5] + High-risk    | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Triple sig [1, 3, 8] + High-risk    | HR: 1.568 | 56.8% increase | ❌ HARMFUL
Triple sig [1, 3, 11] + High-risk   | HR: 1.729 | 72.9% increase | ❌ HARMFUL
Triple sig [1, 3, 14] + High-risk   | HR: 1.712 | 71.2% increase | ❌ HARMFUL
Triple sig [1, 3, 17] + High-risk   | HR: 1.728 | 72.8% increase | ❌ HARMFUL
Triple sig [1, 4, 5] + High-risk    | HR: 1.576 | 57.6% increase | ❌ HARMFUL
Triple sig [1, 4, 8] + High-risk    | HR: 1.608 | 60.8% increase | ❌ HARMFUL
Triple sig [1, 4, 11] + High-risk   | HR: 1.674 | 67.4% increase | ❌ HARMFUL
Triple sig [1, 4, 14] + High-risk   | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Triple sig [1, 4, 17] + High-risk   | HR: 1.740 | 74.0% increase | ❌ HARMFUL
Triple sig [1, 5, 6] + High-risk    | HR: 1.575 | 57.5% increase | ❌ HARMFUL
Triple sig [1, 5, 9] + High-risk    | HR: 1.553 | 55.3% increase | ❌ HARMFUL
Triple sig [1, 5, 12] + High-risk   | HR: 1.602 | 60.2% increase | ❌ HARMFUL
Triple sig [1, 5, 15] + High-risk   | HR: 1.495 | 49.5% increase | ❌ HARMFUL
Triple sig [1, 5, 18] + High-risk   | HR: 1.531 | 53.1% increase | ❌ HARMFUL
Triple sig [1, 6, 8] + High-risk    | HR: 1.521 | 52.1% increase | ❌ HARMFUL
Triple sig [1, 6, 11] + High-risk   | HR: 1.492 | 49.2% increase | ❌ HARMFUL
Triple sig [1, 6, 14] + High-risk   | HR: 1.569 | 56.9% increase | ❌ HARMFUL
Triple sig [1, 6, 17] + High-risk   | HR: 1.587 | 58.7% increase | ❌ HARMFUL
Triple sig [1, 7, 8] + High-risk    | HR: 1.663 | 66.3% increase | ❌ HARMFUL
Triple sig [1, 7, 11] + High-risk   | HR: 1.779 | 77.9% increase | ❌ HARMFUL
Triple sig [1, 7, 14] + High-risk   | HR: 1.775 | 77.5% increase | ❌ HARMFUL
Triple sig [1, 7, 17] + High-risk   | HR: 1.661 | 66.1% increase | ❌ HARMFUL
Triple sig [1, 8, 9] + High-risk    | HR: 1.597 | 59.7% increase | ❌ HARMFUL
Triple sig [1, 8, 12] + High-risk   | HR: 1.601 | 60.1% increase | ❌ HARMFUL
Triple sig [1, 8, 15] + High-risk   | HR: 1.461 | 46.1% increase | ❌ HARMFUL
Triple sig [1, 8, 18] + High-risk   | HR: 1.587 | 58.7% increase | ❌ HARMFUL
Triple sig [1, 9, 11] + High-risk   | HR: 1.670 | 67.0% increase | ❌ HARMFUL
Triple sig [1, 9, 14] + High-risk   | HR: 1.616 | 61.6% increase | ❌ HARMFUL
Triple sig [1, 9, 17] + High-risk   | HR: 1.625 | 62.5% increase | ❌ HARMFUL
Triple sig [1, 10, 11] + High-risk  | HR: 1.572 | 57.2% increase | ❌ HARMFUL
Triple sig [1, 10, 14] + High-risk  | HR: 1.643 | 64.3% increase | ❌ HARMFUL
Triple sig [1, 10, 17] + High-risk  | HR: 1.749 | 74.9% increase | ❌ HARMFUL
Triple sig [1, 11, 12] + High-risk  | HR: 1.586 | 58.6% increase | ❌ HARMFUL
Triple sig [1, 11, 15] + High-risk  | HR: 1.550 | 55.0% increase | ❌ HARMFUL
Triple sig [1, 11, 18] + High-risk  | HR: 1.643 | 64.3% increase | ❌ HARMFUL
Triple sig [1, 12, 14] + High-risk  | HR: 1.748 | 74.8% increase | ❌ HARMFUL
Triple sig [1, 12, 17] + High-risk  | HR: 1.682 | 68.2% increase | ❌ HARMFUL
Triple sig [1, 13, 14] + High-risk  | HR: 1.711 | 71.1% increase | ❌ HARMFUL
Triple sig [1, 13, 17] + High-risk  | HR: 1.723 | 72.3% increase | ❌ HARMFUL
Triple sig [1, 14, 15] + High-risk  | HR: 1.556 | 55.6% increase | ❌ HARMFUL
Triple sig [1, 14, 18] + High-risk  | HR: 1.692 | 69.2% increase | ❌ HARMFUL
Triple sig [1, 15, 17] + High-risk  | HR: 1.612 | 61.2% increase | ❌ HARMFUL
Triple sig [1, 16, 17] + High-risk  | HR: 1.881 | 88.1% increase | ❌ HARMFUL
Triple sig [1, 17, 18] + High-risk  | HR: 1.802 | 80.2% increase | ❌ HARMFUL
Triple sig [2, 3, 4] + High-risk    | HR: 1.821 | 82.1% increase | ❌ HARMFUL
Triple sig [2, 3, 7] + High-risk    | HR: 1.953 | 95.3% increase | ❌ HARMFUL
Triple sig [2, 3, 10] + High-risk   | HR: 1.737 | 73.7% increase | ❌ HARMFUL
Triple sig [2, 3, 13] + High-risk   | HR: 1.793 | 79.3% increase | ❌ HARMFUL
Triple sig [2, 3, 16] + High-risk   | HR: 1.938 | 93.8% increase | ❌ HARMFUL
Triple sig [2, 3, 19] + High-risk   | HR: 1.760 | 76.0% increase | ❌ HARMFUL
Triple sig [2, 4, 7] + High-risk    | HR: 1.665 | 66.5% increase | ❌ HARMFUL
Triple sig [2, 4, 10] + High-risk   | HR: 1.568 | 56.8% increase | ❌ HARMFUL
Triple sig [2, 4, 13] + High-risk   | HR: 1.721 | 72.1% increase | ❌ HARMFUL
Triple sig [2, 4, 16] + High-risk   | HR: 1.689 | 68.9% increase | ❌ HARMFUL
Triple sig [2, 4, 19] + High-risk   | HR: 1.773 | 77.3% increase | ❌ HARMFUL
Triple sig [2, 5, 8] + High-risk    | HR: 1.466 | 46.6% increase | ❌ HARMFUL
Triple sig [2, 5, 11] + High-risk   | HR: 1.475 | 47.5% increase | ❌ HARMFUL
Triple sig [2, 5, 14] + High-risk   | HR: 1.474 | 47.4% increase | ❌ HARMFUL
Triple sig [2, 5, 17] + High-risk   | HR: 1.492 | 49.2% increase | ❌ HARMFUL
Triple sig [2, 6, 7] + High-risk    | HR: 1.611 | 61.1% increase | ❌ HARMFUL
Triple sig [2, 6, 10] + High-risk   | HR: 1.495 | 49.5% increase | ❌ HARMFUL
Triple sig [2, 6, 13] + High-risk   | HR: 1.548 | 54.8% increase | ❌ HARMFUL
Triple sig [2, 6, 16] + High-risk   | HR: 1.617 | 61.7% increase | ❌ HARMFUL
Triple sig [2, 6, 19] + High-risk   | HR: 1.539 | 53.9% increase | ❌ HARMFUL
Triple sig [2, 7, 10] + High-risk   | HR: 1.645 | 64.5% increase | ❌ HARMFUL
Triple sig [2, 7, 13] + High-risk   | HR: 1.734 | 73.4% increase | ❌ HARMFUL
Triple sig [2, 7, 16] + High-risk   | HR: 1.873 | 87.3% increase | ❌ HARMFUL
Triple sig [2, 7, 19] + High-risk   | HR: 1.866 | 86.6% increase | ❌ HARMFUL
Triple sig [2, 8, 11] + High-risk   | HR: 1.616 | 61.6% increase | ❌ HARMFUL
Triple sig [2, 8, 14] + High-risk   | HR: 1.598 | 59.8% increase | ❌ HARMFUL
Triple sig [2, 8, 17] + High-risk   | HR: 1.635 | 63.5% increase | ❌ HARMFUL
Triple sig [2, 9, 10] + High-risk   | HR: 1.484 | 48.4% increase | ❌ HARMFUL
Triple sig [2, 9, 13] + High-risk   | HR: 1.657 | 65.7% increase | ❌ HARMFUL
Triple sig [2, 9, 16] + High-risk   | HR: 1.651 | 65.1% increase | ❌ HARMFUL
Triple sig [2, 9, 19] + High-risk   | HR: 1.679 | 67.9% increase | ❌ HARMFUL
Triple sig [2, 10, 13] + High-risk  | HR: 1.645 | 64.5% increase | ❌ HARMFUL
Triple sig [2, 10, 16] + High-risk  | HR: 1.766 | 76.6% increase | ❌ HARMFUL
Triple sig [2, 10, 19] + High-risk  | HR: 1.724 | 72.4% increase | ❌ HARMFUL
Triple sig [2, 11, 14] + High-risk  | HR: 1.624 | 62.4% increase | ❌ HARMFUL
Triple sig [2, 11, 17] + High-risk  | HR: 1.665 | 66.5% increase | ❌ HARMFUL
Triple sig [2, 12, 13] + High-risk  | HR: 1.706 | 70.6% increase | ❌ HARMFUL
Triple sig [2, 12, 16] + High-risk  | HR: 1.801 | 80.1% increase | ❌ HARMFUL
Triple sig [2, 12, 19] + High-risk  | HR: 1.790 | 79.0% increase | ❌ HARMFUL
Triple sig [2, 13, 16] + High-risk  | HR: 1.896 | 89.6% increase | ❌ HARMFUL
Triple sig [2, 13, 19] + High-risk  | HR: 1.720 | 72.0% increase | ❌ HARMFUL
Triple sig [2, 14, 17] + High-risk  | HR: 1.751 | 75.1% increase | ❌ HARMFUL
Triple sig [2, 15, 16] + High-risk  | HR: 1.660 | 66.0% increase | ❌ HARMFUL
Triple sig [2, 15, 19] + High-risk  | HR: 1.781 | 78.1% increase | ❌ HARMFUL
Triple sig [2, 16, 19] + High-risk  | HR: 1.837 | 83.7% increase | ❌ HARMFUL
Triple sig [2, 18, 19] + High-risk  | HR: 1.690 | 69.0% increase | ❌ HARMFUL
Triple sig [3, 4, 7] + High-risk    | HR: 1.852 | 85.2% increase | ❌ HARMFUL
Triple sig [3, 4, 10] + High-risk   | HR: 1.625 | 62.5% increase | ❌ HARMFUL
Triple sig [3, 4, 13] + High-risk   | HR: 1.686 | 68.6% increase | ❌ HARMFUL
Triple sig [3, 4, 16] + High-risk   | HR: 1.879 | 87.9% increase | ❌ HARMFUL
Triple sig [3, 4, 19] + High-risk   | HR: 1.821 | 82.1% increase | ❌ HARMFUL
Triple sig [3, 5, 8] + High-risk    | HR: 1.519 | 51.9% increase | ❌ HARMFUL
Triple sig [3, 5, 11] + High-risk   | HR: 1.590 | 59.0% increase | ❌ HARMFUL
Triple sig [3, 5, 14] + High-risk   | HR: 1.614 | 61.4% increase | ❌ HARMFUL
Triple sig [3, 5, 17] + High-risk   | HR: 1.606 | 60.6% increase | ❌ HARMFUL
Triple sig [3, 6, 7] + High-risk    | HR: 1.842 | 84.2% increase | ❌ HARMFUL
Triple sig [3, 6, 10] + High-risk   | HR: 1.553 | 55.3% increase | ❌ HARMFUL
Triple sig [3, 6, 13] + High-risk   | HR: 1.626 | 62.6% increase | ❌ HARMFUL
Triple sig [3, 6, 16] + High-risk   | HR: 1.655 | 65.5% increase | ❌ HARMFUL
Triple sig [3, 6, 19] + High-risk   | HR: 1.732 | 73.2% increase | ❌ HARMFUL
Triple sig [3, 7, 10] + High-risk   | HR: 1.943 | 94.3% increase | ❌ HARMFUL
Triple sig [3, 7, 13] + High-risk   | HR: 1.868 | 86.8% increase | ❌ HARMFUL
Triple sig [3, 7, 16] + High-risk   | HR: 1.982 | 98.2% increase | ❌ HARMFUL
Triple sig [3, 7, 19] + High-risk   | HR: 1.843 | 84.3% increase | ❌ HARMFUL
Triple sig [3, 8, 11] + High-risk   | HR: 1.700 | 70.0% increase | ❌ HARMFUL
Triple sig [3, 8, 14] + High-risk   | HR: 1.673 | 67.3% increase | ❌ HARMFUL
Triple sig [3, 8, 17] + High-risk   | HR: 1.658 | 65.8% increase | ❌ HARMFUL
Triple sig [3, 9, 10] + High-risk   | HR: 1.595 | 59.5% increase | ❌ HARMFUL
Triple sig [3, 9, 13] + High-risk   | HR: 1.726 | 72.6% increase | ❌ HARMFUL
Triple sig [3, 9, 16] + High-risk   | HR: 1.681 | 68.1% increase | ❌ HARMFUL
Triple sig [3, 9, 19] + High-risk   | HR: 1.675 | 67.5% increase | ❌ HARMFUL
Triple sig [3, 10, 13] + High-risk  | HR: 1.623 | 62.3% increase | ❌ HARMFUL
Triple sig [3, 10, 16] + High-risk  | HR: 1.865 | 86.5% increase | ❌ HARMFUL
Triple sig [3, 10, 19] + High-risk  | HR: 1.730 | 73.0% increase | ❌ HARMFUL
Triple sig [3, 11, 14] + High-risk  | HR: 1.789 | 78.9% increase | ❌ HARMFUL
Triple sig [3, 11, 17] + High-risk  | HR: 1.674 | 67.4% increase | ❌ HARMFUL
Triple sig [3, 12, 13] + High-risk  | HR: 1.840 | 84.0% increase | ❌ HARMFUL
Triple sig [3, 12, 16] + High-risk  | HR: 1.757 | 75.7% increase | ❌ HARMFUL
Triple sig [3, 12, 19] + High-risk  | HR: 1.767 | 76.7% increase | ❌ HARMFUL
Triple sig [3, 13, 16] + High-risk  | HR: 1.885 | 88.5% increase | ❌ HARMFUL
Triple sig [3, 13, 19] + High-risk  | HR: 1.852 | 85.2% increase | ❌ HARMFUL
Triple sig [3, 14, 17] + High-risk  | HR: 1.769 | 76.9% increase | ❌ HARMFUL
Triple sig [3, 15, 16] + High-risk  | HR: 1.861 | 86.1% increase | ❌ HARMFUL
Triple sig [3, 15, 19] + High-risk  | HR: 1.722 | 72.2% increase | ❌ HARMFUL
Triple sig [3, 16, 19] + High-risk  | HR: 1.825 | 82.5% increase | ❌ HARMFUL
Triple sig [3, 18, 19] + High-risk  | HR: 1.783 | 78.3% increase | ❌ HARMFUL
Triple sig [4, 5, 8] + High-risk    | HR: 1.417 | 41.7% increase | ❌ HARMFUL
Triple sig [4, 5, 11] + High-risk   | HR: 1.464 | 46.4% increase | ❌ HARMFUL
Triple sig [4, 5, 14] + High-risk   | HR: 1.535 | 53.5% increase | ❌ HARMFUL
Triple sig [4, 5, 17] + High-risk   | HR: 1.479 | 47.9% increase | ❌ HARMFUL
Triple sig [4, 6, 7] + High-risk    | HR: 1.566 | 56.6% increase | ❌ HARMFUL
Triple sig [4, 6, 10] + High-risk   | HR: 1.519 | 51.9% increase | ❌ HARMFUL
Triple sig [4, 6, 13] + High-risk   | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Triple sig [4, 6, 16] + High-risk   | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Triple sig [4, 6, 19] + High-risk   | HR: 1.622 | 62.2% increase | ❌ HARMFUL
Triple sig [4, 7, 10] + High-risk   | HR: 1.610 | 61.0% increase | ❌ HARMFUL
Triple sig [4, 7, 13] + High-risk   | HR: 1.669 | 66.9% increase | ❌ HARMFUL
Triple sig [4, 7, 16] + High-risk   | HR: 1.798 | 79.8% increase | ❌ HARMFUL
Triple sig [4, 7, 19] + High-risk   | HR: 1.763 | 76.3% increase | ❌ HARMFUL
Triple sig [4, 8, 11] + High-risk   | HR: 1.564 | 56.4% increase | ❌ HARMFUL
Triple sig [4, 8, 14] + High-risk   | HR: 1.641 | 64.1% increase | ❌ HARMFUL
Triple sig [4, 8, 17] + High-risk   | HR: 1.595 | 59.5% increase | ❌ HARMFUL
Triple sig [4, 9, 10] + High-risk   | HR: 1.542 | 54.2% increase | ❌ HARMFUL
Triple sig [4, 9, 13] + High-risk   | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Triple sig [4, 9, 16] + High-risk   | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Triple sig [4, 9, 19] + High-risk   | HR: 1.627 | 62.7% increase | ❌ HARMFUL
Triple sig [4, 10, 13] + High-risk  | HR: 1.508 | 50.8% increase | ❌ HARMFUL
Triple sig [4, 10, 16] + High-risk  | HR: 1.683 | 68.3% increase | ❌ HARMFUL
Triple sig [4, 10, 19] + High-risk  | HR: 1.688 | 68.8% increase | ❌ HARMFUL
Triple sig [4, 11, 14] + High-risk  | HR: 1.591 | 59.1% increase | ❌ HARMFUL
Triple sig [4, 11, 17] + High-risk  | HR: 1.662 | 66.2% increase | ❌ HARMFUL
Triple sig [4, 12, 13] + High-risk  | HR: 1.674 | 67.4% increase | ❌ HARMFUL
Triple sig [4, 12, 16] + High-risk  | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Triple sig [4, 12, 19] + High-risk  | HR: 1.675 | 67.5% increase | ❌ HARMFUL
Triple sig [4, 13, 16] + High-risk  | HR: 1.857 | 85.7% increase | ❌ HARMFUL
Triple sig [4, 13, 19] + High-risk  | HR: 1.713 | 71.3% increase | ❌ HARMFUL
Triple sig [4, 14, 17] + High-risk  | HR: 1.718 | 71.8% increase | ❌ HARMFUL
Triple sig [4, 15, 16] + High-risk  | HR: 1.622 | 62.2% increase | ❌ HARMFUL
Triple sig [4, 15, 19] + High-risk  | HR: 1.588 | 58.8% increase | ❌ HARMFUL
Triple sig [4, 16, 19] + High-risk  | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Triple sig [4, 18, 19] + High-risk  | HR: 1.773 | 77.3% increase | ❌ HARMFUL
Triple sig [5, 6, 9] + High-risk    | HR: 1.436 | 43.6% increase | ❌ HARMFUL
Triple sig [5, 6, 12] + High-risk   | HR: 1.470 | 47.0% increase | ❌ HARMFUL
Triple sig [5, 6, 15] + High-risk   | HR: 1.339 | 33.9% increase | ❌ HARMFUL
Triple sig [5, 6, 18] + High-risk   | HR: 1.473 | 47.3% increase | ❌ HARMFUL
Triple sig [5, 7, 9] + High-risk    | HR: 1.520 | 52.0% increase | ❌ HARMFUL
Triple sig [5, 7, 12] + High-risk   | HR: 1.584 | 58.4% increase | ❌ HARMFUL
Triple sig [5, 7, 15] + High-risk   | HR: 1.606 | 60.6% increase | ❌ HARMFUL
Triple sig [5, 7, 18] + High-risk   | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Triple sig [5, 8, 10] + High-risk   | HR: 1.442 | 44.2% increase | ❌ HARMFUL
Triple sig [5, 8, 13] + High-risk   | HR: 1.494 | 49.4% increase | ❌ HARMFUL
Triple sig [5, 8, 16] + High-risk   | HR: 1.405 | 40.5% increase | ❌ HARMFUL
Triple sig [5, 8, 19] + High-risk   | HR: 1.516 | 51.6% increase | ❌ HARMFUL
Triple sig [5, 9, 12] + High-risk   | HR: 1.543 | 54.3% increase | ❌ HARMFUL
Triple sig [5, 9, 15] + High-risk   | HR: 1.407 | 40.7% increase | ❌ HARMFUL
Triple sig [5, 9, 18] + High-risk   | HR: 1.482 | 48.2% increase | ❌ HARMFUL
Triple sig [5, 10, 12] + High-risk  | HR: 1.486 | 48.6% increase | ❌ HARMFUL
Triple sig [5, 10, 15] + High-risk  | HR: 1.299 | 29.9% increase | ❌ HARMFUL
Triple sig [5, 10, 18] + High-risk  | HR: 1.482 | 48.2% increase | ❌ HARMFUL
Triple sig [5, 11, 13] + High-risk  | HR: 1.487 | 48.7% increase | ❌ HARMFUL
Triple sig [5, 11, 16] + High-risk  | HR: 1.514 | 51.4% increase | ❌ HARMFUL
Triple sig [5, 11, 19] + High-risk  | HR: 1.559 | 55.9% increase | ❌ HARMFUL
Triple sig [5, 12, 15] + High-risk  | HR: 1.477 | 47.7% increase | ❌ HARMFUL
Triple sig [5, 12, 18] + High-risk  | HR: 1.465 | 46.5% increase | ❌ HARMFUL
Triple sig [5, 13, 15] + High-risk  | HR: 1.471 | 47.1% increase | ❌ HARMFUL
Triple sig [5, 13, 18] + High-risk  | HR: 1.505 | 50.5% increase | ❌ HARMFUL
Triple sig [5, 14, 16] + High-risk  | HR: 1.574 | 57.4% increase | ❌ HARMFUL
Triple sig [5, 14, 19] + High-risk  | HR: 1.509 | 50.9% increase | ❌ HARMFUL
Triple sig [5, 15, 18] + High-risk  | HR: 1.453 | 45.3% increase | ❌ HARMFUL
Triple sig [5, 16, 18] + High-risk  | HR: 1.505 | 50.5% increase | ❌ HARMFUL
Triple sig [5, 17, 19] + High-risk  | HR: 1.552 | 55.2% increase | ❌ HARMFUL
Triple sig [6, 7, 9] + High-risk    | HR: 1.728 | 72.8% increase | ❌ HARMFUL
Triple sig [6, 7, 12] + High-risk   | HR: 1.733 | 73.3% increase | ❌ HARMFUL
Triple sig [6, 7, 15] + High-risk   | HR: 1.650 | 65.0% increase | ❌ HARMFUL
Triple sig [6, 7, 18] + High-risk   | HR: 1.630 | 63.0% increase | ❌ HARMFUL
Triple sig [6, 8, 10] + High-risk   | HR: 1.614 | 61.4% increase | ❌ HARMFUL
Triple sig [6, 8, 13] + High-risk   | HR: 1.579 | 57.9% increase | ❌ HARMFUL
Triple sig [6, 8, 16] + High-risk   | HR: 1.569 | 56.9% increase | ❌ HARMFUL
Triple sig [6, 8, 19] + High-risk   | HR: 1.569 | 56.9% increase | ❌ HARMFUL
Triple sig [6, 9, 12] + High-risk   | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Triple sig [6, 9, 15] + High-risk   | HR: 1.358 | 35.8% increase | ❌ HARMFUL
Triple sig [6, 9, 18] + High-risk   | HR: 1.530 | 53.0% increase | ❌ HARMFUL
Triple sig [6, 10, 12] + High-risk  | HR: 1.443 | 44.3% increase | ❌ HARMFUL
Triple sig [6, 10, 15] + High-risk  | HR: 1.259 | 25.9% increase | ❌ HARMFUL
Triple sig [6, 10, 18] + High-risk  | HR: 1.387 | 38.7% increase | ❌ HARMFUL
Triple sig [6, 11, 13] + High-risk  | HR: 1.607 | 60.7% increase | ❌ HARMFUL
Triple sig [6, 11, 16] + High-risk  | HR: 1.610 | 61.0% increase | ❌ HARMFUL
Triple sig [6, 11, 19] + High-risk  | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Triple sig [6, 12, 15] + High-risk  | HR: 1.448 | 44.8% increase | ❌ HARMFUL
Triple sig [6, 12, 18] + High-risk  | HR: 1.472 | 47.2% increase | ❌ HARMFUL
Triple sig [6, 13, 15] + High-risk  | HR: 1.477 | 47.7% increase | ❌ HARMFUL
Triple sig [6, 13, 18] + High-risk  | HR: 1.602 | 60.2% increase | ❌ HARMFUL
Triple sig [6, 14, 16] + High-risk  | HR: 1.682 | 68.2% increase | ❌ HARMFUL
Triple sig [6, 14, 19] + High-risk  | HR: 1.679 | 67.9% increase | ❌ HARMFUL
Triple sig [6, 15, 18] + High-risk  | HR: 1.377 | 37.7% increase | ❌ HARMFUL
Triple sig [6, 16, 18] + High-risk  | HR: 1.586 | 58.6% increase | ❌ HARMFUL
Triple sig [6, 17, 19] + High-risk  | HR: 1.747 | 74.7% increase | ❌ HARMFUL
Triple sig [7, 8, 10] + High-risk   | HR: 1.610 | 61.0% increase | ❌ HARMFUL
Triple sig [7, 8, 13] + High-risk   | HR: 1.670 | 67.0% increase | ❌ HARMFUL
Triple sig [7, 8, 16] + High-risk   | HR: 1.819 | 81.9% increase | ❌ HARMFUL
Triple sig [7, 8, 19] + High-risk   | HR: 1.744 | 74.4% increase | ❌ HARMFUL
Triple sig [7, 9, 12] + High-risk   | HR: 1.704 | 70.4% increase | ❌ HARMFUL
Triple sig [7, 9, 15] + High-risk   | HR: 1.656 | 65.6% increase | ❌ HARMFUL
Triple sig [7, 9, 18] + High-risk   | HR: 1.706 | 70.6% increase | ❌ HARMFUL
Triple sig [7, 10, 12] + High-risk  | HR: 1.660 | 66.0% increase | ❌ HARMFUL
Triple sig [7, 10, 15] + High-risk  | HR: 1.626 | 62.6% increase | ❌ HARMFUL
Triple sig [7, 10, 18] + High-risk  | HR: 1.710 | 71.0% increase | ❌ HARMFUL
Triple sig [7, 11, 13] + High-risk  | HR: 1.735 | 73.5% increase | ❌ HARMFUL
Triple sig [7, 11, 16] + High-risk  | HR: 1.739 | 73.9% increase | ❌ HARMFUL
Triple sig [7, 11, 19] + High-risk  | HR: 1.821 | 82.1% increase | ❌ HARMFUL
Triple sig [7, 12, 15] + High-risk  | HR: 1.772 | 77.2% increase | ❌ HARMFUL
Triple sig [7, 12, 18] + High-risk  | HR: 1.701 | 70.1% increase | ❌ HARMFUL
Triple sig [7, 13, 15] + High-risk  | HR: 1.708 | 70.8% increase | ❌ HARMFUL
Triple sig [7, 13, 18] + High-risk  | HR: 1.723 | 72.3% increase | ❌ HARMFUL
Triple sig [7, 14, 16] + High-risk  | HR: 1.876 | 87.6% increase | ❌ HARMFUL
Triple sig [7, 14, 19] + High-risk  | HR: 1.839 | 83.9% increase | ❌ HARMFUL
Triple sig [7, 15, 18] + High-risk  | HR: 1.695 | 69.5% increase | ❌ HARMFUL
Triple sig [7, 16, 18] + High-risk  | HR: 1.700 | 70.0% increase | ❌ HARMFUL
Triple sig [7, 17, 19] + High-risk  | HR: 1.929 | 92.9% increase | ❌ HARMFUL
Triple sig [8, 9, 11] + High-risk   | HR: 1.535 | 53.5% increase | ❌ HARMFUL
Triple sig [8, 9, 14] + High-risk   | HR: 1.505 | 50.5% increase | ❌ HARMFUL
Triple sig [8, 9, 17] + High-risk   | HR: 1.590 | 59.0% increase | ❌ HARMFUL
Triple sig [8, 10, 11] + High-risk  | HR: 1.544 | 54.4% increase | ❌ HARMFUL
Triple sig [8, 10, 14] + High-risk  | HR: 1.548 | 54.8% increase | ❌ HARMFUL
Triple sig [8, 10, 17] + High-risk  | HR: 1.575 | 57.5% increase | ❌ HARMFUL
Triple sig [8, 11, 12] + High-risk  | HR: 1.554 | 55.4% increase | ❌ HARMFUL
Triple sig [8, 11, 15] + High-risk  | HR: 1.566 | 56.6% increase | ❌ HARMFUL
Triple sig [8, 11, 18] + High-risk  | HR: 1.557 | 55.7% increase | ❌ HARMFUL
Triple sig [8, 12, 14] + High-risk  | HR: 1.649 | 64.9% increase | ❌ HARMFUL
Triple sig [8, 12, 17] + High-risk  | HR: 1.624 | 62.4% increase | ❌ HARMFUL
Triple sig [8, 13, 14] + High-risk  | HR: 1.635 | 63.5% increase | ❌ HARMFUL
Triple sig [8, 13, 17] + High-risk  | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Triple sig [8, 14, 15] + High-risk  | HR: 1.459 | 45.9% increase | ❌ HARMFUL
Triple sig [8, 14, 18] + High-risk  | HR: 1.648 | 64.8% increase | ❌ HARMFUL
Triple sig [8, 15, 17] + High-risk  | HR: 1.552 | 55.2% increase | ❌ HARMFUL
Triple sig [8, 16, 17] + High-risk  | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Triple sig [8, 17, 18] + High-risk  | HR: 1.552 | 55.2% increase | ❌ HARMFUL
Triple sig [9, 10, 11] + High-risk  | HR: 1.489 | 48.9% increase | ❌ HARMFUL
Triple sig [9, 10, 14] + High-risk  | HR: 1.574 | 57.4% increase | ❌ HARMFUL
Triple sig [9, 10, 17] + High-risk  | HR: 1.687 | 68.7% increase | ❌ HARMFUL
Triple sig [9, 11, 12] + High-risk  | HR: 1.612 | 61.2% increase | ❌ HARMFUL
Triple sig [9, 11, 15] + High-risk  | HR: 1.599 | 59.9% increase | ❌ HARMFUL
Triple sig [9, 11, 18] + High-risk  | HR: 1.519 | 51.9% increase | ❌ HARMFUL
Triple sig [9, 12, 14] + High-risk  | HR: 1.737 | 73.7% increase | ❌ HARMFUL
Triple sig [9, 12, 17] + High-risk  | HR: 1.635 | 63.5% increase | ❌ HARMFUL
Triple sig [9, 13, 14] + High-risk  | HR: 1.650 | 65.0% increase | ❌ HARMFUL
Triple sig [9, 13, 17] + High-risk  | HR: 1.637 | 63.7% increase | ❌ HARMFUL
Triple sig [9, 14, 15] + High-risk  | HR: 1.487 | 48.7% increase | ❌ HARMFUL
Triple sig [9, 14, 18] + High-risk  | HR: 1.694 | 69.4% increase | ❌ HARMFUL
Triple sig [9, 15, 17] + High-risk  | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Triple sig [9, 16, 17] + High-risk  | HR: 1.757 | 75.7% increase | ❌ HARMFUL
Triple sig [9, 17, 18] + High-risk  | HR: 1.705 | 70.5% increase | ❌ HARMFUL
Triple sig [10, 11, 12] + High-risk | HR: 1.549 | 54.9% increase | ❌ HARMFUL
Triple sig [10, 11, 15] + High-risk | HR: 1.363 | 36.3% increase | ❌ HARMFUL
Triple sig [10, 11, 18] + High-risk | HR: 1.546 | 54.6% increase | ❌ HARMFUL
Triple sig [10, 12, 14] + High-risk | HR: 1.700 | 70.0% increase | ❌ HARMFUL
Triple sig [10, 12, 17] + High-risk | HR: 1.628 | 62.8% increase | ❌ HARMFUL
Triple sig [10, 13, 14] + High-risk | HR: 1.544 | 54.4% increase | ❌ HARMFUL
Triple sig [10, 13, 17] + High-risk | HR: 1.609 | 60.9% increase | ❌ HARMFUL
Triple sig [10, 14, 15] + High-risk | HR: 1.491 | 49.1% increase | ❌ HARMFUL
Triple sig [10, 14, 18] + High-risk | HR: 1.664 | 66.4% increase | ❌ HARMFUL
Triple sig [10, 15, 17] + High-risk | HR: 1.558 | 55.8% increase | ❌ HARMFUL
Triple sig [10, 16, 17] + High-risk | HR: 1.765 | 76.5% increase | ❌ HARMFUL
Triple sig [10, 17, 18] + High-risk | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Triple sig [11, 12, 13] + High-risk | HR: 1.614 | 61.4% increase | ❌ HARMFUL
Triple sig [11, 12, 16] + High-risk | HR: 1.672 | 67.2% increase | ❌ HARMFUL
Triple sig [11, 12, 19] + High-risk | HR: 1.790 | 79.0% increase | ❌ HARMFUL
Triple sig [11, 13, 16] + High-risk | HR: 1.794 | 79.4% increase | ❌ HARMFUL
Triple sig [11, 13, 19] + High-risk | HR: 1.762 | 76.2% increase | ❌ HARMFUL
Triple sig [11, 14, 17] + High-risk | HR: 1.798 | 79.8% increase | ❌ HARMFUL
Triple sig [11, 15, 16] + High-risk | HR: 1.591 | 59.1% increase | ❌ HARMFUL
Triple sig [11, 15, 19] + High-risk | HR: 1.601 | 60.1% increase | ❌ HARMFUL
Triple sig [11, 16, 19] + High-risk | HR: 1.817 | 81.7% increase | ❌ HARMFUL
Triple sig [11, 18, 19] + High-risk | HR: 1.761 | 76.1% increase | ❌ HARMFUL
Triple sig [12, 13, 16] + High-risk | HR: 1.719 | 71.9% increase | ❌ HARMFUL
Triple sig [12, 13, 19] + High-risk | HR: 1.713 | 71.3% increase | ❌ HARMFUL
Triple sig [12, 14, 17] + High-risk | HR: 1.886 | 88.6% increase | ❌ HARMFUL
Triple sig [12, 15, 16] + High-risk | HR: 1.723 | 72.3% increase | ❌ HARMFUL
Triple sig [12, 15, 19] + High-risk | HR: 1.666 | 66.6% increase | ❌ HARMFUL
Triple sig [12, 16, 19] + High-risk | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Triple sig [12, 18, 19] + High-risk | HR: 1.568 | 56.8% increase | ❌ HARMFUL
Triple sig [13, 14, 17] + High-risk | HR: 1.792 | 79.2% increase | ❌ HARMFUL
Triple sig [13, 15, 16] + High-risk | HR: 1.683 | 68.3% increase | ❌ HARMFUL
Triple sig [13, 15, 19] + High-risk | HR: 1.692 | 69.2% increase | ❌ HARMFUL
Triple sig [13, 16, 19] + High-risk | HR: 1.766 | 76.6% increase | ❌ HARMFUL
Triple sig [13, 18, 19] + High-risk | HR: 1.883 | 88.3% increase | ❌ HARMFUL
Triple sig [14, 15, 18] + High-risk | HR: 1.518 | 51.8% increase | ❌ HARMFUL
Triple sig [14, 16, 18] + High-risk | HR: 1.858 | 85.8% increase | ❌ HARMFUL
Triple sig [14, 17, 19] + High-risk | HR: 1.791 | 79.1% increase | ❌ HARMFUL
Triple sig [15, 16, 18] + High-risk | HR: 1.690 | 69.0% increase | ❌ HARMFUL
Triple sig [15, 17, 19] + High-risk | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Triple sig [16, 17, 19] + High-risk | HR: 1.753 | 75.3% increase | ❌ HARMFUL
Quad sig [0, 1, 2, 3] + High-risk   | HR: 1.946 | 94.6% increase | ❌ HARMFUL
Quad sig [0, 1, 2, 13] + High-risk  | HR: 1.886 | 88.6% increase | ❌ HARMFUL
Quad sig [0, 1, 3, 7] + High-risk   | HR: 2.097 | 109.7% increase | ❌ HARMFUL
Quad sig [0, 1, 3, 17] + High-risk  | HR: 1.883 | 88.3% increase | ❌ HARMFUL
Quad sig [0, 1, 4, 12] + High-risk  | HR: 1.874 | 87.4% increase | ❌ HARMFUL
Quad sig [0, 1, 5, 8] + High-risk   | HR: 1.618 | 61.8% increase | ❌ HARMFUL
Quad sig [0, 1, 5, 18] + High-risk  | HR: 1.619 | 61.9% increase | ❌ HARMFUL
Quad sig [0, 1, 6, 15] + High-risk  | HR: 1.666 | 66.6% increase | ❌ HARMFUL
Quad sig [0, 1, 7, 13] + High-risk  | HR: 1.895 | 89.5% increase | ❌ HARMFUL
Quad sig [0, 1, 8, 12] + High-risk  | HR: 1.839 | 83.9% increase | ❌ HARMFUL
Quad sig [0, 1, 9, 12] + High-risk  | HR: 1.828 | 82.8% increase | ❌ HARMFUL
Quad sig [0, 1, 10, 13] + High-risk | HR: 1.860 | 86.0% increase | ❌ HARMFUL
Quad sig [0, 1, 11, 15] + High-risk | HR: 1.647 | 64.7% increase | ❌ HARMFUL
Quad sig [0, 1, 12, 18] + High-risk | HR: 1.781 | 78.1% increase | ❌ HARMFUL
Quad sig [0, 1, 14, 17] + High-risk | HR: 1.943 | 94.3% increase | ❌ HARMFUL
Quad sig [0, 1, 17, 18] + High-risk | HR: 1.833 | 83.3% increase | ❌ HARMFUL
Quad sig [0, 2, 3, 11] + High-risk  | HR: 2.022 | 102.2% increase | ❌ HARMFUL
Quad sig [0, 2, 4, 6] + High-risk   | HR: 1.777 | 77.7% increase | ❌ HARMFUL
Quad sig [0, 2, 4, 16] + High-risk  | HR: 1.851 | 85.1% increase | ❌ HARMFUL
Quad sig [0, 2, 5, 12] + High-risk  | HR: 1.647 | 64.7% increase | ❌ HARMFUL
Quad sig [0, 2, 6, 9] + High-risk   | HR: 1.702 | 70.2% increase | ❌ HARMFUL
Quad sig [0, 2, 6, 19] + High-risk  | HR: 1.879 | 87.9% increase | ❌ HARMFUL
Quad sig [0, 2, 7, 17] + High-risk  | HR: 1.898 | 89.8% increase | ❌ HARMFUL
Quad sig [0, 2, 8, 16] + High-risk  | HR: 1.847 | 84.7% increase | ❌ HARMFUL
Quad sig [0, 2, 9, 16] + High-risk  | HR: 1.859 | 85.9% increase | ❌ HARMFUL
Quad sig [0, 2, 10, 17] + High-risk | HR: 1.903 | 90.3% increase | ❌ HARMFUL
Quad sig [0, 2, 11, 19] + High-risk | HR: 1.995 | 99.5% increase | ❌ HARMFUL
Quad sig [0, 2, 13, 16] + High-risk | HR: 1.985 | 98.5% increase | ❌ HARMFUL
Quad sig [0, 2, 15, 17] + High-risk | HR: 1.895 | 89.5% increase | ❌ HARMFUL
Quad sig [0, 3, 4, 6] + High-risk   | HR: 1.854 | 85.4% increase | ❌ HARMFUL
Quad sig [0, 3, 4, 16] + High-risk  | HR: 2.003 | 100.3% increase | ❌ HARMFUL
Quad sig [0, 3, 5, 12] + High-risk  | HR: 1.758 | 75.8% increase | ❌ HARMFUL
Quad sig [0, 3, 6, 9] + High-risk   | HR: 1.690 | 69.0% increase | ❌ HARMFUL
Quad sig [0, 3, 6, 19] + High-risk  | HR: 2.019 | 101.9% increase | ❌ HARMFUL
Quad sig [0, 3, 7, 17] + High-risk  | HR: 1.950 | 95.0% increase | ❌ HARMFUL
Quad sig [0, 3, 8, 16] + High-risk  | HR: 1.926 | 92.6% increase | ❌ HARMFUL
Quad sig [0, 3, 9, 16] + High-risk  | HR: 1.871 | 87.1% increase | ❌ HARMFUL
Quad sig [0, 3, 10, 17] + High-risk | HR: 1.869 | 86.9% increase | ❌ HARMFUL
Quad sig [0, 3, 11, 19] + High-risk | HR: 2.211 | 121.1% increase | ❌ HARMFUL
Quad sig [0, 3, 13, 16] + High-risk | HR: 2.000 | 100.0% increase | ❌ HARMFUL
Quad sig [0, 3, 15, 17] + High-risk | HR: 2.064 | 106.4% increase | ❌ HARMFUL
Quad sig [0, 4, 5, 7] + High-risk   | HR: 1.634 | 63.4% increase | ❌ HARMFUL
Quad sig [0, 4, 5, 17] + High-risk  | HR: 1.557 | 55.7% increase | ❌ HARMFUL
Quad sig [0, 4, 6, 14] + High-risk  | HR: 1.764 | 76.4% increase | ❌ HARMFUL
Quad sig [0, 4, 7, 12] + High-risk  | HR: 1.954 | 95.4% increase | ❌ HARMFUL
Quad sig [0, 4, 8, 11] + High-risk  | HR: 1.710 | 71.0% increase | ❌ HARMFUL
Quad sig [0, 4, 9, 11] + High-risk  | HR: 1.798 | 79.8% increase | ❌ HARMFUL
Quad sig [0, 4, 10, 12] + High-risk | HR: 1.643 | 64.3% increase | ❌ HARMFUL
Quad sig [0, 4, 11, 14] + High-risk | HR: 1.718 | 71.8% increase | ❌ HARMFUL
Quad sig [0, 4, 12, 17] + High-risk | HR: 1.794 | 79.4% increase | ❌ HARMFUL
Quad sig [0, 4, 14, 16] + High-risk | HR: 1.907 | 90.7% increase | ❌ HARMFUL
Quad sig [0, 4, 16, 19] + High-risk | HR: 1.803 | 80.3% increase | ❌ HARMFUL
Quad sig [0, 5, 6, 13] + High-risk  | HR: 1.583 | 58.3% increase | ❌ HARMFUL
Quad sig [0, 5, 7, 11] + High-risk  | HR: 1.683 | 68.3% increase | ❌ HARMFUL
Quad sig [0, 5, 8, 10] + High-risk  | HR: 1.544 | 54.4% increase | ❌ HARMFUL
Quad sig [0, 5, 9, 10] + High-risk  | HR: 1.579 | 57.9% increase | ❌ HARMFUL
Quad sig [0, 5, 10, 11] + High-risk | HR: 1.535 | 53.5% increase | ❌ HARMFUL
Quad sig [0, 5, 11, 13] + High-risk | HR: 1.635 | 63.5% increase | ❌ HARMFUL
Quad sig [0, 5, 12, 16] + High-risk | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Quad sig [0, 5, 14, 15] + High-risk | HR: 1.527 | 52.7% increase | ❌ HARMFUL
Quad sig [0, 5, 16, 18] + High-risk | HR: 1.748 | 74.8% increase | ❌ HARMFUL
Quad sig [0, 6, 7, 13] + High-risk  | HR: 1.961 | 96.1% increase | ❌ HARMFUL
Quad sig [0, 6, 8, 12] + High-risk  | HR: 1.761 | 76.1% increase | ❌ HARMFUL
Quad sig [0, 6, 9, 12] + High-risk  | HR: 1.839 | 83.9% increase | ❌ HARMFUL
Quad sig [0, 6, 10, 13] + High-risk | HR: 1.809 | 80.9% increase | ❌ HARMFUL
Quad sig [0, 6, 11, 15] + High-risk | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Quad sig [0, 6, 12, 18] + High-risk | HR: 1.775 | 77.5% increase | ❌ HARMFUL
Quad sig [0, 6, 14, 17] + High-risk | HR: 1.926 | 92.6% increase | ❌ HARMFUL
Quad sig [0, 6, 17, 18] + High-risk | HR: 1.806 | 80.6% increase | ❌ HARMFUL
Quad sig [0, 7, 8, 16] + High-risk  | HR: 1.997 | 99.7% increase | ❌ HARMFUL
Quad sig [0, 7, 9, 16] + High-risk  | HR: 1.918 | 91.8% increase | ❌ HARMFUL
Quad sig [0, 7, 10, 17] + High-risk | HR: 1.903 | 90.3% increase | ❌ HARMFUL
Quad sig [0, 7, 11, 19] + High-risk | HR: 1.914 | 91.4% increase | ❌ HARMFUL
Quad sig [0, 7, 13, 16] + High-risk | HR: 1.962 | 96.2% increase | ❌ HARMFUL
Quad sig [0, 7, 15, 17] + High-risk | HR: 1.902 | 90.2% increase | ❌ HARMFUL
Quad sig [0, 8, 9, 11] + High-risk  | HR: 1.642 | 64.2% increase | ❌ HARMFUL
Quad sig [0, 8, 10, 12] + High-risk | HR: 1.672 | 67.2% increase | ❌ HARMFUL
Quad sig [0, 8, 11, 14] + High-risk | HR: 1.709 | 70.9% increase | ❌ HARMFUL
Quad sig [0, 8, 12, 17] + High-risk | HR: 1.764 | 76.4% increase | ❌ HARMFUL
Quad sig [0, 8, 14, 16] + High-risk | HR: 1.818 | 81.8% increase | ❌ HARMFUL
Quad sig [0, 8, 16, 19] + High-risk | HR: 1.835 | 83.5% increase | ❌ HARMFUL
Quad sig [0, 9, 10, 17] + High-risk | HR: 1.728 | 72.8% increase | ❌ HARMFUL
Quad sig [0, 9, 11, 19] + High-risk | HR: 1.855 | 85.5% increase | ❌ HARMFUL
Quad sig [0, 9, 13, 16] + High-risk | HR: 1.796 | 79.6% increase | ❌ HARMFUL
Quad sig [0, 9, 15, 17] + High-risk | HR: 1.796 | 79.6% increase | ❌ HARMFUL
Quad sig [0, 10, 11, 13] + High-risk | HR: 1.729 | 72.9% increase | ❌ HARMFUL
Quad sig [0, 10, 12, 16] + High-risk | HR: 1.810 | 81.0% increase | ❌ HARMFUL
Quad sig [0, 10, 14, 15] + High-risk | HR: 1.654 | 65.4% increase | ❌ HARMFUL
Quad sig [0, 10, 16, 18] + High-risk | HR: 1.923 | 92.3% increase | ❌ HARMFUL
Quad sig [0, 11, 12, 18] + High-risk | HR: 1.701 | 70.1% increase | ❌ HARMFUL
Quad sig [0, 11, 14, 17] + High-risk | HR: 1.885 | 88.5% increase | ❌ HARMFUL
Quad sig [0, 11, 17, 18] + High-risk | HR: 1.770 | 77.0% increase | ❌ HARMFUL
Quad sig [0, 12, 14, 16] + High-risk | HR: 1.883 | 88.3% increase | ❌ HARMFUL
Quad sig [0, 12, 16, 19] + High-risk | HR: 1.872 | 87.2% increase | ❌ HARMFUL
Quad sig [0, 13, 15, 17] + High-risk | HR: 1.973 | 97.3% increase | ❌ HARMFUL
Quad sig [0, 14, 15, 17] + High-risk | HR: 1.916 | 91.6% increase | ❌ HARMFUL
Quad sig [0, 15, 16, 18] + High-risk | HR: 2.014 | 101.4% increase | ❌ HARMFUL
Quad sig [1, 2, 3, 5] + High-risk   | HR: 1.647 | 64.7% increase | ❌ HARMFUL
Quad sig [1, 2, 3, 15] + High-risk  | HR: 1.887 | 88.7% increase | ❌ HARMFUL
Quad sig [1, 2, 4, 10] + High-risk  | HR: 1.813 | 81.3% increase | ❌ HARMFUL
Quad sig [1, 2, 5, 6] + High-risk   | HR: 1.628 | 62.8% increase | ❌ HARMFUL
Quad sig [1, 2, 5, 16] + High-risk  | HR: 1.623 | 62.3% increase | ❌ HARMFUL
Quad sig [1, 2, 6, 13] + High-risk  | HR: 1.668 | 66.8% increase | ❌ HARMFUL
Quad sig [1, 2, 7, 11] + High-risk  | HR: 1.832 | 83.2% increase | ❌ HARMFUL
Quad sig [1, 2, 8, 10] + High-risk  | HR: 1.655 | 65.5% increase | ❌ HARMFUL
Quad sig [1, 2, 9, 10] + High-risk  | HR: 1.727 | 72.7% increase | ❌ HARMFUL
Quad sig [1, 2, 10, 11] + High-risk | HR: 1.688 | 68.8% increase | ❌ HARMFUL
Quad sig [1, 2, 11, 13] + High-risk | HR: 1.613 | 61.3% increase | ❌ HARMFUL
Quad sig [1, 2, 12, 16] + High-risk | HR: 1.997 | 99.7% increase | ❌ HARMFUL
Quad sig [1, 2, 14, 15] + High-risk | HR: 1.681 | 68.1% increase | ❌ HARMFUL
Quad sig [1, 2, 16, 18] + High-risk | HR: 1.849 | 84.9% increase | ❌ HARMFUL
Quad sig [1, 3, 4, 10] + High-risk  | HR: 1.720 | 72.0% increase | ❌ HARMFUL
Quad sig [1, 3, 5, 6] + High-risk   | HR: 1.617 | 61.7% increase | ❌ HARMFUL
Quad sig [1, 3, 5, 16] + High-risk  | HR: 1.668 | 66.8% increase | ❌ HARMFUL
Quad sig [1, 3, 6, 13] + High-risk  | HR: 1.659 | 65.9% increase | ❌ HARMFUL
Quad sig [1, 3, 7, 11] + High-risk  | HR: 1.886 | 88.6% increase | ❌ HARMFUL
Quad sig [1, 3, 8, 10] + High-risk  | HR: 1.609 | 60.9% increase | ❌ HARMFUL
Quad sig [1, 3, 9, 10] + High-risk  | HR: 1.653 | 65.3% increase | ❌ HARMFUL
Quad sig [1, 3, 10, 11] + High-risk | HR: 1.747 | 74.7% increase | ❌ HARMFUL
Quad sig [1, 3, 11, 13] + High-risk | HR: 1.835 | 83.5% increase | ❌ HARMFUL
Quad sig [1, 3, 12, 16] + High-risk | HR: 1.932 | 93.2% increase | ❌ HARMFUL
Quad sig [1, 3, 14, 15] + High-risk | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Quad sig [1, 3, 16, 18] + High-risk | HR: 1.905 | 90.5% increase | ❌ HARMFUL
Quad sig [1, 4, 5, 11] + High-risk  | HR: 1.561 | 56.1% increase | ❌ HARMFUL
Quad sig [1, 4, 6, 8] + High-risk   | HR: 1.645 | 64.5% increase | ❌ HARMFUL
Quad sig [1, 4, 6, 18] + High-risk  | HR: 1.742 | 74.2% increase | ❌ HARMFUL
Quad sig [1, 4, 7, 16] + High-risk  | HR: 1.994 | 99.4% increase | ❌ HARMFUL
Quad sig [1, 4, 8, 15] + High-risk  | HR: 1.605 | 60.5% increase | ❌ HARMFUL
Quad sig [1, 4, 9, 15] + High-risk  | HR: 1.663 | 66.3% increase | ❌ HARMFUL
Quad sig [1, 4, 10, 16] + High-risk | HR: 1.761 | 76.1% increase | ❌ HARMFUL
Quad sig [1, 4, 11, 18] + High-risk | HR: 1.725 | 72.5% increase | ❌ HARMFUL
Quad sig [1, 4, 13, 15] + High-risk | HR: 1.696 | 69.6% increase | ❌ HARMFUL
Quad sig [1, 4, 15, 16] + High-risk | HR: 1.838 | 83.8% increase | ❌ HARMFUL
Quad sig [1, 5, 6, 7] + High-risk   | HR: 1.751 | 75.1% increase | ❌ HARMFUL
Quad sig [1, 5, 6, 17] + High-risk  | HR: 1.576 | 57.6% increase | ❌ HARMFUL
Quad sig [1, 5, 7, 15] + High-risk  | HR: 1.812 | 81.2% increase | ❌ HARMFUL
Quad sig [1, 5, 8, 14] + High-risk  | HR: 1.606 | 60.6% increase | ❌ HARMFUL
Quad sig [1, 5, 9, 14] + High-risk  | HR: 1.637 | 63.7% increase | ❌ HARMFUL
Quad sig [1, 5, 10, 15] + High-risk | HR: 1.579 | 57.9% increase | ❌ HARMFUL
Quad sig [1, 5, 11, 17] + High-risk | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Quad sig [1, 5, 13, 14] + High-risk | HR: 1.741 | 74.1% increase | ❌ HARMFUL
Quad sig [1, 5, 14, 19] + High-risk | HR: 1.630 | 63.0% increase | ❌ HARMFUL
Quad sig [1, 5, 18, 19] + High-risk | HR: 1.655 | 65.5% increase | ❌ HARMFUL
Quad sig [1, 6, 7, 17] + High-risk  | HR: 1.758 | 75.8% increase | ❌ HARMFUL
Quad sig [1, 6, 8, 16] + High-risk  | HR: 1.648 | 64.8% increase | ❌ HARMFUL
Quad sig [1, 6, 9, 16] + High-risk  | HR: 1.705 | 70.5% increase | ❌ HARMFUL
Quad sig [1, 6, 10, 17] + High-risk | HR: 1.660 | 66.0% increase | ❌ HARMFUL
Quad sig [1, 6, 11, 19] + High-risk | HR: 1.772 | 77.2% increase | ❌ HARMFUL
Quad sig [1, 6, 13, 16] + High-risk | HR: 1.848 | 84.8% increase | ❌ HARMFUL
Quad sig [1, 6, 15, 17] + High-risk | HR: 1.586 | 58.6% increase | ❌ HARMFUL
Quad sig [1, 7, 8, 10] + High-risk  | HR: 1.665 | 66.5% increase | ❌ HARMFUL
Quad sig [1, 7, 9, 10] + High-risk  | HR: 1.861 | 86.1% increase | ❌ HARMFUL
Quad sig [1, 7, 10, 11] + High-risk | HR: 1.767 | 76.7% increase | ❌ HARMFUL
Quad sig [1, 7, 11, 13] + High-risk | HR: 1.835 | 83.5% increase | ❌ HARMFUL
Quad sig [1, 7, 12, 16] + High-risk | HR: 2.020 | 102.0% increase | ❌ HARMFUL
Quad sig [1, 7, 14, 15] + High-risk | HR: 1.846 | 84.6% increase | ❌ HARMFUL
Quad sig [1, 7, 16, 18] + High-risk | HR: 1.831 | 83.1% increase | ❌ HARMFUL
Quad sig [1, 8, 9, 15] + High-risk  | HR: 1.633 | 63.3% increase | ❌ HARMFUL
Quad sig [1, 8, 10, 16] + High-risk | HR: 1.740 | 74.0% increase | ❌ HARMFUL
Quad sig [1, 8, 11, 18] + High-risk | HR: 1.647 | 64.7% increase | ❌ HARMFUL
Quad sig [1, 8, 13, 15] + High-risk | HR: 1.619 | 61.9% increase | ❌ HARMFUL
Quad sig [1, 8, 15, 16] + High-risk | HR: 1.748 | 74.8% increase | ❌ HARMFUL
Quad sig [1, 9, 10, 11] + High-risk | HR: 1.671 | 67.1% increase | ❌ HARMFUL
Quad sig [1, 9, 11, 13] + High-risk | HR: 1.651 | 65.1% increase | ❌ HARMFUL
Quad sig [1, 9, 12, 16] + High-risk | HR: 1.813 | 81.3% increase | ❌ HARMFUL
Quad sig [1, 9, 14, 15] + High-risk | HR: 1.668 | 66.8% increase | ❌ HARMFUL
Quad sig [1, 9, 16, 18] + High-risk | HR: 1.995 | 99.5% increase | ❌ HARMFUL
Quad sig [1, 10, 11, 17] + High-risk | HR: 1.759 | 75.9% increase | ❌ HARMFUL
Quad sig [1, 10, 13, 14] + High-risk | HR: 1.709 | 70.9% increase | ❌ HARMFUL
Quad sig [1, 10, 14, 19] + High-risk | HR: 1.803 | 80.3% increase | ❌ HARMFUL
Quad sig [1, 10, 18, 19] + High-risk | HR: 1.806 | 80.6% increase | ❌ HARMFUL
Quad sig [1, 11, 13, 16] + High-risk | HR: 1.983 | 98.3% increase | ❌ HARMFUL
Quad sig [1, 11, 15, 17] + High-risk | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Quad sig [1, 12, 13, 15] + High-risk | HR: 1.737 | 73.7% increase | ❌ HARMFUL
Quad sig [1, 12, 15, 16] + High-risk | HR: 1.868 | 86.8% increase | ❌ HARMFUL
Quad sig [1, 13, 14, 15] + High-risk | HR: 1.665 | 66.5% increase | ❌ HARMFUL
Quad sig [1, 13, 16, 18] + High-risk | HR: 1.973 | 97.3% increase | ❌ HARMFUL
Quad sig [1, 14, 16, 18] + High-risk | HR: 1.955 | 95.5% increase | ❌ HARMFUL
Quad sig [1, 15, 18, 19] + High-risk | HR: 1.844 | 84.4% increase | ❌ HARMFUL
Quad sig [2, 3, 4, 10] + High-risk  | HR: 1.888 | 88.8% increase | ❌ HARMFUL
Quad sig [2, 3, 5, 6] + High-risk   | HR: 1.598 | 59.8% increase | ❌ HARMFUL
Quad sig [2, 3, 5, 16] + High-risk  | HR: 1.659 | 65.9% increase | ❌ HARMFUL
Quad sig [2, 3, 6, 13] + High-risk  | HR: 1.855 | 85.5% increase | ❌ HARMFUL
Quad sig [2, 3, 7, 11] + High-risk  | HR: 1.983 | 98.3% increase | ❌ HARMFUL
Quad sig [2, 3, 8, 10] + High-risk  | HR: 1.838 | 83.8% increase | ❌ HARMFUL
Quad sig [2, 3, 9, 10] + High-risk  | HR: 1.769 | 76.9% increase | ❌ HARMFUL
Quad sig [2, 3, 10, 11] + High-risk | HR: 1.849 | 84.9% increase | ❌ HARMFUL
Quad sig [2, 3, 11, 13] + High-risk | HR: 1.927 | 92.7% increase | ❌ HARMFUL
Quad sig [2, 3, 12, 16] + High-risk | HR: 2.041 | 104.1% increase | ❌ HARMFUL
Quad sig [2, 3, 14, 15] + High-risk | HR: 1.773 | 77.3% increase | ❌ HARMFUL
Quad sig [2, 3, 16, 18] + High-risk | HR: 2.002 | 100.2% increase | ❌ HARMFUL
Quad sig [2, 4, 5, 11] + High-risk  | HR: 1.539 | 53.9% increase | ❌ HARMFUL
Quad sig [2, 4, 6, 8] + High-risk   | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Quad sig [2, 4, 6, 18] + High-risk  | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Quad sig [2, 4, 7, 16] + High-risk  | HR: 1.985 | 98.5% increase | ❌ HARMFUL
Quad sig [2, 4, 8, 15] + High-risk  | HR: 1.606 | 60.6% increase | ❌ HARMFUL
Quad sig [2, 4, 9, 15] + High-risk  | HR: 1.629 | 62.9% increase | ❌ HARMFUL
Quad sig [2, 4, 10, 16] + High-risk | HR: 1.728 | 72.8% increase | ❌ HARMFUL
Quad sig [2, 4, 11, 18] + High-risk | HR: 1.707 | 70.7% increase | ❌ HARMFUL
Quad sig [2, 4, 13, 15] + High-risk | HR: 1.675 | 67.5% increase | ❌ HARMFUL
Quad sig [2, 4, 15, 16] + High-risk | HR: 1.769 | 76.9% increase | ❌ HARMFUL
Quad sig [2, 5, 6, 7] + High-risk   | HR: 1.653 | 65.3% increase | ❌ HARMFUL
Quad sig [2, 5, 6, 17] + High-risk  | HR: 1.584 | 58.4% increase | ❌ HARMFUL
Quad sig [2, 5, 7, 15] + High-risk  | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Quad sig [2, 5, 8, 14] + High-risk  | HR: 1.555 | 55.5% increase | ❌ HARMFUL
Quad sig [2, 5, 9, 14] + High-risk  | HR: 1.555 | 55.5% increase | ❌ HARMFUL
Quad sig [2, 5, 10, 15] + High-risk | HR: 1.480 | 48.0% increase | ❌ HARMFUL
Quad sig [2, 5, 11, 17] + High-risk | HR: 1.614 | 61.4% increase | ❌ HARMFUL
Quad sig [2, 5, 13, 14] + High-risk | HR: 1.635 | 63.5% increase | ❌ HARMFUL
Quad sig [2, 5, 14, 19] + High-risk | HR: 1.577 | 57.7% increase | ❌ HARMFUL
Quad sig [2, 5, 18, 19] + High-risk | HR: 1.573 | 57.3% increase | ❌ HARMFUL
Quad sig [2, 6, 7, 17] + High-risk  | HR: 1.774 | 77.4% increase | ❌ HARMFUL
Quad sig [2, 6, 8, 16] + High-risk  | HR: 1.647 | 64.7% increase | ❌ HARMFUL
Quad sig [2, 6, 9, 16] + High-risk  | HR: 1.650 | 65.0% increase | ❌ HARMFUL
Quad sig [2, 6, 10, 17] + High-risk | HR: 1.696 | 69.6% increase | ❌ HARMFUL
Quad sig [2, 6, 11, 19] + High-risk | HR: 1.825 | 82.5% increase | ❌ HARMFUL
Quad sig [2, 6, 13, 16] + High-risk | HR: 1.830 | 83.0% increase | ❌ HARMFUL
Quad sig [2, 6, 15, 17] + High-risk | HR: 1.589 | 58.9% increase | ❌ HARMFUL
Quad sig [2, 7, 8, 10] + High-risk  | HR: 1.680 | 68.0% increase | ❌ HARMFUL
Quad sig [2, 7, 9, 10] + High-risk  | HR: 1.790 | 79.0% increase | ❌ HARMFUL
Quad sig [2, 7, 10, 11] + High-risk | HR: 1.703 | 70.3% increase | ❌ HARMFUL
Quad sig [2, 7, 11, 13] + High-risk | HR: 1.925 | 92.5% increase | ❌ HARMFUL
Quad sig [2, 7, 12, 16] + High-risk | HR: 2.014 | 101.4% increase | ❌ HARMFUL
Quad sig [2, 7, 14, 15] + High-risk | HR: 1.771 | 77.1% increase | ❌ HARMFUL
Quad sig [2, 7, 16, 18] + High-risk | HR: 1.945 | 94.5% increase | ❌ HARMFUL
Quad sig [2, 8, 9, 15] + High-risk  | HR: 1.603 | 60.3% increase | ❌ HARMFUL
Quad sig [2, 8, 10, 16] + High-risk | HR: 1.690 | 69.0% increase | ❌ HARMFUL
Quad sig [2, 8, 11, 18] + High-risk | HR: 1.611 | 61.1% increase | ❌ HARMFUL
Quad sig [2, 8, 13, 15] + High-risk | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Quad sig [2, 8, 15, 16] + High-risk | HR: 1.716 | 71.6% increase | ❌ HARMFUL
Quad sig [2, 9, 10, 11] + High-risk | HR: 1.596 | 59.6% increase | ❌ HARMFUL
Quad sig [2, 9, 11, 13] + High-risk | HR: 1.600 | 60.0% increase | ❌ HARMFUL
Quad sig [2, 9, 12, 16] + High-risk | HR: 1.822 | 82.2% increase | ❌ HARMFUL
Quad sig [2, 9, 14, 15] + High-risk | HR: 1.524 | 52.4% increase | ❌ HARMFUL
Quad sig [2, 9, 16, 18] + High-risk | HR: 1.864 | 86.4% increase | ❌ HARMFUL
Quad sig [2, 10, 11, 17] + High-risk | HR: 1.841 | 84.1% increase | ❌ HARMFUL
Quad sig [2, 10, 13, 14] + High-risk | HR: 1.664 | 66.4% increase | ❌ HARMFUL
Quad sig [2, 10, 14, 19] + High-risk | HR: 1.782 | 78.2% increase | ❌ HARMFUL
Quad sig [2, 10, 18, 19] + High-risk | HR: 1.765 | 76.5% increase | ❌ HARMFUL
Quad sig [2, 11, 13, 16] + High-risk | HR: 1.906 | 90.6% increase | ❌ HARMFUL
Quad sig [2, 11, 15, 17] + High-risk | HR: 1.643 | 64.3% increase | ❌ HARMFUL
Quad sig [2, 12, 13, 15] + High-risk | HR: 1.729 | 72.9% increase | ❌ HARMFUL
Quad sig [2, 12, 15, 16] + High-risk | HR: 1.808 | 80.8% increase | ❌ HARMFUL
Quad sig [2, 13, 14, 15] + High-risk | HR: 1.614 | 61.4% increase | ❌ HARMFUL
Quad sig [2, 13, 16, 18] + High-risk | HR: 1.949 | 94.9% increase | ❌ HARMFUL
Quad sig [2, 14, 16, 18] + High-risk | HR: 1.848 | 84.8% increase | ❌ HARMFUL
Quad sig [2, 15, 18, 19] + High-risk | HR: 1.883 | 88.3% increase | ❌ HARMFUL
Quad sig [3, 4, 5, 11] + High-risk  | HR: 1.661 | 66.1% increase | ❌ HARMFUL
Quad sig [3, 4, 6, 8] + High-risk   | HR: 1.705 | 70.5% increase | ❌ HARMFUL
Quad sig [3, 4, 6, 18] + High-risk  | HR: 1.824 | 82.4% increase | ❌ HARMFUL
Quad sig [3, 4, 7, 16] + High-risk  | HR: 2.108 | 110.8% increase | ❌ HARMFUL
Quad sig [3, 4, 8, 15] + High-risk  | HR: 1.752 | 75.2% increase | ❌ HARMFUL
Quad sig [3, 4, 9, 15] + High-risk  | HR: 1.818 | 81.8% increase | ❌ HARMFUL
Quad sig [3, 4, 10, 16] + High-risk | HR: 1.886 | 88.6% increase | ❌ HARMFUL
Quad sig [3, 4, 11, 18] + High-risk | HR: 1.802 | 80.2% increase | ❌ HARMFUL
Quad sig [3, 4, 13, 15] + High-risk | HR: 1.882 | 88.2% increase | ❌ HARMFUL
Quad sig [3, 4, 15, 16] + High-risk | HR: 1.951 | 95.1% increase | ❌ HARMFUL
Quad sig [3, 5, 6, 7] + High-risk   | HR: 1.744 | 74.4% increase | ❌ HARMFUL
Quad sig [3, 5, 6, 17] + High-risk  | HR: 1.580 | 58.0% increase | ❌ HARMFUL
Quad sig [3, 5, 7, 15] + High-risk  | HR: 1.786 | 78.6% increase | ❌ HARMFUL
Quad sig [3, 5, 8, 14] + High-risk  | HR: 1.622 | 62.2% increase | ❌ HARMFUL
Quad sig [3, 5, 9, 14] + High-risk  | HR: 1.590 | 59.0% increase | ❌ HARMFUL
Quad sig [3, 5, 10, 15] + High-risk | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Quad sig [3, 5, 11, 17] + High-risk | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Quad sig [3, 5, 13, 14] + High-risk | HR: 1.687 | 68.7% increase | ❌ HARMFUL
Quad sig [3, 5, 14, 19] + High-risk | HR: 1.717 | 71.7% increase | ❌ HARMFUL
Quad sig [3, 5, 18, 19] + High-risk | HR: 1.736 | 73.6% increase | ❌ HARMFUL
Quad sig [3, 6, 7, 17] + High-risk  | HR: 1.855 | 85.5% increase | ❌ HARMFUL
Quad sig [3, 6, 8, 16] + High-risk  | HR: 1.658 | 65.8% increase | ❌ HARMFUL
Quad sig [3, 6, 9, 16] + High-risk  | HR: 1.571 | 57.1% increase | ❌ HARMFUL
Quad sig [3, 6, 10, 17] + High-risk | HR: 1.686 | 68.6% increase | ❌ HARMFUL
Quad sig [3, 6, 11, 19] + High-risk | HR: 1.809 | 80.9% increase | ❌ HARMFUL
Quad sig [3, 6, 13, 16] + High-risk | HR: 1.774 | 77.4% increase | ❌ HARMFUL
Quad sig [3, 6, 15, 17] + High-risk | HR: 1.689 | 68.9% increase | ❌ HARMFUL
Quad sig [3, 7, 8, 10] + High-risk  | HR: 1.950 | 95.0% increase | ❌ HARMFUL
Quad sig [3, 7, 9, 10] + High-risk  | HR: 1.937 | 93.7% increase | ❌ HARMFUL
Quad sig [3, 7, 10, 11] + High-risk | HR: 1.822 | 82.2% increase | ❌ HARMFUL
Quad sig [3, 7, 11, 13] + High-risk | HR: 2.089 | 108.9% increase | ❌ HARMFUL
Quad sig [3, 7, 12, 16] + High-risk | HR: 2.040 | 104.0% increase | ❌ HARMFUL
Quad sig [3, 7, 14, 15] + High-risk | HR: 1.906 | 90.6% increase | ❌ HARMFUL
Quad sig [3, 7, 16, 18] + High-risk | HR: 2.066 | 106.6% increase | ❌ HARMFUL
Quad sig [3, 8, 9, 15] + High-risk  | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Quad sig [3, 8, 10, 16] + High-risk | HR: 1.759 | 75.9% increase | ❌ HARMFUL
Quad sig [3, 8, 11, 18] + High-risk | HR: 1.778 | 77.8% increase | ❌ HARMFUL
Quad sig [3, 8, 13, 15] + High-risk | HR: 1.840 | 84.0% increase | ❌ HARMFUL
Quad sig [3, 8, 15, 16] + High-risk | HR: 1.905 | 90.5% increase | ❌ HARMFUL
Quad sig [3, 9, 10, 11] + High-risk | HR: 1.715 | 71.5% increase | ❌ HARMFUL
Quad sig [3, 9, 11, 13] + High-risk | HR: 1.747 | 74.7% increase | ❌ HARMFUL
Quad sig [3, 9, 12, 16] + High-risk | HR: 1.851 | 85.1% increase | ❌ HARMFUL
Quad sig [3, 9, 14, 15] + High-risk | HR: 1.731 | 73.1% increase | ❌ HARMFUL
Quad sig [3, 9, 16, 18] + High-risk | HR: 1.878 | 87.8% increase | ❌ HARMFUL
Quad sig [3, 10, 11, 17] + High-risk | HR: 1.738 | 73.8% increase | ❌ HARMFUL
Quad sig [3, 10, 13, 14] + High-risk | HR: 1.839 | 83.9% increase | ❌ HARMFUL
Quad sig [3, 10, 14, 19] + High-risk | HR: 1.981 | 98.1% increase | ❌ HARMFUL
Quad sig [3, 10, 18, 19] + High-risk | HR: 1.852 | 85.2% increase | ❌ HARMFUL
Quad sig [3, 11, 13, 16] + High-risk | HR: 1.981 | 98.1% increase | ❌ HARMFUL
Quad sig [3, 11, 15, 17] + High-risk | HR: 1.851 | 85.1% increase | ❌ HARMFUL
Quad sig [3, 12, 13, 15] + High-risk | HR: 1.841 | 84.1% increase | ❌ HARMFUL
Quad sig [3, 12, 15, 16] + High-risk | HR: 2.034 | 103.4% increase | ❌ HARMFUL
Quad sig [3, 13, 14, 15] + High-risk | HR: 1.790 | 79.0% increase | ❌ HARMFUL
Quad sig [3, 13, 16, 18] + High-risk | HR: 1.903 | 90.3% increase | ❌ HARMFUL
Quad sig [3, 14, 16, 18] + High-risk | HR: 2.028 | 102.8% increase | ❌ HARMFUL
Quad sig [3, 15, 18, 19] + High-risk | HR: 1.859 | 85.9% increase | ❌ HARMFUL
Quad sig [4, 5, 6, 12] + High-risk  | HR: 1.557 | 55.7% increase | ❌ HARMFUL
Quad sig [4, 5, 7, 10] + High-risk  | HR: 1.557 | 55.7% increase | ❌ HARMFUL
Quad sig [4, 5, 8, 9] + High-risk   | HR: 1.526 | 52.6% increase | ❌ HARMFUL
Quad sig [4, 5, 8, 19] + High-risk  | HR: 1.595 | 59.5% increase | ❌ HARMFUL
Quad sig [4, 5, 9, 19] + High-risk  | HR: 1.643 | 64.3% increase | ❌ HARMFUL
Quad sig [4, 5, 11, 12] + High-risk | HR: 1.648 | 64.8% increase | ❌ HARMFUL
Quad sig [4, 5, 12, 15] + High-risk | HR: 1.609 | 60.9% increase | ❌ HARMFUL
Quad sig [4, 5, 13, 19] + High-risk | HR: 1.701 | 70.1% increase | ❌ HARMFUL
Quad sig [4, 5, 16, 17] + High-risk | HR: 1.614 | 61.4% increase | ❌ HARMFUL
Quad sig [4, 6, 7, 12] + High-risk  | HR: 1.691 | 69.1% increase | ❌ HARMFUL
Quad sig [4, 6, 8, 11] + High-risk  | HR: 1.524 | 52.4% increase | ❌ HARMFUL
Quad sig [4, 6, 9, 11] + High-risk  | HR: 1.579 | 57.9% increase | ❌ HARMFUL
Quad sig [4, 6, 10, 12] + High-risk | HR: 1.581 | 58.1% increase | ❌ HARMFUL
Quad sig [4, 6, 11, 14] + High-risk | HR: 1.570 | 57.0% increase | ❌ HARMFUL
Quad sig [4, 6, 12, 17] + High-risk | HR: 1.752 | 75.2% increase | ❌ HARMFUL
Quad sig [4, 6, 14, 16] + High-risk | HR: 1.809 | 80.9% increase | ❌ HARMFUL
Quad sig [4, 6, 16, 19] + High-risk | HR: 1.839 | 83.9% increase | ❌ HARMFUL
Quad sig [4, 7, 8, 15] + High-risk  | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Quad sig [4, 7, 9, 15] + High-risk  | HR: 1.767 | 76.7% increase | ❌ HARMFUL
Quad sig [4, 7, 10, 16] + High-risk | HR: 1.809 | 80.9% increase | ❌ HARMFUL
Quad sig [4, 7, 11, 18] + High-risk | HR: 1.719 | 71.9% increase | ❌ HARMFUL
Quad sig [4, 7, 13, 15] + High-risk | HR: 1.752 | 75.2% increase | ❌ HARMFUL
Quad sig [4, 7, 15, 16] + High-risk | HR: 1.803 | 80.3% increase | ❌ HARMFUL
Quad sig [4, 8, 9, 10] + High-risk  | HR: 1.516 | 51.6% increase | ❌ HARMFUL
Quad sig [4, 8, 10, 11] + High-risk | HR: 1.568 | 56.8% increase | ❌ HARMFUL
Quad sig [4, 8, 11, 13] + High-risk | HR: 1.624 | 62.4% increase | ❌ HARMFUL
Quad sig [4, 8, 12, 16] + High-risk | HR: 1.648 | 64.8% increase | ❌ HARMFUL
Quad sig [4, 8, 14, 15] + High-risk | HR: 1.528 | 52.8% increase | ❌ HARMFUL
Quad sig [4, 8, 16, 18] + High-risk | HR: 1.803 | 80.3% increase | ❌ HARMFUL
Quad sig [4, 9, 10, 16] + High-risk | HR: 1.702 | 70.2% increase | ❌ HARMFUL
Quad sig [4, 9, 11, 18] + High-risk | HR: 1.633 | 63.3% increase | ❌ HARMFUL
Quad sig [4, 9, 13, 15] + High-risk | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Quad sig [4, 9, 15, 16] + High-risk | HR: 1.770 | 77.0% increase | ❌ HARMFUL
Quad sig [4, 10, 11, 12] + High-risk | HR: 1.692 | 69.2% increase | ❌ HARMFUL
Quad sig [4, 10, 12, 15] + High-risk | HR: 1.559 | 55.9% increase | ❌ HARMFUL
Quad sig [4, 10, 13, 19] + High-risk | HR: 1.809 | 80.9% increase | ❌ HARMFUL
Quad sig [4, 10, 16, 17] + High-risk | HR: 1.830 | 83.0% increase | ❌ HARMFUL
Quad sig [4, 11, 12, 17] + High-risk | HR: 1.710 | 71.0% increase | ❌ HARMFUL
Quad sig [4, 11, 14, 16] + High-risk | HR: 1.746 | 74.6% increase | ❌ HARMFUL
Quad sig [4, 11, 16, 19] + High-risk | HR: 1.843 | 84.3% increase | ❌ HARMFUL
Quad sig [4, 12, 14, 15] + High-risk | HR: 1.659 | 65.9% increase | ❌ HARMFUL
Quad sig [4, 12, 16, 18] + High-risk | HR: 1.680 | 68.0% increase | ❌ HARMFUL
Quad sig [4, 13, 15, 16] + High-risk | HR: 1.823 | 82.3% increase | ❌ HARMFUL
Quad sig [4, 14, 15, 16] + High-risk | HR: 1.743 | 74.3% increase | ❌ HARMFUL
Quad sig [4, 15, 16, 17] + High-risk | HR: 1.767 | 76.7% increase | ❌ HARMFUL
Quad sig [5, 6, 7, 8] + High-risk   | HR: 1.615 | 61.5% increase | ❌ HARMFUL
Quad sig [5, 6, 7, 18] + High-risk  | HR: 1.637 | 63.7% increase | ❌ HARMFUL
Quad sig [5, 6, 8, 17] + High-risk  | HR: 1.534 | 53.4% increase | ❌ HARMFUL
Quad sig [5, 6, 9, 17] + High-risk  | HR: 1.551 | 55.1% increase | ❌ HARMFUL
Quad sig [5, 6, 10, 18] + High-risk | HR: 1.544 | 54.4% increase | ❌ HARMFUL
Quad sig [5, 6, 12, 13] + High-risk | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Quad sig [5, 6, 13, 17] + High-risk | HR: 1.631 | 63.1% increase | ❌ HARMFUL
Quad sig [5, 6, 15, 18] + High-risk | HR: 1.494 | 49.4% increase | ❌ HARMFUL
Quad sig [5, 7, 8, 11] + High-risk  | HR: 1.607 | 60.7% increase | ❌ HARMFUL
Quad sig [5, 7, 9, 11] + High-risk  | HR: 1.632 | 63.2% increase | ❌ HARMFUL
Quad sig [5, 7, 10, 12] + High-risk | HR: 1.759 | 75.9% increase | ❌ HARMFUL
Quad sig [5, 7, 11, 14] + High-risk | HR: 1.694 | 69.4% increase | ❌ HARMFUL
Quad sig [5, 7, 12, 17] + High-risk | HR: 1.621 | 62.1% increase | ❌ HARMFUL
Quad sig [5, 7, 14, 16] + High-risk | HR: 1.702 | 70.2% increase | ❌ HARMFUL
Quad sig [5, 7, 16, 19] + High-risk | HR: 1.673 | 67.3% increase | ❌ HARMFUL
Quad sig [5, 8, 9, 16] + High-risk  | HR: 1.466 | 46.6% increase | ❌ HARMFUL
Quad sig [5, 8, 10, 17] + High-risk | HR: 1.555 | 55.5% increase | ❌ HARMFUL
Quad sig [5, 8, 11, 19] + High-risk | HR: 1.549 | 54.9% increase | ❌ HARMFUL
Quad sig [5, 8, 13, 16] + High-risk | HR: 1.557 | 55.7% increase | ❌ HARMFUL
Quad sig [5, 8, 15, 17] + High-risk | HR: 1.506 | 50.6% increase | ❌ HARMFUL
Quad sig [5, 9, 10, 12] + High-risk | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Quad sig [5, 9, 11, 14] + High-risk | HR: 1.573 | 57.3% increase | ❌ HARMFUL
Quad sig [5, 9, 12, 17] + High-risk | HR: 1.623 | 62.3% increase | ❌ HARMFUL
Quad sig [5, 9, 14, 16] + High-risk | HR: 1.611 | 61.1% increase | ❌ HARMFUL
Quad sig [5, 9, 16, 19] + High-risk | HR: 1.518 | 51.8% increase | ❌ HARMFUL
Quad sig [5, 10, 11, 18] + High-risk | HR: 1.517 | 51.7% increase | ❌ HARMFUL
Quad sig [5, 10, 13, 15] + High-risk | HR: 1.504 | 50.4% increase | ❌ HARMFUL
Quad sig [5, 10, 15, 16] + High-risk | HR: 1.555 | 55.5% increase | ❌ HARMFUL
Quad sig [5, 11, 12, 13] + High-risk | HR: 1.612 | 61.2% increase | ❌ HARMFUL
Quad sig [5, 11, 13, 17] + High-risk | HR: 1.646 | 64.6% increase | ❌ HARMFUL
Quad sig [5, 11, 15, 18] + High-risk | HR: 1.531 | 53.1% increase | ❌ HARMFUL
Quad sig [5, 12, 13, 16] + High-risk | HR: 1.631 | 63.1% increase | ❌ HARMFUL
Quad sig [5, 12, 15, 17] + High-risk | HR: 1.606 | 60.6% increase | ❌ HARMFUL
Quad sig [5, 13, 14, 16] + High-risk | HR: 1.707 | 70.7% increase | ❌ HARMFUL
Quad sig [5, 13, 16, 19] + High-risk | HR: 1.640 | 64.0% increase | ❌ HARMFUL
Quad sig [5, 14, 16, 19] + High-risk | HR: 1.622 | 62.2% increase | ❌ HARMFUL
Quad sig [5, 16, 17, 18] + High-risk | HR: 1.585 | 58.5% increase | ❌ HARMFUL
Quad sig [6, 7, 8, 15] + High-risk  | HR: 1.682 | 68.2% increase | ❌ HARMFUL
Quad sig [6, 7, 9, 15] + High-risk  | HR: 1.804 | 80.4% increase | ❌ HARMFUL
Quad sig [6, 7, 10, 16] + High-risk | HR: 1.769 | 76.9% increase | ❌ HARMFUL
Quad sig [6, 7, 11, 18] + High-risk | HR: 1.665 | 66.5% increase | ❌ HARMFUL
Quad sig [6, 7, 13, 15] + High-risk | HR: 1.788 | 78.8% increase | ❌ HARMFUL
Quad sig [6, 7, 15, 16] + High-risk | HR: 1.743 | 74.3% increase | ❌ HARMFUL
Quad sig [6, 8, 9, 10] + High-risk  | HR: 1.529 | 52.9% increase | ❌ HARMFUL
Quad sig [6, 8, 10, 11] + High-risk | HR: 1.572 | 57.2% increase | ❌ HARMFUL
Quad sig [6, 8, 11, 13] + High-risk | HR: 1.709 | 70.9% increase | ❌ HARMFUL
Quad sig [6, 8, 12, 16] + High-risk | HR: 1.630 | 63.0% increase | ❌ HARMFUL
Quad sig [6, 8, 14, 15] + High-risk | HR: 1.542 | 54.2% increase | ❌ HARMFUL
Quad sig [6, 8, 16, 18] + High-risk | HR: 1.636 | 63.6% increase | ❌ HARMFUL
Quad sig [6, 9, 10, 16] + High-risk | HR: 1.621 | 62.1% increase | ❌ HARMFUL
Quad sig [6, 9, 11, 18] + High-risk | HR: 1.549 | 54.9% increase | ❌ HARMFUL
Quad sig [6, 9, 13, 15] + High-risk | HR: 1.618 | 61.8% increase | ❌ HARMFUL
Quad sig [6, 9, 15, 16] + High-risk | HR: 1.615 | 61.5% increase | ❌ HARMFUL
Quad sig [6, 10, 11, 12] + High-risk | HR: 1.544 | 54.4% increase | ❌ HARMFUL
Quad sig [6, 10, 12, 15] + High-risk | HR: 1.482 | 48.2% increase | ❌ HARMFUL
Quad sig [6, 10, 13, 19] + High-risk | HR: 1.638 | 63.8% increase | ❌ HARMFUL
Quad sig [6, 10, 16, 17] + High-risk | HR: 1.839 | 83.9% increase | ❌ HARMFUL
Quad sig [6, 11, 12, 17] + High-risk | HR: 1.686 | 68.6% increase | ❌ HARMFUL
Quad sig [6, 11, 14, 16] + High-risk | HR: 1.744 | 74.4% increase | ❌ HARMFUL
Quad sig [6, 11, 16, 19] + High-risk | HR: 1.697 | 69.7% increase | ❌ HARMFUL
Quad sig [6, 12, 14, 15] + High-risk | HR: 1.550 | 55.0% increase | ❌ HARMFUL
Quad sig [6, 12, 16, 18] + High-risk | HR: 1.722 | 72.2% increase | ❌ HARMFUL
Quad sig [6, 13, 15, 16] + High-risk | HR: 1.731 | 73.1% increase | ❌ HARMFUL
Quad sig [6, 14, 15, 16] + High-risk | HR: 1.731 | 73.1% increase | ❌ HARMFUL
Quad sig [6, 15, 16, 17] + High-risk | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Quad sig [7, 8, 9, 10] + High-risk  | HR: 1.684 | 68.4% increase | ❌ HARMFUL
Quad sig [7, 8, 10, 11] + High-risk | HR: 1.591 | 59.1% increase | ❌ HARMFUL
Quad sig [7, 8, 11, 13] + High-risk | HR: 1.757 | 75.7% increase | ❌ HARMFUL
Quad sig [7, 8, 12, 16] + High-risk | HR: 1.820 | 82.0% increase | ❌ HARMFUL
Quad sig [7, 8, 14, 15] + High-risk | HR: 1.845 | 84.5% increase | ❌ HARMFUL
Quad sig [7, 8, 16, 18] + High-risk | HR: 1.886 | 88.6% increase | ❌ HARMFUL
Quad sig [7, 9, 10, 16] + High-risk | HR: 1.893 | 89.3% increase | ❌ HARMFUL
Quad sig [7, 9, 11, 18] + High-risk | HR: 1.778 | 77.8% increase | ❌ HARMFUL
Quad sig [7, 9, 13, 15] + High-risk | HR: 1.779 | 77.9% increase | ❌ HARMFUL
Quad sig [7, 9, 15, 16] + High-risk | HR: 1.883 | 88.3% increase | ❌ HARMFUL
Quad sig [7, 10, 11, 12] + High-risk | HR: 1.758 | 75.8% increase | ❌ HARMFUL
Quad sig [7, 10, 12, 15] + High-risk | HR: 1.770 | 77.0% increase | ❌ HARMFUL
Quad sig [7, 10, 13, 19] + High-risk | HR: 2.028 | 102.8% increase | ❌ HARMFUL
Quad sig [7, 10, 16, 17] + High-risk | HR: 1.832 | 83.2% increase | ❌ HARMFUL
Quad sig [7, 11, 12, 17] + High-risk | HR: 1.786 | 78.6% increase | ❌ HARMFUL
Quad sig [7, 11, 14, 16] + High-risk | HR: 1.896 | 89.6% increase | ❌ HARMFUL
Quad sig [7, 11, 16, 19] + High-risk | HR: 1.933 | 93.3% increase | ❌ HARMFUL
Quad sig [7, 12, 14, 15] + High-risk | HR: 1.804 | 80.4% increase | ❌ HARMFUL
Quad sig [7, 12, 16, 18] + High-risk | HR: 1.962 | 96.2% increase | ❌ HARMFUL
Quad sig [7, 13, 15, 16] + High-risk | HR: 1.888 | 88.8% increase | ❌ HARMFUL
Quad sig [7, 14, 15, 16] + High-risk | HR: 1.903 | 90.3% increase | ❌ HARMFUL
Quad sig [7, 15, 16, 17] + High-risk | HR: 1.728 | 72.8% increase | ❌ HARMFUL
Quad sig [8, 9, 10, 11] + High-risk | HR: 1.609 | 60.9% increase | ❌ HARMFUL
Quad sig [8, 9, 11, 13] + High-risk | HR: 1.618 | 61.8% increase | ❌ HARMFUL
Quad sig [8, 9, 12, 16] + High-risk | HR: 1.664 | 66.4% increase | ❌ HARMFUL
Quad sig [8, 9, 14, 15] + High-risk | HR: 1.542 | 54.2% increase | ❌ HARMFUL
Quad sig [8, 9, 16, 18] + High-risk | HR: 1.770 | 77.0% increase | ❌ HARMFUL
Quad sig [8, 10, 11, 17] + High-risk | HR: 1.603 | 60.3% increase | ❌ HARMFUL
Quad sig [8, 10, 13, 14] + High-risk | HR: 1.596 | 59.6% increase | ❌ HARMFUL
Quad sig [8, 10, 14, 19] + High-risk | HR: 1.775 | 77.5% increase | ❌ HARMFUL
Quad sig [8, 10, 18, 19] + High-risk | HR: 1.678 | 67.8% increase | ❌ HARMFUL
Quad sig [8, 11, 13, 16] + High-risk | HR: 1.840 | 84.0% increase | ❌ HARMFUL
Quad sig [8, 11, 15, 17] + High-risk | HR: 1.679 | 67.9% increase | ❌ HARMFUL
Quad sig [8, 12, 13, 15] + High-risk | HR: 1.668 | 66.8% increase | ❌ HARMFUL
Quad sig [8, 12, 15, 16] + High-risk | HR: 1.750 | 75.0% increase | ❌ HARMFUL
Quad sig [8, 13, 14, 15] + High-risk | HR: 1.586 | 58.6% increase | ❌ HARMFUL
Quad sig [8, 13, 16, 18] + High-risk | HR: 1.709 | 70.9% increase | ❌ HARMFUL
Quad sig [8, 14, 16, 18] + High-risk | HR: 1.850 | 85.0% increase | ❌ HARMFUL
Quad sig [8, 15, 18, 19] + High-risk | HR: 1.784 | 78.4% increase | ❌ HARMFUL
Quad sig [9, 10, 11, 17] + High-risk | HR: 1.660 | 66.0% increase | ❌ HARMFUL
Quad sig [9, 10, 13, 14] + High-risk | HR: 1.650 | 65.0% increase | ❌ HARMFUL
Quad sig [9, 10, 14, 19] + High-risk | HR: 1.693 | 69.3% increase | ❌ HARMFUL
Quad sig [9, 10, 18, 19] + High-risk | HR: 1.768 | 76.8% increase | ❌ HARMFUL
Quad sig [9, 11, 13, 16] + High-risk | HR: 1.720 | 72.0% increase | ❌ HARMFUL
Quad sig [9, 11, 15, 17] + High-risk | HR: 1.756 | 75.6% increase | ❌ HARMFUL
Quad sig [9, 12, 13, 15] + High-risk | HR: 1.669 | 66.9% increase | ❌ HARMFUL
Quad sig [9, 12, 15, 16] + High-risk | HR: 1.713 | 71.3% increase | ❌ HARMFUL
Quad sig [9, 13, 14, 15] + High-risk | HR: 1.663 | 66.3% increase | ❌ HARMFUL
Quad sig [9, 13, 16, 18] + High-risk | HR: 1.894 | 89.4% increase | ❌ HARMFUL
Quad sig [9, 14, 16, 18] + High-risk | HR: 1.899 | 89.9% increase | ❌ HARMFUL
Quad sig [9, 15, 18, 19] + High-risk | HR: 1.935 | 93.5% increase | ❌ HARMFUL
Quad sig [10, 11, 12, 18] + High-risk | HR: 1.579 | 57.9% increase | ❌ HARMFUL
Quad sig [10, 11, 14, 17] + High-risk | HR: 1.785 | 78.5% increase | ❌ HARMFUL
Quad sig [10, 11, 17, 18] + High-risk | HR: 1.680 | 68.0% increase | ❌ HARMFUL
Quad sig [10, 12, 14, 16] + High-risk | HR: 1.797 | 79.7% increase | ❌ HARMFUL
Quad sig [10, 12, 16, 19] + High-risk | HR: 1.736 | 73.6% increase | ❌ HARMFUL
Quad sig [10, 13, 15, 17] + High-risk | HR: 1.642 | 64.2% increase | ❌ HARMFUL
Quad sig [10, 14, 15, 17] + High-risk | HR: 1.644 | 64.4% increase | ❌ HARMFUL
Quad sig [10, 15, 16, 18] + High-risk | HR: 1.760 | 76.0% increase | ❌ HARMFUL
Quad sig [11, 12, 13, 15] + High-risk | HR: 1.646 | 64.6% increase | ❌ HARMFUL
Quad sig [11, 12, 15, 16] + High-risk | HR: 1.726 | 72.6% increase | ❌ HARMFUL
Quad sig [11, 13, 14, 15] + High-risk | HR: 1.793 | 79.3% increase | ❌ HARMFUL
Quad sig [11, 13, 16, 18] + High-risk | HR: 1.788 | 78.8% increase | ❌ HARMFUL
Quad sig [11, 14, 16, 18] + High-risk | HR: 1.848 | 84.8% increase | ❌ HARMFUL
Quad sig [11, 15, 18, 19] + High-risk | HR: 1.845 | 84.5% increase | ❌ HARMFUL
Quad sig [12, 13, 15, 16] + High-risk | HR: 1.777 | 77.7% increase | ❌ HARMFUL
Quad sig [12, 14, 15, 16] + High-risk | HR: 1.784 | 78.4% increase | ❌ HARMFUL
Quad sig [12, 15, 16, 17] + High-risk | HR: 1.814 | 81.4% increase | ❌ HARMFUL
Quad sig [13, 14, 15, 16] + High-risk | HR: 1.878 | 87.8% increase | ❌ HARMFUL
Quad sig [13, 15, 16, 17] + High-risk | HR: 1.889 | 88.9% increase | ❌ HARMFUL
Quad sig [14, 15, 16, 17] + High-risk | HR: 1.773 | 77.3% increase | ❌ HARMFUL
Quad sig [15, 16, 17, 18] + High-risk | HR: 1.856 | 85.6% increase | ❌ HARMFUL

❌ No protective effects found. Need to investigate further.