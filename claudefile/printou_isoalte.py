================================================================================
EVALUATING ALL 5 CONFIGURATIONS - FIRST 5 BATCHES (50K PATIENTS)
================================================================================
Batch size: 10000
Number of batches: 5
Total patients: 50,000
Bootstrap iterations: 10
================================================================================

Loading data files...
✓ Loaded Y: torch.Size([50000, 348, 52]), E: torch.Size([50000, 348]), pce_df: 50000 rows

================================================================================
CONFIGURATION: FIXEDK_FREEG
Directory: /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedk_freeg_vectorized/
================================================================================
  Pooling 5 batches from enrollment_predictions_fixedphi_fixedk_freeg_vectorized...
    ✓ Batch 0: torch.Size([10000, 348, 52])
    ✓ Batch 1: torch.Size([10000, 348, 52])
    ✓ Batch 2: torch.Size([10000, 348, 52])
    ✓ Batch 3: torch.Size([10000, 348, 52])
    ✓ Batch 4: torch.Size([10000, 348, 52])
  ✓ Pooled shape: torch.Size([50000, 348, 52])

================================================================================
EVALUATING: FIXEDK_FREEG
================================================================================

Evaluating static 10-year AUC...

Evaluating ASCVD (10-Year Outcome, 1-Year Score)...
AUC: 0.732 (0.728-0.736) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 4333 (8.7%) (from 50000 individuals)
Excluded 0 prevalent cases for ASCVD.

   Sex-stratified analysis:
   Female: AUC = 0.710, Events = 1446/27107
   Male: AUC = 0.714, Events = 2887/22893

   ASCVD risk in patients with pre-existing conditions:
   RA: AUC = 0.710, Events = 36/248
   Breast_Cancer: AUC = 0.658, Events = 48/840

Evaluating Diabetes (10-Year Outcome, 1-Year Score)...
AUC: 0.629 (0.620-0.635) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2921 (5.8%) (from 50000 individuals)
Excluded 0 prevalent cases for Diabetes.

   Sex-stratified analysis:
   Female: AUC = 0.623, Events = 1193/27107
   Male: AUC = 0.625, Events = 1728/22893

Evaluating Atrial_Fib (10-Year Outcome, 1-Year Score)...
AUC: 0.709 (0.702-0.715) (calculated on 49353 individuals)
Events (10-Year in Eval Cohort): 1919 (3.8%) (from 50000 individuals)
Excluded 647 prevalent cases for Atrial_Fib.

   Sex-stratified analysis:
   Female: AUC = 0.713, Events = 1041/26933
   Male: AUC = 0.712, Events = 852/22420

Evaluating CKD (10-Year Outcome, 1-Year Score)...
AUC: 0.703 (0.685-0.712) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1135 (2.3%) (from 50000 individuals)
Excluded 0 prevalent cases for CKD.

   Sex-stratified analysis:
   Female: AUC = 0.708, Events = 530/27107
   Male: AUC = 0.696, Events = 605/22893

Evaluating All_Cancers (10-Year Outcome, 1-Year Score)...
AUC: 0.674 (0.668-0.678) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2521 (5.0%) (from 50000 individuals)
Excluded 0 prevalent cases for All_Cancers.

   Sex-stratified analysis:
   Female: AUC = 0.640, Events = 785/27107
   Male: AUC = 0.676, Events = 1736/22893

Evaluating Stroke (10-Year Outcome, 1-Year Score)...
AUC: 0.676 (0.655-0.692) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 692 (1.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Stroke.

   Sex-stratified analysis:
   Female: AUC = 0.673, Events = 313/27107
   Male: AUC = 0.674, Events = 379/22893

Evaluating Heart_Failure (10-Year Outcome, 1-Year Score)...
AUC: 0.705 (0.701-0.719) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 991 (2.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Heart_Failure.

   Sex-stratified analysis:
   Female: AUC = 0.732, Events = 352/27107
   Male: AUC = 0.680, Events = 639/22893

Evaluating Pneumonia (10-Year Outcome, 1-Year Score)...
AUC: 0.650 (0.635-0.657) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1758 (3.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Pneumonia.

   Sex-stratified analysis:
   Female: AUC = 0.650, Events = 773/27107
   Male: AUC = 0.644, Events = 985/22893

Evaluating COPD (10-Year Outcome, 1-Year Score)...
AUC: 0.665 (0.655-0.671) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2071 (4.1%) (from 50000 individuals)
Excluded 0 prevalent cases for COPD.

   Sex-stratified analysis:
   Female: AUC = 0.667, Events = 925/27107
   Male: AUC = 0.658, Events = 1146/22893

Evaluating Osteoporosis (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.666-0.686) (calculated on 49858 individuals)
Events (10-Year in Eval Cohort): 1101 (2.2%) (from 50000 individuals)
Excluded 142 prevalent cases for Osteoporosis.

   Sex-stratified analysis:
   Female: AUC = 0.675, Events = 611/26995
   Male: AUC = 0.677, Events = 489/22863

Evaluating Anemia (10-Year Outcome, 1-Year Score)...
AUC: 0.595 (0.587-0.600) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2676 (5.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Anemia.

   Sex-stratified analysis:
   Female: AUC = 0.568, Events = 1430/27107
   Male: AUC = 0.625, Events = 1246/22893

Evaluating Colorectal_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.653 (0.652-0.673) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 613 (1.2%) (from 50000 individuals)
Excluded 0 prevalent cases for Colorectal_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.630, Events = 282/27107
   Male: AUC = 0.669, Events = 331/22893

Evaluating Breast_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Female: Found 27107 individuals in cohort
AUC: 0.548 (0.539-0.565) (calculated on 27107 individuals)
Events (10-Year in Eval Cohort): 1098 (4.1%) (from 27107 individuals)
Excluded 0 prevalent cases for Breast_Cancer.

Evaluating Prostate_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Male: Found 22893 individuals in cohort
AUC: 0.681 (0.670-0.687) (calculated on 22640 individuals)
Events (10-Year in Eval Cohort): 952 (4.2%) (from 22893 individuals)
Excluded 253 prevalent cases for Prostate_Cancer.

Evaluating Lung_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.658-0.704) (calculated on 49963 individuals)
Events (10-Year in Eval Cohort): 432 (0.9%) (from 50000 individuals)
Excluded 37 prevalent cases for Lung_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.681, Events = 218/27088
   Male: AUC = 0.671, Events = 212/22875

Evaluating Bladder_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.716 (0.686-0.738) (calculated on 49870 individuals)
Events (10-Year in Eval Cohort): 255 (0.5%) (from 50000 individuals)
Excluded 130 prevalent cases for Bladder_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.693, Events = 148/27077
   Male: AUC = 0.749, Events = 107/22793

Evaluating Secondary_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.607 (0.590-0.616) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1497 (3.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Secondary_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.586, Events = 857/27107
   Male: AUC = 0.636, Events = 640/22893

Evaluating Depression (10-Year Outcome, 1-Year Score)...
AUC: 0.478 (0.462-0.486) (calculated on 49510 individuals)
Events (10-Year in Eval Cohort): 1995 (4.0%) (from 50000 individuals)
Excluded 490 prevalent cases for Depression.

   Sex-stratified analysis:
   Female: AUC = 0.475, Events = 1071/26809
   Male: AUC = 0.493, Events = 909/22701

Evaluating Anxiety (10-Year Outcome, 1-Year Score)...
AUC: 0.515 (0.501-0.525) (calculated on 49863 individuals)
Events (10-Year in Eval Cohort): 1289 (2.6%) (from 50000 individuals)
Excluded 137 prevalent cases for Anxiety.

   Sex-stratified analysis:
   Female: AUC = 0.523, Events = 692/27021
   Male: AUC = 0.510, Events = 597/22842

Evaluating Bipolar_Disorder (10-Year Outcome, 1-Year Score)...
AUC: 0.451 (0.415-0.494) (calculated on 49925 individuals)
Events (10-Year in Eval Cohort): 121 (0.2%) (from 50000 individuals)
Excluded 75 prevalent cases for Bipolar_Disorder.

   Sex-stratified analysis:
   Female: AUC = 0.467, Events = 69/27055
   Male: AUC = 0.421, Events = 51/22870

Evaluating Rheumatoid_Arthritis (10-Year Outcome, 1-Year Score)...
AUC: 0.611 (0.606-0.628) (calculated on 49752 individuals)
Events (10-Year in Eval Cohort): 595 (1.2%) (from 50000 individuals)
Excluded 248 prevalent cases for Rheumatoid_Arthritis.

   Sex-stratified analysis:
   Female: AUC = 0.617, Events = 314/26913
   Male: AUC = 0.608, Events = 279/22839

Evaluating Psoriasis (10-Year Outcome, 1-Year Score)...
AUC: 0.550 (0.521-0.579) (calculated on 49901 individuals)
Events (10-Year in Eval Cohort): 236 (0.5%) (from 50000 individuals)
Excluded 99 prevalent cases for Psoriasis.

   Sex-stratified analysis:
   Female: AUC = 0.532, Events = 111/27069
   Male: AUC = 0.565, Events = 124/22832

Evaluating Ulcerative_Colitis (10-Year Outcome, 1-Year Score)...
AUC: 0.572 (0.560-0.599) (calculated on 49750 individuals)
Events (10-Year in Eval Cohort): 248 (0.5%) (from 50000 individuals)
Excluded 250 prevalent cases for Ulcerative_Colitis.

   Sex-stratified analysis:
   Female: AUC = 0.575, Events = 144/26979
   Male: AUC = 0.571, Events = 103/22771

Evaluating Crohns_Disease (10-Year Outcome, 1-Year Score)...
AUC: 0.577 (0.526-0.610) (calculated on 49850 individuals)
Events (10-Year in Eval Cohort): 136 (0.3%) (from 50000 individuals)
Excluded 150 prevalent cases for Crohns_Disease.

   Sex-stratified analysis:
   Female: AUC = 0.591, Events = 77/27034
   Male: AUC = 0.561, Events = 59/22816

Evaluating Asthma (10-Year Outcome, 1-Year Score)...
AUC: 0.526 (0.522-0.532) (calculated on 48407 individuals)
Events (10-Year in Eval Cohort): 3038 (6.1%) (from 50000 individuals)
Excluded 1593 prevalent cases for Asthma.

   Sex-stratified analysis:
   Female: AUC = 0.544, Events = 1591/26140
   Male: AUC = 0.543, Events = 1350/22267

Evaluating Parkinsons (10-Year Outcome, 1-Year Score)...
AUC: 0.728 (0.719-0.751) (calculated on 49971 individuals)
Events (10-Year in Eval Cohort): 223 (0.4%) (from 50000 individuals)
Excluded 29 prevalent cases for Parkinsons.

   Sex-stratified analysis:
   Female: AUC = 0.740, Events = 124/27095
   Male: AUC = 0.714, Events = 99/22876

Evaluating Multiple_Sclerosis (10-Year Outcome, 1-Year Score)...
AUC: 0.529 (0.482-0.567) (calculated on 49869 individuals)
Events (10-Year in Eval Cohort): 97 (0.2%) (from 50000 individuals)
Excluded 131 prevalent cases for Multiple_Sclerosis.

   Sex-stratified analysis:
   Female: AUC = 0.521, Events = 61/27009
   Male: AUC = 0.544, Events = 36/22860

Evaluating Thyroid_Disorders (10-Year Outcome, 1-Year Score)...
AUC: 0.574 (0.571-0.584) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2238 (4.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Thyroid_Disorders.

   Sex-stratified analysis:
   Female: AUC = 0.574, Events = 1792/27107
   Male: AUC = 0.561, Events = 446/22893

Summary of Results (Prospective 10-Year Outcome, 1-Year Score, Sex-Adjusted):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)  
--------------------------------------------------------------------------------
ASCVD                0.732 (0.728-0.736)       4333       8.7
Diabetes             0.629 (0.620-0.635)       2921       5.8
Atrial_Fib           0.709 (0.702-0.715)       1919       3.8
CKD                  0.703 (0.685-0.712)       1135       2.3
All_Cancers          0.674 (0.668-0.678)       2521       5.0
Stroke               0.676 (0.655-0.692)       692        1.4
Heart_Failure        0.705 (0.701-0.719)       991        2.0
Pneumonia            0.650 (0.635-0.657)       1758       3.5
COPD                 0.665 (0.655-0.671)       2071       4.1
Osteoporosis         0.675 (0.666-0.686)       1101       2.2
Anemia               0.595 (0.587-0.600)       2676       5.4
Colorectal_Cancer    0.653 (0.652-0.673)       613        1.2
Breast_Cancer        0.548 (0.539-0.565)       1098       4.1
Prostate_Cancer      0.681 (0.670-0.687)       952        4.2
Lung_Cancer          0.675 (0.658-0.704)       432        0.9
Bladder_Cancer       0.716 (0.686-0.738)       255        0.5
Secondary_Cancer     0.607 (0.590-0.616)       1497       3.0
Depression           0.478 (0.462-0.486)       1995       4.0
Anxiety              0.515 (0.501-0.525)       1289       2.6
Bipolar_Disorder     0.451 (0.415-0.494)       121        0.2
Rheumatoid_Arthritis 0.611 (0.606-0.628)       595        1.2
Psoriasis            0.550 (0.521-0.579)       236        0.5
Ulcerative_Colitis   0.572 (0.560-0.599)       248        0.5
Crohns_Disease       0.577 (0.526-0.610)       136        0.3
Asthma               0.526 (0.522-0.532)       3038       6.1
Parkinsons           0.728 (0.719-0.751)       223        0.4
Multiple_Sclerosis   0.529 (0.482-0.567)       97         0.2
Thyroid_Disorders    0.574 (0.571-0.584)       2238       4.5
--------------------------------------------------------------------------------
Evaluating dynamic 1-year AUC...
Filtering for 1: Found 22893 individuals in cohort

Summary of Results (1-Year Risk, Sex-Adjusted, Offset=0):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)   C-index   
--------------------------------------------------------------------------------
ASCVD                0.873 (0.859-0.883)       397        0.8        N/A
Diabetes             0.715 (0.675-0.755)       195        0.4        N/A
Atrial_Fib           0.837 (0.792-0.867)       120        0.2        N/A
CKD                  0.683 (0.545-0.795)       16         0.0        N/A
All_Cancers          0.737 (0.714-0.782)       115        0.2        N/A
Stroke               0.659 (0.594-0.688)       30         0.1        N/A
Heart_Failure        0.812 (0.775-0.851)       51         0.1        N/A
Pneumonia            0.611 (0.593-0.666)       56         0.1        N/A
COPD                 0.725 (0.661-0.739)       99         0.2        N/A
Osteoporosis         0.786 (0.734-0.834)       44         0.1        N/A
Anemia               0.632 (0.595-0.630)       144        0.3        N/A
Colorectal_Cancer    0.851 (0.808-0.913)       37         0.1        N/A
Breast_Cancer        0.834 (0.779-0.895)       89         0.2        N/A
Prostate_Cancer      0.801 (0.759-0.818)       56         0.2        N/A
Lung_Cancer          0.745 (0.518-0.900)       11         0.0        N/A
Bladder_Cancer       0.824 (0.585-0.913)       9          0.0        N/A
Secondary_Cancer     0.599 (0.565-0.672)       51         0.1        N/A
Depression           0.669 (0.570-0.713)       62         0.1        N/A
Anxiety              0.684 (0.572-0.767)       24         0.0        N/A
Bipolar_Disorder     0.734 (0.496-0.954)       4          0.0        N/A
Rheumatoid_Arthritis 0.804 (0.718-0.828)       34         0.1        N/A
Psoriasis            0.685 (0.604-0.828)       9          0.0        N/A
Ulcerative_Colitis   0.935 (0.913-0.983)       17         0.0        N/A
Crohns_Disease       0.970 (0.951-0.977)       7          0.0        N/A
Asthma               0.666 (0.663-0.700)       251        0.5        N/A
Parkinsons           0.769 (0.731-0.903)       10         0.0        N/A
Multiple_Sclerosis   0.938 (0.894-0.968)       13         0.0        N/A
Thyroid_Disorders    0.658 (0.612-0.712)       137        0.3        N/A
--------------------------------------------------------------------------------
  ✓ Static 10-year: 28 diseases
  ✓ Dynamic 1-year: 28 diseases

================================================================================
CONFIGURATION: FIXEDG_FREEK
Directory: /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedg_freek_vectorized/
================================================================================
  Pooling 5 batches from enrollment_predictions_fixedphi_fixedg_freek_vectorized...
    ✓ Batch 0: torch.Size([10000, 348, 52])
    ✓ Batch 1: torch.Size([10000, 348, 52])
    ✓ Batch 2: torch.Size([10000, 348, 52])
    ✓ Batch 3: torch.Size([10000, 348, 52])
    ✓ Batch 4: torch.Size([10000, 348, 52])
  ✓ Pooled shape: torch.Size([50000, 348, 52])

================================================================================
EVALUATING: FIXEDG_FREEK
================================================================================

Evaluating static 10-year AUC...

Evaluating ASCVD (10-Year Outcome, 1-Year Score)...
AUC: 0.695 (0.691-0.705) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 4333 (8.7%) (from 50000 individuals)
Excluded 0 prevalent cases for ASCVD.

   Sex-stratified analysis:
   Female: AUC = 0.692, Events = 1446/27107
   Male: AUC = 0.688, Events = 2887/22893

   ASCVD risk in patients with pre-existing conditions:
   RA: AUC = 0.675, Events = 36/248
   Breast_Cancer: AUC = 0.644, Events = 48/840

Evaluating Diabetes (10-Year Outcome, 1-Year Score)...
AUC: 0.603 (0.594-0.611) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2921 (5.8%) (from 50000 individuals)
Excluded 0 prevalent cases for Diabetes.

   Sex-stratified analysis:
   Female: AUC = 0.598, Events = 1193/27107
   Male: AUC = 0.601, Events = 1728/22893

Evaluating Atrial_Fib (10-Year Outcome, 1-Year Score)...
AUC: 0.704 (0.701-0.711) (calculated on 49353 individuals)
Events (10-Year in Eval Cohort): 1919 (3.8%) (from 50000 individuals)
Excluded 647 prevalent cases for Atrial_Fib.

   Sex-stratified analysis:
   Female: AUC = 0.708, Events = 1041/26933
   Male: AUC = 0.708, Events = 852/22420

Evaluating CKD (10-Year Outcome, 1-Year Score)...
AUC: 0.705 (0.694-0.714) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1135 (2.3%) (from 50000 individuals)
Excluded 0 prevalent cases for CKD.

   Sex-stratified analysis:
   Female: AUC = 0.709, Events = 530/27107
   Male: AUC = 0.697, Events = 605/22893

Evaluating All_Cancers (10-Year Outcome, 1-Year Score)...
AUC: 0.669 (0.661-0.676) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2521 (5.0%) (from 50000 individuals)
Excluded 0 prevalent cases for All_Cancers.

   Sex-stratified analysis:
   Female: AUC = 0.639, Events = 785/27107
   Male: AUC = 0.674, Events = 1736/22893

Evaluating Stroke (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.665-0.691) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 692 (1.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Stroke.

   Sex-stratified analysis:
   Female: AUC = 0.672, Events = 313/27107
   Male: AUC = 0.673, Events = 379/22893

Evaluating Heart_Failure (10-Year Outcome, 1-Year Score)...
AUC: 0.704 (0.688-0.718) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 991 (2.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Heart_Failure.

   Sex-stratified analysis:
   Female: AUC = 0.732, Events = 352/27107
   Male: AUC = 0.680, Events = 639/22893

Evaluating Pneumonia (10-Year Outcome, 1-Year Score)...
AUC: 0.649 (0.637-0.660) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1758 (3.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Pneumonia.

   Sex-stratified analysis:
   Female: AUC = 0.649, Events = 773/27107
   Male: AUC = 0.645, Events = 985/22893

Evaluating COPD (10-Year Outcome, 1-Year Score)...
AUC: 0.664 (0.662-0.670) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2071 (4.1%) (from 50000 individuals)
Excluded 0 prevalent cases for COPD.

   Sex-stratified analysis:
   Female: AUC = 0.666, Events = 925/27107
   Male: AUC = 0.657, Events = 1146/22893

Evaluating Osteoporosis (10-Year Outcome, 1-Year Score)...
AUC: 0.669 (0.657-0.687) (calculated on 49858 individuals)
Events (10-Year in Eval Cohort): 1101 (2.2%) (from 50000 individuals)
Excluded 142 prevalent cases for Osteoporosis.

   Sex-stratified analysis:
   Female: AUC = 0.669, Events = 611/26995
   Male: AUC = 0.670, Events = 489/22863

Evaluating Anemia (10-Year Outcome, 1-Year Score)...
AUC: 0.595 (0.590-0.597) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2676 (5.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Anemia.

   Sex-stratified analysis:
   Female: AUC = 0.567, Events = 1430/27107
   Male: AUC = 0.626, Events = 1246/22893

Evaluating Colorectal_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.647 (0.624-0.672) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 613 (1.2%) (from 50000 individuals)
Excluded 0 prevalent cases for Colorectal_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.624, Events = 282/27107
   Male: AUC = 0.663, Events = 331/22893

Evaluating Breast_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Female: Found 27107 individuals in cohort
AUC: 0.542 (0.530-0.570) (calculated on 27107 individuals)
Events (10-Year in Eval Cohort): 1098 (4.1%) (from 27107 individuals)
Excluded 0 prevalent cases for Breast_Cancer.

Evaluating Prostate_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Male: Found 22893 individuals in cohort
AUC: 0.678 (0.669-0.680) (calculated on 22640 individuals)
Events (10-Year in Eval Cohort): 952 (4.2%) (from 22893 individuals)
Excluded 253 prevalent cases for Prostate_Cancer.

Evaluating Lung_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.673 (0.657-0.689) (calculated on 49963 individuals)
Events (10-Year in Eval Cohort): 432 (0.9%) (from 50000 individuals)
Excluded 37 prevalent cases for Lung_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.679, Events = 218/27088
   Male: AUC = 0.671, Events = 212/22875

Evaluating Bladder_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.704 (0.679-0.725) (calculated on 49870 individuals)
Events (10-Year in Eval Cohort): 255 (0.5%) (from 50000 individuals)
Excluded 130 prevalent cases for Bladder_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.680, Events = 148/27077
   Male: AUC = 0.738, Events = 107/22793

Evaluating Secondary_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.605 (0.595-0.619) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1497 (3.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Secondary_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.583, Events = 857/27107
   Male: AUC = 0.635, Events = 640/22893

Evaluating Depression (10-Year Outcome, 1-Year Score)...
AUC: 0.471 (0.462-0.479) (calculated on 49510 individuals)
Events (10-Year in Eval Cohort): 1995 (4.0%) (from 50000 individuals)
Excluded 490 prevalent cases for Depression.

   Sex-stratified analysis:
   Female: AUC = 0.467, Events = 1071/26809
   Male: AUC = 0.486, Events = 909/22701

Evaluating Anxiety (10-Year Outcome, 1-Year Score)...
AUC: 0.509 (0.488-0.521) (calculated on 49863 individuals)
Events (10-Year in Eval Cohort): 1289 (2.6%) (from 50000 individuals)
Excluded 137 prevalent cases for Anxiety.

   Sex-stratified analysis:
   Female: AUC = 0.518, Events = 692/27021
   Male: AUC = 0.502, Events = 597/22842

Evaluating Bipolar_Disorder (10-Year Outcome, 1-Year Score)...
AUC: 0.452 (0.422-0.513) (calculated on 49925 individuals)
Events (10-Year in Eval Cohort): 121 (0.2%) (from 50000 individuals)
Excluded 75 prevalent cases for Bipolar_Disorder.

   Sex-stratified analysis:
   Female: AUC = 0.477, Events = 69/27055
   Male: AUC = 0.409, Events = 51/22870

Evaluating Rheumatoid_Arthritis (10-Year Outcome, 1-Year Score)...
AUC: 0.606 (0.574-0.627) (calculated on 49752 individuals)
Events (10-Year in Eval Cohort): 595 (1.2%) (from 50000 individuals)
Excluded 248 prevalent cases for Rheumatoid_Arthritis.

   Sex-stratified analysis:
   Female: AUC = 0.608, Events = 314/26913
   Male: AUC = 0.606, Events = 279/22839

Evaluating Psoriasis (10-Year Outcome, 1-Year Score)...
AUC: 0.549 (0.527-0.591) (calculated on 49901 individuals)
Events (10-Year in Eval Cohort): 236 (0.5%) (from 50000 individuals)
Excluded 99 prevalent cases for Psoriasis.

   Sex-stratified analysis:
   Female: AUC = 0.537, Events = 111/27069
   Male: AUC = 0.558, Events = 124/22832

Evaluating Ulcerative_Colitis (10-Year Outcome, 1-Year Score)...
AUC: 0.568 (0.541-0.597) (calculated on 49750 individuals)
Events (10-Year in Eval Cohort): 248 (0.5%) (from 50000 individuals)
Excluded 250 prevalent cases for Ulcerative_Colitis.

   Sex-stratified analysis:
   Female: AUC = 0.570, Events = 144/26979
   Male: AUC = 0.569, Events = 103/22771

Evaluating Crohns_Disease (10-Year Outcome, 1-Year Score)...
AUC: 0.557 (0.513-0.577) (calculated on 49850 individuals)
Events (10-Year in Eval Cohort): 136 (0.3%) (from 50000 individuals)
Excluded 150 prevalent cases for Crohns_Disease.

   Sex-stratified analysis:
   Female: AUC = 0.574, Events = 77/27034
   Male: AUC = 0.538, Events = 59/22816

Evaluating Asthma (10-Year Outcome, 1-Year Score)...
AUC: 0.519 (0.509-0.526) (calculated on 48407 individuals)
Events (10-Year in Eval Cohort): 3038 (6.1%) (from 50000 individuals)
Excluded 1593 prevalent cases for Asthma.

   Sex-stratified analysis:
   Female: AUC = 0.538, Events = 1591/26140
   Male: AUC = 0.535, Events = 1350/22267

Evaluating Parkinsons (10-Year Outcome, 1-Year Score)...
AUC: 0.731 (0.718-0.748) (calculated on 49971 individuals)
Events (10-Year in Eval Cohort): 223 (0.4%) (from 50000 individuals)
Excluded 29 prevalent cases for Parkinsons.

   Sex-stratified analysis:
   Female: AUC = 0.744, Events = 124/27095
   Male: AUC = 0.715, Events = 99/22876

Evaluating Multiple_Sclerosis (10-Year Outcome, 1-Year Score)...
AUC: 0.542 (0.497-0.593) (calculated on 49869 individuals)
Events (10-Year in Eval Cohort): 97 (0.2%) (from 50000 individuals)
Excluded 131 prevalent cases for Multiple_Sclerosis.

   Sex-stratified analysis:
   Female: AUC = 0.523, Events = 61/27009
   Male: AUC = 0.577, Events = 36/22860

Evaluating Thyroid_Disorders (10-Year Outcome, 1-Year Score)...
AUC: 0.563 (0.554-0.579) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2238 (4.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Thyroid_Disorders.

   Sex-stratified analysis:
   Female: AUC = 0.572, Events = 1792/27107
   Male: AUC = 0.557, Events = 446/22893

Summary of Results (Prospective 10-Year Outcome, 1-Year Score, Sex-Adjusted):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)  
--------------------------------------------------------------------------------
ASCVD                0.695 (0.691-0.705)       4333       8.7
Diabetes             0.603 (0.594-0.611)       2921       5.8
Atrial_Fib           0.704 (0.701-0.711)       1919       3.8
CKD                  0.705 (0.694-0.714)       1135       2.3
All_Cancers          0.669 (0.661-0.676)       2521       5.0
Stroke               0.675 (0.665-0.691)       692        1.4
Heart_Failure        0.704 (0.688-0.718)       991        2.0
Pneumonia            0.649 (0.637-0.660)       1758       3.5
COPD                 0.664 (0.662-0.670)       2071       4.1
Osteoporosis         0.669 (0.657-0.687)       1101       2.2
Anemia               0.595 (0.590-0.597)       2676       5.4
Colorectal_Cancer    0.647 (0.624-0.672)       613        1.2
Breast_Cancer        0.542 (0.530-0.570)       1098       4.1
Prostate_Cancer      0.678 (0.669-0.680)       952        4.2
Lung_Cancer          0.673 (0.657-0.689)       432        0.9
Bladder_Cancer       0.704 (0.679-0.725)       255        0.5
Secondary_Cancer     0.605 (0.595-0.619)       1497       3.0
Depression           0.471 (0.462-0.479)       1995       4.0
Anxiety              0.509 (0.488-0.521)       1289       2.6
Bipolar_Disorder     0.452 (0.422-0.513)       121        0.2
Rheumatoid_Arthritis 0.606 (0.574-0.627)       595        1.2
Psoriasis            0.549 (0.527-0.591)       236        0.5
Ulcerative_Colitis   0.568 (0.541-0.597)       248        0.5
Crohns_Disease       0.557 (0.513-0.577)       136        0.3
Asthma               0.519 (0.509-0.526)       3038       6.1
Parkinsons           0.731 (0.718-0.748)       223        0.4
Multiple_Sclerosis   0.542 (0.497-0.593)       97         0.2
Thyroid_Disorders    0.563 (0.554-0.579)       2238       4.5
--------------------------------------------------------------------------------
Evaluating dynamic 1-year AUC...
Filtering for 1: Found 22893 individuals in cohort

Summary of Results (1-Year Risk, Sex-Adjusted, Offset=0):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)   C-index   
--------------------------------------------------------------------------------
ASCVD                0.870 (0.845-0.882)       397        0.8        N/A
Diabetes             0.685 (0.665-0.724)       195        0.4        N/A
Atrial_Fib           0.828 (0.793-0.872)       120        0.2        N/A
CKD                  0.680 (0.472-0.808)       16         0.0        N/A
All_Cancers          0.731 (0.705-0.751)       115        0.2        N/A
Stroke               0.657 (0.605-0.702)       30         0.1        N/A
Heart_Failure        0.803 (0.757-0.836)       51         0.1        N/A
Pneumonia            0.609 (0.528-0.654)       56         0.1        N/A
COPD                 0.726 (0.680-0.758)       99         0.2        N/A
Osteoporosis         0.782 (0.754-0.827)       44         0.1        N/A
Anemia               0.636 (0.596-0.683)       144        0.3        N/A
Colorectal_Cancer    0.851 (0.837-0.896)       37         0.1        N/A
Breast_Cancer        0.840 (0.796-0.861)       89         0.2        N/A
Prostate_Cancer      0.799 (0.768-0.823)       56         0.2        N/A
Lung_Cancer          0.747 (0.681-0.917)       11         0.0        N/A
Bladder_Cancer       0.825 (0.664-0.942)       9          0.0        N/A
Secondary_Cancer     0.598 (0.517-0.685)       51         0.1        N/A
Depression           0.664 (0.614-0.739)       62         0.1        N/A
Anxiety              0.679 (0.561-0.796)       24         0.0        N/A
Bipolar_Disorder     0.847 (0.767-0.978)       4          0.0        N/A
Rheumatoid_Arthritis 0.799 (0.739-0.863)       34         0.1        N/A
Psoriasis            0.720 (0.653-0.845)       9          0.0        N/A
Ulcerative_Colitis   0.946 (0.897-0.990)       17         0.0        N/A
Crohns_Disease       0.965 (0.955-0.978)       7          0.0        N/A
Asthma               0.663 (0.640-0.688)       251        0.5        N/A
Parkinsons           0.782 (0.714-0.849)       10         0.0        N/A
Multiple_Sclerosis   0.947 (0.926-0.969)       13         0.0        N/A
Thyroid_Disorders    0.652 (0.589-0.664)       137        0.3        N/A
--------------------------------------------------------------------------------
  ✓ Static 10-year: 28 diseases
  ✓ Dynamic 1-year: 28 diseases

================================================================================
CONFIGURATION: ORIGINAL
Directory: /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_correctedE_vectorized/
================================================================================
  Pooling 5 batches from enrollment_predictions_fixedphi_correctedE_vectorized...
    ✓ Batch 0: torch.Size([10000, 348, 52])
    ✓ Batch 1: torch.Size([10000, 348, 52])
    ✓ Batch 2: torch.Size([10000, 348, 52])
    ✓ Batch 3: torch.Size([10000, 348, 52])
    ✓ Batch 4: torch.Size([10000, 348, 52])
  ✓ Pooled shape: torch.Size([50000, 348, 52])

================================================================================
EVALUATING: ORIGINAL
================================================================================

Evaluating static 10-year AUC...

Evaluating ASCVD (10-Year Outcome, 1-Year Score)...
AUC: 0.732 (0.726-0.738) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 4333 (8.7%) (from 50000 individuals)
Excluded 0 prevalent cases for ASCVD.

   Sex-stratified analysis:
   Female: AUC = 0.710, Events = 1446/27107
   Male: AUC = 0.714, Events = 2887/22893

   ASCVD risk in patients with pre-existing conditions:
   RA: AUC = 0.710, Events = 36/248
   Breast_Cancer: AUC = 0.657, Events = 48/840

Evaluating Diabetes (10-Year Outcome, 1-Year Score)...
AUC: 0.629 (0.624-0.639) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2921 (5.8%) (from 50000 individuals)
Excluded 0 prevalent cases for Diabetes.

   Sex-stratified analysis:
   Female: AUC = 0.624, Events = 1193/27107
   Male: AUC = 0.624, Events = 1728/22893

Evaluating Atrial_Fib (10-Year Outcome, 1-Year Score)...
AUC: 0.709 (0.702-0.718) (calculated on 49353 individuals)
Events (10-Year in Eval Cohort): 1919 (3.8%) (from 50000 individuals)
Excluded 647 prevalent cases for Atrial_Fib.

   Sex-stratified analysis:
   Female: AUC = 0.713, Events = 1041/26933
   Male: AUC = 0.712, Events = 852/22420

Evaluating CKD (10-Year Outcome, 1-Year Score)...
AUC: 0.703 (0.689-0.708) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1135 (2.3%) (from 50000 individuals)
Excluded 0 prevalent cases for CKD.

   Sex-stratified analysis:
   Female: AUC = 0.708, Events = 530/27107
   Male: AUC = 0.696, Events = 605/22893

Evaluating All_Cancers (10-Year Outcome, 1-Year Score)...
AUC: 0.674 (0.669-0.685) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2521 (5.0%) (from 50000 individuals)
Excluded 0 prevalent cases for All_Cancers.

   Sex-stratified analysis:
   Female: AUC = 0.640, Events = 785/27107
   Male: AUC = 0.676, Events = 1736/22893

Evaluating Stroke (10-Year Outcome, 1-Year Score)...
AUC: 0.676 (0.652-0.686) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 692 (1.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Stroke.

   Sex-stratified analysis:
   Female: AUC = 0.673, Events = 313/27107
   Male: AUC = 0.673, Events = 379/22893

Evaluating Heart_Failure (10-Year Outcome, 1-Year Score)...
AUC: 0.706 (0.692-0.722) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 991 (2.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Heart_Failure.

   Sex-stratified analysis:
   Female: AUC = 0.732, Events = 352/27107
   Male: AUC = 0.681, Events = 639/22893

Evaluating Pneumonia (10-Year Outcome, 1-Year Score)...
AUC: 0.650 (0.640-0.654) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1758 (3.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Pneumonia.

   Sex-stratified analysis:
   Female: AUC = 0.651, Events = 773/27107
   Male: AUC = 0.644, Events = 985/22893

Evaluating COPD (10-Year Outcome, 1-Year Score)...
AUC: 0.665 (0.654-0.672) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2071 (4.1%) (from 50000 individuals)
Excluded 0 prevalent cases for COPD.

   Sex-stratified analysis:
   Female: AUC = 0.667, Events = 925/27107
   Male: AUC = 0.659, Events = 1146/22893

Evaluating Osteoporosis (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.664-0.681) (calculated on 49858 individuals)
Events (10-Year in Eval Cohort): 1101 (2.2%) (from 50000 individuals)
Excluded 142 prevalent cases for Osteoporosis.

   Sex-stratified analysis:
   Female: AUC = 0.675, Events = 611/26995
   Male: AUC = 0.677, Events = 489/22863

Evaluating Anemia (10-Year Outcome, 1-Year Score)...
AUC: 0.595 (0.588-0.605) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2676 (5.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Anemia.

   Sex-stratified analysis:
   Female: AUC = 0.568, Events = 1430/27107
   Male: AUC = 0.625, Events = 1246/22893

Evaluating Colorectal_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.653 (0.641-0.668) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 613 (1.2%) (from 50000 individuals)
Excluded 0 prevalent cases for Colorectal_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.629, Events = 282/27107
   Male: AUC = 0.669, Events = 331/22893

Evaluating Breast_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Female: Found 27107 individuals in cohort
AUC: 0.548 (0.538-0.564) (calculated on 27107 individuals)
Events (10-Year in Eval Cohort): 1098 (4.1%) (from 27107 individuals)
Excluded 0 prevalent cases for Breast_Cancer.

Evaluating Prostate_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Male: Found 22893 individuals in cohort
AUC: 0.681 (0.671-0.693) (calculated on 22640 individuals)
Events (10-Year in Eval Cohort): 952 (4.2%) (from 22893 individuals)
Excluded 253 prevalent cases for Prostate_Cancer.

Evaluating Lung_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.662-0.689) (calculated on 49963 individuals)
Events (10-Year in Eval Cohort): 432 (0.9%) (from 50000 individuals)
Excluded 37 prevalent cases for Lung_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.681, Events = 218/27088
   Male: AUC = 0.671, Events = 212/22875

Evaluating Bladder_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.716 (0.709-0.729) (calculated on 49870 individuals)
Events (10-Year in Eval Cohort): 255 (0.5%) (from 50000 individuals)
Excluded 130 prevalent cases for Bladder_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.693, Events = 148/27077
   Male: AUC = 0.748, Events = 107/22793

Evaluating Secondary_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.607 (0.600-0.620) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1497 (3.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Secondary_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.586, Events = 857/27107
   Male: AUC = 0.636, Events = 640/22893

Evaluating Depression (10-Year Outcome, 1-Year Score)...
AUC: 0.478 (0.473-0.489) (calculated on 49510 individuals)
Events (10-Year in Eval Cohort): 1995 (4.0%) (from 50000 individuals)
Excluded 490 prevalent cases for Depression.

   Sex-stratified analysis:
   Female: AUC = 0.475, Events = 1071/26809
   Male: AUC = 0.493, Events = 909/22701

Evaluating Anxiety (10-Year Outcome, 1-Year Score)...
AUC: 0.515 (0.507-0.531) (calculated on 49863 individuals)
Events (10-Year in Eval Cohort): 1289 (2.6%) (from 50000 individuals)
Excluded 137 prevalent cases for Anxiety.

   Sex-stratified analysis:
   Female: AUC = 0.523, Events = 692/27021
   Male: AUC = 0.510, Events = 597/22842

Evaluating Bipolar_Disorder (10-Year Outcome, 1-Year Score)...
AUC: 0.449 (0.422-0.471) (calculated on 49925 individuals)
Events (10-Year in Eval Cohort): 121 (0.2%) (from 50000 individuals)
Excluded 75 prevalent cases for Bipolar_Disorder.

   Sex-stratified analysis:
   Female: AUC = 0.464, Events = 69/27055
   Male: AUC = 0.420, Events = 51/22870

Evaluating Rheumatoid_Arthritis (10-Year Outcome, 1-Year Score)...
AUC: 0.612 (0.594-0.628) (calculated on 49752 individuals)
Events (10-Year in Eval Cohort): 595 (1.2%) (from 50000 individuals)
Excluded 248 prevalent cases for Rheumatoid_Arthritis.

   Sex-stratified analysis:
   Female: AUC = 0.617, Events = 314/26913
   Male: AUC = 0.609, Events = 279/22839

Evaluating Psoriasis (10-Year Outcome, 1-Year Score)...
AUC: 0.550 (0.501-0.591) (calculated on 49901 individuals)
Events (10-Year in Eval Cohort): 236 (0.5%) (from 50000 individuals)
Excluded 99 prevalent cases for Psoriasis.

   Sex-stratified analysis:
   Female: AUC = 0.532, Events = 111/27069
   Male: AUC = 0.565, Events = 124/22832

Evaluating Ulcerative_Colitis (10-Year Outcome, 1-Year Score)...
AUC: 0.572 (0.519-0.604) (calculated on 49750 individuals)
Events (10-Year in Eval Cohort): 248 (0.5%) (from 50000 individuals)
Excluded 250 prevalent cases for Ulcerative_Colitis.

   Sex-stratified analysis:
   Female: AUC = 0.576, Events = 144/26979
   Male: AUC = 0.570, Events = 103/22771

Evaluating Crohns_Disease (10-Year Outcome, 1-Year Score)...
AUC: 0.578 (0.547-0.630) (calculated on 49850 individuals)
Events (10-Year in Eval Cohort): 136 (0.3%) (from 50000 individuals)
Excluded 150 prevalent cases for Crohns_Disease.

   Sex-stratified analysis:
   Female: AUC = 0.593, Events = 77/27034
   Male: AUC = 0.562, Events = 59/22816

Evaluating Asthma (10-Year Outcome, 1-Year Score)...
AUC: 0.526 (0.518-0.537) (calculated on 48407 individuals)
Events (10-Year in Eval Cohort): 3038 (6.1%) (from 50000 individuals)
Excluded 1593 prevalent cases for Asthma.

   Sex-stratified analysis:
   Female: AUC = 0.544, Events = 1591/26140
   Male: AUC = 0.543, Events = 1350/22267

Evaluating Parkinsons (10-Year Outcome, 1-Year Score)...
AUC: 0.728 (0.703-0.742) (calculated on 49971 individuals)
Events (10-Year in Eval Cohort): 223 (0.4%) (from 50000 individuals)
Excluded 29 prevalent cases for Parkinsons.

   Sex-stratified analysis:
   Female: AUC = 0.740, Events = 124/27095
   Male: AUC = 0.713, Events = 99/22876

Evaluating Multiple_Sclerosis (10-Year Outcome, 1-Year Score)...
AUC: 0.525 (0.475-0.558) (calculated on 49869 individuals)
Events (10-Year in Eval Cohort): 97 (0.2%) (from 50000 individuals)
Excluded 131 prevalent cases for Multiple_Sclerosis.

   Sex-stratified analysis:
   Female: AUC = 0.517, Events = 61/27009
   Male: AUC = 0.542, Events = 36/22860

Evaluating Thyroid_Disorders (10-Year Outcome, 1-Year Score)...
AUC: 0.574 (0.566-0.580) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2238 (4.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Thyroid_Disorders.

   Sex-stratified analysis:
   Female: AUC = 0.574, Events = 1792/27107
   Male: AUC = 0.561, Events = 446/22893

Summary of Results (Prospective 10-Year Outcome, 1-Year Score, Sex-Adjusted):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)  
--------------------------------------------------------------------------------
ASCVD                0.732 (0.726-0.738)       4333       8.7
Diabetes             0.629 (0.624-0.639)       2921       5.8
Atrial_Fib           0.709 (0.702-0.718)       1919       3.8
CKD                  0.703 (0.689-0.708)       1135       2.3
All_Cancers          0.674 (0.669-0.685)       2521       5.0
Stroke               0.676 (0.652-0.686)       692        1.4
Heart_Failure        0.706 (0.692-0.722)       991        2.0
Pneumonia            0.650 (0.640-0.654)       1758       3.5
COPD                 0.665 (0.654-0.672)       2071       4.1
Osteoporosis         0.675 (0.664-0.681)       1101       2.2
Anemia               0.595 (0.588-0.605)       2676       5.4
Colorectal_Cancer    0.653 (0.641-0.668)       613        1.2
Breast_Cancer        0.548 (0.538-0.564)       1098       4.1
Prostate_Cancer      0.681 (0.671-0.693)       952        4.2
Lung_Cancer          0.675 (0.662-0.689)       432        0.9
Bladder_Cancer       0.716 (0.709-0.729)       255        0.5
Secondary_Cancer     0.607 (0.600-0.620)       1497       3.0
Depression           0.478 (0.473-0.489)       1995       4.0
Anxiety              0.515 (0.507-0.531)       1289       2.6
Bipolar_Disorder     0.449 (0.422-0.471)       121        0.2
Rheumatoid_Arthritis 0.612 (0.594-0.628)       595        1.2
Psoriasis            0.550 (0.501-0.591)       236        0.5
Ulcerative_Colitis   0.572 (0.519-0.604)       248        0.5
Crohns_Disease       0.578 (0.547-0.630)       136        0.3
Asthma               0.526 (0.518-0.537)       3038       6.1
Parkinsons           0.728 (0.703-0.742)       223        0.4
Multiple_Sclerosis   0.525 (0.475-0.558)       97         0.2
Thyroid_Disorders    0.574 (0.566-0.580)       2238       4.5
--------------------------------------------------------------------------------
Evaluating dynamic 1-year AUC...
Filtering for 1: Found 22893 individuals in cohort

Summary of Results (1-Year Risk, Sex-Adjusted, Offset=0):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)   C-index   
--------------------------------------------------------------------------------
ASCVD                0.873 (0.850-0.888)       397        0.8        N/A
Diabetes             0.714 (0.652-0.740)       195        0.4        N/A
Atrial_Fib           0.838 (0.785-0.866)       120        0.2        N/A
CKD                  0.684 (0.475-0.720)       16         0.0        N/A
All_Cancers          0.737 (0.709-0.770)       115        0.2        N/A
Stroke               0.658 (0.576-0.730)       30         0.1        N/A
Heart_Failure        0.812 (0.792-0.857)       51         0.1        N/A
Pneumonia            0.611 (0.567-0.680)       56         0.1        N/A
COPD                 0.724 (0.672-0.753)       99         0.2        N/A
Osteoporosis         0.786 (0.701-0.822)       44         0.1        N/A
Anemia               0.633 (0.584-0.656)       144        0.3        N/A
Colorectal_Cancer    0.852 (0.828-0.900)       37         0.1        N/A
Breast_Cancer        0.833 (0.803-0.897)       89         0.2        N/A
Prostate_Cancer      0.801 (0.767-0.839)       56         0.2        N/A
Lung_Cancer          0.744 (0.642-0.848)       11         0.0        N/A
Bladder_Cancer       0.822 (0.787-0.883)       9          0.0        N/A
Secondary_Cancer     0.598 (0.556-0.647)       51         0.1        N/A
Depression           0.668 (0.627-0.726)       62         0.1        N/A
Anxiety              0.682 (0.646-0.776)       24         0.0        N/A
Bipolar_Disorder     0.715 (0.434-0.963)       4          0.0        N/A
Rheumatoid_Arthritis 0.803 (0.712-0.862)       34         0.1        N/A
Psoriasis            0.683 (0.578-0.716)       9          0.0        N/A
Ulcerative_Colitis   0.937 (0.860-0.979)       17         0.0        N/A
Crohns_Disease       0.970 (0.954-0.989)       7          0.0        N/A
Asthma               0.666 (0.627-0.700)       251        0.5        N/A
Parkinsons           0.767 (0.634-0.839)       10         0.0        N/A
Multiple_Sclerosis   0.935 (0.874-0.970)       13         0.0        N/A
Thyroid_Disorders    0.658 (0.623-0.716)       137        0.3        N/A
--------------------------------------------------------------------------------
  ✓ Static 10-year: 28 diseases
  ✓ Dynamic 1-year: 28 diseases

================================================================================
CONFIGURATION: FIXEDGK
Directory: /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_vectorized/
================================================================================
  Pooling 5 batches from enrollment_predictions_fixedphi_fixedgk_vectorized...
    ✓ Batch 0: torch.Size([10000, 348, 52])
    ✓ Batch 1: torch.Size([10000, 348, 52])
    ✓ Batch 2: torch.Size([10000, 348, 52])
    ✓ Batch 3: torch.Size([10000, 348, 52])
    ✓ Batch 4: torch.Size([10000, 348, 52])
  ✓ Pooled shape: torch.Size([50000, 348, 52])

================================================================================
EVALUATING: FIXEDGK
================================================================================

Evaluating static 10-year AUC...

Evaluating ASCVD (10-Year Outcome, 1-Year Score)...
AUC: 0.695 (0.689-0.702) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 4333 (8.7%) (from 50000 individuals)
Excluded 0 prevalent cases for ASCVD.

   Sex-stratified analysis:
   Female: AUC = 0.692, Events = 1446/27107
   Male: AUC = 0.688, Events = 2887/22893

   ASCVD risk in patients with pre-existing conditions:
   RA: AUC = 0.676, Events = 36/248
   Breast_Cancer: AUC = 0.643, Events = 48/840

Evaluating Diabetes (10-Year Outcome, 1-Year Score)...
AUC: 0.603 (0.597-0.616) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2921 (5.8%) (from 50000 individuals)
Excluded 0 prevalent cases for Diabetes.

   Sex-stratified analysis:
   Female: AUC = 0.598, Events = 1193/27107
   Male: AUC = 0.601, Events = 1728/22893

Evaluating Atrial_Fib (10-Year Outcome, 1-Year Score)...
AUC: 0.704 (0.699-0.712) (calculated on 49353 individuals)
Events (10-Year in Eval Cohort): 1919 (3.8%) (from 50000 individuals)
Excluded 647 prevalent cases for Atrial_Fib.

   Sex-stratified analysis:
   Female: AUC = 0.708, Events = 1041/26933
   Male: AUC = 0.708, Events = 852/22420

Evaluating CKD (10-Year Outcome, 1-Year Score)...
AUC: 0.704 (0.691-0.716) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1135 (2.3%) (from 50000 individuals)
Excluded 0 prevalent cases for CKD.

   Sex-stratified analysis:
   Female: AUC = 0.709, Events = 530/27107
   Male: AUC = 0.697, Events = 605/22893

Evaluating All_Cancers (10-Year Outcome, 1-Year Score)...
AUC: 0.669 (0.657-0.676) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2521 (5.0%) (from 50000 individuals)
Excluded 0 prevalent cases for All_Cancers.

   Sex-stratified analysis:
   Female: AUC = 0.639, Events = 785/27107
   Male: AUC = 0.675, Events = 1736/22893

Evaluating Stroke (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.659-0.693) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 692 (1.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Stroke.

   Sex-stratified analysis:
   Female: AUC = 0.672, Events = 313/27107
   Male: AUC = 0.673, Events = 379/22893

Evaluating Heart_Failure (10-Year Outcome, 1-Year Score)...
AUC: 0.704 (0.687-0.712) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 991 (2.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Heart_Failure.

   Sex-stratified analysis:
   Female: AUC = 0.732, Events = 352/27107
   Male: AUC = 0.680, Events = 639/22893

Evaluating Pneumonia (10-Year Outcome, 1-Year Score)...
AUC: 0.649 (0.643-0.655) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1758 (3.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Pneumonia.

   Sex-stratified analysis:
   Female: AUC = 0.649, Events = 773/27107
   Male: AUC = 0.645, Events = 985/22893

Evaluating COPD (10-Year Outcome, 1-Year Score)...
AUC: 0.663 (0.655-0.671) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2071 (4.1%) (from 50000 individuals)
Excluded 0 prevalent cases for COPD.

   Sex-stratified analysis:
   Female: AUC = 0.666, Events = 925/27107
   Male: AUC = 0.657, Events = 1146/22893

Evaluating Osteoporosis (10-Year Outcome, 1-Year Score)...
AUC: 0.669 (0.653-0.682) (calculated on 49858 individuals)
Events (10-Year in Eval Cohort): 1101 (2.2%) (from 50000 individuals)
Excluded 142 prevalent cases for Osteoporosis.

   Sex-stratified analysis:
   Female: AUC = 0.669, Events = 611/26995
   Male: AUC = 0.670, Events = 489/22863

Evaluating Anemia (10-Year Outcome, 1-Year Score)...
AUC: 0.595 (0.588-0.603) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2676 (5.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Anemia.

   Sex-stratified analysis:
   Female: AUC = 0.567, Events = 1430/27107
   Male: AUC = 0.626, Events = 1246/22893

Evaluating Colorectal_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.647 (0.624-0.671) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 613 (1.2%) (from 50000 individuals)
Excluded 0 prevalent cases for Colorectal_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.624, Events = 282/27107
   Male: AUC = 0.663, Events = 331/22893

Evaluating Breast_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Female: Found 27107 individuals in cohort
AUC: 0.541 (0.527-0.555) (calculated on 27107 individuals)
Events (10-Year in Eval Cohort): 1098 (4.1%) (from 27107 individuals)
Excluded 0 prevalent cases for Breast_Cancer.

Evaluating Prostate_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Male: Found 22893 individuals in cohort
AUC: 0.678 (0.667-0.684) (calculated on 22640 individuals)
Events (10-Year in Eval Cohort): 952 (4.2%) (from 22893 individuals)
Excluded 253 prevalent cases for Prostate_Cancer.

Evaluating Lung_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.673 (0.663-0.693) (calculated on 49963 individuals)
Events (10-Year in Eval Cohort): 432 (0.9%) (from 50000 individuals)
Excluded 37 prevalent cases for Lung_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.679, Events = 218/27088
   Male: AUC = 0.671, Events = 212/22875

Evaluating Bladder_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.705 (0.685-0.725) (calculated on 49870 individuals)
Events (10-Year in Eval Cohort): 255 (0.5%) (from 50000 individuals)
Excluded 130 prevalent cases for Bladder_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.681, Events = 148/27077
   Male: AUC = 0.739, Events = 107/22793

Evaluating Secondary_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.604 (0.594-0.615) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1497 (3.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Secondary_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.583, Events = 857/27107
   Male: AUC = 0.635, Events = 640/22893

Evaluating Depression (10-Year Outcome, 1-Year Score)...
AUC: 0.471 (0.458-0.483) (calculated on 49510 individuals)
Events (10-Year in Eval Cohort): 1995 (4.0%) (from 50000 individuals)
Excluded 490 prevalent cases for Depression.

   Sex-stratified analysis:
   Female: AUC = 0.467, Events = 1071/26809
   Male: AUC = 0.487, Events = 909/22701

Evaluating Anxiety (10-Year Outcome, 1-Year Score)...
AUC: 0.509 (0.503-0.524) (calculated on 49863 individuals)
Events (10-Year in Eval Cohort): 1289 (2.6%) (from 50000 individuals)
Excluded 137 prevalent cases for Anxiety.

   Sex-stratified analysis:
   Female: AUC = 0.518, Events = 692/27021
   Male: AUC = 0.502, Events = 597/22842

Evaluating Bipolar_Disorder (10-Year Outcome, 1-Year Score)...
AUC: 0.454 (0.421-0.530) (calculated on 49925 individuals)
Events (10-Year in Eval Cohort): 121 (0.2%) (from 50000 individuals)
Excluded 75 prevalent cases for Bipolar_Disorder.

   Sex-stratified analysis:
   Female: AUC = 0.481, Events = 69/27055
   Male: AUC = 0.410, Events = 51/22870

Evaluating Rheumatoid_Arthritis (10-Year Outcome, 1-Year Score)...
AUC: 0.606 (0.584-0.618) (calculated on 49752 individuals)
Events (10-Year in Eval Cohort): 595 (1.2%) (from 50000 individuals)
Excluded 248 prevalent cases for Rheumatoid_Arthritis.

   Sex-stratified analysis:
   Female: AUC = 0.608, Events = 314/26913
   Male: AUC = 0.605, Events = 279/22839

Evaluating Psoriasis (10-Year Outcome, 1-Year Score)...
AUC: 0.549 (0.513-0.582) (calculated on 49901 individuals)
Events (10-Year in Eval Cohort): 236 (0.5%) (from 50000 individuals)
Excluded 99 prevalent cases for Psoriasis.

   Sex-stratified analysis:
   Female: AUC = 0.537, Events = 111/27069
   Male: AUC = 0.558, Events = 124/22832

Evaluating Ulcerative_Colitis (10-Year Outcome, 1-Year Score)...
AUC: 0.567 (0.555-0.616) (calculated on 49750 individuals)
Events (10-Year in Eval Cohort): 248 (0.5%) (from 50000 individuals)
Excluded 250 prevalent cases for Ulcerative_Colitis.

   Sex-stratified analysis:
   Female: AUC = 0.569, Events = 144/26979
   Male: AUC = 0.569, Events = 103/22771

Evaluating Crohns_Disease (10-Year Outcome, 1-Year Score)...
AUC: 0.555 (0.544-0.575) (calculated on 49850 individuals)
Events (10-Year in Eval Cohort): 136 (0.3%) (from 50000 individuals)
Excluded 150 prevalent cases for Crohns_Disease.

   Sex-stratified analysis:
   Female: AUC = 0.571, Events = 77/27034
   Male: AUC = 0.538, Events = 59/22816

Evaluating Asthma (10-Year Outcome, 1-Year Score)...
AUC: 0.519 (0.510-0.527) (calculated on 48407 individuals)
Events (10-Year in Eval Cohort): 3038 (6.1%) (from 50000 individuals)
Excluded 1593 prevalent cases for Asthma.

   Sex-stratified analysis:
   Female: AUC = 0.538, Events = 1591/26140
   Male: AUC = 0.535, Events = 1350/22267

Evaluating Parkinsons (10-Year Outcome, 1-Year Score)...
AUC: 0.731 (0.712-0.759) (calculated on 49971 individuals)
Events (10-Year in Eval Cohort): 223 (0.4%) (from 50000 individuals)
Excluded 29 prevalent cases for Parkinsons.

   Sex-stratified analysis:
   Female: AUC = 0.744, Events = 124/27095
   Male: AUC = 0.715, Events = 99/22876

Evaluating Multiple_Sclerosis (10-Year Outcome, 1-Year Score)...
AUC: 0.546 (0.482-0.591) (calculated on 49869 individuals)
Events (10-Year in Eval Cohort): 97 (0.2%) (from 50000 individuals)
Excluded 131 prevalent cases for Multiple_Sclerosis.

   Sex-stratified analysis:
   Female: AUC = 0.528, Events = 61/27009
   Male: AUC = 0.579, Events = 36/22860

Evaluating Thyroid_Disorders (10-Year Outcome, 1-Year Score)...
AUC: 0.563 (0.552-0.571) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2238 (4.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Thyroid_Disorders.

   Sex-stratified analysis:
   Female: AUC = 0.572, Events = 1792/27107
   Male: AUC = 0.557, Events = 446/22893

Summary of Results (Prospective 10-Year Outcome, 1-Year Score, Sex-Adjusted):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)  
--------------------------------------------------------------------------------
ASCVD                0.695 (0.689-0.702)       4333       8.7
Diabetes             0.603 (0.597-0.616)       2921       5.8
Atrial_Fib           0.704 (0.699-0.712)       1919       3.8
CKD                  0.704 (0.691-0.716)       1135       2.3
All_Cancers          0.669 (0.657-0.676)       2521       5.0
Stroke               0.675 (0.659-0.693)       692        1.4
Heart_Failure        0.704 (0.687-0.712)       991        2.0
Pneumonia            0.649 (0.643-0.655)       1758       3.5
COPD                 0.663 (0.655-0.671)       2071       4.1
Osteoporosis         0.669 (0.653-0.682)       1101       2.2
Anemia               0.595 (0.588-0.603)       2676       5.4
Colorectal_Cancer    0.647 (0.624-0.671)       613        1.2
Breast_Cancer        0.541 (0.527-0.555)       1098       4.1
Prostate_Cancer      0.678 (0.667-0.684)       952        4.2
Lung_Cancer          0.673 (0.663-0.693)       432        0.9
Bladder_Cancer       0.705 (0.685-0.725)       255        0.5
Secondary_Cancer     0.604 (0.594-0.615)       1497       3.0
Depression           0.471 (0.458-0.483)       1995       4.0
Anxiety              0.509 (0.503-0.524)       1289       2.6
Bipolar_Disorder     0.454 (0.421-0.530)       121        0.2
Rheumatoid_Arthritis 0.606 (0.584-0.618)       595        1.2
Psoriasis            0.549 (0.513-0.582)       236        0.5
Ulcerative_Colitis   0.567 (0.555-0.616)       248        0.5
Crohns_Disease       0.555 (0.544-0.575)       136        0.3
Asthma               0.519 (0.510-0.527)       3038       6.1
Parkinsons           0.731 (0.712-0.759)       223        0.4
Multiple_Sclerosis   0.546 (0.482-0.591)       97         0.2
Thyroid_Disorders    0.563 (0.552-0.571)       2238       4.5
--------------------------------------------------------------------------------
Evaluating dynamic 1-year AUC...
Filtering for 1: Found 22893 individuals in cohort

Summary of Results (1-Year Risk, Sex-Adjusted, Offset=0):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)   C-index   
--------------------------------------------------------------------------------
ASCVD                0.870 (0.842-0.891)       397        0.8        N/A
Diabetes             0.685 (0.665-0.697)       195        0.4        N/A
Atrial_Fib           0.828 (0.786-0.857)       120        0.2        N/A
CKD                  0.680 (0.572-0.820)       16         0.0        N/A
All_Cancers          0.731 (0.702-0.767)       115        0.2        N/A
Stroke               0.659 (0.598-0.704)       30         0.1        N/A
Heart_Failure        0.804 (0.753-0.871)       51         0.1        N/A
Pneumonia            0.609 (0.560-0.636)       56         0.1        N/A
COPD                 0.727 (0.679-0.755)       99         0.2        N/A
Osteoporosis         0.782 (0.697-0.819)       44         0.1        N/A
Anemia               0.635 (0.593-0.680)       144        0.3        N/A
Colorectal_Cancer    0.847 (0.832-0.877)       37         0.1        N/A
Breast_Cancer        0.841 (0.781-0.888)       89         0.2        N/A
Prostate_Cancer      0.799 (0.749-0.846)       56         0.2        N/A
Lung_Cancer          0.748 (0.649-0.895)       11         0.0        N/A
Bladder_Cancer       0.827 (0.712-0.914)       9          0.0        N/A
Secondary_Cancer     0.598 (0.542-0.646)       51         0.1        N/A
Depression           0.665 (0.588-0.722)       62         0.1        N/A
Anxiety              0.680 (0.586-0.790)       24         0.0        N/A
Bipolar_Disorder     0.869 (0.735-0.957)       4          0.0        N/A
Rheumatoid_Arthritis 0.801 (0.707-0.821)       34         0.1        N/A
Psoriasis            0.722 (0.630-0.868)       9          0.0        N/A
Ulcerative_Colitis   0.943 (0.914-0.981)       17         0.0        N/A
Crohns_Disease       0.966 (0.953-0.970)       7          0.0        N/A
Asthma               0.663 (0.647-0.686)       251        0.5        N/A
Parkinsons           0.783 (0.724-0.895)       10         0.0        N/A
Multiple_Sclerosis   0.951 (0.929-0.958)       13         0.0        N/A
Thyroid_Disorders    0.653 (0.591-0.683)       137        0.3        N/A
--------------------------------------------------------------------------------
  ✓ Static 10-year: 28 diseases
  ✓ Dynamic 1-year: 28 diseases

================================================================================
CONFIGURATION: FIXEDGK_NOLR
Directory: /Users/sarahurbut/Library/CloudStorage/Dropbox/enrollment_predictions_fixedphi_fixedgk_nolr_vectorized/
================================================================================
  Pooling 5 batches from enrollment_predictions_fixedphi_fixedgk_nolr_vectorized...
    ✓ Batch 0: torch.Size([10000, 348, 52])
    ✓ Batch 1: torch.Size([10000, 348, 52])
    ✓ Batch 2: torch.Size([10000, 348, 52])
    ✓ Batch 3: torch.Size([10000, 348, 52])
    ✓ Batch 4: torch.Size([10000, 348, 52])
  ✓ Pooled shape: torch.Size([50000, 348, 52])

================================================================================
EVALUATING: FIXEDGK_NOLR
================================================================================

Evaluating static 10-year AUC...

Evaluating ASCVD (10-Year Outcome, 1-Year Score)...
AUC: 0.730 (0.721-0.733) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 4333 (8.7%) (from 50000 individuals)
Excluded 0 prevalent cases for ASCVD.

   Sex-stratified analysis:
   Female: AUC = 0.705, Events = 1446/27107
   Male: AUC = 0.711, Events = 2887/22893

   ASCVD risk in patients with pre-existing conditions:
   RA: AUC = 0.713, Events = 36/248
   Breast_Cancer: AUC = 0.666, Events = 48/840

Evaluating Diabetes (10-Year Outcome, 1-Year Score)...
AUC: 0.629 (0.622-0.635) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2921 (5.8%) (from 50000 individuals)
Excluded 0 prevalent cases for Diabetes.

   Sex-stratified analysis:
   Female: AUC = 0.622, Events = 1193/27107
   Male: AUC = 0.625, Events = 1728/22893

Evaluating Atrial_Fib (10-Year Outcome, 1-Year Score)...
AUC: 0.710 (0.697-0.716) (calculated on 49353 individuals)
Events (10-Year in Eval Cohort): 1919 (3.8%) (from 50000 individuals)
Excluded 647 prevalent cases for Atrial_Fib.

   Sex-stratified analysis:
   Female: AUC = 0.714, Events = 1041/26933
   Male: AUC = 0.713, Events = 852/22420

Evaluating CKD (10-Year Outcome, 1-Year Score)...
AUC: 0.702 (0.701-0.716) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1135 (2.3%) (from 50000 individuals)
Excluded 0 prevalent cases for CKD.

   Sex-stratified analysis:
   Female: AUC = 0.708, Events = 530/27107
   Male: AUC = 0.695, Events = 605/22893

Evaluating All_Cancers (10-Year Outcome, 1-Year Score)...
AUC: 0.674 (0.669-0.680) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2521 (5.0%) (from 50000 individuals)
Excluded 0 prevalent cases for All_Cancers.

   Sex-stratified analysis:
   Female: AUC = 0.639, Events = 785/27107
   Male: AUC = 0.676, Events = 1736/22893

Evaluating Stroke (10-Year Outcome, 1-Year Score)...
AUC: 0.675 (0.654-0.694) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 692 (1.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Stroke.

   Sex-stratified analysis:
   Female: AUC = 0.672, Events = 313/27107
   Male: AUC = 0.673, Events = 379/22893

Evaluating Heart_Failure (10-Year Outcome, 1-Year Score)...
AUC: 0.705 (0.692-0.715) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 991 (2.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Heart_Failure.

   Sex-stratified analysis:
   Female: AUC = 0.732, Events = 352/27107
   Male: AUC = 0.679, Events = 639/22893

Evaluating Pneumonia (10-Year Outcome, 1-Year Score)...
AUC: 0.650 (0.635-0.659) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1758 (3.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Pneumonia.

   Sex-stratified analysis:
   Female: AUC = 0.650, Events = 773/27107
   Male: AUC = 0.644, Events = 985/22893

Evaluating COPD (10-Year Outcome, 1-Year Score)...
AUC: 0.664 (0.660-0.673) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2071 (4.1%) (from 50000 individuals)
Excluded 0 prevalent cases for COPD.

   Sex-stratified analysis:
   Female: AUC = 0.667, Events = 925/27107
   Male: AUC = 0.657, Events = 1146/22893

Evaluating Osteoporosis (10-Year Outcome, 1-Year Score)...
AUC: 0.676 (0.661-0.685) (calculated on 49858 individuals)
Events (10-Year in Eval Cohort): 1101 (2.2%) (from 50000 individuals)
Excluded 142 prevalent cases for Osteoporosis.

   Sex-stratified analysis:
   Female: AUC = 0.676, Events = 611/26995
   Male: AUC = 0.677, Events = 489/22863

Evaluating Anemia (10-Year Outcome, 1-Year Score)...
AUC: 0.594 (0.585-0.599) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2676 (5.4%) (from 50000 individuals)
Excluded 0 prevalent cases for Anemia.

   Sex-stratified analysis:
   Female: AUC = 0.567, Events = 1430/27107
   Male: AUC = 0.624, Events = 1246/22893

Evaluating Colorectal_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.652 (0.634-0.663) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 613 (1.2%) (from 50000 individuals)
Excluded 0 prevalent cases for Colorectal_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.629, Events = 282/27107
   Male: AUC = 0.667, Events = 331/22893

Evaluating Breast_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Female: Found 27107 individuals in cohort
AUC: 0.545 (0.533-0.555) (calculated on 27107 individuals)
Events (10-Year in Eval Cohort): 1098 (4.1%) (from 27107 individuals)
Excluded 0 prevalent cases for Breast_Cancer.

Evaluating Prostate_Cancer (10-Year Outcome, 1-Year Score)...
Filtering for Male: Found 22893 individuals in cohort
AUC: 0.683 (0.672-0.699) (calculated on 22640 individuals)
Events (10-Year in Eval Cohort): 952 (4.2%) (from 22893 individuals)
Excluded 253 prevalent cases for Prostate_Cancer.

Evaluating Lung_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.673 (0.666-0.697) (calculated on 49963 individuals)
Events (10-Year in Eval Cohort): 432 (0.9%) (from 50000 individuals)
Excluded 37 prevalent cases for Lung_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.679, Events = 218/27088
   Male: AUC = 0.671, Events = 212/22875

Evaluating Bladder_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.714 (0.693-0.742) (calculated on 49870 individuals)
Events (10-Year in Eval Cohort): 255 (0.5%) (from 50000 individuals)
Excluded 130 prevalent cases for Bladder_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.691, Events = 148/27077
   Male: AUC = 0.747, Events = 107/22793

Evaluating Secondary_Cancer (10-Year Outcome, 1-Year Score)...
AUC: 0.605 (0.598-0.615) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 1497 (3.0%) (from 50000 individuals)
Excluded 0 prevalent cases for Secondary_Cancer.

   Sex-stratified analysis:
   Female: AUC = 0.584, Events = 857/27107
   Male: AUC = 0.635, Events = 640/22893

Evaluating Depression (10-Year Outcome, 1-Year Score)...
AUC: 0.478 (0.462-0.493) (calculated on 49510 individuals)
Events (10-Year in Eval Cohort): 1995 (4.0%) (from 50000 individuals)
Excluded 490 prevalent cases for Depression.

   Sex-stratified analysis:
   Female: AUC = 0.474, Events = 1071/26809
   Male: AUC = 0.494, Events = 909/22701

Evaluating Anxiety (10-Year Outcome, 1-Year Score)...
AUC: 0.514 (0.502-0.528) (calculated on 49863 individuals)
Events (10-Year in Eval Cohort): 1289 (2.6%) (from 50000 individuals)
Excluded 137 prevalent cases for Anxiety.

   Sex-stratified analysis:
   Female: AUC = 0.521, Events = 692/27021
   Male: AUC = 0.508, Events = 597/22842

Evaluating Bipolar_Disorder (10-Year Outcome, 1-Year Score)...
AUC: 0.458 (0.402-0.498) (calculated on 49925 individuals)
Events (10-Year in Eval Cohort): 121 (0.2%) (from 50000 individuals)
Excluded 75 prevalent cases for Bipolar_Disorder.

   Sex-stratified analysis:
   Female: AUC = 0.481, Events = 69/27055
   Male: AUC = 0.419, Events = 51/22870

Evaluating Rheumatoid_Arthritis (10-Year Outcome, 1-Year Score)...
AUC: 0.611 (0.603-0.624) (calculated on 49752 individuals)
Events (10-Year in Eval Cohort): 595 (1.2%) (from 50000 individuals)
Excluded 248 prevalent cases for Rheumatoid_Arthritis.

   Sex-stratified analysis:
   Female: AUC = 0.615, Events = 314/26913
   Male: AUC = 0.610, Events = 279/22839

Evaluating Psoriasis (10-Year Outcome, 1-Year Score)...
AUC: 0.549 (0.516-0.598) (calculated on 49901 individuals)
Events (10-Year in Eval Cohort): 236 (0.5%) (from 50000 individuals)
Excluded 99 prevalent cases for Psoriasis.

   Sex-stratified analysis:
   Female: AUC = 0.538, Events = 111/27069
   Male: AUC = 0.558, Events = 124/22832

Evaluating Ulcerative_Colitis (10-Year Outcome, 1-Year Score)...
AUC: 0.570 (0.525-0.602) (calculated on 49750 individuals)
Events (10-Year in Eval Cohort): 248 (0.5%) (from 50000 individuals)
Excluded 250 prevalent cases for Ulcerative_Colitis.

   Sex-stratified analysis:
   Female: AUC = 0.573, Events = 144/26979
   Male: AUC = 0.571, Events = 103/22771

Evaluating Crohns_Disease (10-Year Outcome, 1-Year Score)...
AUC: 0.565 (0.534-0.628) (calculated on 49850 individuals)
Events (10-Year in Eval Cohort): 136 (0.3%) (from 50000 individuals)
Excluded 150 prevalent cases for Crohns_Disease.

   Sex-stratified analysis:
   Female: AUC = 0.584, Events = 77/27034
   Male: AUC = 0.542, Events = 59/22816

Evaluating Asthma (10-Year Outcome, 1-Year Score)...
AUC: 0.527 (0.519-0.534) (calculated on 48407 individuals)
Events (10-Year in Eval Cohort): 3038 (6.1%) (from 50000 individuals)
Excluded 1593 prevalent cases for Asthma.

   Sex-stratified analysis:
   Female: AUC = 0.546, Events = 1591/26140
   Male: AUC = 0.542, Events = 1350/22267

Evaluating Parkinsons (10-Year Outcome, 1-Year Score)...
AUC: 0.731 (0.711-0.758) (calculated on 49971 individuals)
Events (10-Year in Eval Cohort): 223 (0.4%) (from 50000 individuals)
Excluded 29 prevalent cases for Parkinsons.

   Sex-stratified analysis:
   Female: AUC = 0.744, Events = 124/27095
   Male: AUC = 0.716, Events = 99/22876

Evaluating Multiple_Sclerosis (10-Year Outcome, 1-Year Score)...
AUC: 0.549 (0.510-0.633) (calculated on 49869 individuals)
Events (10-Year in Eval Cohort): 97 (0.2%) (from 50000 individuals)
Excluded 131 prevalent cases for Multiple_Sclerosis.

   Sex-stratified analysis:
   Female: AUC = 0.530, Events = 61/27009
   Male: AUC = 0.582, Events = 36/22860

Evaluating Thyroid_Disorders (10-Year Outcome, 1-Year Score)...
AUC: 0.573 (0.563-0.595) (calculated on 50000 individuals)
Events (10-Year in Eval Cohort): 2238 (4.5%) (from 50000 individuals)
Excluded 0 prevalent cases for Thyroid_Disorders.

   Sex-stratified analysis:
   Female: AUC = 0.573, Events = 1792/27107
   Male: AUC = 0.560, Events = 446/22893

Summary of Results (Prospective 10-Year Outcome, 1-Year Score, Sex-Adjusted):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)  
--------------------------------------------------------------------------------
ASCVD                0.730 (0.721-0.733)       4333       8.7
Diabetes             0.629 (0.622-0.635)       2921       5.8
Atrial_Fib           0.710 (0.697-0.716)       1919       3.8
CKD                  0.702 (0.701-0.716)       1135       2.3
All_Cancers          0.674 (0.669-0.680)       2521       5.0
Stroke               0.675 (0.654-0.694)       692        1.4
Heart_Failure        0.705 (0.692-0.715)       991        2.0
Pneumonia            0.650 (0.635-0.659)       1758       3.5
COPD                 0.664 (0.660-0.673)       2071       4.1
Osteoporosis         0.676 (0.661-0.685)       1101       2.2
Anemia               0.594 (0.585-0.599)       2676       5.4
Colorectal_Cancer    0.652 (0.634-0.663)       613        1.2
Breast_Cancer        0.545 (0.533-0.555)       1098       4.1
Prostate_Cancer      0.683 (0.672-0.699)       952        4.2
Lung_Cancer          0.673 (0.666-0.697)       432        0.9
Bladder_Cancer       0.714 (0.693-0.742)       255        0.5
Secondary_Cancer     0.605 (0.598-0.615)       1497       3.0
Depression           0.478 (0.462-0.493)       1995       4.0
Anxiety              0.514 (0.502-0.528)       1289       2.6
Bipolar_Disorder     0.458 (0.402-0.498)       121        0.2
Rheumatoid_Arthritis 0.611 (0.603-0.624)       595        1.2
Psoriasis            0.549 (0.516-0.598)       236        0.5
Ulcerative_Colitis   0.570 (0.525-0.602)       248        0.5
Crohns_Disease       0.565 (0.534-0.628)       136        0.3
Asthma               0.527 (0.519-0.534)       3038       6.1
Parkinsons           0.731 (0.711-0.758)       223        0.4
Multiple_Sclerosis   0.549 (0.510-0.633)       97         0.2
Thyroid_Disorders    0.573 (0.563-0.595)       2238       4.5
--------------------------------------------------------------------------------
Evaluating dynamic 1-year AUC...
Filtering for 1: Found 22893 individuals in cohort

Summary of Results (1-Year Risk, Sex-Adjusted, Offset=0):
--------------------------------------------------------------------------------
Disease Group        AUC                       Events     Rate (%)   C-index   
--------------------------------------------------------------------------------
ASCVD                0.872 (0.851-0.883)       397        0.8        N/A
Diabetes             0.712 (0.682-0.748)       195        0.4        N/A
Atrial_Fib           0.833 (0.800-0.854)       120        0.2        N/A
CKD                  0.676 (0.537-0.778)       16         0.0        N/A
All_Cancers          0.738 (0.702-0.773)       115        0.2        N/A
Stroke               0.659 (0.634-0.705)       30         0.1        N/A
Heart_Failure        0.805 (0.762-0.841)       51         0.1        N/A
Pneumonia            0.610 (0.567-0.662)       56         0.1        N/A
COPD                 0.727 (0.671-0.752)       99         0.2        N/A
Osteoporosis         0.788 (0.698-0.848)       44         0.1        N/A
Anemia               0.633 (0.605-0.672)       144        0.3        N/A
Colorectal_Cancer    0.853 (0.812-0.900)       37         0.1        N/A
Breast_Cancer        0.846 (0.800-0.911)       89         0.2        N/A
Prostate_Cancer      0.806 (0.785-0.848)       56         0.2        N/A
Lung_Cancer          0.749 (0.610-0.839)       11         0.0        N/A
Bladder_Cancer       0.831 (0.604-0.860)       9          0.0        N/A
Secondary_Cancer     0.600 (0.566-0.624)       51         0.1        N/A
Depression           0.673 (0.582-0.702)       62         0.1        N/A
Anxiety              0.679 (0.496-0.742)       24         0.0        N/A
Bipolar_Disorder     0.849 (0.800-0.983)       4          0.0        N/A
Rheumatoid_Arthritis 0.802 (0.772-0.866)       34         0.1        N/A
Psoriasis            0.718 (0.624-0.793)       9          0.0        N/A
Ulcerative_Colitis   0.932 (0.913-0.965)       17         0.0        N/A
Crohns_Disease       0.967 (0.960-0.975)       7          0.0        N/A
Asthma               0.668 (0.631-0.680)       251        0.5        N/A
Parkinsons           0.790 (0.681-0.927)       10         0.0        N/A
Multiple_Sclerosis   0.948 (0.935-0.954)       13         0.0        N/A
Thyroid_Disorders    0.660 (0.617-0.682)       137        0.3        N/A
--------------------------------------------------------------------------------
  ✓ Static 10-year: 28 diseases
  ✓ Dynamic 1-year: 28 diseases

================================================================================
SUMMARY
================================================================================
Evaluated 5 configurations
Static 10-year AUC: 140 disease-config combinations
Dynamic 1-year AUC: 140 disease-config combinations

Results saved to:
  - /Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_5_configs_5batches_static_10yr_auc_results.csv
  - /Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/all_5_configs_5batches_dynamic_1yr_auc_results.csv
================================================================================
COMPLETED
================================================================================