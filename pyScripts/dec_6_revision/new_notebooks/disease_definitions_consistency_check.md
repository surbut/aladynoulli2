# Disease Category Definitions Consistency Check

## Summary

After checking disease category definitions across evaluation functions, here are the findings:

## Key Findings

### 1. **All_Cancers Definition - INCONSISTENCY FOUND**

**Most evaluation functions (fig5utils.py, evaluatetdccode.py):**
```python
'All_Cancers': [
    'Colon cancer', 
    'Cancer of bronchus; lung', 
    'Cancer of prostate', 
    'Malignant neoplasm of bladder', 
    'Secondary malignant neoplasm',
    'Secondary malignant neoplasm of digestive systems', 
    'Secondary malignant neoplasm of liver'
]
```
**Does NOT include breast cancer** ✓ (This is intentional - breast cancer is analyzed separately)

**Some older functions (utils.py):**
```python
'All_Cancers': [
    'Colon cancer', 
    'Malignant neoplasm of rectum, rectosigmoid junction, and anus',
    'Cancer of bronchus; lung',
    'Breast cancer [female]',  # ← INCLUDES breast cancer
    'Malignant neoplasm of female breast',  # ← INCLUDES breast cancer
    'Cancer of prostate',
    'Malignant neoplasm of bladder',
    'Secondary malignant neoplasm',
    'Secondary malignancy of lymph nodes',
    'Secondary malignancy of respiratory organs',
    'Secondary malignant neoplasm of digestive systems',
    'Secondary malignant neoplasm of liver',
    'Secondary malignancy of bone'
]
```

### 2. **Secondary_Cancer Definition - MOSTLY CONSISTENT**

**Standard definition (used in fig5utils.py, evaluatetdccode.py):**
```python
'Secondary_Cancer': [
    'Secondary malignant neoplasm', 
    'Secondary malignancy of lymph nodes', 
    'Secondary malignancy of respiratory organs', 
    'Secondary malignant neoplasm of digestive systems'
]
```

**Some variations include:**
- `'Secondary malignant neoplasm of liver'` (in some analysis scripts)

**Note:** Some secondary cancer types are also included in `All_Cancers`, which creates overlap.

### 3. **ASCVD Definition - CONSISTENT**

All functions use the same definition:
```python
'ASCVD': [
    'Myocardial infarction', 
    'Coronary atherosclerosis', 
    'Other acute and subacute forms of ischemic heart disease', 
    'Unstable angina (intermediate coronary syndrome)', 
    'Angina pectoris', 
    'Other chronic ischemic heart disease, unspecified'
]
```

## Recommendations

### For Performance Comparison Plot

**Use functions from `fig5utils.py`** because:
1. ✅ Most up-to-date definitions
2. ✅ Consistent with `evaluatetdccode.py` 
3. ✅ `All_Cancers` correctly excludes breast cancer (analyzed separately)
4. ✅ Used in the performance notebook

**Recommended function:**
- `evaluate_major_diseases_wsex_with_bootstrap_from_pi()` (line 1685 in fig5utils.py)
  - Accepts pre-computed pi tensor (faster)
  - Has bootstrap CIs
  - Uses consistent disease definitions
  - Handles sex-specific diseases correctly

### Disease Definition Standard (from fig5utils.py)

```python
major_diseases = {
    'ASCVD': ['Myocardial infarction', 'Coronary atherosclerosis', 'Other acute and subacute forms of ischemic heart disease', 
              'Unstable angina (intermediate coronary syndrome)', 'Angina pectoris', 'Other chronic ischemic heart disease, unspecified'],
    'Diabetes': ['Type 2 diabetes'],
    'Atrial_Fib': ['Atrial fibrillation and flutter'],
    'CKD': ['Chronic renal failure [CKD]', 'Chronic Kidney Disease, Stage III'],
    'All_Cancers': ['Colon cancer', 'Cancer of bronchus; lung', 'Cancer of prostate', 'Malignant neoplasm of bladder', 
                    'Secondary malignant neoplasm','Secondary malignant neoplasm of digestive systems', 'Secondary malignant neoplasm of liver'],
    'Stroke': ['Cerebral artery occlusion, with cerebral infarction', 'Cerebral ischemia'],
    'Heart_Failure': ['Congestive heart failure (CHF) NOS', 'Heart failure NOS'],
    'Pneumonia': ['Pneumonia', 'Bacterial pneumonia', 'Pneumococcal pneumonia'],
    'COPD': ['Chronic airway obstruction', 'Emphysema', 'Obstructive chronic bronchitis'],
    'Osteoporosis': ['Osteoporosis NOS'],
    'Anemia': ['Iron deficiency anemias, unspecified or not due to blood loss', 'Other anemias'],
    'Colorectal_Cancer': ['Colon cancer', 'Malignant neoplasm of rectum, rectosigmoid junction, and anus'],
    'Breast_Cancer': ['Breast cancer [female]', 'Malignant neoplasm of female breast'],  # Sex-specific
    'Prostate_Cancer': ['Cancer of prostate'],  # Sex-specific
    'Lung_Cancer': ['Cancer of bronchus; lung'],
    'Bladder_Cancer': ['Malignant neoplasm of bladder'],
    'Secondary_Cancer': ['Secondary malignant neoplasm', 'Secondary malignancy of lymph nodes', 
                         'Secondary malignancy of respiratory organs', 'Secondary malignant neoplasm of digestive systems'],
    'Depression': ['Major depressive disorder'],
    'Anxiety': ['Anxiety disorder'],
    'Bipolar_Disorder': ['Bipolar'],
    'Rheumatoid_Arthritis': ['Rheumatoid arthritis'],
    'Psoriasis': ['Psoriasis vulgaris'],
    'Ulcerative_Colitis': ['Ulcerative colitis'],
    'Crohns_Disease': ['Regional enteritis'],
    'Asthma': ['Asthma'],
    'Parkinsons': ["Parkinson's disease"],
    'Multiple_Sclerosis': ['Multiple sclerosis'],
    'Thyroid_Disorders': ['Thyrotoxicosis with or without goiter', 'Secondary hypothyroidism', 'Hypothyroidism NOS']
}
```

## Notes

1. **All_Cancers excluding breast cancer is intentional** - Breast cancer is analyzed separately as a sex-specific disease, which makes sense for fair comparisons.

2. **Overlap between All_Cancers and Secondary_Cancer** - Some secondary cancer types appear in both categories. This is acceptable as they serve different analytical purposes:
   - `All_Cancers`: Primary + secondary cancers combined
   - `Secondary_Cancer`: Only metastatic/secondary cancers

3. **Consistency across evaluation functions** - The definitions in `fig5utils.py` and `evaluatetdccode.py` are consistent, which is what matters for the performance comparison.

## Action Items

✅ **No changes needed** - The current definitions in `fig5utils.py` are appropriate and consistent for the performance comparison plot.

The slight inconsistency with older `utils.py` functions is not a concern since those are not being used for the current performance evaluation.
