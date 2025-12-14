# Disease Definitions Verification for Performance Comparison

## ✅ CONFIRMED: All Definitions Match

### Functions Used to Generate Results

1. **1-Year Baseline (washout_0yr_results.csv)**
   - Function: `evaluate_major_diseases_wsex_with_bootstrap_dynamic_1year_different_start_end_numeric_sex`
   - Source: `evaluatetdccode.py` (line 3)
   - Script: `generate_washout_predictions.py`

2. **1-Year Median (medians_with_global0.csv)**
   - Uses same function as above, aggregated across offsets 0-9

3. **10-Year Static (static_10yr_results.csv)**
   - Function: `evaluate_major_diseases_wsex_with_bootstrap_from_pi`
   - Source: `fig5utils.py` (line 1685)
   - Script: `generate_time_horizon_predictions.py`

### Disease Definitions (Consistent Across All Functions)

All three functions use **identical** disease category definitions:

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

### Key Points

1. ✅ **All_Cancers excludes breast cancer** - This is intentional and consistent across all functions
2. ✅ **Secondary_Cancer definition** - Consistent across all functions
3. ✅ **ASCVD definition** - Consistent across all functions
4. ✅ **All other categories** - Consistent across all functions

### Performance Comparison Script

The `generate_performance_comparison.py` script:
- ✅ Only **loads** CSV files (doesn't regenerate results)
- ✅ Uses disease names from CSV files to merge data
- ✅ No disease definition logic - relies on pre-generated results

**Conclusion**: The performance comparison plot will correctly match all results because:
1. All source functions use identical disease definitions
2. The comparison script only loads and merges pre-generated CSV files
3. Disease names in CSV files match across all sources

## Recommendation

✅ **No changes needed** - Everything is consistent and correct!

The 10-year static function (`evaluate_major_diseases_wsex_with_bootstrap_from_pi` from `fig5utils.py`) is the best choice because:
- It's the function actually used to generate the results
- It has bootstrap CIs
- It handles sex-specific diseases correctly
- It accepts pre-computed pi tensor (efficient)
