# MGB Data Requirements for Pathway Analysis

To run the pathway analysis on MGB data, you'll need the following data files:

## Core Required Files

### 1. **Y_tensor.pt** - Binary Disease Event Matrix
- **Format**: PyTorch tensor file (`.pt`)
- **Shape**: `(N_patients, N_diseases, T_timepoints)`
- **Content**: Binary indicators (0/1) for each disease at each time point for each patient
- **Time points**: Ages 30-81 (52 time points, one per year)
- **How to generate**: From MGB EHR data, create a 3D tensor where:
  - Dimension 0: Patient index (ordered, matches processed_ids)
  - Dimension 1: Disease index (matches disease_names order)
  - Dimension 2: Time/age index (age 30 = index 0, age 81 = index 51)
  - Value: 1 if patient had disease at that age, 0 otherwise

### 2. **thetas.npy or thetas.pt** - Signature Loadings
- **Format**: NumPy array (`.npy`) or PyTorch tensor (`.pt`)
- **Shape**: `(N_patients, K_signatures, T_timepoints)`
- **Content**: Signature loading values (proportions) for each patient at each time point
- **Critical**: These must be generated from MGB data using your **trained Aladyn model** (not just copied from UKB)
- **How to generate**: 
  1. Use the trained model from UKB (`model.pt` or trained weights)
  2. Run the MGB data through the model to generate individual signature trajectories
  3. The model should output `thetas` (softmax-normalized λ values) for each patient
- **Note**: This is the most complex part - you need the full model inference pipeline

### 3. **disease_names.csv** - Disease Name Mapping
- **Format**: CSV file with one column
- **Content**: List of disease names in the same order as dimension 1 of Y_tensor
- **Example**:
  ```
  x
  myocardial infarction
  type 2 diabetes
  essential hypertension
  ...
  ```
- **Must match**: The disease order in Y_tensor exactly

### 4. **processed_ids.csv** - Patient ID Mapping
- **Format**: CSV file with column `eid` (or patient identifier column)
- **Content**: Patient IDs (MGB medical record numbers) in the same order as patients in Y_tensor and thetas
- **Shape**: `(N_patients,)` - one row per patient
- **Example**:
  ```
  eid
  12345
  12346
  12347
  ...
  ```
- **Critical**: Row i in processed_ids must correspond to patient index i in Y_tensor and thetas

## Optional Files (for full analysis)

### 5. **gp_scripts.txt** (or similar) - Medication/Prescription Data
- **Format**: Tab-separated file or CSV
- **Required columns**:
  - `eid`: Patient ID (must match processed_ids)
  - `read_2`: Medication code (BNF, READ, or ATC code)
  - `bnf_code`: BNF category code (if available)
  - Date fields: Prescription date or age at prescription
- **Content**: All prescription records for patients in the analysis
- **Used for**: Medication pathway analysis (which medications differ by pathway)

### 6. **PRS data** (optional, for genetic validation)
- **Format**: CSV with patient IDs and PRS scores
- **Required columns**:
  - Patient ID (matching processed_ids)
  - PRS scores (CAD PRS, CVD PRS, T2D PRS, etc.)
- **Used for**: Validating pathway differences with genetic risk scores

## Data Consistency Requirements

### Critical Mappings:
1. **Patient order must be identical** across:
   - `Y_tensor` (row i)
   - `thetas` (row i)
   - `processed_ids` (row i)

2. **Disease order must match**:
   - `Y_tensor` (column j) 
   - `disease_names` (row j)

3. **Time alignment**:
   - All tensors must use the same time grid (ages 30-81)
   - Age 30 = time index 0, Age 81 = time index 51

## How to Generate Thetas from MGB Data

This is the most complex step. You'll need:

1. **Trained model weights** (from UKB training):
   - `model.pt` or equivalent
   - The full Aladyn model architecture with trained parameters (φ, ψ, λ, γ, etc.)

2. **MGB genetic/demographic data** (G matrix):
   - PRS scores for each patient
   - Sex (0/1)
   - Principal components (PC1-PC10)
   - Other genetic covariates used in the model

3. **Inference pipeline**:
   - Load trained model
   - For each MGB patient:
     - Extract their genetic/demographic features (G_i)
     - Use model to compute signature loadings (λ_i)
     - Normalize to get thetas (θ_i = softmax(λ_i))
   - Stack all patient thetas into array: `(N_patients, K, T)`

4. **Script location**: This inference would typically be done in:
   - `run_aladyn_predict.py` (if it exists) or similar prediction script
   - Or create new script: `generate_mgb_thetas.py`

## File Paths to Update

Once you have the MGB data files, update these paths in the code:

1. **`pathway_discovery.py`** (line 23):
   ```python
   Y_full = torch.load('/path/to/mgb/Y_tensor.pt')
   ```

2. **`pathway_discovery.py`** (line 27):
   ```python
   thetas = torch.load('/path/to/mgb/thetas.pt').numpy()
   ```

3. **`pathway_discovery.py`** (line 32):
   ```python
   processed_ids_df = pd.read_csv('/path/to/mgb/processed_ids.csv')
   ```

4. **`pathway_discovery.py`** (line 42):
   ```python
   disease_names_df = pd.read_csv('/path/to/mgb/disease_names.csv')
   ```

5. **`medication_integration.py`** (line 19):
   ```python
   gp_scripts_path = '/path/to/mgb/gp_scripts.txt'
   ```

## Validation Checklist

Before running pathway analysis, verify:

- [ ] Y_tensor shape: `(N, D, T)` where N = number of patients, D = number of diseases, T = 52
- [ ] thetas shape: `(N, K, T)` where K = number of signatures (typically 21)
- [ ] processed_ids length = N (number of patients)
- [ ] disease_names length = D (number of diseases)
- [ ] Y_tensor and thetas have same N (number of patients)
- [ ] All patient IDs in medication data (if using) exist in processed_ids
- [ ] Thetas are properly normalized (should sum to ~1.0 across signatures for each patient/time)
- [ ] Time alignment: Age 30-81 matches across all tensors

## Minimum Sample Size

- **Minimum patients with target disease**: 50 (for pathway discovery)
- **Recommended**: 100+ patients with target disease for stable pathway identification
- **Ideal**: 1000+ patients for robust statistical validation

## Questions to Ask MGB Data Team

1. "Can you provide a binary disease event matrix (Y_tensor) with shape (N, D, T) for ages 30-81?"
2. "Do you have the trained model weights that we can use to generate signature loadings (thetas) from MGB patient data?"
3. "Can you provide patient IDs in the same order as the Y_tensor rows?"
4. "Do you have prescription/medication data that can be linked to patient IDs?"
5. "What disease codes are used in the MGB data? (ICD-10, PheCodes, etc.)"

