# Fuzzy Matching: How Disease Names Are Matched Between Cohorts

## The Problem

UKB and MGB use **different disease naming conventions**:
- UKB: `"coronary_artery_disease"` (underscores)
- MGB: `"coronary artery disease"` (spaces) or `"coronary_artery_disease"` (same)
- UKB: `"type_2_diabetes"` 
- MGB: `"diabetes_mellitus"` or `"type 2 diabetes"` (different format)

**Exact string matching would fail!** We need to match diseases even when names are slightly different.

---

## What is Fuzzy Matching?

**Fuzzy matching** is a technique to find similar strings even when they're not exactly the same. It handles:
- Different separators (underscores vs spaces)
- Different word order
- Slight spelling differences
- Abbreviations vs full names

---

## How Our Fuzzy Matching Works

The `match_disease_names_by_keywords()` function uses a **multi-step scoring system**:

### Step 1: Extract Key Terms

**Remove common words** that don't help matching:
```python
common_words = {'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
               'disease', 'disorder', 'syndrome', 'condition', 'acute', 'chronic'}
```

**Example**:
- UKB: `"coronary_artery_disease"` → key terms: `{"coronary", "artery"}`
- MGB: `"coronary artery disease"` → key terms: `{"coronary", "artery"}`

### Step 2: Calculate Multiple Similarity Scores

#### Score 1: Exact Word Match (70% weight)
```python
overlap = ukb_terms & mgb_terms  # Set intersection
exact_match_score = len(overlap) / max(len(ukb_terms), len(mgb_terms))
```

**Example**:
- UKB terms: `{"coronary", "artery"}`
- MGB terms: `{"coronary", "artery"}`
- Overlap: `{"coronary", "artery"}` (2 words)
- Score: `2 / max(2, 2) = 1.0` (perfect match!)

#### Score 2: Substring Match (20% weight)
```python
# Check if one term is contained in another
if "coronary" in "coronary_artery" or "coronary_artery" in "coronary":
    substring_match = True
```

**Example**:
- `"myocardial"` in `"myocardial_infarction"` → match!
- `"diabetes"` in `"diabetes_mellitus"` → match!

#### Score 3: Character Overlap (10% weight)
```python
# Compare character sets (ignoring spaces, underscores, hyphens)
ukb_chars = set("coronaryarterydisease")
mgb_chars = set("coronaryarterydisease")
char_overlap = len(ukb_chars & mgb_chars) / max(len(ukb_chars), len(mgb_chars))
```

**Example**:
- UKB: `"coronary_artery_disease"` → chars: `{'c','o','r','n','a','y','_','t','e','r','y','d','i','s','e'}`
- MGB: `"coronary artery disease"` → chars: `{'c','o','r','n','a','y',' ','t','e','r','y','d','i','s','e'}`
- Overlap: Most characters match (ignoring `_` vs ` `)
- Score: High character overlap

### Step 3: Combine Scores

```python
total_score = 0.0
if exact_match_score > 0:
    total_score += exact_match_score * 0.7  # 70% weight
if substring_match:
    total_score += 0.2  # 20% weight
if char_overlap > 0.5:
    total_score += char_overlap * 0.1  # 10% weight
```

**Threshold**: Only keep matches with `total_score > 0.2`

---

## Examples

### Example 1: Perfect Match
- **UKB**: `"coronary_artery_disease"`
- **MGB**: `"coronary artery disease"`
- **Key terms**: Both → `{"coronary", "artery"}`
- **Exact match**: 2/2 = 1.0
- **Substring**: N/A (exact match)
- **Character overlap**: High
- **Total score**: `1.0 * 0.7 + 0.2 + 0.1 = 1.0` ✅ **MATCH!**

### Example 2: Partial Match
- **UKB**: `"type_2_diabetes"`
- **MGB**: `"diabetes_mellitus"`
- **Key terms**: UKB → `{"type", "2", "diabetes"}`, MGB → `{"diabetes", "mellitus"}`
- **Exact match**: 1/3 = 0.33 (only "diabetes" matches)
- **Substring**: `"diabetes"` in both → True
- **Character overlap**: Moderate
- **Total score**: `0.33 * 0.7 + 0.2 + 0.05 = 0.48` ✅ **MATCH!**

### Example 3: No Match
- **UKB**: `"myocardial_infarction"`
- **MGB**: `"diabetes_mellitus"`
- **Key terms**: UKB → `{"myocardial", "infarction"}`, MGB → `{"diabetes", "mellitus"}`
- **Exact match**: 0/2 = 0.0 (no overlap)
- **Substring**: False
- **Character overlap**: Low
- **Total score**: `0.0` ❌ **NO MATCH**

---

## Why Fuzzy Matching is Necessary

### Without Fuzzy Matching:
- `"coronary_artery_disease"` ≠ `"coronary artery disease"` (exact match fails)
- `"type_2_diabetes"` ≠ `"diabetes_mellitus"` (exact match fails)
- **Result**: No diseases matched, pathways can't be compared!

### With Fuzzy Matching:
- `"coronary_artery_disease"` ≈ `"coronary artery disease"` (fuzzy match succeeds)
- `"type_2_diabetes"` ≈ `"diabetes_mellitus"` (partial match, both have "diabetes")
- **Result**: Diseases matched, pathways can be compared!

---

## Limitations

1. **False Positives**: Might match unrelated diseases that share words
   - Example: `"heart_disease"` might match `"heart_failure"` (both have "heart")
   - **Mitigation**: Require enrichment in both pathways (>1.1x)

2. **False Negatives**: Might miss matches if names are very different
   - Example: `"MI"` vs `"myocardial_infarction"` (abbreviation)
   - **Mitigation**: Lower threshold (0.2) to catch more matches

3. **Ambiguity**: One UKB disease might match multiple MGB diseases
   - **Solution**: Use the **best match** (highest score)

---

## Code Reference

See `match_pathways_by_disease_patterns.py`:
- `match_disease_names_by_keywords()`: Lines 90-151
- Uses multi-step scoring (exact match + substring + character overlap)
- Returns sorted list of potential matches (best first)

---

## Summary

**Fuzzy matching** allows us to match disease names between cohorts even when:
- Names use different separators (underscores vs spaces)
- Names have different formats (abbreviations vs full names)
- Names are slightly different but refer to the same disease

**This is essential** for comparing pathways across cohorts with different naming conventions!

