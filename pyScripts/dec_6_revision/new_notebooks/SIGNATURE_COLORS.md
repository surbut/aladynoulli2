# Signature Colors for Figure3 and R3_Q8_Heterogeneity_MainPaper_Method

## Source
Both notebooks now use: `sns.color_palette("tab20", 21)` for signatures 0-20 (21 total signatures)

## Color Formats for Python

### Format 1: RGB Tuples (0-1 range) - **Recommended for matplotlib/seaborn**
This is what Python uses internally. Values range from 0.0 to 1.0.

```python
import seaborn as sns
colors = sns.color_palette("tab20", 21)
# Returns: [(r, g, b), (r, g, b), ...] where each value is 0.0-1.0
```

### Format 2: Hex Codes - **For web/graphics software**
Standard hex format (e.g., `#1f77b4`)

### Format 3: RGB Tuples (0-255 range) - **Standard RGB**
Integer values from 0-255, commonly used in image processing

## Quick Reference

**Python accepts all of these formats:**
- RGB tuples (0-1 range): `(0.12, 0.47, 0.71)` ← **Most common for matplotlib**
- RGB tuples (0-255 range): `(31, 119, 180)`
- Hex codes: `"#1f77b4"`
- Matplotlib color names: `"blue"`, `"red"`, etc.

**Best practice:** Use RGB tuples (0-1 range) or hex codes for maximum compatibility.

## To Generate Colors in Python

```python
import seaborn as sns
import numpy as np

K = 21  # Signatures 0-20
colors = sns.color_palette("tab20", K)

# Convert to different formats if needed:
# Hex codes:
hex_colors = [f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}" for c in colors]

# RGB (0-255):
rgb255_colors = [(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
```

## Notes
- Figure3_Individual_Trajectories.ipynb uses: `colors = sns.color_palette("tab20", K_total)`
- R3_Q8_Heterogeneity_MainPaper_Method.ipynb now uses the same: `colors = sns.color_palette("tab20", K)`
- This ensures signature 0 always has the same color, signature 1 always has the same color, etc., across all plots









