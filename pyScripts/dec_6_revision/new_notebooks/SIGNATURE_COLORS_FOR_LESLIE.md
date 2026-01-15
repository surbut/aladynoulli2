# Signature Colors for Aladynoulli Plots

**For Leslie - Figure Guru**

## Overview
- **21 signatures** (numbered 0-20)
- **Color source:** Matplotlib/seaborn `tab20` palette (first 20) + `tab20b` palette (signature 20)
- **Purpose:** Ensure consistent colors across all plots (Figure3 timeline plots, R3 heterogeneity plots, etc.)

---

## Python Format (RGB tuples, 0-1 range)
**This is what Python/matplotlib uses directly. Recommended format.**

```python
signature_colors = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Signature 0
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),  # Signature 1
    (1.0, 0.4980392156862745, 0.054901960784313725),               # Signature 2
    (1.0, 0.7333333333333333, 0.47058823529411764),                # Signature 3
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Signature 4
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Signature 5 (Red - cardiovascular, swapped from Sig 6)
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),   # Signature 6 (Light green - swapped from Sig 5)
    (1.0, 0.596078431372549, 0.5882352941176471),                  # Signature 7
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),   # Signature 8
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),  # Signature 9
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Signature 10
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),   # Signature 11
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),  # Signature 12
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),  # Signature 13
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),  # Signature 14
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),  # Signature 15
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # Signature 16
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),  # Signature 17
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), # Signature 18
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),  # Signature 19
    (0.24705882352941178, 0.3176470588235294, 0.7098039215686275)  # Signature 20 (from tab20b)
]
```

---

## Hex Codes
**Standard hex format for web/graphics software.**

```
Signature 0:  #1f77b4
Signature 1:  #aec7e8
Signature 2:  #ff7f0e
Signature 3:  #ffbb78
Signature 4:  #2ca02c
Signature 5:  #d62728  (Red - cardiovascular signature)
Signature 6:  #98df8a  (Light green - swapped from Sig 5)
Signature 7:  #ff9896
Signature 8:  #9467bd
Signature 9:  #c5b0d5
Signature 10: #8c564b
Signature 11: #c49c94
Signature 12: #e377c2
Signature 13: #f7b6d2
Signature 14: #7f7f7f
Signature 15: #c7c7c7
Signature 16: #bcbd22
Signature 17: #dbdb8d
Signature 18: #17becf
Signature 19: #9edae5
Signature 20: #3f3fbf  (unique color from tab20b - not a duplicate of signature 0!)
```

---

## RGB (0-255 range)
**Standard RGB integer values (0-255).**

```
Signature 0:  (31, 119, 180)
Signature 1:  (174, 199, 232)
Signature 2:  (255, 127, 14)
Signature 3:  (255, 187, 120)
Signature 4:  (44, 160, 44)
Signature 5:  (214, 39, 40)  (Red - cardiovascular signature)
Signature 6:  (152, 223, 138)  (Light green - swapped from Sig 5)
Signature 7:  (255, 152, 150)
Signature 8:  (148, 103, 189)
Signature 9:  (197, 176, 213)
Signature 10: (140, 86, 75)
Signature 11: (196, 156, 148)
Signature 12: (227, 119, 194)
Signature 13: (247, 182, 210)
Signature 14: (127, 127, 127)
Signature 15: (199, 199, 199)
Signature 16: (188, 189, 34)
Signature 17: (219, 219, 141)
Signature 18: (23, 190, 207)
Signature 19: (158, 218, 229)
Signature 20: (63, 63, 191)  (unique color from tab20b)
```

---

## How to Use in Python

### Option 1: Direct import from seaborn (Recommended)
```python
import seaborn as sns

def get_signature_colors(K=21):
    """Get signature colors matching Figure3."""
    if K <= 20:
        colors = sns.color_palette("tab20", K)
    else:
        colors_20 = sns.color_palette("tab20", 20)
        colors_b = sns.color_palette("tab20b", 20)
        colors = list(colors_20) + [colors_b[0]]  # Signature 20 gets unique color
    
    # Swap signature 5 and 6: Sig 5 (cardiovascular) gets red, Sig 6 gets light green
    if K > 5:
        colors[5], colors[6] = colors[6], colors[5]
    
    return colors

# Usage:
colors = get_signature_colors(21)
# colors[0] is signature 0, colors[20] is signature 20, etc.
```

### Option 2: Use the exact RGB tuples from above
```python
signature_colors = [(r, g, b), ...]  # Copy from "Python Format" section above
```

### Option 3: Use hex codes
```python
signature_colors_hex = ['#1f77b4', '#aec7e8', ...]  # Copy from "Hex Codes" section above
# Convert to RGB if needed:
import matplotlib.colors as mcolors
colors = [mcolors.hex2color(h) for h in signature_colors_hex]
```

---

## Notes for Leslie

1. **All formats work in Python** - RGB tuples (0-1), hex codes, and RGB (0-255) are all accepted by matplotlib/seaborn
2. **Most common format:** RGB tuples in 0-1 range (this is what seaborn returns natively)
3. **Signature 5 is red:** Signature 5 (cardiovascular) is swapped to red (#d62728) for biological interpretability. Signature 6 gets the light green color.
4. **Signature 20 is unique:** It uses the first color from `tab20b` palette, NOT a duplicate of signature 0
4. **Consistency:** These exact colors are used in:
   - `Figure3_Individual_Trajectories.ipynb` (timeline plots)
   - `R3_Q8_Heterogeneity_MainPaper_Method.ipynb` (normalized theta plots, cluster projections)
   - All related visualization scripts

---

## Visual Reference

Signature 20 (the 21st signature) has hex code `#3f3fbf` which is a distinct blue-purple color, different from signature 0's `#1f77b4` blue. This ensures all 21 signatures have visually distinct colors in the plots.

---

**Last updated:** January 2025
**Source:** Matplotlib tab20 + tab20b colormaps via seaborn

