"""Generate multi-panel genetic validation figure for paper.

Panels (if inputs available):
- Panel A: GAMMA loadings for a given signature (barplot of top traits)
- Panel B: Lead loci for the signature (known vs novel)
- Panel C: Evidence matrix for top novel genes (GWAS / component traits / RVAS / known GWAS)
- Panel D: RVAS summary for genes overlapping signature loci

Outputs are saved as PDFs under
  new_notebooks/results/paper_figs/fig4/

Adjust the hard-coded paths/column names at the top as needed.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11


# =============================================================================
# CONFIG (EDIT AS NEEDED)
# =============================================================================

# Default paths (edit to match your setup)
ALL_LOCI_FILE = Path("/Users/sarahurbut/Library/CloudStorage/Dropbox-Personal/all_loci_annotated.tsv")
RVAS_FILE = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/genetic/RVAS_signature_results.tsv")
GAMMA_FILE = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/genetic/gamma_matrix_sig_by_trait.npy")
GAMMA_TRAITS_FILE = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/genetic/gamma_traits.csv")

OUTPUT_DIR = Path("/Users/sarahurbut/aladynoulli2/pyScripts/dec_6_revision/new_notebooks/results/paper_gigs/fig4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs(loci_path: Path, rvas_path: Path, gamma_path: Path, gamma_traits_path: Path):
    """Load loci, RVAS, and (optionally) GAMMA data."""
    print("Loading lead loci from:", loci_path)
    loci = pd.read_csv(loci_path, sep="\t")

    # --- Normalize column names ---
    # Signature column
    if "signature" not in loci.columns and "SIG" in loci.columns:
        loci = loci.rename(columns={"SIG": "signature"})
    # Chromosome column
    if "chr" not in loci.columns and "#CHR" in loci.columns:
        loci = loci.rename(columns={"#CHR": "chr"})
    # Nearest gene column
    if "nearest_gene" not in loci.columns and "nearestgene" in loci.columns:
        loci = loci.rename(columns={"nearestgene": "nearest_gene"})
    # P-value column
    if "pval" not in loci.columns and "p" in loci.columns:
        loci = loci.rename(columns={"p": "pval"})
    # Novelty flag: derive from KNOWN if present
    if "is_novel" not in loci.columns and "KNOWN" in loci.columns:
        # KNOWN == 1 => known locus, KNOWN == 0 => novel
        loci["is_novel"] = loci["KNOWN"].apply(lambda x: False if str(x) == "1" else True)

    rvas = None
    if rvas_path.exists():
        print("Loading RVAS results from:", rvas_path)
        rvas = pd.read_csv(rvas_path, sep="\t")
    else:
        print("[INFO] RVAS file not found; RVAS panels will be skipped.")

    gamma = None
    gamma_traits = None
    if gamma_path.exists() and gamma_traits_path.exists():
        print("Loading GAMMA matrix from:", gamma_path)
        gamma = np.load(gamma_path)
        print("Loading GAMMA trait names from:", gamma_traits_path)
        gamma_traits = pd.read_csv(gamma_traits_path)
    else:
        print("[INFO] GAMMA files not found; GAMMA panel will be skipped.")

    return loci, rvas, gamma, gamma_traits


# =============================================================================
# PANEL A: GAMMA LOADINGS FOR ONE SIGNATURE
# =============================================================================

def plot_gamma_panel(ax, gamma: np.ndarray, gamma_traits: pd.DataFrame, sig_id: int, top_k: int = 20):
    """Barplot of top-k trait loadings for a given signature."""
    sig_row = gamma[sig_id, :]
    trait_names = gamma_traits.iloc[:, 0].astype(str).values

    gamma_df = pd.DataFrame({"trait": trait_names, "gamma": sig_row})
    gamma_df = gamma_df.sort_values("gamma", ascending=False).head(top_k)

    sns.barplot(data=gamma_df, x="gamma", y="trait", ax=ax, palette="viridis")
    ax.set_xlabel(f"GAMMA loading (Signature {sig_id})")
    ax.set_ylabel("")
    ax.set_title(f"Signature {sig_id}: Top {top_k} Trait Loadings")


# =============================================================================
# PANEL B: LEAD LOCI STRIP PLOT (KNOWN VS NOVEL)
# =============================================================================

def plot_loci_panel(ax, sig_loci: pd.DataFrame):
    """Strip/lollipop plot of lead loci for a signature, colored by known vs novel."""
    if "chr" not in sig_loci.columns:
        raise ValueError("Expected 'chr' column in loci table.")
    if "pval" not in sig_loci.columns:
        raise ValueError("Expected 'pval' column in loci table.")

    df = sig_loci.copy()
    df["chr"] = df["chr"].astype(str)
    df["-log10p"] = -np.log10(df["pval"])

    # Define novelty column if not present
    if "is_novel" not in df.columns:
        # Fallback: treat everything as novel
        print("[WARN] 'is_novel' column not found; marking all loci as novel.")
        df["is_novel"] = True

    # Order chromosomes numerically where possible
    def chr_key(c):
        return (not c.isdigit(), int(c) if c.isdigit() else 99)

    chr_order = sorted(df["chr"].unique(), key=chr_key)

    x_vals = []
    y_vals = []
    hue_vals = []
    for i, chrom in enumerate(chr_order):
        df_c = df[df["chr"] == chrom]
        jitter = (np.random.rand(len(df_c)) - 0.5) * 0.3
        x_vals.extend(i + jitter)
        y_vals.extend(df_c["-log10p"].values)
        hue_vals.extend(df_c["is_novel"].map({True: "Novel", False: "Known"}).values)

    plot_df = pd.DataFrame({
        "x": x_vals,
        "y": y_vals,
        "Novelty": hue_vals,
    })

    sns.scatterplot(data=plot_df, x="x", y="y", hue="Novelty", style="Novelty",
                    ax=ax, s=50, alpha=0.85, edgecolor="none")

    ax.set_xticks(range(len(chr_order)))
    ax.set_xticklabels(chr_order)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("-log10(P)")
    ax.set_title("Signature lead loci (Known vs Novel)")
    ax.legend(title="Locus class", fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))


# =============================================================================
# PANEL C: EVIDENCE MATRIX FOR TOP NOVEL GENES
# =============================================================================

def build_evidence_matrix(sig_loci: pd.DataFrame, rvas: Optional[pd.DataFrame],
                           n_genes: int = 20) -> pd.DataFrame:
    """Construct gene x evidence matrix for top novel genes.

    Expects columns in sig_loci:
      - 'nearest_gene' (or 'gene')
      - 'is_novel' (bool)
      - 'component_trait' (optional)
      - 'known_gwas_bool' (optional; True if previously reported)
    """
    df = sig_loci.copy()

    gene_col = None
    if "nearest_gene" in df.columns:
        gene_col = "nearest_gene"
    elif "gene" in df.columns:
        gene_col = "gene"
    else:
        raise ValueError("Expected a 'nearest_gene' or 'gene' column in loci table.")

    novel = df[df["is_novel"]].copy()
    novel = novel.sort_values("pval").head(n_genes)

    novel_genes = novel[gene_col].dropna().astype(str).unique()

    rvas_sig = set()
    if rvas is not None and "gene" in rvas.columns and "pval" in rvas.columns:
        # Use a fairly stringent default threshold; adjust as needed
        rvas_sig = set(rvas.loc[rvas["pval"] < 5e-6, "gene"].astype(str).unique())

    rows = []
    for g in novel_genes:
        loci_g = novel[novel[gene_col] == g]
        has_trait = int("component_trait" in loci_g.columns and loci_g["component_trait"].notna().any())

        if "known_gwas_bool" in loci_g.columns:
            known_flag = int(bool(loci_g["known_gwas_bool"].any()))
        else:
            known_flag = 0

        row = {
            "gene": g,
            "SigLead": 1,
            "n_lead_variants": int(loci_g.shape[0]),
            "Component_trait_hit": has_trait,
            "RVAS_hit": int(g in rvas_sig),
            "Known_GWAS": known_flag,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    evidence_df = pd.DataFrame(rows).set_index("gene")
    return evidence_df


def plot_evidence_matrix(ax, evidence_df: pd.DataFrame):
    """Plot a heatmap for the evidence matrix."""
    if evidence_df.empty:
        ax.text(0.5, 0.5, "No novel genes / evidence matrix empty",
                ha="center", va="center")
        ax.axis("off")
        return

    sns.heatmap(evidence_df, annot=True, fmt="d", cmap="YlGnBu",
                cbar=False, ax=ax)
    ax.set_title("Novel genes: multi-source evidence")
    ax.set_xlabel("Evidence type")
    ax.set_ylabel("Gene")


# =============================================================================
# PANEL D: RVAS SUMMARY BARPLOT
# =============================================================================

def plot_rvas_panel(ax, rvas: Optional[pd.DataFrame], genes_of_interest: list[str], top_k: int = 15):
    """Barplot of -log10(p) for RVAS genes overlapping genes_of_interest."""
    if rvas is None or "gene" not in rvas.columns or "pval" not in rvas.columns:
        ax.text(0.5, 0.5, "RVAS data not available",
                ha="center", va="center")
        ax.axis("off")
        return

    rvas_sub = rvas[rvas["gene"].astype(str).isin(genes_of_interest)].copy()
    if rvas_sub.empty:
        ax.text(0.5, 0.5, "No RVAS hits overlapping novel genes",
                ha="center", va="center")
        ax.axis("off")
        return

    rvas_sub["-log10p"] = -np.log10(rvas_sub["pval"])
    rvas_sub = rvas_sub.sort_values("pval").head(top_k)

    sns.barplot(data=rvas_sub, x="-log10p", y="gene", ax=ax, palette="magma")
    ax.set_xlabel("-log10(P) (RVAS)")
    ax.set_ylabel("")
    ax.set_title("Top RVAS genes overlapping signature loci")


# =============================================================================
# MAIN DRIVER
# =============================================================================


def main(signature: int, loci_path: Path, rvas_path: Path, gamma_path: Path, gamma_traits_path: Path,
         output_dir: Path):
    loci, rvas, gamma, gamma_traits = load_inputs(loci_path, rvas_path, gamma_path, gamma_traits_path)

    # Expect either 'signature' or 'SIG' in loci table
    if "signature" not in loci.columns and "SIG" not in loci.columns:
        raise ValueError("Expected 'signature' or 'SIG' column in loci table.")

    if "signature" not in loci.columns and "SIG" in loci.columns:
        loci = loci.rename(columns={"SIG": "signature"})

    # If values look like 'SIG5', extract numeric part
    if loci["signature"].dtype == object:
        sig_nums = pd.to_numeric(
            loci["signature"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        )
        loci = loci.assign(signature_num=sig_nums)
        sig_loci = loci[loci["signature_num"] == signature].copy()
    else:
        sig_loci = loci[loci["signature"] == signature].copy()
    if sig_loci.empty:
        raise ValueError(f"No loci found for signature {signature}.")

    print(f"Found {len(sig_loci)} lead loci for signature {signature}.")

    # Build evidence matrix for panel C
    evidence_df = build_evidence_matrix(sig_loci, rvas, n_genes=20)
    genes_for_rvas = list(evidence_df.index) if not evidence_df.empty else []

    # --- Create multi-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axA, axB, axC, axD = axes.flat

    # Panel A: GAMMA
    if gamma is not None and gamma_traits is not None and signature < gamma.shape[0]:
        plot_gamma_panel(axA, gamma, gamma_traits, signature, top_k=20)
    else:
        axA.text(0.5, 0.5, "GAMMA data not available",
                 ha="center", va="center")
        axA.axis("off")

    # Panel B: loci strip
    plot_loci_panel(axB, sig_loci)

    # Panel C: evidence matrix
    plot_evidence_matrix(axC, evidence_df)

    # Panel D: RVAS summary
    plot_rvas_panel(axD, rvas, genes_for_rvas, top_k=15)

    fig.suptitle(f"Genetic validation for Signature {signature}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_file = output_dir / f"genetic_validation_signature{signature}_multipanel.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    print(f"\n✓ Saved multi-panel figure to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-panel genetic validation figure.")
    parser.add_argument("--signature", type=int, default=5, help="Signature index to plot (default: 5)")
    parser.add_argument("--loci_path", type=str, default=str(ALL_LOCI_FILE), help="Path to all_loci_annotated.tsv")
    parser.add_argument("--rvas_path", type=str, default=str(RVAS_FILE), help="Path to RVAS results TSV")
    parser.add_argument("--gamma_path", type=str, default=str(GAMMA_FILE), help="Path to GAMMA .npy file")
    parser.add_argument("--gamma_traits_path", type=str, default=str(GAMMA_TRAITS_FILE), help="Path to GAMMA traits CSV")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory for figures")

    args = parser.parse_args()

    main(
        signature=args.signature,
        loci_path=Path(args.loci_path),
        rvas_path=Path(args.rvas_path),
        gamma_path=Path(args.gamma_path),
        gamma_traits_path=Path(args.gamma_traits_path),
        output_dir=Path(args.output_dir),
    )
