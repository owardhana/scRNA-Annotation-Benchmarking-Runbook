"""
figstyle.py — one palette, one ramp, one type spec for every figure in the paper.

Everything visual that has to agree across Figures 1-4 and Supplementary Figs S1-S16
lives here. Build scripts import from this module and never hard-code a hex.

Colour-blind safety
-------------------
Family colours are the Okabe-Ito qualitative set (validated with colour-blind observers)
with Tol wine for classical ML and Tol indigo for foundation models. At 100% simulated
severity the minimum pairwise separation across the seven families is dE 15.0
(protanomaly) and 15.2 (deuteranomaly) in CAM02-UCS. Accent sets for non-family
categoricals come from Paul Tol's "muted" scheme. Colour is never the only cue: every
family is also named or spatially grouped in the panel where it appears.

Good vs bad
-----------
One convention throughout: warm/red = poor agreement, cool/blue = good agreement.
The ramp is ColorBrewer RdBu reversed, diverging about 0.5 — a real reference point
for Cohen's kappa (Landis-Koch "moderate") and for the 0 / 0.5 / 1 ontology score.
Failed runs are a neutral mid grey that appears nowhere in the ramp.
"""

# ---------------------------------------------------------------- type
FONT = "Liberation Sans"
PT_TAG, PT_LAB, PT_TICK, PT_ROW = 8, 7, 6, 5.5
INK, MUTED, RULE, LW = "#000000", "#4d4d4d", "#bfbfbf", 0.5

# ---------------------------------------------------------------- families
# Okabe-Ito blue / orange / bluish-green / sky-blue / vermillion, with Tol wine and
# indigo. Both the catmap key spelling and the display spelling map to the same hex
# so no script has to translate.
_FAM = [
    (("Marker-Based",     "Marker-based"),     "#0072B2"),
    (("Similarity-Based", "Similarity-based"), "#E69F00"),
    (("Classic ML",       "Classical ML"),     "#882255"),
    (("Deep Learning",    "Deep learning"),    "#009E73"),
    (("Semi-Supervised",  "Semi-supervised"),  "#56B4E9"),
    (("LLM",              "LLM"),              "#D55E00"),
    (("Foundation model", "Foundation Model"), "#332288"),
]
FAMILY_COLORS = {k: hexv for keys, hexv in _FAM for k in keys}

FAMILY_ORDER = ["Marker-Based", "Similarity-Based", "Classic ML",
                "Deep Learning", "Semi-Supervised"]
ALL_FAMILY_ORDER = FAMILY_ORDER + ["LLM", "Foundation model"]

# catmap spelling -> the spelling that appears in figure labels
SHORT = {"Marker-Based": "Marker-based", "Similarity-Based": "Similarity-based",
         "Classic ML": "Classical ML", "Deep Learning": "Deep learning",
         "Semi-Supervised": "Semi-supervised", "LLM": "LLM",
         "Foundation model": "Foundation model"}

# the three evaluation regimes of Figure 3 / Supplementary Fig. S16
C_DB, C_LLM, C_FM = FAMILY_COLORS["Marker-Based"], FAMILY_COLORS["LLM"], FAMILY_COLORS["Foundation model"]

# ---------------------------------------------------------------- good <-> bad ramp
FAIL_GREY = "#a6a6a6"          # no valid prediction; not a kappa value
_RDBU_R = ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
           "#f7f2ee",                                   # 0.5, barely tinted, never neutral grey
           "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"]


def score_ramp():
    """Diverging red(poor) -> blue(good) map for kappa and ontology score, 0-1."""
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("score_rdbu", _RDBU_R).copy()
    cm.set_bad(FAIL_GREY)
    return cm


# ---------------------------------------------------------------- non-family categoricals
# Paul Tol "muted" — used where the categories are NOT method families, so that a
# reader never has to wonder whether a colour means a family.
TOL = {"indigo": "#332288", "cyan": "#88CCEE", "teal": "#44AA99", "green": "#117733",
       "olive": "#999933", "sand": "#DDCC77", "rose": "#CC6677", "wine": "#882255",
       "purple": "#AA4499", "grey": "#AAAAAA"}

# Figure 2d — variance components of the crossed dataset x tool model
VARIANCE_COLORS = {"Dataset": TOL["indigo"], "Tool × dataset": TOL["sand"],
                   "Tool": TOL["cyan"], "Replicate": "#5a5a5a"}

# Supplementary Fig. S14 — the three cross-platform blocks
BLOCK_COLORS = {"CellBench": "#7E1E9C", "Pancreas": "#999933", "PBMCbench": "#40B0A6"}

# Figure 1 panel a — three ordered levels of a design factor, neutral so the only
# colour in Figure 1 is the family column
LEVEL_SHADES = ["#f0f0f0", "#c9c9c9", "#6e6e6e"]
LEVEL_INK = ["#000000", "#000000", "#ffffff"]


# ---------------------------------------------------------------- matplotlib rc
def rc(tick=PT_TICK, lab=PT_LAB):
    return {
        "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
        "font.size": tick, "axes.labelsize": lab,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": tick,
        "axes.linewidth": LW, "axes.edgecolor": INK,
        "xtick.major.width": LW, "ytick.major.width": LW,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "figure.dpi": 600, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.grid": False, "figure.autolayout": False,
        "xtick.top": False, "ytick.right": False,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
    }


def family_key(ax, families=None, x=0.42, y=0.36, title="Family", pitch=0.072,
               fontsize=PT_ROW, swatch=0.075, width=0.58, boxed=True):
    """Titled (optionally boxed) family key in axes coordinates.

    x, y  : left edge and the baseline of the TITLE
    pitch : vertical step between entries
    width : box width in axes coordinates
    """
    fams = families or FAMILY_ORDER
    first = y - 0.055                       # baseline of the first entry
    if boxed:
        from matplotlib.patches import FancyBboxPatch
        top = y + 0.048
        bottom = first - (len(fams) - 1) * pitch - 0.030
        ax.add_patch(FancyBboxPatch(
            (x - 0.030, bottom), width, top - bottom, transform=ax.transAxes,
            boxstyle="square,pad=0", facecolor="white", edgecolor=RULE,
            linewidth=LW, zorder=5, clip_on=False))
    ax.text(x, y, title, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="baseline", ha="left", color=INK, zorder=6)
    for i, f in enumerate(fams):
        yy = first - i * pitch
        ax.plot([x, x + swatch], [yy + 0.014] * 2, transform=ax.transAxes,
                color=FAMILY_COLORS[f], lw=1.5, clip_on=False,
                solid_capstyle="butt", zorder=6)
        ax.text(x + swatch + 0.030, yy, SHORT.get(f, f), transform=ax.transAxes,
                fontsize=fontsize, va="baseline", ha="left", color=INK, zorder=6)
