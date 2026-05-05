import shutil

import matplotlib as mpl
import matplotlib.ticker as mticker


PAPER_PLOT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    "figure.titlesize": 15,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "darkgray",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "grid.linestyle": "--",
    "grid.color": "lightgray",
    "grid.alpha": 0.8,
}


LATEX_PLOT_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["cm10"],
    "text.latex.preamble": r"\usepackage{amsmath,bm}",
    "axes.formatter.use_mathtext": True,
}


percent_formatter = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")


DEFAULT_COLOR_CYCLE = [
    "#0072B2",  # blue
    "#E85234",  # reddish purple
    "#6F9B3C",  # green
    "#B57EDC",  # orange
    "#E69F00",  # sky blue
    "#F0E442",  # yellow
    "#6E6E6E",  # black
]

DEFAULT_MARKER_CYCLE = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "p", "*"]
DEFAULT_LINESTYLE_CYCLE = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]

def get_plot_style(label, index=None, overrides=None):
    if index is None:
        index = sum((idx + 1) * ord(char) for idx, char in enumerate(str(label)))

    style = {
        "color": DEFAULT_COLOR_CYCLE[index % len(DEFAULT_COLOR_CYCLE)],
        "marker": DEFAULT_MARKER_CYCLE[index % len(DEFAULT_MARKER_CYCLE)],
        "linestyle": DEFAULT_LINESTYLE_CYCLE[index % len(DEFAULT_LINESTYLE_CYCLE)],
    }
    if overrides and label in overrides:
        style.update(overrides[label])
    return style


def get_color(label, index=None, overrides=None):
    return get_plot_style(label, index=index, overrides=overrides)["color"]


def get_line_style(label, index=None, overrides=None):
    style = get_plot_style(label, index=index, overrides=overrides)
    style.setdefault("linewidth", 3.0)
    style.setdefault("markersize", 11.0)
    return style


def get_scatter_style(label, index=None, overrides=None):
    style = get_plot_style(label, index=index, overrides=overrides)
    return {
        "color": style["color"],
        "marker": style["marker"],
        "s": style.get("s", 38),
        "alpha": style.get("alpha", 0.85),
        "edgecolors": style.get("edgecolors", "none"),
    }


def build_style_map(labels, kind="line", overrides=None):
    style_fn = get_scatter_style if kind == "scatter" else get_line_style
    return {label: style_fn(label, index=idx, overrides=overrides) for idx, label in enumerate(labels)}


def build_color_map(labels, overrides=None):
    return {label: get_color(label, index=idx, overrides=overrides) for idx, label in enumerate(labels)}


def _should_use_tex(use_tex):
    if use_tex == "auto":
        return shutil.which("latex") is not None
    return bool(use_tex)


def paper_plot_rc(overrides=None, use_tex=False):
    rc = PAPER_PLOT_RC.copy()
    if _should_use_tex(use_tex):
        rc.update(LATEX_PLOT_RC)
    if overrides:
        rc.update(overrides)
    return rc


def apply_paper_plot_style(overrides=None, use_tex=False):
    rc = paper_plot_rc(overrides, use_tex=use_tex)
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="paper", rc=rc)
    except ImportError:
        pass
    mpl.rcParams.update(rc)



def apply_latex_plot_style(overrides=None, use_tex="auto"):
    mpl.rcParams.update(paper_plot_rc(overrides, use_tex=use_tex))


def paper_plot_context(overrides=None, use_tex=False):
    return mpl.rc_context(paper_plot_rc(overrides, use_tex=use_tex))
