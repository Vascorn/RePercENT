import matplotlib as mpl


PAPER_PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def paper_plot_rc(overrides=None):
    rc = PAPER_PLOT_RC.copy()
    if overrides:
        rc.update(overrides)
    return rc


def apply_paper_plot_style(overrides=None):
    mpl.rcParams.update(paper_plot_rc(overrides))


def paper_plot_context(overrides=None):
    return mpl.rc_context(paper_plot_rc(overrides))
