"""Method schematic of the PAF output head (Fig. 1 of the manuscript).

  python tools/make_method_figure.py   -> figures/figM_method.pdf / .png

Pure drawing; the inset power curve is the real g(v) fitted on the training segment (wpf/physics.py), taken from the
result file of the TCN+PAF H=16 run (as in fig4). Style as tools/make_figures.py (174 mm, Arial, Okabe-Ito).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import make_figures as MF  # noqa: E402
from wpf import physics  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "0.35"


def box(ax, x, y, w, h, text, fc="white", ec="0.2", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color="0.2", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=color, linestyle=ls,
                                 connectionstyle="arc3,rad=%g" % rad, shrinkA=0, shrinkB=0))


def main():
    MF.style()
    fig = plt.figure(figsize=(MF.W2, 62 * MF.MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.36)
    ax.axis("off")

    box(ax, 0.015, 0.12, 0.125, 0.12, "Look-back window\n$X_t$ (L = 96, C = 22)\npower, farm and\ncluster SCADA")
    box(ax, 0.17, 0.12, 0.12, 0.12, "Backbone $f_\\theta$\n(unchanged)\nLSTM, GRU, TCN,\nTransformer, ...", fc="0.93")
    arrow(ax, (0.14, 0.18), (0.17, 0.18))
    box(ax, 0.325, 0.12, 0.10, 0.12, "Output\n$\\hat{Y}\\in\\mathbb{R}^{H\\times C}$\n(all channels)")
    arrow(ax, (0.29, 0.18), (0.325, 0.18))

    # direct path (top) and driver forecasts (bottom)
    box(ax, 0.47, 0.265, 0.13, 0.065, "direct power $\\hat{y}^{dir}_h$", ec=BLUE)
    box(ax, 0.47, 0.03, 0.13, 0.085, "wind speed $\\hat{v}_h$\nunavailable share $\\hat{s}_h$", ec=GREEN)
    arrow(ax, (0.425, 0.22), (0.47, 0.297), rad=-0.2)
    arrow(ax, (0.425, 0.14), (0.47, 0.072), rad=0.2)
    ax.text(0.535, 0.005, "auxiliary loss vs observed $v$, $s$ (Eq. 3)", ha="center", fontsize=6, color=GREEN)

    # physics path with real curve inset
    box(ax, 0.64, 0.02, 0.16, 0.20, "", ec=ORANGE)
    ax.text(0.72, 0.203, "physics path (Eq. 1)", ha="center", va="center", fontsize=7)
    ax.text(0.72, 0.04, "$C_{inst}(1-\\hat{s}_h)\\,g(\\hat{v}_h)$", ha="center", va="center", fontsize=7)
    arrow(ax, (0.60, 0.072), (0.64, 0.072))
    # the curve actually used by the runs (stored with the TCN+PAF, H = 16 result, seed 2021)
    pc = json.load(open(glob.glob(os.path.join(MF.HERE, "results", MF.GRID, MF.FEAT, "TCN+paf", "16", "s2021.json"))[0],
                        encoding="utf-8"))["hparams"]["power_curve"]
    curve = physics.FarmPowerCurve(pc["knots"], pc["values"])
    ins = fig.add_axes([0.69, 0.27, 0.085, 0.2])
    v = np.linspace(0, 20, 200)
    ins.plot(v, curve(v), color=ORANGE, lw=1.0)
    ins.set_xlim(0, 20)
    ins.set_ylim(0, 1.0)
    ins.set_xticks([0, 10, 20])
    ins.set_yticks([0, 0.5, 1])
    ins.tick_params(labelsize=5, length=1.5, pad=1)
    ins.set_xlabel("$v$ (m/s)", fontsize=5, labelpad=0)
    ins.set_ylabel("$g(v)$", fontsize=5, labelpad=0)
    for s in ins.spines.values():
        s.set_linewidth(0.4)

    # gate and output
    box(ax, 0.835, 0.14, 0.075, 0.09, "gate\n$\\sigma(\\lambda_h)$", fc="0.93")
    arrow(ax, (0.60, 0.297), (0.835, 0.21), rad=-0.15, color=BLUE)
    arrow(ax, (0.80, 0.12), (0.835, 0.16), rad=0.15, color=ORANGE)
    box(ax, 0.925, 0.14, 0.065, 0.09, "fused\n$\\hat{y}_h$ (Eq. 2)", bold=False)
    arrow(ax, (0.91, 0.185), (0.925, 0.185))
    ax.text(0.872, 0.255, "one weight\nper lead time", ha="center", fontsize=6, color=GREY)
    MF.save(fig, "figM_method")


if __name__ == "__main__":
    main()
