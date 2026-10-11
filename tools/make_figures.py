"""M5 validation figures from the REAL results (results/main, dataset/), style A (dot + interval) chosen by the
author on 2026-10-10; style C (annotated heat map) for the supplementary overview.

  fig1_main_gain     RMSE reduction of backbone+PAF and backbone+ens2 vs the same backbone, 7 backbones x 3 horizons,
                     plus a separate TimeMixer strip (unstable backbone, own x scale).
  fig2_per_step      per-lead-step test RMSE: persistence, strongest baseline, proposed model, its +ens2 control;
                     bottom row = RMSE change of proposed / ens2 vs the strongest baseline per step.
  fig3_gate          learned per-step gate weight of the direct path, PAF vs ens2, H = 16 (descriptive).
  fig4_data          (a) turbine layout by cluster, (b) farm power vs farm wind speed coloured by Unav with the
                     training-fitted curve C*g(V), (c) daily mean farm power over two years with split and gaps.
  figS1_heatmap      style C overview of the fig1 quantity, all 8 backbones.

Gain quantity = the one of tools/m3_tests.py: 100 (1 - RMSE_A / RMSE_B), RMSE pooled over the 3 seeds and all
test windows (analyze.window_losses), 95% CI = circular block bootstrap, 7-day blocks, 2000 reps, seed 0. The
CI is descriptive; only the PAF-vs-backbone and PAF-vs-ens2 families were tested (reports/m3_tests.md).

  python tools/make_figures.py   -> figures/*.pdf, figures/*.png, figures/source_data/*.csv
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze as A  # noqa: E402
from wpf.data import load_table  # noqa: E402

OUT = os.path.join(HERE, "figures")
SRC = os.path.join(OUT, "source_data")
GRID, FEAT = "main", "farm+cluster"
BB7 = ["LSTM", "GRU", "TCN", "Transformer", "DLinear", "PatchTST", "iTransformer"]
BB8 = BB7 + ["TimeMixer"]
HS = [4, 8, 16]
HLAB = {4: "H = 4 (1 h)", 8: "H = 8 (2 h)", 16: "H = 16 (4 h)"}
C_PAF, C_ENS, C_BB, C_PER = "#0072B2", "#E69F00", "#555555", "#999999"  # Okabe-Ito + greys
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
HEAD_LAB = {"paf": "+PAF (physics second path)", "ens2": "+ens2 (learned second path)"}
MM = 1 / 25.4
W2 = 174 * MM  # full width for large-sized Springer journals (Electrical Engineering guidelines, checked 2026-10-11)


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],  # journal asks for Helvetica/Arial lettering
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def write_csv(name, rows):
    with open(os.path.join(SRC, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def spec(label):
    return "%s/%s" % (FEAT, label)


# ------------------------------------------------------------------ fig 1 / S1: gains vs backbone

def gain_rows():
    rows = []
    for H in HS:
        for m in BB8:
            data = A.aligned(GRID, [spec(m), spec(m + "+paf"), spec(m + "+ens2")], H)
            y = data[spec(m)][1]
            se_b, _ = A.window_losses(data[spec(m)][0], y)
            for head in ("paf", "ens2"):
                se_h, _ = A.window_losses(data[spec(m + "+" + head)][0], y)
                p, lo, hi = A.block_bootstrap_rmse_gain(se_h, se_b, reps=2000)
                rows.append(dict(H=H, backbone=m, head=head, rmse_backbone=float(np.sqrt(se_b.mean())),
                                 rmse_head=float(np.sqrt(se_h.mean())), gain_pct=p, ci_lo=lo, ci_hi=hi,
                                 n_windows=len(y), n_seeds=data[spec(m)][0].shape[0]))
    return rows


MARK = {"paf": "o", "ens2": "D"}  # distinct markers: the figure must survive grayscale printing


def fig1(rows):
    fig = plt.figure(figsize=(W2, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[len(BB7), 1.4], hspace=0.62, wspace=0.08)
    main7 = [v for v in rows if v["backbone"] in BB7]
    xlo = np.floor(min(v["ci_lo"] for v in main7)) - 0.5
    xhi = np.ceil(max(v["ci_hi"] for v in main7)) + 0.5
    for j, H in enumerate(HS):
        ax = fig.add_subplot(gs[0, j])
        tm = fig.add_subplot(gs[1, j])
        for a, names in ((ax, BB7), (tm, ["TimeMixer"])):
            yy = np.arange(len(names))
            for head, c, dy in (("paf", C_PAF, -0.17), ("ens2", C_ENS, 0.17)):
                r = [next(v for v in rows if v["H"] == H and v["backbone"] == m and v["head"] == head) for m in names]
                g = np.array([v["gain_pct"] for v in r])
                err = np.array([[v["gain_pct"] - v["ci_lo"] for v in r], [v["ci_hi"] - v["gain_pct"] for v in r]])
                a.errorbar(g, yy + dy, xerr=err, fmt=MARK[head], ms=3.2, color=c, elinewidth=0.8, capsize=0,
                           label=HEAD_LAB[head])
            a.axvline(0, color="0.5", lw=0.6, ls="--", zorder=0)
            a.set_yticks(yy)
            a.set_yticklabels(names if j == 0 else [])
            a.set_ylim(len(names) - 0.5, -0.5)
            a.tick_params(axis="y", length=0)
            a.grid(axis="x", color="0.92", lw=0.5)
            a.set_axisbelow(True)
        ax.set_xlim(xlo, xhi)
        ax.set_title(HLAB[H])
        tm.set_title("TimeMixer, own x scale", fontsize=7, color="0.3")
        if H == 4:  # the ens2 point is the seed-pooled RMSE with one diverged seed (52.9 MW vs 15.6 / 16.4 MW)
            v = next(v for v in rows if v["H"] == H and v["backbone"] == "TimeMixer" and v["head"] == "ens2")
            tm.annotate("ens2: 1 of 3 seeds diverged", (v["ci_hi"], 0.17), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=6, color=C_ENS)
        if H == 16:  # here the BACKBONE has one diverged seed (63.7 MW vs 30.5 / 29.5 MW), so both "gains" are large
            tm.text(0.42, 0.5, "backbone: 1 of 3 seeds diverged", transform=tm.transAxes, ha="center",
                    va="center", fontsize=6, color="0.3",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
    fig.text(0.5, 0.0, "RMSE reduction vs the same backbone without head (%)", ha="center")
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "fig1_main_gain")


def figS1(rows):
    lim = 4.0  # covers every non-TimeMixer cell; cells beyond it are marked with a dagger
    fig, axs = plt.subplots(1, 2, figsize=(W2 * 0.8, 3.0), sharey=True)
    for ax, head in zip(axs, ("paf", "ens2")):
        M = np.array([[next(v["gain_pct"] for v in rows if v["H"] == H and v["backbone"] == m and v["head"] == head)
                       for H in HS] for m in BB8])
        im = ax.imshow(np.clip(M, -lim, lim), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(len(BB8)):
            for j in range(len(HS)):
                out = abs(M[i, j]) > lim
                ax.text(j, i, "%+.1f%s" % (M[i, j], "†" if out else ""), ha="center", va="center", fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 * lim else "black")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["1 h", "2 h", "4 h"])
        ax.set_yticks(range(len(BB8)))
        ax.set_yticklabels(BB8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(HEAD_LAB[head])
    cb = fig.colorbar(im, ax=axs, shrink=0.85, extend="both")
    cb.set_label("RMSE reduction vs backbone (%)")
    cb.outline.set_linewidth(0.5)
    save(fig, "figS1_heatmap")


# ------------------------------------------------------------------ fig 2: per-step RMSE

def step_rmse(pred, true):
    return np.sqrt(((pred - true[None]) ** 2).mean(axis=(0, 1)))


def fig2():
    prim = {r["H"]: r for r in json.load(open(os.path.join(HERE, "reports", "analysis", "m3_tests.json"),
                                              encoding="utf-8"))["primary"]}
    fig, axs = plt.subplots(2, 3, figsize=(W2, 3.6), sharex="col", gridspec_kw=dict(height_ratios=[1.5, 1],
                                                                                    hspace=0.12, wspace=0.3))
    rows = []
    for j, H in enumerate(HS):
        prop, base = prim[H]["a"], prim[H]["b"]
        ens = prop.replace("+paf", "+ens2")
        specs = [spec("Persistence"), base, prop, ens]
        data = A.aligned(GRID, specs, H)
        y = data[base][1]
        curves = {s: step_rmse(data[s][0], y) for s in specs}
        x = np.arange(1, H + 1) * 15
        role_lab = {spec("Persistence"): "Persistence", base: "Strongest baseline (no head)",
                    prop: "Proposed: backbone + PAF", ens: "Same backbone + ens2"}
        sty = {spec("Persistence"): dict(color=C_PER, ls=":", marker=None), base: dict(color=C_BB, ls="-", marker="s"),
               prop: dict(color=C_PAF, ls="-", marker="o"), ens: dict(color=C_ENS, ls="--", marker="^")}
        ax = axs[0, j]
        for s in specs:
            ax.plot(x, curves[s], lw=1.0, ms=2.5, label=role_lab[s], **sty[s])
        ax.set_title("%s\nbaseline %s, proposed %s + PAF" % (HLAB[H], base.split("/")[1], prop.split("/")[1][:-4]),
                     fontsize=7)
        bx = axs[1, j]
        for s in (prop, ens):
            bx.plot(x, 100 * (1 - curves[s] / curves[base]), lw=1.0, ms=2.5, **sty[s])
        bx.axhline(0, color="0.5", lw=0.6, ls="--")
        bx.set_xlabel("Lead time (min)")
        bx.set_xticks({4: [15, 30, 45, 60], 8: [15, 30, 60, 90, 120], 16: [15, 60, 120, 180, 240]}[H])
        for s in specs:
            for k in range(H):
                rows.append(dict(H=H, model=s.split("/")[1], role={spec("Persistence"): "persistence", base: "baseline",
                                                                   prop: "proposed", ens: "ens2_control"}[s],
                                 step=k + 1, lead_min=int(x[k]), rmse_mw=float(curves[s][k]),
                                 change_vs_baseline_pct=float(100 * (1 - curves[s][k] / curves[base][k]))))
    axs[0, 0].set_ylabel("Test RMSE (MW)")
    axs[1, 0].set_ylabel("RMSE reduction vs\nstrongest baseline (%)")
    fig.align_ylabels(axs[:, 0])
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06), handlelength=2.4)
    write_csv("fig2_per_step", rows)
    save(fig, "fig2_per_step")


# ------------------------------------------------------------------ fig 3: gate weights

def gate(label, H):
    g = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", GRID, FEAT, label, str(H), "s*.json"))):
        g.append(json.load(open(f, encoding="utf-8"))["result"]["gate_direct_weight"])
    return np.array(g)


def fig3(H=16):
    fig, axs = plt.subplots(2, 4, figsize=(W2, 3.0), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.35,
                                                                                               wspace=0.12))
    x = np.arange(1, H + 1) * 15
    rows = []
    for ax, m in zip(axs.ravel(), BB8):
        for head, c in (("paf", C_PAF), ("ens2", C_ENS)):
            G = gate("%s+%s" % (m, head), H)
            mu, sd = G.mean(0), G.std(0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.2, lw=0)
            ax.plot(x, mu, color=c, lw=1.1, ls="-" if head == "paf" else "--", label=HEAD_LAB[head])
            for k in range(H):
                rows.append(dict(H=H, backbone=m, head=head, step=k + 1, lead_min=int(x[k]), n_seeds=len(G),
                                 direct_weight_mean=float(mu[k]), direct_weight_sd=float(sd[k])))
        ax.axhline(0.5, color="0.7", lw=0.5, ls=":")
        ax.set_title(m)
        ax.set_ylim(0, 1)
        ax.set_xticks([60, 120, 180, 240])
    for ax in axs[1]:
        ax.set_xlabel("Lead time (min)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Weight of direct path")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    write_csv("fig3_gate_H%d" % H, rows)
    save(fig, "fig3_gate")


# ------------------------------------------------------------------ fig 4: data overview

def fig4():
    t = load_table(os.path.join(HERE, "dataset"), "farm+cluster")
    meta = t["meta"]
    df = pd.read_csv(os.path.join(HERE, "dataset", "sdwpf_15min.csv"), usecols=["date", "Wspd", "Unav", "OT"])
    df["date"] = pd.to_datetime(df["date"])
    loc = pd.read_csv(os.path.join(HERE, "..", "data_raw", "sdwpf", "sdwpf_turb_location_elevation.csv"))
    cl = {tid: int(k) for k, v in meta["clusters"].items() for tid in v}
    pc = json.load(open(glob.glob(os.path.join(HERE, "results", GRID, FEAT, "TCN+paf", "16", "s2021.json"))[0],
                        encoding="utf-8"))["hparams"]["power_curve"]
    cap = meta["capacity_mw"]

    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[0.6, 1.4], hspace=0.38, wspace=0.22)
    # (a) layout
    ax = fig.add_subplot(gs[0, 0])
    for k in sorted(meta["clusters"], key=int):
        s = loc[loc["TurbID"].map(cl) == int(k)]
        ax.scatter(s["x"] / 1000, s["y"] / 1000, s=9, color=OKABE[int(k)], label="C%s" % k, edgecolor="none")
        cx, cy = meta["cluster_centers_xy"][k]
        ax.text(cx / 1000, cy / 1000, "C%s" % k, fontsize=7, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("a", loc="left", weight="bold")
    # (b) power vs wind speed
    bx = fig.add_subplot(gs[0, 1])
    v = df.dropna(subset=["Wspd", "Unav", "OT"])
    sc = bx.scatter(v["Wspd"], v["OT"], c=v["Unav"], s=0.6, cmap="viridis", vmin=0, vmax=0.5, rasterized=True,
                    edgecolor="none")
    kn, gv = np.array(pc["knots"]), np.array(pc["values"])
    bx.plot(kn, cap * gv, color="#D55E00", lw=1.3, label="C·g(V)\n(training-bin fit)")
    bx.set_xlabel("Farm mean wind speed (m s$^{-1}$)")
    bx.set_ylabel("Farm power (MW)")
    bx.set_xlim(0, 16)
    bx.set_ylim(-5, cap)
    bx.legend(frameon=False, loc="upper left")
    cb = fig.colorbar(sc, ax=bx, extend="max", pad=0.02)
    cb.set_label("Unav (share of turbines)")
    cb.outline.set_linewidth(0.5)
    bx.set_title("b", loc="left", weight="bold")
    # (c) two-year series, daily mean, split and gaps
    cx = fig.add_subplot(gs[1, :])
    d = df.set_index("date")["OT"].resample("D").mean()
    cx.plot(d.index, d.values, color="0.25", lw=0.6)
    dates = t["dates"]
    b1, b2 = t["b1"], t["b2"]
    for k, (name, c) in enumerate((("train", "#FFFFFF"), ("validation", "#E8F1F8"), ("test", "#FBEBD8"))):
        cx.axvspan(dates[b1[k]], dates[b2[k] - 1], color=c, zorder=0, lw=0)
        cx.text(dates[(b1[k] + b2[k]) // 2], cap * 0.93, name, ha="center", fontsize=7, color="0.3")
    bad = df["OT"].isna().values
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    runs = list(zip(edges[::2], edges[1::2]))
    for a0, a1 in runs:
        cx.axvspan(df["date"].iloc[a0], df["date"].iloc[a1 - 1], ymin=0, ymax=0.06, color="#D55E00", lw=0.8)
    cx.set_ylim(0, cap)
    cx.set_xlim(dates[0], dates[-1])
    cx.set_ylabel("Daily mean power (MW)")
    cx.set_title("c", loc="left", weight="bold")  # caption: orange marks = invalid bins (runs / bins printed below)
    write_csv("fig4_daily_power", [dict(date=str(i.date()), daily_mean_mw=float(x)) for i, x in d.items()])
    write_csv("fig4_invalid_runs", [dict(start=str(df["date"].iloc[a0]), end=str(df["date"].iloc[a1 - 1]),
                                         n_bins=int(a1 - a0)) for a0, a1 in runs])
    save(fig, "fig4_data")
    return dict(n_invalid=int(bad.sum()), n_runs=len(runs), n_scatter=len(v))


def main():
    style()
    os.makedirs(SRC, exist_ok=True)
    rows = gain_rows()
    write_csv("fig1_gain_vs_backbone", rows)
    fig1(rows)
    figS1(rows)
    fig2()
    fig3()
    print(fig4())


if __name__ == "__main__":
    main()
