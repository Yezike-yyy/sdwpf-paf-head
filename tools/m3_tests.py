"""Pre-registered M3 tests (统计分析计划.md, revision 1) on results/main.

Primary family (Holm, 3 tests): proposed = backbone+paf with the lowest validation RMSE at each horizon vs the
strongest baseline = head-free model (either input set) with the lowest validation RMSE at that horizon.
Secondary family (Holm, 24 tests): backbone+paf vs backbone, farm+cluster inputs, 8 backbones x 3 horizons.
Test: Ibragimov-Mueller group t-test (q = 8) on seed-averaged per-window MSE; daily DM and window DM as
sensitivity; 95% CI of the RMSE reduction from a circular block bootstrap (7-day blocks, 2000 reps).
Diagnostic: ACF of the VALIDATION loss differential (decay time in hours).
Third family (plan revision 2, Holm, 24 tests; only when the `control` grid results exist): backbone+paf vs
backbone+ens2 (learned linear second head instead of the physics path).

  python tools/m3_tests.py            -> reports/m3_tests.md and reports/analysis/m3_tests.json
"""
import glob
import json
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np  # noqa: E402

import analyze as A  # noqa: E402

GRID = "main"
HS = [4, 8, 16]
BACKBONES = ["LSTM", "GRU", "TCN", "Transformer", "DLinear", "PatchTST", "iTransformer", "TimeMixer"]


def val_rmse(spec, H):
    feat, label = spec.split("/")
    fs = glob.glob(os.path.join(HERE, "results", GRID, feat, label, str(H), "s*.json"))
    return np.mean([json.load(open(f, encoding="utf-8"))["result"]["val"]["rmse"] for f in fs])


def acf_decay_hours(d, thr=0.1, max_lag=2000):
    d = d - d.mean()
    v = np.dot(d, d)
    for k in range(1, max_lag):
        if np.dot(d[k:], d[:-k]) / v < thr:
            return k * 0.25
    return float("inf")


def compare(a, b, H, reps=2000):
    data = A.aligned(GRID, [a, b], H)
    (pa, y, st, _), (pb, _, _, _) = data[a], data[b]
    se_a, ae_a = A.window_losses(pa, y)
    se_b, ae_b = A.window_losses(pb, y)
    day = A.origin_day(st, A.seq_len_of(GRID, a, H))
    point, lo, hi = A.block_bootstrap_rmse_gain(se_a, se_b, reps=reps)
    im = A.group_t_test(se_a, se_b)
    # validation diagnostic
    va = A.aligned(GRID, [a, b], H, split="val")
    sv_a, _ = A.window_losses(va[a][0], va[a][1])
    sv_b, _ = A.window_losses(va[b][0], va[b][1])
    return dict(a=a, b=b, H=H, rmse_a=float(np.sqrt(se_a.mean())), rmse_b=float(np.sqrt(se_b.mean())),
                gain_pct=point, ci=[lo, hi], mae_gain_pct=float(100 * (1 - ae_a.mean() / ae_b.mean())),
                p_im=im["p"], groups_a_better=im["groups_a_better"],
                p_daily_dm=A.daily_dm(se_a, se_b, day)["p"],
                p_window_dm=A.dm_test(se_a, se_b, A.default_lag(len(y), H), H)["p"],
                p_im_mae=A.group_t_test(ae_a, ae_b)["p"],
                val_acf_decay_h=acf_decay_hours(sv_a - sv_b))


def main():
    primary, secondary, control = [], [], []
    for H in HS:
        prop = min(("farm+cluster/%s+paf" % m for m in BACKBONES), key=lambda s: val_rmse(s, H))
        base = min(("%s/%s" % (f, m) for f in ("farm+cluster", "target") for m in BACKBONES),
                   key=lambda s: val_rmse(s, H))
        primary.append(compare(prop, base, H))
        for m in BACKBONES:
            secondary.append(compare("farm+cluster/%s+paf" % m, "farm+cluster/%s" % m, H))
            if A.pred_files(GRID, "farm+cluster/%s+ens2" % m, H):
                control.append(compare("farm+cluster/%s+paf" % m, "farm+cluster/%s+ens2" % m, H))
    for fam in (primary, secondary, control):
        if not fam:
            continue
        adj = A.holm([r["p_im"] for r in fam])
        for r, p in zip(fam, adj):
            r["p_im_holm"] = float(p)
    lines = ["# M3 pre-registered tests (grid main, 3 seeds, test segment)", "",
             "Gain = 100 (1 - RMSE_A / RMSE_B) of seed-averaged forecasts; CI = 7-day block bootstrap; "
             "p = Ibragimov-Mueller group t (q = 8), Holm within family; daily / window DM = sensitivity; "
             "ACF = decay time (h) of the validation loss differential to 0.1.", ""]
    for name, fam in (("Primary family: proposed vs strongest baseline", primary),
                      ("Secondary family: backbone + PAF vs backbone", secondary),
                      ("Third family (revision 2): backbone + PAF vs backbone + ens2 control", control)):
        if not fam:
            continue
        lines += ["## " + name, "", "| H | A | B | RMSE A | RMSE B | gain % [95% CI] | MAE gain % | groups A better |"
                  " p (IM) | p Holm | p daily DM | p window DM | val ACF h |", "|---" * 13 + "|"]
        for r in fam:
            lines.append("| %d | %s | %s | %.2f | %.2f | %+.2f [%+.2f, %+.2f] | %+.2f | %d/8 | %.4f | %.4f | %.4f | %.4f | %.1f |" % (
                r["H"], r["a"], r["b"], r["rmse_a"], r["rmse_b"], r["gain_pct"], r["ci"][0], r["ci"][1],
                r["mae_gain_pct"], r["groups_a_better"], r["p_im"], r["p_im_holm"], r["p_daily_dm"],
                r["p_window_dm"], r["val_acf_decay_h"]))
        lines.append("")
    os.makedirs(os.path.join(HERE, "reports", "analysis"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "m3_tests.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "reports", "analysis", "m3_tests.json"), "w", encoding="utf-8") as f:
        json.dump(dict(primary=primary, secondary=secondary, control=control), f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
