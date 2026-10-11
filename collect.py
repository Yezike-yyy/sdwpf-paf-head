"""Summarize a grid: mean +- std over seeds of test RMSE / MAE (MW), R2, skill vs persistence.

  python collect.py --grid main             -> reports/main.md and reports/main.csv
  python collect.py --grid ablation         (backbone and persistence references are taken from --ref main)
  python collect.py --grid lrscan/lr1e-03   (one learning rate of the scan; grid.py --show-lr gives the choice)
  python collect.py --grid main --farm kelmarsh   -> reports/kelmarsh_main.md (plan revision 3 farms)
Skill = 1 - RMSE_model / RMSE_persistence (same horizon, same test windows). Head rows (model+head) also show
the RMSE change vs the same backbone without head ("vs bb", positive = better), computed per seed and averaged.
Validation RMSE is the only number allowed for choosing the backbone of the proposed model. "ep" = mean best
epoch; "cap" = runs whose best epoch was the last allowed one (possibly truncated training).
"""
import argparse
import csv
import glob
import json
import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wpf import farms  # noqa: E402
BASE_ORDER = ["Persistence", "LSTM", "GRU", "TCN", "Transformer", "DLinear", "PatchTST", "iTransformer", "TimeMixer"]
HEAD_ORDER = ["none", "paf", "ens2", "aux", "paf_noavail", "paf_noaux", "phys_only"]


def load(grid, farm="sdwpf"):
    R = {}
    for p in glob.glob(os.path.join(farms.results_dir(farm), grid, "*", "*", "*", "s*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        R.setdefault((r["features"], int(r["pred_len"]), r.get("label", r["model"])), []).append(r)
    return R


def order_key(label):
    m, _, h = label.partition("+")
    return (BASE_ORDER.index(m) if m in BASE_ORDER else 99, HEAD_ORDER.index(h or "none"))


def ms(v, fmt="%.2f"):
    if len(v) == 1:
        return fmt % v[0]
    return (fmt + " ± " + fmt) % (np.mean(v), np.std(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--ref", default="main", help="grid holding persistence / backbone results if absent here")
    ap.add_argument("--farm", default="sdwpf", choices=sorted(farms.FARMS))
    a = ap.parse_args()
    R = load(a.grid, a.farm)
    REF = load(a.ref, a.farm) if a.ref != a.grid else {}
    feats = sorted({k[0] for k in R}, key=lambda f: f != "farm+cluster")
    Hs = sorted({k[1] for k in R})
    with open(glob.glob(os.path.join(farms.data_dir(a.farm), "*_meta.json"))[0], encoding="utf-8") as f:
        cap = json.load(f)["capacity_mw"]
    lines = ["# %s: farm-level %s, 15 min, look-back 96 (24 h)" % (a.grid, farms.display_name(a.farm)), "",
             "Test metrics in MW (capacity %g MW), mean ± std over seeds; Persistence is deterministic." % cap, ""]
    rows = []
    for feat in feats:
        for H in Hs:
            pers = R.get((feat, H, "Persistence")) or REF.get((feat, H, "Persistence"))
            prmse = pers[0]["result"]["test"]["rmse"] if pers else None
            labels = sorted({k[2] for k in R if k[0] == feat and k[1] == H}, key=order_key)
            lines += ["## inputs = %s, H = %d (%d min)" % (feat, H, 15 * H), "",
                      "| model | seeds | test RMSE | test MAE | test R² | skill % | vs bb % | val RMSE |"
                      " lag_1 Δ% test/val | ep | cap | min/run |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            trained = [m for m in labels if m != "Persistence"]
            best_val = min((np.mean([r["result"]["val"]["rmse"] for r in R[(feat, H, m)]]), m)
                           for m in trained)[1] if trained else None
            for m in labels:
                rs = R[(feat, H, m)]
                te = [r["result"]["test"] for r in rs]
                rmse = [t["rmse"] for t in te]
                vr = [r["result"]["val"]["rmse"] for r in rs]
                skill = [100 * (1 - x / prmse) for x in rmse] if prmse else [float("nan")]
                vs = "-"
                if "+" in m:
                    bb = m.split("+")[0]
                    ref = {r["seed"]: r["result"]["test"]["rmse"]
                           for r in (R.get((feat, H, bb)) or REF.get((feat, H, bb)) or [])}
                    g = [100 * (1 - r["result"]["test"]["rmse"] / ref[r["seed"]]) for r in rs if r["seed"] in ref]
                    vs = ms(g, "%+.1f") if g else "-"

                def lag(k, split):
                    g = [100 * (r["result"][split]["rmse"] - r["result"][split + "_lag"][k]["rmse"])
                         / r["result"][split]["rmse"] for r in rs if r["result"].get(split + "_lag")]
                    return "%+.2f" % np.mean(g) if g else "-"

                eps = [r["result"].get("best_epoch") for r in rs if "best_epoch" in r["result"]]
                cap = sum(1 for r in rs if "best_epoch" in r["result"]
                          and r["result"]["best_epoch"] == r["hparams"].get("train_epochs"))
                mins = np.mean([r["result"].get("wall_sec", 0) for r in rs]) / 60
                tag = " (best val)" if m == best_val else ""
                lines.append("| %s%s | %d | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %.1f |" % (
                    m, tag, len(rs), ms(rmse), ms([t["mae"] for t in te]), ms([t["r2"] for t in te], "%.4f"),
                    ms(skill, "%.1f"), vs, ms(vr), lag("1", "test"), lag("1", "val"),
                    "%.1f" % np.mean(eps) if eps else "-", cap if eps else "-", mins))
                rows.append(dict(features=feat, H=H, model=m, seeds=len(rs), test_rmse=np.mean(rmse),
                                 test_rmse_std=np.std(rmse), test_mae=np.mean([t["mae"] for t in te]),
                                 test_r2=np.mean([t["r2"] for t in te]), skill_pct=np.mean(skill),
                                 val_rmse=np.mean(vr), vs_backbone_pct=vs))
            lines.append("")
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    name = farms.report_name(a.farm, a.grid.replace("/", "_"))
    md = os.path.join(HERE, "reports", name + ".md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if rows:
        with open(os.path.join(HERE, "reports", name + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("\n".join(lines))
    print("wrote", md)


if __name__ == "__main__":
    main()
