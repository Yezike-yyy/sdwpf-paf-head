"""Ensemble control computed from saved predictions (no training): 2-member seed ensembles of a backbone vs one
backbone+PAF model. If a single PAF model matches or beats averaging two independently trained backbones, the
PAF gain is not just "averaging two forecasts".

  python tools/seed_ensemble.py   -> reports/seed_ensemble.md
Each number is the mean over the 3 seeds (single models) or over the 3 seed pairs (ensembles) of the test RMSE (MW).
"""
import itertools
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np  # noqa: E402

import analyze as A  # noqa: E402

BACKBONES = ["LSTM", "GRU", "TCN", "Transformer", "DLinear", "PatchTST", "iTransformer", "TimeMixer"]


def rmse(p, y):
    return float(np.sqrt(((p - y) ** 2).mean()))


def main():
    lines = ["# Seed-ensemble control (grid main, farm+cluster, test RMSE in MW)", "",
             "single = mean over 3 seeds; ens2 = mean over the 3 pairs of the RMSE of the 2-seed average forecast; "
             "Δ = 100 (1 - RMSE / RMSE_single backbone).", "",
             "| H | backbone | single | ens2 of backbone | Δ ens2 % | +PAF single | Δ PAF % | ens2 of +PAF | Δ % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for H in (4, 8, 16):
        for m in BACKBONES:
            b, p = "farm+cluster/%s" % m, "farm+cluster/%s+paf" % m
            d = A.aligned("main", [b, p], H)
            pb, y = d[b][0], d[b][1]
            pp = d[p][0]
            single = np.mean([rmse(x, y) for x in pb])
            ens = np.mean([rmse((pb[i] + pb[j]) / 2, y) for i, j in itertools.combinations(range(len(pb)), 2)])
            paf = np.mean([rmse(x, y) for x in pp])
            ensp = np.mean([rmse((pp[i] + pp[j]) / 2, y) for i, j in itertools.combinations(range(len(pp)), 2)])
            g = lambda v: 100 * (1 - v / single)  # noqa: E731
            lines.append("| %d | %s | %.2f | %.2f | %+.2f | %.2f | %+.2f | %.2f | %+.2f |" % (
                H, m, single, ens, g(ens), paf, g(paf), ensp, g(ensp)))
    with open(os.path.join(HERE, "reports", "seed_ensemble.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
