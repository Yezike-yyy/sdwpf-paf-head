"""Placebo calibration of the revision-3 group tests on SDWPF (discovery data; no confirmatory farm involved).

Question: is the registered group t-test (statistic computed within each of q = 8 time groups) or the candidate
linearized version (group values = full-sample statistic + group mean of the delta-method influence values) close to
its nominal 5 % size on data like ours? The SDWPF dry run gave very different H2 p-values for the two at H = 4 and 8.

Randomization null: on randomly chosen units of the forecast-origin time (calendar days, or 7-day blocks) the PAF and
ens2 predictions (H2), or the PAF and backbone predictions (H1), are swapped for all backbones at once. theta and G
are then 0 in expectation while the real heterogeneity across groups (flat-high content) and the dependence of the
errors are kept. Day swaps keep within-day dependence only; 7-day swaps impose week-long dependence, a harsh case
(22 units over 8 groups).

  python tools/r3_placebo.py [--reps 1000]   -> reports/r3_placebo_sdwpf.md (+ reports/analysis/r3_placebo_sdwpf.json)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import analyze as A  # noqa: E402
import r3_tests as R  # noqa: E402

FARM = "sdwpf"
ALPHA = 0.05
UNITS = [("day", 1), ("7-day", 7)]


def load(H, region, ws):
    keep = {b: R.kept_seeds(FARM, b, H)[0] for b in R.BB7}
    se, st = {}, None
    for b in R.BB7:
        for suf in ("", "+paf", "+ens2"):
            se[b + suf], _, st_m = R.seed_se(FARM, "%s/%s%s" % (R.FC, b, suf), H, keep[b])
            if st is not None and not np.array_equal(st, st_m):
                raise ValueError("windows differ (%s H=%d)" % (b + suf, H))
            st = st_m
    L = A.seq_len_of(R.GRID, "%s/%s" % (R.FC, R.BB7[0]), H, FARM)
    reg_t = region(ws[st[:, None] + L + np.arange(H)[None]])
    day = (st + L - 1) // 96  # the SDWPF table starts at 00:00, 96 bins per day

    def stack(suf):
        return np.stack([se[b + suf] for b in R.BB7])

    return stack(""), stack("+paf"), stack("+ens2"), reg_t, day


def h2_tests(S, ok):
    theta = R.h2_parts(S, np.arange(S[0].shape[1]))[0]
    p_group = R.t_two_sided([R.h2_parts(S, g)[0] for g in ok])[1]
    a_fh, b_fh, a_st, b_st = S
    psi = (R.influence(a_fh, b_fh) - R.influence(a_st, b_st)).mean(axis=0)
    p_lin = R.t_two_sided([theta + psi[g].mean() for g in ok])[1]
    return theta, p_group, p_lin


def h1_tests(Aw, Bw, groups):
    G = R.gain(Aw.sum(1), Bw.sum(1))
    p_group = R.t_two_sided([R.gain(Aw[:, g].sum(1), Bw[:, g].sum(1)) for g in groups])[1]
    psi = R.influence(Aw, Bw).mean(axis=0)
    p_lin = R.t_two_sided([G + psi[g].mean() for g in groups])[1]
    return G, p_group, p_lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    region, ws, _ = R.region_fn(FARM)
    rng = np.random.RandomState(1)
    mc = 196.0 * np.sqrt(ALPHA * (1 - ALPHA) / a.reps)
    res = dict(reps=a.reps, alpha=ALPHA, mc_half_width_pct=float(mc), cells={})
    for H in R.HS:
        t0 = time.time()
        bb, paf, ens, reg_t, day = load(H, region, ws)
        m_fh, m_st = reg_t == 2, reg_t == 1
        S = R.h2_sums(paf, ens, m_fh, m_st)
        groups = np.array_split(np.arange(paf.shape[1]), R.Q)
        counts = [int(m_fh[g].sum()) for g in groups]
        ok = [g for g, c in zip(groups, counts) if c >= R.MIN_FH_STEPS]
        Aw, Bw = paf.sum(axis=2), bb.sum(axis=2)
        th, pg, pl = h2_tests(S, ok)
        G, pg1, pl1 = h1_tests(Aw, Bw, groups)
        cell = dict(n_windows=int(paf.shape[1]), flat_high_steps_per_group=counts,
                    observed=dict(theta=th, h2_p_group=pg, h2_p_lin=pl, G=G, h1_p_group=pg1, h1_p_lin=pl1), placebo={})
        for name, days in UNITS:
            uid = np.unique(day // days, return_inverse=True)[1]
            nu = int(uid.max()) + 1
            rej = np.zeros(4)
            null_theta, null_G = [], []
            for _ in range(a.reps):
                sw = (rng.rand(nu) < 0.5)[uid][None, :]
                S2 = [np.where(sw, S[1], S[0]), np.where(sw, S[0], S[1]),
                      np.where(sw, S[3], S[2]), np.where(sw, S[2], S[3])]
                t2, p2g, p2l = h2_tests(S2, ok)
                g1, p1g, p1l = h1_tests(np.where(sw, Bw, Aw), np.where(sw, Aw, Bw), groups)
                rej += [p2g < ALPHA, p2l < ALPHA, p1g < ALPHA, p1l < ALPHA]
                null_theta.append(t2)
                null_G.append(g1)
            size = 100.0 * rej / a.reps
            cell["placebo"][name] = dict(units=nu, size_pct=dict(h2_group=size[0], h2_lin=size[1],
                                                                  h1_group=size[2], h1_lin=size[3]),
                                         sd_theta=float(np.std(null_theta)), sd_G=float(np.std(null_G)))
        res["cells"][str(H)] = cell
        print("H=%d done in %.0fs" % (H, time.time() - t0), flush=True)

    os.makedirs(os.path.join(HERE, "reports", "analysis"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "analysis", "r3_placebo_sdwpf.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    L = ["# Placebo calibration of the revision-3 group tests (SDWPF, discovery data)", "",
         "Generated by `tools/r3_placebo.py`. The PAF and ens2 predictions (H2), or the PAF and backbone predictions "
         "(H1), are swapped for all 7 stable backbones on random units of the forecast-origin time, so the true "
         "statistic is 0 in expectation while the heterogeneity and dependence of the real errors are kept. Size = "
         "share of placebo replicates with p < %.2f (two-sided). %d replicates per cell; Monte Carlo 95 %% half-width "
         "at 5 %%: %.1f points. 'group' = registered test (statistic within each time group); 'lin' = candidate "
         "amendment (full-sample statistic + group mean of the delta-method influence values)." % (ALPHA, a.reps, mc),
         "", "| H | swap unit | units | H2 group | H2 lin | H1 group | H1 lin | placebo sd theta | observed theta "
         "| placebo sd G | observed G |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for H in R.HS:
        c = res["cells"][str(H)]
        for name, _ in UNITS:
            p = c["placebo"][name]
            s = p["size_pct"]
            L.append("| %d | %s | %d | %.1f %% | %.1f %% | %.1f %% | %.1f %% | %.2f | %+.2f | %.2f | %+.2f |" % (
                H, name, p["units"], s["h2_group"], s["h2_lin"], s["h1_group"], s["h1_lin"], p["sd_theta"],
                c["observed"]["theta"], p["sd_G"], c["observed"]["G"]))
    L += ["", "Observed p-values (unswapped):", "",
          "| H | flat-high steps per group | H2 p group | H2 p lin | H1 p group | H1 p lin |", "|---|---|---|---|---|---|"]
    for H in R.HS:
        o = res["cells"][str(H)]["observed"]
        L.append("| %d | %s | %.4f | %.4f | %.4f | %.4f |" % (
            H, res["cells"][str(H)]["flat_high_steps_per_group"], o["h2_p_group"], o["h2_p_lin"], o["h1_p_group"],
            o["h1_p_lin"]))
    L.append("")
    with open(os.path.join(HERE, "reports", "r3_placebo_sdwpf.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
