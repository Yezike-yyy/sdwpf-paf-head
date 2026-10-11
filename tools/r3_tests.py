"""Pre-registered tests of plan revision 3 (统计分析计划.md, 2026-10-10): validation on two independent farms.

Fourth family, H1 (Holm over 2 farms x 3 H = 6): the backbone-averaged RMSE reduction G of backbone + PAF vs
  backbone is > 0.
Fifth family, H2 (Holm over 6): theta > 0, theta = backbone mean of
  [gain(PAF vs ens2 | target flat-high) - gain(PAF vs ens2 | target steep)].
Backbones: the 7 stable ones (TimeMixer is reported separately). gain = 100 (1 - RMSE_A / RMSE_B), RMSE from the
seed-averaged squared error of the test window-steps (the pre-registered estimand).
Test: Ibragimov-Mueller type group t-test. The time-ordered test windows are cut into q = 8 contiguous groups, the
statistic is computed within each group, and the group values get a two-sided one-sample t-test (df = groups - 1).
H2 uses only the groups with >= 50 flat-high window-steps; with fewer than 6 such groups, or flat-high < 3 % of the
test target steps, the cell is "not testable" (descriptive only; it enters the Holm family with p = 1, so the family
size stays 6).
Divergence rule: a run (model, head, H, seed) whose validation RMSE exceeds 1.5 x the persistence validation RMSE at
the same H has diverged; that seed is dropped from every comparison of that backbone at that H (paired).
Regions: r = g(V) / max g with g = the farm's training-segment power curve (wpf.physics, exactly as run.py fits it);
flat-low r < 0.1, steep 0.1 <= r < 0.9, flat-high r >= 0.9. Primary: V = observed farm-mean wind speed at each
target step (ex post); secondary: V at the forecast origin (ex ante), descriptive only.
CIs: joint circular block bootstrap (7-day blocks = 672 windows, 2000 reps, seed 0, the same blocks for all
backbones and both regions).
A linearized variant of the group test (group values = full-sample statistic + group mean of the delta-method
influence values) was evaluated on SDWPF before any new-farm result and NOT adopted (2026-10-10, user audit): it is
liberal under week-long dependence in the placebo calibration (tools/r3_placebo.py). Its values ("p_lin") stay in
the JSON as a diagnostic only; they are not adjusted, not reported in the tables and not used for any conclusion.

  python tools/r3_tests.py --farms penmanshiel kelmarsh   -> reports/r3_tests.md (+ reports/analysis/r3_tests.json)
  python tools/r3_tests.py --farms sdwpf                  -> reports/r3_sdwpf_exploratory.md (discovery data: the
                                                             regions were defined after seeing SDWPF test results)
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np  # noqa: E402

import analyze as A  # noqa: E402
from wpf import farms  # noqa: E402
from wpf.data import load_table  # noqa: E402
from wpf.physics import fit_curve_from_table  # noqa: E402

BB7 = ["LSTM", "GRU", "TCN", "Transformer", "DLinear", "PatchTST", "iTransformer"]
ALL8 = BB7 + ["TimeMixer"]
HS = [4, 8, 16]
GRID = "main"
FC = "farm+cluster"
DIVERGE = 1.5
Q = 8
MIN_FH_STEPS = 50
MIN_GROUPS = 6
MIN_FH_SHARE = 0.03
REPS = 2000
REGION_NAMES = ["flat-low", "steep", "flat-high"]


# ---------------------------------------------------------------- loading

def run_records(farm, label, H):
    out = {}
    for f in glob.glob(os.path.join(farms.results_dir(farm), GRID, FC, label, str(H), "s*.json")):
        with open(f, encoding="utf-8") as fh:
            out[os.path.basename(f).split(".")[0]] = json.load(fh)
    return out


def persistence_val_rmse(farm, H):
    rec = run_records(farm, "Persistence", H)
    if not rec:
        raise FileNotFoundError("no persistence result for %s H=%d" % (farm, H))
    return float(list(rec.values())[0]["result"]["val"]["rmse"])


def kept_seeds(farm, bb, H, heads=("", "+paf", "+ens2")):
    """Seeds present for every head of the backbone, minus the seeds of any diverged run (paired drop)."""
    thr = DIVERGE * persistence_val_rmse(farm, H)
    present, bad = None, {}
    for suf in heads:
        rec = run_records(farm, bb + suf, H)
        present = set(rec) if present is None else present & set(rec)
        for s, r in rec.items():
            v = r["result"]["val"]["rmse"]
            if v > thr:
                bad["%s%s %s" % (bb, suf, s)] = round(float(v), 3)
    dropped = {k.split()[-1] for k in bad}
    return sorted(present - dropped), bad, thr


def seed_se(farm, spec, H, keep, split="test"):
    """Seed-averaged squared error (N, H) over the kept seeds."""
    p, y, st, seeds = A.load_model(GRID, spec, H, split, farm)
    idx = [i for i, s in enumerate(seeds) if s in keep]
    if not idx:
        raise ValueError("no kept seed for %s H=%d (%s)" % (spec, H, farm))
    return ((p[idx] - y[None]) ** 2).mean(axis=0), y, st


def region_fn(farm):
    """Region of a wind speed from the farm's training power curve (fitted exactly as run.py does)."""
    t = load_table(farms.data_dir(farm), FC)
    curve = fit_curve_from_table(t)
    gmax = float(curve.values.max())
    iw = t["cols"].index("Wspd")
    ws = t["data"][:, iw].astype(np.float64) * t["std"][iw] + t["mean"][iw]

    def region(v):
        r = curve(v) / gmax
        return np.where(r < 0.1, 0, np.where(r < 0.9, 1, 2))

    return region, ws, curve


# ---------------------------------------------------------------- statistics

def gain(a_sum, b_sum):
    """Backbone-mean relative RMSE reduction (%) from per-backbone sums of squared errors (arrays of shape (B,))."""
    return float(np.mean(100.0 * (1.0 - np.sqrt(a_sum / b_sum))))


def t_two_sided(v):
    from scipy import stats
    v = np.asarray(v, dtype=np.float64)
    q = len(v)
    sd = v.std(ddof=1) if q > 1 else 0.0
    if q < 2 or sd == 0:
        return float("nan"), 1.0
    stat = v.mean() / (sd / math.sqrt(q))
    return float(stat), float(2.0 * stats.t.sf(abs(stat), df=q - 1))


def boot_indices(n, reps=REPS, block=A.BLOCK, seed=0):
    rng = np.random.RandomState(seed)
    nb = int(math.ceil(n / float(block)))
    offs = np.arange(block)
    return [((rng.randint(0, n, nb)[:, None] + offs[None]) % n).ravel()[:n] for _ in range(reps)]


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def influence(a_w, b_w):
    """Per-window influence values (B, N) of the backbone gains 100 (1 - sqrt(sum a / sum b)) (delta method):
    psi_w = -50 r^(-1/2) (a_w - r b_w) / mean(b), r = sum a / sum b; they sum to 0 over the windows, and
    gain + mean of psi over a window subset is the first-order expansion of the gain on that subset."""
    r = a_w.sum(axis=1, keepdims=True) / b_w.sum(axis=1, keepdims=True)
    return -50.0 / np.sqrt(r) * (a_w - r * b_w) / b_w.mean(axis=1, keepdims=True)


def h1_cell(A_w, B_w, boots):
    """A_w, B_w: (B, N) per-window sums of squared error (candidate, reference)."""
    n = A_w.shape[1]
    groups = np.array_split(np.arange(n), Q)
    G = gain(A_w.sum(1), B_w.sum(1))
    gj = [gain(A_w[:, g].sum(1), B_w[:, g].sum(1)) for g in groups]
    stat, p = t_two_sided(gj)
    psi = influence(A_w, B_w).mean(axis=0)  # backbone mean
    gl = [G + psi[g].mean() for g in groups]
    stat_l, p_l = t_two_sided(gl)
    bs = [gain(A_w[:, i].sum(1), B_w[:, i].sum(1)) for i in boots]
    return dict(G=G, ci=ci(bs), groups=gj, groups_positive=int(np.sum(np.array(gj) > 0)), t=stat, p=p,
                groups_lin=[float(x) for x in gl], t_lin=stat_l, p_lin=p_l)


def h2_sums(paf, ens, m_fh, m_st):
    """Per-window sums (B, N) of the squared errors in each region: PAF / ens2 x flat-high / steep."""
    return [(x * m[None]).sum(axis=2) for x, m in ((paf, m_fh), (ens, m_fh), (paf, m_st), (ens, m_st))]


def h2_parts(S, idx):
    """theta, gain_flat_high, gain_steep (backbone means) on the window subset idx."""
    a_fh, b_fh, a_st, b_st = (s[:, idx].sum(axis=1) for s in S)
    g_fh = 100.0 * (1.0 - np.sqrt(a_fh / b_fh))
    g_st = 100.0 * (1.0 - np.sqrt(a_st / b_st))
    return float(np.mean(g_fh - g_st)), float(np.mean(g_fh)), float(np.mean(g_st))


def h2_cell(paf, ens, m_fh, m_st, boots):
    n = paf.shape[1]
    share = float(m_fh.mean())
    groups = np.array_split(np.arange(n), Q)
    counts = [int(m_fh[g].sum()) for g in groups]
    ok_groups = [g for g, c in zip(groups, counts) if c >= MIN_FH_STEPS]
    S = h2_sums(paf, ens, m_fh, m_st)
    theta, g_fh, g_st = h2_parts(S, np.arange(n))
    res = dict(theta=theta, gain_flat_high=g_fh, gain_steep=g_st, flat_high_share_pct=100 * share,
               flat_high_steps_per_group=counts, n_groups_used=len(ok_groups))
    testable = share >= MIN_FH_SHARE and len(ok_groups) >= MIN_GROUPS
    res["testable"] = bool(testable)
    if testable:
        tj = [h2_parts(S, g)[0] for g in ok_groups]
        res["groups"] = tj
        res["groups_positive"] = int(np.sum(np.array(tj) > 0))
        res["t"], res["p"] = t_two_sided(tj)
        a_fh, b_fh, a_st, b_st = S
        psi = (influence(a_fh, b_fh) - influence(a_st, b_st)).mean(axis=0)  # backbone mean, (N,)
        tl = [theta + psi[g].mean() for g in ok_groups]
        res["groups_lin"] = [float(x) for x in tl]
        res["t_lin"], res["p_lin"] = t_two_sided(tl)
    else:
        res["groups"], res["groups_positive"], res["t"], res["p"] = [], 0, float("nan"), 1.0
        res["groups_lin"], res["t_lin"], res["p_lin"] = [], float("nan"), 1.0
    if share > 0:
        bs = [h2_parts(S, i) for i in boots]
        res["ci"] = ci([b[0] for b in bs])
        res["ci_flat_high"] = ci([b[1] for b in bs])
        res["ci_steep"] = ci([b[2] for b in bs])
    return res


def region_gain(se_a, se_b, mask, boots):
    """Backbone-mean gain of A vs B on the window-steps in mask, with the joint bootstrap CI."""
    a = (se_a * mask[None]).sum(axis=2)  # (B, N)
    b = (se_b * mask[None]).sum(axis=2)
    if b.sum() == 0:
        return dict(gain=float("nan"), ci=[float("nan")] * 2)
    return dict(gain=gain(a.sum(1), b.sum(1)), ci=ci([gain(a[:, i].sum(1), b[:, i].sum(1)) for i in boots]))


# ---------------------------------------------------------------- per farm

def val_rmse_mean(farm, label, H, seeds):
    rec = run_records(farm, label, H)
    v = [r["result"]["val"]["rmse"] for s, r in rec.items() if s in seeds]
    return float(np.mean(v)) if v else float("inf")


def analyze_farm(farm):
    region, ws, curve = region_fn(farm)
    out = dict(farm=farm, power_curve=curve.to_dict(), g_max=float(curve.values.max()), cells={})
    for H in HS:
        keep, diverged, thr = {}, {}, None
        for bb in ALL8:
            keep[bb], bad, thr = kept_seeds(farm, bb, H)
            diverged.update(bad)
        se, st = {}, None
        for bb in ALL8:
            for suf in ("", "+paf", "+ens2"):
                se[bb + suf], y, st_m = seed_se(farm, "%s/%s%s" % (FC, bb, suf), H, keep[bb])
                if st is not None and not np.array_equal(st, st_m):
                    raise ValueError("windows differ (%s %s H=%d)" % (farm, bb + suf, H))
                st = st_m
        n = len(y)
        L = A.seq_len_of(GRID, "%s/%s" % (FC, BB7[0]), H, farm)
        boots = boot_indices(n)
        reg_t = region(ws[st[:, None] + L + np.arange(H)[None]])  # ex post, per target step
        reg_o = np.repeat(region(ws[st + L - 1])[:, None], H, axis=1)  # ex ante, per window
        bb_se = np.stack([se[b] for b in BB7])
        paf_se = np.stack([se[b + "+paf"] for b in BB7])
        ens_se = np.stack([se[b + "+ens2"] for b in BB7])
        cell = dict(H=H, n_windows=n, kept_seeds={b: keep[b] for b in ALL8}, diverged_runs=diverged,
                    divergence_threshold_val_rmse=thr)
        cell["H1"] = h1_cell(paf_se.sum(axis=2), bb_se.sum(axis=2), boots)
        cell["H2"] = h2_cell(paf_se, ens_se, reg_t == 2, reg_t == 1, boots)
        desc = {}
        for rname, rmat in (("ex post", reg_t), ("ex ante", reg_o)):
            for k, nm in enumerate(REGION_NAMES):
                m = rmat == k
                desc["%s %s" % (rname, nm)] = dict(
                    share_pct=float(100 * m.mean()),
                    paf_vs_ens2=region_gain(paf_se, ens_se, m, boots),
                    paf_vs_bb=region_gain(paf_se, bb_se, m, boots),
                    ens2_vs_bb=region_gain(ens_se, bb_se, m, boots))
        allm = np.ones_like(reg_t, dtype=bool)
        desc["all"] = dict(share_pct=100.0, paf_vs_ens2=region_gain(paf_se, ens_se, allm, boots),
                           paf_vs_bb=region_gain(paf_se, bb_se, allm, boots),
                           ens2_vs_bb=region_gain(ens_se, bb_se, allm, boots))
        cell["regions"] = desc
        per = {}
        for b in ALL8:
            def g(a, r):
                return float(100 * (1 - math.sqrt(se[a].mean() / se[r].mean())))
            per[b] = dict(seeds=keep[b], rmse_bb=float(math.sqrt(se[b].mean())),
                          rmse_paf=float(math.sqrt(se[b + "+paf"].mean())),
                          rmse_ens2=float(math.sqrt(se[b + "+ens2"].mean())),
                          paf_vs_bb=g(b + "+paf", b), paf_vs_ens2=g(b + "+paf", b + "+ens2"),
                          ens2_vs_bb=g(b + "+ens2", b))
        cell["per_backbone"] = per
        # descriptive: proposed (backbone + PAF with the lowest validation RMSE) vs strongest baseline (head-free
        # backbone with the lowest validation RMSE), farm+cluster inputs, kept seeds of each
        prop = min(ALL8, key=lambda b: val_rmse_mean(farm, b + "+paf", H, keep[b]))
        base = min(ALL8, key=lambda b: val_rmse_mean(farm, b, H, keep[b]))
        sa, sb = se[prop + "+paf"].mean(1), se[base].mean(1)
        pt, lo, hi = A.block_bootstrap_rmse_gain(sa, sb)
        cell["proposed_vs_strongest"] = dict(proposed=prop + "+paf", strongest=base, gain=pt, ci=[lo, hi],
                                             p_im=A.group_t_test(sa, sb)["p"],
                                             rmse_proposed=float(math.sqrt(sa.mean())),
                                             rmse_strongest=float(math.sqrt(sb.mean())))
        out["cells"][str(H)] = cell
        print("%s H=%d: G %+.2f%% p=%.4f | theta %+.2f (testable %s) p=%.4f | diverged %s" % (
            farm, H, cell["H1"]["G"], cell["H1"]["p"], cell["H2"]["theta"], cell["H2"]["testable"], cell["H2"]["p"],
            diverged or "none"), flush=True)
    return out


# ---------------------------------------------------------------- report

def report(res, confirmatory, complete):
    fam_h1 = [(f, H) for f in res for H in HS]
    if confirmatory and complete:
        for key in ("H1", "H2"):
            adj = A.holm([res[f]["cells"][str(H)][key]["p"] for f, H in fam_h1])
            for (f, H), p in zip(fam_h1, adj):
                res[f]["cells"][str(H)][key]["p_holm"] = float(p)
    title = ("# Plan revision 3: pre-registered tests on independent farms" if confirmatory else
             "# Plan revision 3 statistics on SDWPF (EXPLORATORY: discovery data, not in the fourth/fifth family)")
    L = [title, ""]
    if confirmatory and not complete:
        L += ["**INCOMPLETE**: the confirmatory families need both farms; no Holm adjustment is reported.", ""]
    L += ["Gain = 100 (1 - RMSE_A / RMSE_B), backbone mean over %s; RMSE from seed-averaged squared errors. "
          "CI = joint 7-day block bootstrap (2000 reps). p = registered two-sided group t-test: the statistic is "
          "recomputed within each of q = 8 contiguous groups of the test windows (H2: groups with >= %d flat-high "
          "window-steps). Calibration of the test: reports/r3_placebo_sdwpf.md." % (", ".join(BB7), MIN_FH_STEPS), ""]

    def ph(c):
        return "%.4f" % c["p_holm"] if "p_holm" in c else "-"

    hd = "Holm (family of 6)" if confirmatory else "Holm"
    L += ["## H1: backbone + PAF vs backbone (G > 0)", "",
          "| farm | H | windows | G %% [95%% CI] | groups G>0 | t | p | p %s |" % hd, "|---" * 8 + "|"]
    for f, H in fam_h1:
        c = res[f]["cells"][str(H)]["H1"]
        L.append("| %s | %d | %d | %+.2f [%+.2f, %+.2f] | %d/8 | %.2f | %.4f | %s |" % (
            f, H, res[f]["cells"][str(H)]["n_windows"], c["G"], c["ci"][0], c["ci"][1], c["groups_positive"], c["t"],
            c["p"], ph(c)))
    L += ["", "## H2: regime complementarity, theta = gain(PAF vs ens2 | flat-high) - gain(PAF vs ens2 | steep) > 0",
          "", "| farm | H | flat-high %% | gain flat-high %% | gain steep %% | theta [95%% CI] | testable | groups used "
          "| groups theta>0 | t | p | p %s |" % hd, "|---" * 12 + "|"]
    for f, H in fam_h1:
        c = res[f]["cells"][str(H)]["H2"]
        cis = c.get("ci", [float("nan")] * 2)
        if c["testable"]:
            test_cols = "%d | %.2f | %.4f | %s" % (c["groups_positive"], c["t"], c["p"], ph(c))
        else:  # not testable: descriptive only, enters Holm with p = 1
            test_cols = "- | - | - (not testable) | %s" % ph(c)
        L.append("| %s | %d | %.1f | %+.2f | %+.2f | %+.2f [%+.2f, %+.2f] | %s | %d | %s |" % (
            f, H, c["flat_high_share_pct"], c["gain_flat_high"], c["gain_steep"], c["theta"], cis[0], cis[1],
            "yes" if c["testable"] else "NO", c["n_groups_used"], test_cols))
    L += ["", "## Descriptive: backbone-mean gains by region (%)", ""]
    for f in res:
        for H in HS:
            c = res[f]["cells"][str(H)]
            L += ["### %s, H = %d" % (f, H), "", "| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |",
                  "|---|---|---|---|---|"]
            for rn, d in c["regions"].items():
                L.append("| %s | %.1f | %s | %s | %s |" % (rn, d["share_pct"], *[
                    "%+.2f [%+.2f, %+.2f]" % (d[k]["gain"], d[k]["ci"][0], d[k]["ci"][1]) for k in
                    ("paf_vs_ens2", "paf_vs_bb", "ens2_vs_bb")]))
            L += ["", "| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |",
                  "|---" * 8 + "|"]
            for b, d in c["per_backbone"].items():
                L.append("| %s%s | %d | %.3f | %.3f | %.3f | %+.2f | %+.2f | %+.2f |" % (
                    b, " (separate)" if b not in BB7 else "", len(d["seeds"]), d["rmse_bb"], d["rmse_paf"],
                    d["rmse_ens2"], d["paf_vs_bb"], d["paf_vs_ens2"], d["ens2_vs_bb"]))
            ps = c["proposed_vs_strongest"]
            L += ["", "Proposed (val-selected) %s vs strongest baseline (val-selected) %s: RMSE %.3f vs %.3f, gain "
                  "%+.2f%% [%+.2f, %+.2f], p (group t) %.4f (descriptive)." % (
                      ps["proposed"], ps["strongest"], ps["rmse_proposed"], ps["rmse_strongest"], ps["gain"],
                      ps["ci"][0], ps["ci"][1], ps["p_im"]),
                  "Diverged runs (val RMSE > %.3f): %s" % (c["divergence_threshold_val_rmse"],
                                                          c["diverged_runs"] or "none"), ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", nargs="+", default=list(farms.NEW_FARMS), choices=sorted(farms.FARMS))
    ap.add_argument("--from-json", dest="from_json", action="store_true",
                    help="rebuild the report from reports/analysis/<name>.json without recomputing")
    a = ap.parse_args()
    confirmatory = "sdwpf" not in a.farms
    if not confirmatory and len(a.farms) > 1:
        sys.exit("run SDWPF (exploratory) separately from the confirmatory farms")
    complete = sorted(a.farms) == sorted(farms.NEW_FARMS)
    name = "r3_tests" if confirmatory else "r3_sdwpf_exploratory"
    os.makedirs(os.path.join(HERE, "reports", "analysis"), exist_ok=True)
    jpath = os.path.join(HERE, "reports", "analysis", name + ".json")
    if a.from_json:
        with open(jpath, encoding="utf-8") as f:
            res = json.load(f)["results"]
    else:
        res = {f: analyze_farm(f) for f in a.farms}
        with open(jpath, "w", encoding="utf-8") as f:  # saved before formatting; --from-json reuses it
            json.dump(dict(confirmatory=confirmatory, complete=complete, results=res), f, indent=1)
    L = report(res, confirmatory, complete)  # adds the Holm-adjusted p values to res
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(dict(confirmatory=confirmatory, complete=complete, results=res), f, indent=1)
    with open(os.path.join(HERE, "reports", name + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
