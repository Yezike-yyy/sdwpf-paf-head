"""Pre-registered comparison and exploratory analyses of saved test forecasts (see ../统计分析计划.md).

Inputs: prediction files written by `run.py --save_preds` next to each result JSON:
    results/<grid>/<features>/<model>/<H>/s<seed>.preds.npz  with keys
    test_pred (N, H; MW, already clipped at 0), test_starts (N,), seq_len, and the same for val_; the observed
    values are rebuilt from dataset/ (observed()).

  python tools/analyze.py compare --grid G --a FEAT/MODEL --b FEAT/MODEL --H 16     # DM + block bootstrap
  python tools/analyze.py strata  --grid G --models FEAT/MODEL ... --H 16           # power level / state / ramps
  python tools/analyze.py curve   --grid G --models FEAT/MODEL ... --H 16 --lead 16 --start 2021-11-01 --days 7
                                   -> reports/curve_<...>.csv (truth and forecasts at a fixed lead, for plotting)
Every command takes --farm (default sdwpf; plan revision 3 farms: penmanshiel, kelmarsh); the loaders take farm=.
Seeds: per-window losses are averaged over the available seeds (seeds are training replicates, not samples).
Primary test (plan revision 2026-10-09, after review rev-001 showed that window-level DM over-rejects when the
loss differential decays over ~a day): Ibragimov-Mueller group t-test -- the time-ordered test windows are cut
into Q_GROUPS = 8 contiguous groups (~17 days each), and the 8 group means of the loss differential get a
one-sample t-test with 7 df (Ibragimov & Mueller 2010, JBES 28:453). Simulated size 0.055-0.062 for AR(1)
differentials with phi 0.8-0.995 at 15-min steps (scratchpad im_size.py). Secondary: DM on daily mean
differentials (lag floor(n_days^(1/3)), h = 1); sensitivity: window-level DM with the 4x lag. The block
bootstrap of the RMSE gain uses 7-day blocks (672 windows).
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from wpf import farms  # noqa: E402

CAP = 201.0  # SDWPF; other farms: capacity(farm)
POWER_BINS = [0.0, 0.10, 0.30, 0.60, 0.90, np.inf]
UNAV_BINS = [0.0, 0.05, 0.20, 1.01]
RAMP_FRAC = 0.20  # AUTHOR_INPUT_NEEDED in the plan; change there first
BLOCK = 672  # bootstrap block, windows (7 days of 15-min origins)
Q_GROUPS = 8  # groups of the primary Ibragimov-Mueller test
MIN_DAY_WINDOWS = 48  # a day enters the daily DM test only with >= 12 h of valid forecast origins


# ---------------------------------------------------------------- statistics (pure functions, unit-tested)

def newey_west_var(d, lag):
    """Long-run variance of the mean of d (Bartlett kernel)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    u = d - d.mean()
    s = np.dot(u, u) / n
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * np.dot(u[k:], u[:-k]) / n
    return s / n


def dm_test(loss_a, loss_b, lag, h=1):
    """Diebold-Mariano test of equal expected loss, two-sided, HAC variance with `lag`,
    Harvey-Leybourne-Newbold small-sample factor for horizon h, t distribution with n-1 df.
    Returns dict(mean_diff, stat, p, n, lag); mean_diff < 0 means A has the lower loss."""
    from scipy import stats
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = len(d)
    var = newey_west_var(d, lag)
    stat = d.mean() / math.sqrt(var) if var > 0 else 0.0
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return dict(mean_diff=float(d.mean()), stat=float(stat), p=float(p), n=n, lag=int(lag))


def default_lag(n, H):
    """Pre-registered HAC lag: 4 * ((H - 1) + floor(n^(1/3))). In simulations with N = 13,600 and AR(1)
    loss differentials (phi 0.8-0.95) the plain (H - 1) + n^(1/3) lag over-rejected (size 0.07-0.12); four times
    that lag gave sizes 0.04-0.08 (scratchpad dm_size.py, 2026-10-09)."""
    return 4 * ((H - 1) + int(math.floor(n ** (1.0 / 3.0))))


def group_t_test(loss_a, loss_b, q=Q_GROUPS):
    """Primary test: Ibragimov-Mueller t-test on q contiguous group means of the loss differential.
    Two-sided, t(q-1). mean_diff < 0 means A has the lower loss."""
    from scipy import stats
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    m = np.array([g.mean() for g in np.array_split(d, q)])
    sd = m.std(ddof=1)
    stat = m.mean() / (sd / math.sqrt(q)) if sd > 0 else 0.0
    p = 2.0 * stats.t.sf(abs(stat), df=q - 1) if sd > 0 else 1.0
    return dict(mean_diff=float(d.mean()), stat=float(stat), p=float(p), q=q,
                groups_a_better=int((m < 0).sum()))


def daily_means(x, day):
    """Mean of x per day label (sorted), keeping days with >= MIN_DAY_WINDOWS entries."""
    days, inv, cnt = np.unique(day, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=np.asarray(x, dtype=np.float64))
    keep = cnt >= MIN_DAY_WINDOWS
    return sums[keep] / cnt[keep], days[keep]


def daily_dm(loss_a, loss_b, day):
    """Primary test: DM on daily mean losses, lag floor(n^(1/3)), h = 1."""
    a, _ = daily_means(loss_a, day)
    b, _ = daily_means(loss_b, day)
    n = len(a)
    return dm_test(a, b, int(math.floor(n ** (1.0 / 3.0))), 1)


def block_bootstrap_rmse_gain(se_a, se_b, block=BLOCK, reps=2000, seed=0):
    """Circular moving-block bootstrap of the relative RMSE reduction 100 * (1 - RMSE_a / RMSE_b).
    se_a, se_b: per-window mean squared errors (aligned). Returns (point, lo, hi) of a 95% percentile CI."""
    se_a, se_b = np.asarray(se_a, dtype=np.float64), np.asarray(se_b, dtype=np.float64)
    n = len(se_a)
    point = 100.0 * (1.0 - math.sqrt(se_a.mean() / se_b.mean()))
    rng = np.random.RandomState(seed)
    nb = int(math.ceil(n / float(block)))
    offs = np.arange(block)
    out = np.empty(reps)
    for r in range(reps):
        idx = ((rng.randint(0, n, nb)[:, None] + offs[None]) % n).ravel()[:n]
        out[r] = 100.0 * (1.0 - math.sqrt(se_a[idx].mean() / se_b[idx].mean()))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def holm(pvals):
    """Holm step-down adjusted p values (same order as the input)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ---------------------------------------------------------------- loading

def capacity(farm="sdwpf"):
    """Installed capacity (MW) from the farm's dataset meta."""
    from wpf.data import load_table
    return float(load_table(farms.data_dir(farm), "target")["meta"]["capacity_mw"])


def pred_files(grid, spec, H, farm="sdwpf"):
    feat, model = spec.split("/")
    return sorted(glob.glob(os.path.join(farms.results_dir(farm), grid, feat, model, str(H), "s*.preds.npz")))


def origin_day(starts, seq_len, farm="sdwpf"):
    """Calendar day (YYYYMMDD int) of each window's forecast origin (last observed bin)."""
    from wpf.data import load_table
    t = load_table(farms.data_dir(farm), "target")
    d = t["dates"][np.asarray(starts) + seq_len - 1]
    return np.asarray(d.year * 10000 + d.month * 100 + d.day)


def observed(starts, seq_len, H, farm="sdwpf"):
    """Observed farm power (MW) of each window's horizon, rebuilt from the dataset exactly as run.py scores it."""
    from wpf.data import load_table
    t = load_table(farms.data_dir(farm), "target")
    rows = np.asarray(starts)[:, None] + seq_len + np.arange(H)[None]
    return t["data"][rows, -1].astype(np.float64) * t["std"][-1] + t["mean"][-1]


def seq_len_of(grid, spec, H, farm="sdwpf"):
    z = np.load(pred_files(grid, spec, H, farm)[0])
    return int(z["seq_len"]) if "seq_len" in z.files else 96


def load_model(grid, spec, H, split="test", farm="sdwpf"):
    """Seed-stacked forecasts: pred (S, N, H), true (N, H), starts (N,), seeds."""
    files = pred_files(grid, spec, H, farm)
    if not files:
        raise FileNotFoundError("no prediction files for %s H=%d in grid %s of %s (run with --save_preds)"
                                % (spec, H, grid, farm))
    preds, true, starts, seeds = [], None, None, []
    for f in files:
        z = np.load(f)
        if true is None:
            starts = z[split + "_starts"]
            L = int(z["seq_len"]) if "seq_len" in z.files else 96
            true = observed(starts, L, z[split + "_pred"].shape[1], farm)
            if split + "_true" in z.files:  # older files stored the truth: must agree
                assert np.allclose(true, z[split + "_true"], atol=1e-3)
        elif not np.array_equal(starts, z[split + "_starts"]):
            raise ValueError("windows differ between seeds of %s" % spec)
        preds.append(z[split + "_pred"].astype(np.float64))
        seeds.append(os.path.basename(f).split(".")[0])
    return np.stack(preds), true, starts, seeds


def aligned(grid, specs, H, split="test", farm="sdwpf"):
    """Load several models and check they were evaluated on exactly the same windows."""
    data = {s: load_model(grid, s, H, split, farm) for s in specs}
    ref = data[specs[0]][2]
    for s in specs[1:]:
        if not np.array_equal(ref, data[s][2]):
            raise ValueError("%s and %s were evaluated on different windows" % (specs[0], s))
    return data


def window_losses(pred, true):
    """Seed-averaged per-window MSE and MAE (MW^2, MW)."""
    e = pred - true[None]
    return (e ** 2).mean(axis=2).mean(axis=0), np.abs(e).mean(axis=2).mean(axis=0)


# ---------------------------------------------------------------- commands

def cmd_compare(a):
    data = aligned(a.grid, [a.a, a.b], a.H, farm=a.farm)
    (pa, y, st, sa), (pb, _, _, sb) = data[a.a], data[a.b]
    se_a, ae_a = window_losses(pa, y)
    se_b, ae_b = window_losses(pb, y)
    n = len(y)
    lag = a.lag or default_lag(n, a.H)
    day = origin_day(st, seq_len_of(a.grid, a.a, a.H, a.farm), a.farm)
    res = dict(a=a.a, b=a.b, H=a.H, n_windows=n, n_days=int(len(daily_means(se_a, day)[0])), seeds_a=sa, seeds_b=sb,
               rmse_a=float(np.sqrt(se_a.mean())), rmse_b=float(np.sqrt(se_b.mean())),
               mae_a=float(ae_a.mean()), mae_b=float(ae_b.mean()),
               primary_im_mse=group_t_test(se_a, se_b), im_mae=group_t_test(ae_a, ae_b),
               dm_daily_mse=daily_dm(se_a, se_b, day), dm_daily_mae=daily_dm(ae_a, ae_b, day),
               sensitivity_dm_window_mse=dm_test(se_a, se_b, lag, a.H),
               sensitivity_dm_window_mse_lag2x=dm_test(se_a, se_b, 2 * lag, a.H))
    res["rmse_gain_pct"], res["ci_lo"], res["ci_hi"] = block_bootstrap_rmse_gain(se_a, se_b, a.block, a.reps)
    res["rmse_gain_pct_block2x"] = block_bootstrap_rmse_gain(se_a, se_b, 2 * a.block, a.reps)[1:]
    res["mae_gain_pct"] = float(100.0 * (1.0 - ae_a.mean() / ae_b.mean()))
    # per-seed DM (robustness): seed i of A vs seed i of B where both exist
    per = []
    for i in range(min(len(sa), len(sb))):
        per.append(group_t_test(((pa[i] - y) ** 2).mean(1), ((pb[i] - y) ** 2).mean(1))["p"])
    res["per_seed_im_p"] = per
    print(json.dumps(res, indent=1))
    _save(a, "compare_%s_vs_%s_H%d" % (a.a.replace("/", "-"), a.b.replace("/", "-"), a.H), res)


def strata_masks(y, starts, H, table_unav, seq_len, cap=CAP):
    """Window-step masks for the exploratory strata (power level, operating state at origin, ramp windows)."""
    frac = y / cap
    out = {}
    for lo, hi in zip(POWER_BINS[:-1], POWER_BINS[1:]):
        out["power %.0f-%s%%" % (100 * lo, "inf" if np.isinf(hi) else "%.0f" % (100 * hi))] = (frac >= lo) & (frac < hi)
    u0 = table_unav[starts + seq_len - 1]  # last observed bin of each window
    for lo, hi in zip(UNAV_BINS[:-1], UNAV_BINS[1:]):
        m = (u0 >= lo) & (u0 < hi)
        out["unav@origin %.2f-%.2f" % (lo, min(hi, 1.0))] = np.repeat(m[:, None], H, axis=1)
    return out


def cmd_strata(a):
    from wpf.data import load_table
    t = load_table(farms.data_dir(a.farm), "farm+cluster")
    cap = capacity(a.farm)
    i_s, i_y = t["cols"].index("Unav"), t["cols"].index("OT")
    unav = t["data"][:, i_s] * t["std"][i_s] + t["mean"][i_s]
    ot = t["data"][:, i_y] * t["std"][i_y] + t["mean"][i_y]
    data = aligned(a.grid, a.models, a.H, farm=a.farm)
    y, starts = data[a.models[0]][1], data[a.models[0]][2]
    L = seq_len_of(a.grid, a.models[0], a.H, a.farm)
    masks = strata_masks(y, starts, a.H, unav, L, cap)
    origin = ot[starts + L - 1]
    ramp = (np.abs(y - origin[:, None]).max(axis=1) >= RAMP_FRAC * cap)
    masks["ramp windows (>= %.0f%% cap)" % (100 * RAMP_FRAC)] = np.repeat(ramp[:, None], a.H, axis=1)
    rows = []
    for name, m in masks.items():
        row = {"stratum": name, "share_%": float(100 * m.mean())}
        for s in a.models:
            e = data[s][0].mean(axis=0) - y  # seed-averaged forecast error
            row[s] = float(np.sqrt((e[m] ** 2).mean())) if m.any() else float("nan")
        rows.append(row)
    print("exploratory RMSE (MW) by stratum, H=%d" % a.H)
    for r in rows:
        print("%-28s %5.1f%% " % (r["stratum"], r["share_%"]) + " ".join("%s=%.2f" % (s, r[s]) for s in a.models))
    _save(a, "strata_H%d" % a.H, rows)


def cmd_curve(a):
    import pandas as pd
    from wpf.data import load_table
    t = load_table(farms.data_dir(a.farm), "farm+cluster")
    data = aligned(a.grid, a.models, a.H, farm=a.farm)
    y, starts = data[a.models[0]][1], data[a.models[0]][2]
    k = a.lead - 1
    target_rows = starts + seq_len_of(a.grid, a.models[0], a.H, a.farm) + k
    dates = pd.DatetimeIndex(t["dates"][target_rows])
    sel = (dates >= pd.Timestamp(a.start)) & (dates < pd.Timestamp(a.start) + pd.Timedelta(days=a.days))
    df = pd.DataFrame({"time": dates[sel], "observed_MW": y[sel, k]})
    for s in a.models:
        df[s] = data[s][0].mean(axis=0)[sel, k]
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    p = os.path.join(HERE, "reports", farms.report_name(a.farm, "curve_H%d_lead%d_%s.csv" % (a.H, a.lead, a.start)))
    df.to_csv(p, index=False)
    print("wrote", p, len(df), "rows")


def _save(a, name, obj):
    os.makedirs(os.path.join(HERE, "reports", "analysis"), exist_ok=True)
    p = os.path.join(HERE, "reports", "analysis", farms.report_name(a.farm, a.grid + "_" + name + ".json"))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("compare")
    c.add_argument("--grid", required=True)
    c.add_argument("--a", required=True, help="FEATURES/MODEL of the candidate, e.g. farm+cluster/iTransformer+paf")
    c.add_argument("--b", required=True, help="FEATURES/MODEL of the reference")
    c.add_argument("--H", type=int, required=True)
    c.add_argument("--lag", type=int, default=None)
    c.add_argument("--block", type=int, default=BLOCK)
    c.add_argument("--reps", type=int, default=2000)
    s = sub.add_parser("strata")
    s.add_argument("--grid", required=True)
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--H", type=int, required=True)
    v = sub.add_parser("curve")
    v.add_argument("--grid", required=True)
    v.add_argument("--models", nargs="+", required=True)
    v.add_argument("--H", type=int, required=True)
    v.add_argument("--lead", type=int, required=True)
    v.add_argument("--start", required=True)
    v.add_argument("--days", type=int, default=7)
    for p in (c, s, v):
        p.add_argument("--farm", default="sdwpf", choices=sorted(farms.FARMS))
    a = ap.parse_args()
    {"compare": cmd_compare, "strata": cmd_strata, "curve": cmd_curve}[a.cmd](a)


if __name__ == "__main__":
    main()
