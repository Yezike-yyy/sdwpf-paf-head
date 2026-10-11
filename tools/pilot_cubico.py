"""Data pilot of a Cubico farm (data-acquisition gate, plan revision 3): profile of the raw 10-min signals for the
user's audit, and the statistics that fix the Unav thresholds. Every threshold statistic uses the TRAINING segment
only (the first 70 % of the 15-min grid: 2021-01-01 .. 2022-05-26); no model has been run on these farms.

  python prepare_cubico.py extract --farm kelmarsh
  python tools/pilot_cubico.py --farm kelmarsh            -> reports/pilot_kelmarsh.md (+ .json)
  python tools/pilot_cubico.py --sdwpf                    -> reports/pilot_sdwpf_thresholds.md: the same threshold
                                                             statistics on SDWPF (training segment), to see which
                                                             rule reproduces its documented 2.5 m/s / 89 deg
Threshold statistics:
  pitch: feather position = mode (1-deg bins) of the max blade pitch of stopped records with wind
         (Power <= 0, Wspd > 5 m/s, max pitch > 45 deg); operating range = percentiles of the max blade pitch
         of producing records (Power > 0.05 x rated).
  wind : share of producing records (Power > 0) per 0.5 m/s bin among non-feathered records (max pitch <= 45 deg).
Proposed rule (it reproduces the documented SDWPF thresholds when applied to SDWPF's training segment: feather mode
90 deg -> 89 deg; first bin with >= 95 % producing = 2.5-3.0 m/s -> 2.5 m/s; reports/pilot_sdwpf_thresholds.md):
  pitch threshold = feather position - 1 deg; wind threshold = lower edge of the first 0.5 m/s bin in which
  >= 95 % of the non-feathered records produce power (above it, a stopped turbine is not in low-wind standby).
"""
import argparse
import json
import os
import sys

HERE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prepare_cubico as PC  # noqa: E402

TRAIN_END = {"cubico": "2022-05-27 00:00", "sdwpf": None}  # sdwpf: computed from its grid below
WBINS = np.arange(0.0, 6.01, 0.5)


def fmt_table(df, floatfmt="%.3g"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [(floatfmt % v) if isinstance(v, (float, np.floating)) else str(v)
                                                 for v in r.values]) + " |")
    return out


def threshold_stats(wspd, patv, pmax, rated_kw, feather_floor=45.0):
    """Pitch and wind statistics of the training records (arrays of equal length, NaN allowed)."""
    ok = np.isfinite(wspd) & np.isfinite(patv) & np.isfinite(pmax)
    w, p, m = wspd[ok], patv[ok], pmax[ok]
    stopped = (p <= 0) & (w > 5.0) & (m > feather_floor)
    hist = np.bincount(np.clip(np.floor(m[stopped]).astype(int), 0, 120), minlength=121)
    feather_mode = int(np.argmax(hist)) if stopped.any() else None
    producing = p > 0.05 * rated_kw
    q = np.percentile(m[producing], [50, 99, 99.9, 99.99]) if producing.any() else [np.nan] * 4
    pitch_hist = {}
    for name, sel in (("producing", producing), ("stopped_with_wind", (p <= 0) & (w > 5.0))):
        h, e = np.histogram(m[sel], bins=np.arange(-180, 226, 5))
        pitch_hist[name] = {"%d..%d" % (e[i], e[i + 1]): int(h[i]) for i in range(len(h)) if h[i]}
    out = dict(n_records=int(ok.sum()), feather_mode_deg=feather_mode,
               feather_mode_count=int(hist.max()) if stopped.any() else 0,
               n_stopped_with_wind_feathered=int(stopped.sum()),
               producing_pitch_pct={"p50": float(q[0]), "p99": float(q[1]), "p99.9": float(q[2]), "p99.99": float(q[3])},
               pitch_hist_5deg=pitch_hist)
    rows = []
    for lo in WBINS[:-1]:
        for thr_name, thr in (("not feathered (max pitch <= 45)", 45.0),):
            sel = (w >= lo) & (w < lo + 0.5) & (m <= thr)
            rows.append(dict(bin="%.1f-%.1f" % (lo, lo + 0.5), n=int(sel.sum()),
                             share_producing=float((p[sel] > 0).mean()) if sel.any() else float("nan")))
    out["producing_share_by_wind_bin"] = rows
    for lvl in (0.5, 0.95):
        cut = [r for r in rows if r["n"] >= 100 and r["share_producing"] >= lvl]
        out["first_bin_producing_%d_ms" % round(100 * lvl)] = float(cut[0]["bin"].split("-")[0]) if cut else None
    out["proposed_rule"] = dict(wspd=out["first_bin_producing_95_ms"],
                                pitch=float(feather_mode - 1) if feather_mode is not None else None)
    return out


def cmd_sdwpf(a):
    p = os.path.join(os.path.dirname(HERE), "data_raw", "sdwpf", "sdwpf_2001_2112_full.csv")
    df = pd.read_csv(p, usecols=["Tmstamp", "Wspd", "Pab1", "Pab2", "Pab3", "Patv"], parse_dates=["Tmstamp"])
    grid = pd.date_range("2020-01-01 00:00", "2021-12-31 23:45", freq="15min")
    train_end = grid[int(len(grid) * 0.7)]
    tr = df[df.Tmstamp < train_end]
    st = threshold_stats(tr.Wspd.values, tr.Patv.values, tr[["Pab1", "Pab2", "Pab3"]].max(axis=1).values, 1500.0)
    st["train_end"] = str(train_end)
    st["documented_rule"] = "Wspd > 2.5 m/s and (Patv <= 0 or a blade pitch > 89 deg) (prepare_sdwpf.py)"
    _write("pilot_sdwpf_thresholds", st, ["# SDWPF: Unav threshold statistics (training segment, before %s)" % train_end,
                                          "", "```json", json.dumps(st, indent=1), "```"])


def _write(name, obj, lines):
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", name + ".json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "reports", name + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote reports/%s.md" % name)


def gap_runs(bad, index):
    runs, start = [], None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        if not b and start is not None:
            runs.append((str(index[start]), str(index[i - 1]), i - start))
            start = None
    if start is not None:
        runs.append((str(index[start]), str(index[-1]), len(bad) - start))
    return runs


def cmd_farm(a):
    farm = a.farm
    st = PC.static_table(farm)
    rated = float(st["Rated power (kW)"].iloc[0])
    n_turb = len(st)
    with open(os.path.join(PC.raw_dir(farm), "extract_inventory.json"), encoding="utf-8") as f:
        inv = json.load(f)
    df = PC.load_extract(farm)
    R = {"farm": farm, "n_turbines_static": n_turb, "rated_kw": rated,
         "capacity_mw": float(st["Rated power (kW)"].sum()) / 1000.0}
    L = ["# Data pilot: %s (raw 10-min SCADA, 2021-2022)" % PC.FARMS[farm]["title"], "",
         "Source: Zenodo %s, CC BY 4.0. Profile only: nothing filtered or cleaned; threshold statistics use the "
         "training segment (before %s) only." % (PC.FARMS[farm]["zenodo"], TRAIN_END["cubico"]), ""]

    # 1. inventory
    other = [m["member"] for m in inv["members"] if not m["turbine_file"]]
    tf = [m for m in inv["members"] if m["turbine_file"]]
    turbs = sorted(df.turb.unique())
    R["zips"] = inv["zips"]
    R["n_turbine_files"] = len(tf)
    R["other_members"] = other
    R["turbines_in_data"] = [int(t) for t in turbs]
    R["turbines_static_not_in_data"] = sorted(int(t) for t in set(st.turb) - set(turbs))
    rows_per = df.groupby([df.turb, df.stamp.dt.year]).size().unstack()
    dup = int(df.duplicated(["turb", "stamp"]).sum())
    off_grid = int(((df.stamp.dt.minute % 10) != 0).sum() + (df.stamp.dt.second != 0).sum())
    steps = df.groupby("turb").stamp.diff().dropna().dt.total_seconds() / 60
    R.update(duplicates=dup, stamps_off_10min_grid=off_grid,
             step_minutes=steps.value_counts().head(5).to_dict(),
             stamp_min=str(df.stamp.min()), stamp_max=str(df.stamp.max()))
    L += ["## 1. Inventory", "",
          "- zips: %s" % ", ".join(inv["zips"]),
          "- turbine files: %d; other files in the zips: %s" % (len(tf), ", ".join(other) if other else "none"),
          "- turbines in the data: %s (static file: %d turbines%s)" % (
              ", ".join(str(t) for t in turbs), n_turb,
              "; missing " + str(R["turbines_static_not_in_data"]) if R["turbines_static_not_in_data"] else ""),
          "- time stamps %s .. %s; duplicates (turbine, stamp): %d; stamps off the 10-min grid: %d" % (
              R["stamp_min"], R["stamp_max"], dup, off_grid),
          "- step between consecutive records (min: count): %s" % R["step_minutes"], "",
          "Records per turbine and year (expected 52,560 = 365 x 144):", ""]
    rows_per.index.name = "turbine"
    L += fmt_table(rows_per, "%d") + [""]

    # 2. missingness
    sig = ["Wspd", "Wdir", "Ndir", "Patv", "Pab1", "Pab2", "Pab3", "Etmp", "Itmp", "Prtv", "Avail", "Setp"]
    miss = df.groupby("turb")[sig].apply(lambda x: x.isna().mean() * 100)
    miss.index.name = "turbine"
    R["missing_pct_by_turbine"] = miss.round(3).to_dict(orient="index")
    L += ["## 2. Missing values (% of records, NaN)", ""] + fmt_table(miss, "%.2f") + [""]
    mon = df.assign(ok=df.Patv.notna()).groupby([df.turb, df.stamp.dt.to_period("M")]).ok.mean().unstack(0) * 100
    low = mon[(mon < 90).any(axis=1)]
    L += ["Months in which some turbine has power in < 90 % of its records (% present):", ""]
    if len(low):
        low.index = low.index.astype(str)
        low.index.name = "month"
        L += fmt_table(low, "%.0f")
    else:
        L += ["none"]
    L += [""]

    # 3. ranges
    pmax = df[["Pab1", "Pab2", "Pab3"]].max(axis=1)
    d = ((df.Wdir - df.Ndir + 180) % 360) - 180
    rng = {
        "Power kW": [df.Patv.min(), df.Patv.quantile(0.001), df.Patv.quantile(0.999), df.Patv.max()],
        "Wind speed m/s": [df.Wspd.min(), df.Wspd.quantile(0.001), df.Wspd.quantile(0.999), df.Wspd.max()],
        "max blade pitch deg": [pmax.min(), pmax.quantile(0.001), pmax.quantile(0.999), pmax.max()],
        "Nacelle ambient temp C": [df.Etmp.min(), df.Etmp.quantile(0.001), df.Etmp.quantile(0.999), df.Etmp.max()],
        "Nacelle temp C": [df.Itmp.min(), df.Itmp.quantile(0.001), df.Itmp.quantile(0.999), df.Itmp.max()],
        "Reactive power kvar": [df.Prtv.min(), df.Prtv.quantile(0.001), df.Prtv.quantile(0.999), df.Prtv.max()],
        "Wind dir - nacelle pos deg": [d.min(), d.quantile(0.001), d.quantile(0.999), d.max()],
        "Setpoint kW": [df.Setp.min(), df.Setp.quantile(0.001), df.Setp.quantile(0.999), df.Setp.max()],
        "Time-based avail.": [df.Avail.min(), df.Avail.quantile(0.001), df.Avail.quantile(0.999), df.Avail.max()],
    }
    rt = pd.DataFrame(rng, index=["min", "p0.1", "p99.9", "max"]).T
    rt.index.name = "signal"
    R["ranges"] = rt.round(3).to_dict(orient="index")
    neg = float((df.Patv < 0).mean() * 100)
    over = float((df.Patv > 1.02 * rated).mean() * 100)
    pmin = df[["Pab1", "Pab2", "Pab3"]].min(axis=1)
    odd = (pmax > 100) | (pmin < -10)
    R.update(power_negative_pct=neg, power_over_102pct_rated_pct=over,
             wdir_minus_ndir_abs_median=float(d.abs().median()),
             pitch_out_of_range=dict(n_records=int(odd.sum()), max_over_100=int((pmax > 100).sum()),
                                     min_below_minus10=int((pmin < -10).sum()),
                                     by_turbine={int(k): int(v) for k, v in df[odd].groupby("turb").size().items()},
                                     power_median_kw=float(df.Patv[odd].median()) if odd.any() else None,
                                     share_with_power_le_0=float((df.Patv[odd] <= 0).mean()) if odd.any() else None))
    stop = (df.Patv <= 0) & (df.Wspd > 5) & (pmax > 45)
    R["feather_mode_by_turbine_all_data"] = {
        int(t): int(np.bincount(np.floor(g).astype(int).clip(0, 300)).argmax()) for t, g in pmax[stop].groupby(df.turb[stop])}
    sp = {}
    for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 0.93 * rated), (0.93 * rated, 1.1 * rated)):
        m = (df.Patv >= lo) & (df.Patv < hi)
        sp["%d-%d kW" % (lo, hi)] = float(df.Setp[m].median())
    R["setpoint_median_by_power"] = sp
    L += ["## 3. Value ranges (all records)", ""] + fmt_table(rt, "%.2f") + [
        "", "- power < 0 (idling consumption): %.2f %% of records; power > 1.02 x rated (%.0f kW): %.3f %%" % (
            neg, rated, over),
        "- |wind direction - nacelle position|: median %.1f deg (small = 'Wind direction' is an absolute direction "
        "derived from the nacelle position)" % R["wdir_minus_ndir_abs_median"],
        "- blade angle outside [-10, 100] deg: %(n_records)d records (max > 100: %(max_over_100)d, min < -10: "
        "%(min_below_minus10)d), by turbine %(by_turbine)s; median power %(power_median_kw)s kW, share with power <= 0: "
        "%(share_with_power_le_0)s" % R["pitch_out_of_range"],
        "- feather position by turbine (mode of max blade pitch, stopped with wind, all records): %s deg"
        % R["feather_mode_by_turbine_all_data"],
        "- turbine power setpoint, median by power class: %s (it follows the power, so it is not a curtailment flag; "
        "not used)" % {k: round(v) for k, v in sp.items()}, ""]

    # 4. threshold statistics, training segment only
    tr = df[df.stamp < pd.Timestamp(TRAIN_END["cubico"])]
    ts = threshold_stats(tr.Wspd.values, tr.Patv.values, tr[["Pab1", "Pab2", "Pab3"]].max(axis=1).values, rated)
    R["threshold_stats_train"] = ts
    wb = pd.DataFrame(ts["producing_share_by_wind_bin"]).set_index("bin")
    wb.index.name = "wind bin m/s"
    L += ["## 4. Unav threshold statistics (training segment only)", "",
          "- feather position (mode of max blade pitch, stopped with wind > 5 m/s): %s deg (%d records)" % (
              ts["feather_mode_deg"], ts["feather_mode_count"]),
          "- max blade pitch of producing records (power > 5 %% rated): p50 %.1f, p99 %.1f, p99.9 %.1f, "
          "p99.99 %.1f deg" % tuple(ts["producing_pitch_pct"][k] for k in ("p50", "p99", "p99.9", "p99.99")),
          "- first 0.5 m/s bin in which >= 50 %% / >= 95 %% of the non-feathered records produce: %s / %s m/s" % (
              ts["first_bin_producing_50_ms"], ts["first_bin_producing_95_ms"]),
          "- **proposed thresholds (SDWPF-reproducing rule): wind %s m/s, pitch %s deg**" % (
              ts["proposed_rule"]["wspd"], ts["proposed_rule"]["pitch"]), "",
          "Share of producing records by wind bin (non-feathered records):", ""] + fmt_table(wb, "%.3f") + [
          "", "Max blade pitch histogram (5-deg bins, training records):", "",
          "```", json.dumps(ts["pitch_hist_5deg"], indent=1), "```", ""]

    # 5. cross-check of candidate rules against Greenbyte availability and setpoint (training segment)
    cands = []
    prop = ts["proposed_rule"]
    pm = tr[["Pab1", "Pab2", "Pab3"]].max(axis=1)
    ok = tr.Wspd.notna() & tr.Patv.notna() & tr.Pab1.notna()
    for vw in sorted({2.5, prop["wspd"] if prop["wspd"] is not None else 2.5}):
        for pt in sorted({89.0, prop["pitch"] if prop["pitch"] is not None else 89.0}):
            un = ((tr.Wspd > vw) & ((tr.Patv <= 0) | (pm > pt)))[ok]
            av = tr.Avail[ok]
            m = av.notna() & (tr.Wspd[ok] > 4.0)  # agreement on the same population (above every threshold)
            gb_unav = av[m] < 0.5
            cands.append(dict(wspd=vw, pitch=pt, unav_share_pct=float(un.mean() * 100),
                              agree_with_greenbyte_pct=float((un[m] == gb_unav).mean() * 100),
                              rule_unav_but_gb_available_pct=float((un[m] & ~gb_unav).mean() * 100),
                              rule_avail_but_gb_unavailable_pct=float((~un[m] & gb_unav).mean() * 100)))
    R["candidate_rules_train"] = cands
    ct = pd.DataFrame(cands).set_index(["wspd", "pitch"])
    ct.index = ["%.1f m/s, %g deg" % i for i in ct.index]
    ct.index.name = "rule"
    L += ["## 5. Candidate Unav rules (training segment): share flagged and agreement with Greenbyte availability",
          "", "Unav share: all training records. Agreement: records with wind > 4 m/s; Greenbyte 'Time-based System "
          "Avail.' < 0.5 is taken as unavailable (it does not count curtailment).",
          ""] + fmt_table(ct, "%.2f") + [""]

    # 6. 15-min binning preview (all data, descriptive)
    df["t"] = df.stamp.dt.floor(PC.FREQ)
    rec = df.groupby("stamp").agg(nrep=("Patv", "count"))
    rec["t"] = rec.index.floor(PC.FREQ)
    rep = rec.groupby("t").nrep.mean() / n_turb
    allrep = rec.groupby("t").nrep.min() == n_turb
    grid = pd.date_range(PC.START, PC.END, freq=PC.FREQ)
    rep = rep.reindex(grid).fillna(0.0)
    allrep = allrep.reindex(grid).fillna(False).astype(bool)
    ok_half = rep >= PC.MIN_REPORT
    runs = gap_runs((~ok_half).values, grid)
    R.update(n_bins=len(grid), share_bins_half_reporting_pct=float(ok_half.mean() * 100),
             share_bins_all_reporting_pct=float(allrep.mean() * 100),
             gaps_ge_1h=len([r for r in runs if r[2] >= 4]),
             longest_gaps=sorted(runs, key=lambda r: -r[2])[:5])
    n_tr = int(len(grid) * 0.7)
    split = pd.Series(np.where(np.arange(len(grid)) < n_tr, "train",
                               np.where(np.arange(len(grid)) < len(grid) - int(len(grid) * 0.2), "val", "test")),
                      index=grid)
    sh = pd.DataFrame({"bins": split.value_counts(), "half reporting %": ok_half.groupby(split).mean() * 100,
                       "all reporting %": allrep.groupby(split).mean() * 100}).loc[["train", "val", "test"]]
    # preview of the build rules (same code as prepare_cubico.py build, nothing written), proposed thresholds
    thr = dict(wspd=prop["wspd"], pitch=prop["pitch"])
    if dup:
        raise SystemExit("duplicated (turbine, stamp) records: resolve them before the build preview")
    out, info = PC.make_table(farm, df.drop(columns=["t"]), thr)
    bad = info["bad"].reindex(grid).values
    sh["usable (build rules) %"] = pd.Series(~bad, index=grid).groupby(split).mean() * 100
    meta = info["meta"]
    R["build_preview"] = dict(thresholds=thr, n_bad_bins=meta["n_bad_bins"], clusters=meta["clusters"],
                              n_bins_dropped_low_report=meta["n_bins_dropped_low_report"],
                              n_bins_missing_by_column=meta["n_bins_missing_by_column"],
                              gaps_ge_1h=len(meta["gaps"]), longest_gaps=sorted(meta["gaps"], key=lambda r: -r[2])[:5])
    causes = []  # why each of the longest unusable stretches is unusable
    for s, e, nb in R["build_preview"]["longest_gaps"]:
        sl = slice(pd.Timestamp(s), pd.Timestamp(e))
        frac = info["cols_missing"].loc[sl].mean()
        causes.append(dict(start=s, end=e, bins=nb,
                           low_report_pct=round(100.0 * float(info["low_report"].loc[sl].mean()), 1),
                           columns_missing_in_half_the_bins=[c for c, v in frac.items() if v >= 0.5]))
    R["build_preview"]["longest_gap_causes"] = causes
    sh.index.name = "split"
    R["split_bounds"] = {s: [str(grid[split == s][0]), str(grid[split == s][-1])] for s in ("train", "val", "test")}
    L += ["## 6. 15-min grid preview (descriptive)", "",
          "- %d bins %s .. %s; splits: %s" % (len(grid), grid[0], grid[-1], R["split_bounds"]),
          "- bins where >= half of the turbines report power (SDWPF rule): %.2f %%" % R["share_bins_half_reporting_pct"],
          "- bins where every turbine reports power at every record: %.2f %% (relevant when a cluster is a single "
          "turbine: a cluster feature is then missing whenever that turbine is)" % R["share_bins_all_reporting_pct"],
          "- gaps >= 1 h (bins failing the half rule): %d; longest: %s" % (R["gaps_ge_1h"], R["longest_gaps"]),
          "- build preview (proposed thresholds, all SDWPF rules incl. 'every feature present'): %d unusable bins; "
          "clusters %s; bins lost by column (beyond the reporting rule): %s; gaps >= 1 h: %d; longest: %s" % (
              meta["n_bad_bins"], meta["clusters"], meta["n_bins_missing_by_column"] or "none", len(meta["gaps"]),
              R["build_preview"]["longest_gaps"]),
          "- causes of the longest unusable stretches (bins below the reporting rule; columns missing in at least "
          "half of the bins):"] + [
          "  - %s .. %s (%d bins): low reporting %.1f %%; %s" % (
              c["start"], c["end"], c["bins"], c["low_report_pct"],
              ", ".join(c["columns_missing_in_half_the_bins"]) or "no single column") for c in causes] + [
          ""] + fmt_table(sh, "%.2f") + [""]
    _write("pilot_%s" % farm, R, L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", choices=sorted(PC.FARMS))
    ap.add_argument("--sdwpf", action="store_true")
    a = ap.parse_args()
    if a.sdwpf:
        cmd_sdwpf(a)
    elif a.farm:
        cmd_farm(a)
    else:
        ap.error("--farm or --sdwpf")


if __name__ == "__main__":
    main()
