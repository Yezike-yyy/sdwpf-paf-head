"""SDWPF (Zhou et al., Sci Data 2024; figshare 10.6084/m9.figshare.24798654) -> farm-level 15-min table.

Input : sdwpf_2001_2112_full.csv (134 turbines, SCADA + hourly ERA5, 2020-01 .. 2021-12; the SCADA records are
        15 min apart in 2020 (stamps :10/:25/:40/:55, one record per bin) and 10 min apart in 2021 (one or two
        records per bin), so a 15-min bin averages one record in 2020 and one or two in 2021)
        sdwpf_turb_location_elevation.csv
Output: dataset/sdwpf_15min.csv   one row per 15-min bin on a complete grid (gap bins are NaN)
        dataset/sdwpf_meta.json   capacity, clusters, gaps, column groups

Target `OT` = farm active power in MW: mean of clip(Patv, 0) over the turbines reporting in the bin,
times 134 (missing turbine records imputed by the farm mean). Curtailed / stopped turbines are NOT masked:
the target is what the farm actually delivered. Installed capacity 134 x 1.5 MW = 201 MW. A bin is usable only
if, on average over its records, at least half of the turbines (MIN_REPORT) report active power.

Column groups (selected by --features in run.py):
  farm    : Wspd (mean speed), Wu / Wv (mean wind vector, absolute direction = Ndir + Wdir), Wstd (speed
            spread over turbines at each record, averaged in the bin), Etmp, Itmp, Pitch (mean blade pitch),
            Unav (share of turbines not producing although the wind blows: Wspd > 2.5 m/s and Patv <= 0 or a
            blade pitch > 89 deg, i.e. the abnormal-state rules of the SDWPF documentation; low-wind standby is
            NOT counted), Prtv (reactive power, MVar)
  cluster : P_c{k} (MW) and Ws_c{k} for K spatial clusters (k-means on turbine x, y)
  era5    : T2m, Sp, RelH, Wspd_w, Wu_w, Wv_w, Tp  (reanalysis, NOT available in real time; ablation only)

Run: python prepare_sdwpf.py --raw ../data_raw/sdwpf
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

N_TURB = 134
RATED_MW = 1.5
FREQ = "15min"
MIN_REPORT = 0.5  # minimum mean share of turbines reporting Patv in a usable bin
UNAV_WSPD = 2.5  # m/s; feathered / zero-power turbines below this speed are in normal low-wind standby


def kmeans(X, k, seed=0, restarts=10, iters=100):
    """Plain Lloyd k-means (sklearn's KMeans crashes in threadpoolctl on some Windows builds)."""
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C2 = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"))
    ap.add_argument("--clusters", type=int, default=6)
    a = ap.parse_args()

    loc = pd.read_csv(os.path.join(a.raw, "sdwpf_turb_location_elevation.csv"))
    centers, labels = kmeans(loc[["x", "y"]].values.astype(float), a.clusters)
    # order clusters west -> east so names are stable
    order = np.argsort(centers[:, 0])
    relabel = {int(old): new for new, old in enumerate(order)}
    loc["cluster"] = [relabel[int(c)] for c in labels]
    cl_of = dict(zip(loc.TurbID, loc.cluster))

    cols = ["TurbID", "Tmstamp", "Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv",
            "T2m", "Sp", "RelH", "Wspd_w", "Wdir_w", "Tp", "Patv"]
    df = pd.read_csv(os.path.join(a.raw, "sdwpf_2001_2112_full.csv"), usecols=cols, parse_dates=["Tmstamp"])
    df["t"] = df.Tmstamp.dt.floor(FREQ)
    df["P"] = df.Patv.clip(lower=0) / 1000.0  # MW
    theta = np.deg2rad(df.Ndir + df.Wdir)
    df["Wu"] = df.Wspd * np.cos(theta)
    df["Wv"] = df.Wspd * np.sin(theta)
    df["Pitch"] = df[["Pab1", "Pab2", "Pab3"]].mean(axis=1)
    feathered = df[["Pab1", "Pab2", "Pab3"]].max(axis=1) > 89
    df["Unav"] = ((df.Wspd > UNAV_WSPD) & ((df.Patv <= 0) | feathered)).astype(float)
    df.loc[df.Pab1.isna() | df.Patv.isna() | df.Wspd.isna(), "Unav"] = np.nan
    df["Prtv"] = df.Prtv / 1000.0
    df["cluster"] = df.TurbID.map(cl_of)
    thw = np.deg2rad(df.Wdir_w)
    df["Wu_w"] = df.Wspd_w * np.cos(thw)
    df["Wv_w"] = df.Wspd_w * np.sin(thw)

    g = df.groupby("t")
    # per record time stamp: spread of wind speed across turbines and number of turbines reporting power
    rec = df.groupby("Tmstamp").agg(wstd=("Wspd", "std"), nrep=("Patv", "count"))
    rec["t"] = rec.index.floor(FREQ)
    rg = rec.groupby("t")
    report = rg.nrep.mean() / N_TURB
    rec_per_bin = rg.size()
    rpb = {str(y): {str(int(k)): int(v) for k, v in rec_per_bin[rec_per_bin.index.year == y].value_counts().items()}
           for y in sorted(set(rec_per_bin.index.year))}
    farm = pd.DataFrame({
        "Wspd": g.Wspd.mean(), "Wu": g.Wu.mean(), "Wv": g.Wv.mean(), "Wstd": rg.wstd.mean(),
        "Etmp": g.Etmp.mean(), "Itmp": g.Itmp.mean(), "Pitch": g.Pitch.mean(), "Unav": g.Unav.mean(),
        "Prtv": g.Prtv.mean() * N_TURB,
    })
    era = g[["T2m", "Sp", "RelH", "Wspd_w", "Wu_w", "Wv_w", "Tp"]].mean()
    n_cl = loc.cluster.value_counts().sort_index()
    gc = df.groupby(["t", "cluster"])
    pc = (gc.P.mean().unstack() * n_cl.values[None, :])
    pc.columns = ["P_c%d" % k for k in pc.columns]
    wc = gc.Wspd.mean().unstack()
    wc.columns = ["Ws_c%d" % k for k in wc.columns]
    target = (g.P.mean() * N_TURB).rename("OT")

    out = pd.concat([farm, pc, wc, era, target], axis=1)
    grid = pd.date_range(out.index.min(), "2021-12-31 23:45", freq=FREQ)
    out = out.reindex(grid)
    out = out[out.index >= "2020-01-01 00:00"]
    # a bin is usable only if the target and every SCADA feature exist
    need = list(farm.columns) + list(pc.columns) + list(wc.columns) + ["OT"]
    missing = out[need].isna().any(axis=1)
    low_report = report.reindex(out.index).fillna(0.0) < MIN_REPORT
    bad = missing | low_report
    out.loc[bad, need] = np.nan
    out.index.name = "date"
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "sdwpf_15min.csv")
    out.reset_index().to_csv(p, index=False, float_format="%.6g")

    runs, start = [], None
    for i, b in enumerate(bad.values):
        if b and start is None:
            start = i
        if not b and start is not None:
            runs.append((str(out.index[start]), str(out.index[i - 1]), i - start))
            start = None
    if start is not None:
        runs.append((str(out.index[start]), str(out.index[-1]), len(bad) - start))
    meta = dict(
        source="SDWPF full (Zhou et al., Scientific Data 11:649, 2024), figshare 10.6084/m9.figshare.24798654",
        freq=FREQ, n_bins=len(out), n_bad_bins=int(bad.sum()), capacity_mw=N_TURB * RATED_MW,
        target="OT (farm active power, MW)",
        groups=dict(farm=list(farm.columns), cluster=list(pc.columns) + list(wc.columns), era5=list(era.columns)),
        clusters={str(k): sorted(int(t) for t in loc.TurbID[loc.cluster == k]) for k in range(a.clusters)},
        cluster_centers_xy={str(relabel[int(i)]): [round(float(v), 1) for v in c]
                            for i, c in enumerate(centers)},
        gaps=[r for r in runs if r[2] >= 4], short_gaps=sum(1 for r in runs if r[2] < 4),
        records_per_bin_by_year=rpb, min_report_share=MIN_REPORT,
        n_bins_dropped_low_report=int((low_report & ~missing).sum()),
        unav_rule="Wspd > %.1f m/s and (Patv <= 0 or max blade pitch > 89 deg)" % UNAV_WSPD,
    )
    with open(os.path.join(a.out, "sdwpf_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print("wrote", p, out.shape, "bad bins", int(bad.sum()))
    print("gaps >= 1h:", meta["gaps"], "short gaps:", meta["short_gaps"])
    print("cluster sizes", n_cl.to_dict())
    print("records per bin by year", rpb, "| bins dropped for low reporting", meta["n_bins_dropped_low_report"])
    print(out[["OT", "Wspd", "Unav", "Wstd"]].describe().round(3).to_string())
    print("corr(OT, Wspd) %.3f" % out.OT.corr(out.Wspd))


if __name__ == "__main__":
    main()
