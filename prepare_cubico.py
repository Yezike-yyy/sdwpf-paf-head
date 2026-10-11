"""Cubico wind farms (plan revision 3: independent farms) -> farm-level 15-min table, same layout as SDWPF.

Farms (Cubico Sustainable Investments, Greenbyte 10-min SCADA exports, CC BY 4.0, raw files not redistributed):
  penmanshiel  Zenodo 10.5281/zenodo.16807304 (v3), 14 x Senvion MM82 (no WT03), 28.7 MW
  kelmarsh     Zenodo 10.5281/zenodo.16807551 (v4),  6 x Senvion MM92,           12.3 MW
Input : data_raw/<farm>/<Farm>_WT_static.csv and <Farm>_SCADA_<year>_*.zip for 2021 and 2022; each zip holds one
        Turbine_Data_<Farm>_<id>_<from>_-_<to>_<n>.csv per turbine (UTC, time stamp = start of the 10-min period,
        missing values "NaN"). A 15-min bin averages one or two records, as SDWPF in 2021.
Steps:
  python prepare_cubico.py extract --farm kelmarsh   -> data_raw/kelmarsh/extract_10min.csv.gz
                                                        (the columns below, all records, no filtering)
  python tools/pilot_cubico.py --farm kelmarsh       -> reports/pilot_kelmarsh.md (profile for the user audit)
  python prepare_cubico.py build --farm kelmarsh     -> datasets/kelmarsh/kelmarsh_15min.csv + kelmarsh_meta.json
`build` refuses to run until the Unav thresholds of the farm are filled in UNAV below; they are fixed from the
pilot (raw-signal distributions of the training segment only), after the user's audit, before any model run.

Rules (= prepare_sdwpf.py, where the signal exists):
  OT     farm active power (MW) = mean of clip(Power, 0) over the turbines reporting in the bin x N_TURB;
         curtailed / stopped turbines are not masked; capacity = sum of the rated powers (static file)
  farm   Wspd, Wu / Wv (absolute direction: Greenbyte "Wind direction"), Wstd (speed spread over turbines at
         each record, averaged in the bin), Etmp (nacelle ambient temperature), Itmp (nacelle temperature),
         Pitch (mean blade pitch), Unav (Wspd > UNAV wspd and (Power <= 0 or max blade pitch > UNAV pitch)),
         Prtv (reactive power, MVar)
  cluster P_c{k}, Ws_c{k} for K = min(6, N_TURB) k-means clusters of the turbine positions, ordered west -> east
         (Kelmarsh: one turbine per cluster)
  era5   none (the ERA5 ablation is SDWPF-only)
  a bin is usable only if the target and every SCADA feature exist and at least half of the turbines report
  power on average over its records (MIN_REPORT).
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from prepare_sdwpf import FREQ, MIN_REPORT, kmeans  # noqa: E402  (same binning, reporting rule and k-means)

FARMS = {
    "penmanshiel": dict(title="Penmanshiel", zenodo="10.5281/zenodo.16807304 (v3)"),
    "kelmarsh": dict(title="Kelmarsh", zenodo="10.5281/zenodo.16807551 (v4)"),
}
YEARS = (2021, 2022)
START, END = "2021-01-01 00:00", "2022-12-31 23:45"
# Unav thresholds per farm (None = not fixed yet, build refuses). Fixed 2026-10-10 from the pilot (registered rule:
# training-segment records only; pitch = feather mode - 1 deg, wind = lower edge of the first 0.5 m/s bin where
# >= 95 % of the non-feathered records produce; reports/pilot_<farm>.md), user audit passed, before any model run.
UNAV = {
    "penmanshiel": dict(wspd=4.0, pitch=89.0),
    "kelmarsh": dict(wspd=3.5, pitch=91.0),
}
SIGNALS = [  # short name, Greenbyte column
    ("Wspd", "Wind speed (m/s)"),
    ("Wdir", "Wind direction (°)"),
    ("Ndir", "Nacelle position (°)"),
    ("Patv", "Power (kW)"),
    ("Pab1", "Blade angle (pitch position) A (°)"),
    ("Pab2", "Blade angle (pitch position) B (°)"),
    ("Pab3", "Blade angle (pitch position) C (°)"),
    ("Etmp", "Nacelle ambient temperature (°C)"),
    ("Itmp", "Nacelle temperature (°C)"),
    ("Prtv", "Reactive power (kvar)"),
    ("Avail", "Time-based System Avail."),  # pilot only: cross-check of the Unav rule
    ("Setp", "Turbine Power setpoint (kW)"),  # pilot only: curtailment
]
MEMBER = re.compile(r"Turbine_Data_([A-Za-z]+)_(\d+)_(\d{4}-\d{2}-\d{2})_-_(\d{4}-\d{2}-\d{2})_\d+\.csv$")


def raw_dir(farm):
    return os.path.join(os.path.dirname(HERE), "data_raw", farm)


def static_table(farm):
    """Turbine id, rated kW, lat, lon (rows of the static file with a turbine title)."""
    p = os.path.join(raw_dir(farm), "%s_WT_static.csv" % FARMS[farm]["title"])
    st = pd.read_csv(p, encoding="utf-8-sig")
    st = st[st["Title"].notna()].copy()
    st["turb"] = st["Title"].str.extract(r"(\d+)\s*$")[0].astype(int)
    return st[["turb", "Rated power (kW)", "Latitude", "Longitude"]].reset_index(drop=True)


def read_member(zf, name):
    """One turbine-year file -> DataFrame(stamp, SIGNALS short names); the '#' header block is skipped."""
    with zf.open(name) as raw:
        f = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        while True:
            line = f.readline()
            if not line:
                raise ValueError("no header line in %s" % name)
            if line.startswith("# Date and time"):
                header = next(csv.reader([line[2:].rstrip("\r\n")]))
                break
            if not line.startswith("#"):
                raise ValueError("unexpected line before the header in %s: %r" % (name, line[:80]))
        missing = [c for _, c in SIGNALS if c not in header]
        if missing:
            raise KeyError("%s lacks %s" % (name, missing))
        idx = [0] + [header.index(c) for _, c in SIGNALS]
        df = pd.read_csv(f, header=None, usecols=idx, na_values=["NaN"], low_memory=False)
    df = df[idx]
    df.columns = ["stamp"] + [s for s, _ in SIGNALS]
    df["stamp"] = pd.to_datetime(df["stamp"], format="%Y-%m-%d %H:%M:%S")
    return df


def cmd_extract(a):
    title = FARMS[a.farm]["title"]
    zips = sorted(z for y in YEARS for z in glob.glob(os.path.join(raw_dir(a.farm), "%s_SCADA_%d_*.zip" % (title, y))))
    if not zips:
        sys.exit("no SCADA zips for %s in %s" % (a.farm, raw_dir(a.farm)))
    parts, inventory = [], []
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                m = MEMBER.search(info.filename)
                inventory.append(dict(zip=os.path.basename(z), member=info.filename, size=info.file_size,
                                      turbine_file=bool(m)))
                if not m:
                    continue
                if m.group(1) != title:
                    raise ValueError("member %s in %s is not from %s" % (info.filename, z, title))
                df = read_member(zf, info.filename)
                df.insert(0, "turb", int(m.group(2)))
                print("%-60s %7d rows  %s .. %s" % (info.filename, len(df), df.stamp.min(), df.stamp.max()),
                      flush=True)
                parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values(["turb", "stamp"], kind="mergesort")
    p = os.path.join(raw_dir(a.farm), "extract_10min.csv.gz")
    out.to_csv(p, index=False, float_format="%.6g", date_format="%Y-%m-%d %H:%M:%S")
    with open(os.path.join(raw_dir(a.farm), "extract_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zips=[os.path.basename(z) for z in zips], members=inventory,
                       signals=dict(SIGNALS)), f, indent=1, ensure_ascii=False)
    print("wrote", p, out.shape)


def load_extract(farm):
    df = pd.read_csv(os.path.join(raw_dir(farm), "extract_10min.csv.gz"), parse_dates=["stamp"])
    return df


def lonlat_km(st):
    """Local equirectangular projection of the turbine positions (km)."""
    lat0 = np.deg2rad(st["Latitude"].mean())
    x = (st["Longitude"] - st["Longitude"].mean()) * 111.32 * np.cos(lat0)
    y = (st["Latitude"] - st["Latitude"].mean()) * 110.57
    return np.stack([x.values, y.values], axis=1)


def clusters_of(st, n_clusters=6):
    """K = min(n_clusters, turbines) k-means clusters of the turbine positions, numbered west -> east."""
    k = min(n_clusters, len(st))
    centers, labels = kmeans(lonlat_km(st), k)
    order = np.argsort(centers[:, 0])
    relabel = {int(old): new for new, old in enumerate(order)}
    cl = np.array([relabel[int(c)] for c in labels])
    cent = {str(relabel[int(i)]): [round(float(v), 3) for v in c] for i, c in enumerate(centers)}
    return cl, cent


def make_table(farm, df, thr, n_clusters=6):
    """Farm-level 15-min table from the 10-min extract (no file is written). Returns (out, info): out has every bin
    of START..END, bad bins all-NaN in the SCADA columns; info holds the masks and counts for the meta file."""
    st = static_table(farm)
    n_turb = len(st)
    st["cluster"], centers = clusters_of(st, n_clusters)
    k = int(st.cluster.max()) + 1
    cl_of = dict(zip(st.turb, st.cluster))
    df = df.copy()
    unknown = sorted(set(df.turb) - set(cl_of))
    if unknown:
        raise ValueError("turbines %s are not in the static file" % unknown)
    dup = df.duplicated(["turb", "stamp"])
    if dup.any():
        raise ValueError("%d duplicated (turbine, stamp) records" % int(dup.sum()))
    df["t"] = df.stamp.dt.floor(FREQ)
    df["P"] = df.Patv.clip(lower=0) / 1000.0  # MW
    theta = np.deg2rad(df.Wdir)
    df["Wu"] = df.Wspd * np.cos(theta)
    df["Wv"] = df.Wspd * np.sin(theta)
    pitch = df[["Pab1", "Pab2", "Pab3"]]
    df["Pitch"] = pitch.mean(axis=1)
    feathered = pitch.max(axis=1) > thr["pitch"]
    df["Unav"] = ((df.Wspd > thr["wspd"]) & ((df.Patv <= 0) | feathered)).astype(float)
    df.loc[df.Pab1.isna() | df.Patv.isna() | df.Wspd.isna(), "Unav"] = np.nan
    df["Prtv"] = df.Prtv / 1000.0
    df["cluster"] = df.turb.map(cl_of)

    g = df.groupby("t")
    rec = df.groupby("stamp").agg(wstd=("Wspd", "std"), nrep=("Patv", "count"))
    rec["t"] = rec.index.floor(FREQ)
    rg = rec.groupby("t")
    report = rg.nrep.mean() / n_turb
    rec_per_bin = rg.size()
    rpb = {str(y): {str(int(c)): int(v) for c, v in rec_per_bin[rec_per_bin.index.year == y].value_counts().items()}
           for y in sorted(set(rec_per_bin.index.year))}
    fa = pd.DataFrame({  # farm-level feature group
        "Wspd": g.Wspd.mean(), "Wu": g.Wu.mean(), "Wv": g.Wv.mean(), "Wstd": rg.wstd.mean(),
        "Etmp": g.Etmp.mean(), "Itmp": g.Itmp.mean(), "Pitch": g.Pitch.mean(), "Unav": g.Unav.mean(),
        "Prtv": g.Prtv.mean() * n_turb,
    })
    n_cl = st.cluster.value_counts().sort_index()
    gc = df.groupby(["t", "cluster"])
    pc = gc.P.mean().unstack() * n_cl.values[None, :]
    pc.columns = ["P_c%d" % c for c in pc.columns]
    wc = gc.Wspd.mean().unstack()
    wc.columns = ["Ws_c%d" % c for c in wc.columns]
    target = (g.P.mean() * n_turb).rename("OT")

    out = pd.concat([fa, pc, wc, target], axis=1)
    out = out.reindex(pd.date_range(START, END, freq=FREQ))
    need = list(fa.columns) + list(pc.columns) + list(wc.columns) + ["OT"]
    missing = out[need].isna().any(axis=1)
    low_report = report.reindex(out.index).fillna(0.0) < MIN_REPORT
    bad = missing | low_report
    cols_missing = out[need].isna() & ~low_report.values[:, None]  # which feature makes a bin unusable
    out.loc[bad, need] = np.nan
    out.index.name = "date"

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
        source="%s wind farm SCADA (Cubico Sustainable Investments), Zenodo %s, CC BY 4.0"
               % (FARMS[farm]["title"], FARMS[farm]["zenodo"]),
        farm=farm, freq=FREQ, n_bins=len(out), n_bad_bins=int(bad.sum()),
        capacity_mw=float(st["Rated power (kW)"].sum()) / 1000.0, n_turbines=n_turb,
        target="OT (farm active power, MW)",
        groups=dict(farm=list(fa.columns), cluster=list(pc.columns) + list(wc.columns), era5=[]),
        clusters={str(c): sorted(int(t) for t in st.turb[st.cluster == c]) for c in range(k)},
        cluster_centers_km=centers,
        gaps=[r for r in runs if r[2] >= 4], short_gaps=sum(1 for r in runs if r[2] < 4),
        records_per_bin_by_year=rpb, min_report_share=MIN_REPORT,
        n_bins_dropped_low_report=int((low_report & ~missing).sum()),
        n_bins_missing_feature=int(missing.sum()),
        n_bins_missing_by_column={c: int(v) for c, v in cols_missing.sum().items() if v},
        unav_rule="Wspd > %.1f m/s and (Power <= 0 or max blade pitch > %g deg)" % (thr["wspd"], thr["pitch"]),
        unav_thresholds=thr,
    )
    return out, dict(meta=meta, bad=bad, n_cl=n_cl.to_dict(), cols_missing=cols_missing, low_report=low_report)


def cmd_build(a):
    thr = UNAV[a.farm]
    if thr["wspd"] is None or thr["pitch"] is None:
        sys.exit("Unav thresholds for %s are not fixed yet (pilot + user audit first, then fill UNAV)" % a.farm)
    out, info = make_table(a.farm, load_extract(a.farm), thr, a.clusters)
    meta = info["meta"]
    dst = os.path.join(HERE, "datasets", a.farm)
    os.makedirs(dst, exist_ok=True)
    p = os.path.join(dst, "%s_15min.csv" % a.farm)
    out.reset_index().to_csv(p, index=False, float_format="%.6g")
    with open(os.path.join(dst, "%s_meta.json" % a.farm), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    rpb = meta["records_per_bin_by_year"]
    print("wrote", p, out.shape, "bad bins", meta["n_bad_bins"])
    print("gaps >= 1h:", len(meta["gaps"]), "short gaps:", meta["short_gaps"])
    print("cluster sizes", info["n_cl"])
    print("records per bin by year", rpb, "| bins dropped for low reporting", meta["n_bins_dropped_low_report"])
    print(out[["OT", "Wspd", "Unav", "Wstd"]].describe().round(3).to_string())
    print("corr(OT, Wspd) %.3f" % out.OT.corr(out.Wspd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "build"])
    ap.add_argument("--farm", required=True, choices=sorted(FARMS))
    ap.add_argument("--clusters", type=int, default=6)
    a = ap.parse_args()
    {"extract": cmd_extract, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
