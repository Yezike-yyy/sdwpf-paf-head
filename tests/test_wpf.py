"""Data windows, persistence, time-lagged averaging and metrics; farm paths, Greenbyte reader, Unav threshold rule
(CPU, seconds).

Run: python tests/test_wpf.py
"""
import io
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np  # noqa: E402

from wpf import farms  # noqa: E402
from wpf.data import WindowSplit, load_table, table_files  # noqa: E402
from wpf.exp import lag_average, metrics, persistence  # noqa: E402

ROOT = os.path.join(HERE, "dataset")


def test_windows():
    t = load_table(ROOT, "farm+cluster")
    assert t["cols"][-1] == "OT"
    tr_rows = np.arange(t["b2"][0])
    tr_valid = t["valid"][tr_rows]
    # scaler from valid train rows only: scaled train target has mean 0, std 1
    z = t["data"][tr_rows][tr_valid, -1]
    assert abs(z.mean()) < 1e-3 and abs(z.std() - 1) < 1e-3
    for flag in ("train", "val", "test"):
        s = WindowSplit(ROOT, "farm+cluster", flag, 96, 48, 16)
        W = 96 + 16
        # every window lies inside valid rows and inside its split (val/test may look back by seq_len)
        for st in s.starts[:: max(1, len(s) // 500)]:
            assert t["valid"][st:st + W].all()
        k = {"train": 0, "val": 1, "test": 2}[flag]
        assert s.starts.min() >= t["b1"][k] - (96 if k else 0)
        assert s.starts.max() + W <= t["b2"][k]
        # horizons of test windows never start before the test border (no overlap with val targets)
        if k:
            assert (s.starts + 96).min() >= t["b1"][k]
    print("ok windows (valid rows only, split borders, train-only scaler)")


def test_persistence_and_metrics():
    s = WindowSplit(ROOT, "target", "test", 96, 48, 4)
    P, Y = persistence(s)
    assert np.allclose(P[:, 0], s.data[s.starts + 95, -1])
    assert np.allclose(Y[:, 2], s.data[s.starts + 98, -1])
    m = metrics(s, Y, Y)
    assert m["rmse"] < 1e-3 and m["r2"] > 0.999999
    print("ok persistence and metrics (perfect forecast -> RMSE 0, R2 1)")


def test_lag_average_with_gaps():
    starts = np.array([10, 11, 12, 20, 21])  # a gap between 12 and 20
    H = 4
    P = np.random.RandomState(0).randn(len(starts), H)
    out = lag_average(P, starts, 2)
    assert np.allclose(lag_average(P, starts, 0), P)
    # window 12 (index 2), step 0 <- origins 12 (h0), 11 (h1), 10 (h2)
    assert np.isclose(out[2, 0], (P[2, 0] + P[1, 1] + P[0, 2]) / 3)
    # window 20 (index 3) has no stride-1 predecessor -> unchanged
    assert np.allclose(out[3], P[3])
    # window 21, step 0 <- origins 21 (h0), 20 (h1); 19 does not exist
    assert np.isclose(out[4, 0], (P[4, 0] + P[3, 1]) / 2)
    # last step never has an earlier origin covering it
    assert np.allclose(out[:, H - 1], P[:, H - 1])
    print("ok time-lagged averaging (indices, gaps, k=0 identity)")


def test_farm_paths_and_grid():
    import grid
    csv, meta = table_files(ROOT)
    assert os.path.basename(csv) == "sdwpf_15min.csv" and os.path.basename(meta) == "sdwpf_meta.json"
    tmp = tempfile.mkdtemp()
    try:
        for name in ("a_meta.json", "b_meta.json"):
            open(os.path.join(tmp, name), "w").close()
        try:
            table_files(tmp)
            raise AssertionError("two meta files must be rejected")
        except FileNotFoundError:
            pass
        os.remove(os.path.join(tmp, "b_meta.json"))
        try:
            table_files(tmp)  # a_15min.csv is missing
            raise AssertionError("a meta file without its table must be rejected")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp)
    assert os.path.samefile(farms.data_dir("sdwpf"), ROOT)
    assert farms.report_name("sdwpf", "main") == "main" and farms.report_name("kelmarsh", "main") == "kelmarsh_main"
    # SDWPF jobs keep their old result paths; another farm writes under results/<farm>/ and reads datasets/<farm>/
    j = grid.jobs_of("smoke")[0]
    assert os.path.normpath(j["out"]).startswith(os.path.normpath(os.path.join(HERE, "results", "smoke")))
    k = grid.jobs_of("lrscan", "kelmarsh")
    assert len(k) == 72 and all(os.path.normpath(x["out"]).startswith(
        os.path.normpath(os.path.join(HERE, "results", "kelmarsh", "lrscan"))) for x in k)
    c = grid.cmd_of(k[0], "lrscan")
    assert os.path.normpath(c[c.index("--root") + 1]) == os.path.normpath(os.path.join(HERE, "datasets", "kelmarsh"))
    assert os.path.normpath(c[c.index("--results") + 1]) == os.path.normpath(os.path.join(HERE, "results", "kelmarsh"))
    try:
        grid.build("ablation", "kelmarsh")
        raise AssertionError("the ablation grid is SDWPF-only")
    except SystemExit:
        pass
    print("ok farm paths (table discovery, SDWPF paths unchanged, per-farm results and data, no ablation grid)")


def test_greenbyte_reader():
    import prepare_cubico as PC
    cols = ["Date and time", "Extra 1 (x)"] + [g for _, g in reversed(PC.SIGNALS)] + ["Extra, quoted (y)"]
    lines = ["# This file was exported by Greenbyte", "#", "# Turbine: Kelmarsh 3", "#",
             "# " + ",".join('"%s"' % c if "," in c else c for c in cols)]
    vals = {s: float(i + 1) for i, (s, _) in enumerate(PC.SIGNALS)}
    for t in ("2021-01-01 00:00:00", "2021-01-01 00:10:00"):
        row = [t, "9"] + ["%g" % vals[s] for s, _ in reversed(PC.SIGNALS)] + ["7"]
        lines.append(",".join(row))
    lines.append("2021-01-01 00:20:00,9," + ",".join("NaN" for _ in PC.SIGNALS) + ",7")
    buf = io.BytesIO()
    name = "Turbine_Data_Kelmarsh_3_2021-01-01_-_2022-01-01_228.csv"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, ("﻿" + "\n".join(lines) + "\n").encode("utf-8"))
    with zipfile.ZipFile(buf) as zf:
        df = PC.read_member(zf, name)
    assert list(df.columns) == ["stamp"] + [s for s, _ in PC.SIGNALS] and len(df) == 3
    assert all(df[s].iloc[0] == vals[s] for s, _ in PC.SIGNALS) and df.iloc[2, 1:].isna().all()
    assert str(df.stamp.iloc[1]) == "2021-01-01 00:10:00"
    m = PC.MEMBER.search(name)
    assert m.group(1) == "Kelmarsh" and int(m.group(2)) == 3
    print("ok Greenbyte reader (header block, columns by name in any order, NaN, member names)")


def test_unav_threshold_rule():
    import pilot_cubico
    rng = np.random.RandomState(0)
    n = 40000
    w = rng.uniform(0, 12, n)
    pitch = np.where(w > 11, 20.0, 2.0) + rng.randn(n)
    # below 3 m/s most non-feathered records idle; from 3 m/s on 97 % produce
    p = np.where(w < 3.0, np.where(rng.rand(n) < 0.3, 50.0, -5.0), np.where(rng.rand(n) < 0.97, 500.0, 0.0))
    stop = rng.rand(n) < 0.05
    p[stop], pitch[stop] = -3.0, 88.2 + 0.3 * rng.rand(stop.sum())  # feathered stops at 88 deg
    st = pilot_cubico.threshold_stats(w, p, pitch, 2050.0)
    assert st["feather_mode_deg"] == 88
    assert st["proposed_rule"] == dict(wspd=3.0, pitch=87.0), st["proposed_rule"]
    print("ok Unav threshold rule (feather mode - 1 deg; first bin with >= 95 % producing)")


if __name__ == "__main__":
    test_windows()
    test_persistence_and_metrics()
    test_lag_average_with_gaps()
    test_farm_paths_and_grid()
    test_greenbyte_reader()
    test_unav_threshold_rule()
    print("OK test_wpf")
