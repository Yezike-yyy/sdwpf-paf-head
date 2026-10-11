"""tools/analyze.py statistics on simulated loss series (TEST_ONLY_SIMULATION: code-path and size checks only).

Run: python tests/test_analyze.py
"""
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))

import numpy as np  # noqa: E402

import analyze  # noqa: E402


def test_newey_west():
    rng = np.random.RandomState(0)
    d = rng.randn(500)
    assert np.isclose(analyze.newey_west_var(d, 0), d.var() / len(d))
    print("ok Newey-West lag 0 = var/n")


def test_holm():
    adj = analyze.holm([0.01, 0.04, 0.03, 0.5])
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> monotone 0.09, 0.5*1=0.5
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
    print("ok Holm adjustment")


def ar1(n, phi, rng):
    e = rng.randn(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_dm_size_and_power():
    rng = np.random.RandomState(1)
    n, H = 3000, 16
    lag = analyze.default_lag(n, H)
    rej = 0
    reps = 200
    for _ in range(reps):
        base = np.abs(ar1(n, 0.8, rng)) + 1.0
        noise = ar1(n, 0.8, rng) * 0.3
        rej += analyze.dm_test(base + noise, base, lag, H)["p"] < 0.05  # equal expected loss (noise mean 0)
    size = rej / reps
    print("DM empirical size under the null (AR(1) 0.8 differential, lag %d): %.3f" % (lag, size))
    assert size < 0.15
    base = np.abs(ar1(n, 0.8, rng)) + 1.0
    r = analyze.dm_test(base * 0.9, base, lag, H)
    assert r["mean_diff"] < 0 and r["p"] < 1e-6
    same = analyze.dm_test(base, base, lag, H)
    assert same["stat"] == 0.0 and same["p"] == 1.0
    print("ok DM test (size, power, identical inputs)")


def test_group_t_and_daily_dm():
    rng = np.random.RandomState(3)
    n, rej_im, rej_day, reps = 6000, 0, 0, 200
    day = np.arange(n) // 96
    for _ in range(reps):
        d = ar1(n, 0.99, rng)
        rej_im += analyze.group_t_test(d, np.zeros(n))["p"] < 0.05
        rej_day += analyze.daily_dm(d, np.zeros(n), day)["p"] < 0.05
    print("null rejection rate, AR(1) 0.99: group t %.3f, daily DM %.3f" % (rej_im / reps, rej_day / reps))
    assert rej_im / reps < 0.10
    base = np.abs(ar1(n, 0.8, rng)) + 1.0
    r = analyze.group_t_test(base * 0.9, base)
    assert r["mean_diff"] < 0 and r["p"] < 1e-4 and r["groups_a_better"] == analyze.Q_GROUPS
    assert analyze.group_t_test(base, base)["p"] == 1.0
    m, days = analyze.daily_means(np.ones(200), np.r_[np.zeros(100), np.ones(60), 2 * np.ones(40)])
    assert list(days) == [0.0, 1.0] and np.allclose(m, 1.0)  # day 2 has < 48 windows -> dropped
    print("ok group t-test (size, power, identical inputs) and daily means")


def test_block_bootstrap():
    rng = np.random.RandomState(2)
    se_b = np.abs(ar1(2000, 0.7, rng)) + 1.0
    point, lo, hi = analyze.block_bootstrap_rmse_gain(se_b, se_b, block=96, reps=300)
    assert point == 0.0 and lo <= 0.0 <= hi
    se_a = se_b * 0.81  # RMSE 10 % lower everywhere
    point, lo, hi = analyze.block_bootstrap_rmse_gain(se_a, se_b, block=96, reps=300)
    assert np.isclose(point, 10.0) and lo > 9.9 and hi < 10.1
    again = analyze.block_bootstrap_rmse_gain(se_a, se_b, block=96, reps=300)
    assert again == (point, lo, hi)  # fixed seed -> reproducible
    print("ok block bootstrap (null CI covers 0, exact 10 %% gain, reproducible)")


def test_r3_statistics():
    """Plan revision 3 statistics (tools/r3_tests.py) on simulated squared errors: exact gains, theta, the
    testability rule and the paired divergence drop."""
    import json
    import shutil
    import tempfile
    import r3_tests as R
    from wpf import farms
    rng = np.random.RandomState(5)
    B, N, H = 3, 4000, 4
    ens = np.abs(rng.randn(B, N, H)) + 0.5
    boots = R.boot_indices(N, reps=200, block=96)
    # H1: A = 0.81 B everywhere -> RMSE 10 % lower in every backbone and every group
    h1 = R.h1_cell((0.81 * ens).sum(2), ens.sum(2), boots)
    assert np.isclose(h1["G"], 10.0) and h1["groups_positive"] == 8 and h1["ci"][0] > 9.99
    same = R.h1_cell(ens.sum(2), ens.sum(2), boots)
    assert same["G"] == 0.0 and same["p"] == 1.0
    # H2: PAF 10 % better than ens2 on flat-high steps, equal on steep steps -> theta = 10
    reg = rng.randint(0, 3, (N, H))
    paf = np.where((reg == 2)[None], 0.81 * ens, ens)
    h2 = R.h2_cell(paf, ens, reg == 2, reg == 1, boots)
    assert h2["testable"] and np.isclose(h2["theta"], 10.0) and np.isclose(h2["gain_steep"], 0.0)
    assert h2["groups_positive"] == 8 and h2["ci"][0] > 9.99
    # delta-method group values: their mean over equal groups is the full-sample statistic, and with many windows
    # per group they are close to the recomputed group statistics
    a, b = np.abs(rng.randn(B, N)) + 0.5, np.abs(rng.randn(B, N)) + 0.6
    hl = R.h1_cell(a, b, boots)
    assert np.isclose(np.mean(hl["groups_lin"]), hl["G"]) and np.allclose(hl["groups_lin"], hl["groups"], atol=0.2)
    # flat-high on 2 % of the steps -> not testable (p = 1)
    rare = np.zeros((N, H), bool)
    rare[: int(0.02 * N)] = True
    h2r = R.h2_cell(paf, ens, rare, ~rare, boots)
    assert not h2r["testable"] and h2r["p"] == 1.0
    # divergence: val RMSE > 1.5 x persistence drops that seed for every head of the backbone
    tmp = tempfile.mkdtemp()
    farms.FARMS["_test"] = dict(data="dataset", results=tmp)
    try:
        def put(label, seed, v):
            d = os.path.join(tmp, "main", "farm+cluster", label, "4")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "s%d.json" % seed), "w", encoding="utf-8") as f:
                json.dump({"result": {"val": {"rmse": v}}}, f)
        put("Persistence", 2021, 10.0)
        for s in (2021, 2022, 2023):
            put("LSTM", s, 8.0)
            put("LSTM+paf", s, 16.0 if s == 2022 else 8.0)
            put("LSTM+ens2", s, 8.0)
        keep, bad, thr = R.kept_seeds("_test", "LSTM", 4)
        assert keep == ["s2021", "s2023"] and list(bad) == ["LSTM+paf s2022"] and thr == 15.0
    finally:
        del farms.FARMS["_test"]
        shutil.rmtree(tmp)
    print("ok revision-3 statistics (H1 gain, H2 theta, testability, paired divergence drop)")


if __name__ == "__main__":
    test_newey_west()
    test_holm()
    test_dm_size_and_power()
    test_group_t_and_daily_dm()
    test_block_bootstrap()
    test_r3_statistics()
    print("OK test_analyze")
