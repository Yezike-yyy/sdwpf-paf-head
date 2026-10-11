"""Windows over a farm-level 15-min table, TSLib 'MS' layout: target `OT` is the last column.

A dataset directory holds one `<name>_15min.csv` and one `<name>_meta.json`: SDWPF (prepare_sdwpf.py, `dataset/`)
or a Cubico farm (prepare_cubico.py, `datasets/<farm>/`, plan revision 3).
* chronological split of the 15-min grid: 70 % train / 10 % val / 20 % test (TSLib Dataset_Custom ratios);
  val/test windows may look back into the previous split, as in TSLib;
* a window is kept only if every bin of [look-back; horizon] is valid (gap bins are NaN in the table);
* StandardScaler statistics from the valid train bins; metrics are reported in MW after inverse scaling.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from utils.timefeatures import time_features

FEATURES = {
    "target": [],
    "farm": ["farm"],
    "cluster": ["cluster"],
    "farm+cluster": ["farm", "cluster"],
    "farm+cluster+era5": ["farm", "cluster", "era5"],
}
_CACHE = {}


def table_files(root):
    """(csv, meta) of the single `<name>_meta.json` / `<name>_15min.csv` pair in a dataset directory."""
    metas = sorted(glob.glob(os.path.join(root, "*_meta.json")))
    if len(metas) != 1:
        raise FileNotFoundError("expected one *_meta.json in %s, found %d" % (root, len(metas)))
    csv = metas[0][:-len("_meta.json")] + "_15min.csv"
    if not os.path.exists(csv):
        raise FileNotFoundError(csv)
    return csv, metas[0]


def load_table(root, features):
    key = (os.path.abspath(root), features)
    if key in _CACHE:
        return _CACHE[key]
    csv, meta_path = table_files(root)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    df = pd.read_csv(csv)
    cols = [c for g in FEATURES[features] for c in meta["groups"][g]] + ["OT"]
    values = df[cols].values.astype(np.float64)
    valid = ~np.isnan(values).any(axis=1)
    T = len(df)
    n_tr, n_te = int(T * 0.7), int(T * 0.2)
    n_va = T - n_tr - n_te
    b1, b2 = [0, n_tr, T - n_te], [n_tr, n_tr + n_va, T]
    tr = values[:n_tr][valid[:n_tr]]
    mean, std = tr.mean(0), tr.std(0)
    std[std == 0] = 1.0
    data = np.nan_to_num((values - mean) / std).astype(np.float32)
    dates = pd.to_datetime(df["date"].values)
    out = dict(data=data, valid=valid, dates=dates, b1=b1, b2=b2, mean=mean, std=std, cols=cols, meta=meta)
    _CACHE[key] = out
    return out


class WindowSplit(object):
    """Valid windows of one split. Index i of the split = window starting at row starts[i] of the table."""

    def __init__(self, root, features, flag, seq_len, label_len, pred_len, freq="t"):
        t = load_table(root, features)
        k = {"train": 0, "val": 1, "test": 2}[flag]
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.cols, self.meta = t["cols"], t["meta"]
        self.n_vars = len(t["cols"])
        self.y_mean, self.y_std = float(t["mean"][-1]), float(t["std"][-1])
        self.data = t["data"]
        self.stamp = time_features(t["dates"], freq=freq).transpose(1, 0).astype(np.float32)
        lo = t["b1"][k] - (seq_len if k > 0 else 0)
        hi = t["b2"][k]
        W = seq_len + pred_len
        ok = np.convolve(t["valid"][lo:hi].astype(np.int64), np.ones(W, np.int64), "valid") == W
        self.starts = lo + np.nonzero(ok)[0]
        # consecutive windows (stride 1, no gap in between): needed by cross-origin averaging
        self.dates = t["dates"]
        if len(self.starts) == 0:
            raise ValueError("no valid window in %s" % flag)

    def __len__(self):
        return len(self.starts)

    def to_mw(self, z):
        return z * self.y_std + self.y_mean
