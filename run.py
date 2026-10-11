"""One wind-power forecasting run at farm level, 15 min (SDWPF by default; --root datasets/<farm> for another farm).

  python run.py --model iTransformer --pred_len 16 --features farm+cluster --seed 2021
  python run.py --model TCN --head paf --pred_len 16 --save_preds   # backbone + PAF output head
  python run.py --model Persistence --pred_len 16            # no training
Result: <results>/<tag>/<features>/<label>/<H>/s<seed>.json (skipped if it exists), label = model or model+head;
with --save_preds also s<seed>.preds.npz (val/test forecasts in MW clipped at 0, window starts, seq_len) for
tools/analyze.py, which rebuilds the observed values from the dataset.
"""
import argparse
import json
import os
import platform
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wpf.data import FEATURES, WindowSplit, load_table, table_files  # noqa: E402
from wpf.exp import lag_average, metrics, persistence, run_experiment  # noqa: E402
from wpf.hparams import MODELS, get_hparams  # noqa: E402
from wpf.physics import HEAD_VARIANTS, fit_curve_from_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS) + ["Persistence"])
    ap.add_argument("--pred_len", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--features", default="farm+cluster", choices=list(FEATURES))
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--root", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--epochs", type=int, default=None, help="override train_epochs (smoke tests)")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--head", default="none", choices=list(HEAD_VARIANTS))
    ap.add_argument("--aux_weight", type=float, default=None, help="override the head variant's aux loss weight")
    ap.add_argument("--lr", type=float, default=None, help="override the learning rate (lrscan grid)")
    a = ap.parse_args()

    label = a.model if a.head == "none" else "%s+%s" % (a.model, a.head)
    if a.model == "Persistence" and a.head != "none":
        ap.error("Persistence has no output head")
    out = os.path.join(a.results, a.tag, a.features, label, str(a.pred_len), "s%d.json" % a.seed)
    if os.path.exists(out):
        print("exists", out)
        return 0
    sz = dict(seq_len=a.seq_len, label_len=48, pred_len=a.pred_len)
    splits = {f: WindowSplit(a.root, a.features, f, sz["seq_len"], sz["label_len"], a.pred_len)
              for f in ("train", "val", "test")}
    n_vars = splits["train"].n_vars
    t0 = time.time()
    if a.model == "Persistence":
        res, preds = {}, {}
        for f in ("val", "test"):
            P, Y = persistence(splits[f])
            res[f] = metrics(splits[f], P, Y)
            res[f + "_lag"] = {}
            preds[f + "_pred"] = np.clip(splits[f].to_mw(P), 0.0, None).astype(np.float32)
            preds[f + "_starts"] = splits[f].starts.astype(np.int64)
        hp = dict(model="Persistence", pred_len=a.pred_len, seq_len=a.seq_len)
    else:
        over = dict(seq_len=a.seq_len, seed=a.seed)
        if a.epochs:
            over["train_epochs"] = a.epochs
        if a.lr:
            over["learning_rate"] = a.lr
        wrap, _, _, _, aux_w = HEAD_VARIANTS[a.head]
        over["head"] = a.head
        over["aux_weight"] = aux_w if a.aux_weight is None else a.aux_weight
        if wrap:
            t = load_table(a.root, a.features)
            if "Wspd" not in t["cols"] or "Unav" not in t["cols"]:
                ap.error("--head %s needs the farm features (Wspd, Unav)" % a.head)
            curve = fit_curve_from_table(t)
            over["_head_ctx"] = dict(cols=t["cols"], mean=t["mean"], std=t["std"], curve=curve,
                                     capacity=float(t["meta"]["capacity_mw"]))
            over["power_curve"] = curve.to_dict()
        hp = get_hparams(a.model, a.pred_len, n_vars, **over)
        if hp["label_len"] > a.seq_len:
            hp["label_len"] = a.seq_len
        args = SimpleNamespace(**hp)
        for s in splits.values():
            s.label_len = args.label_len
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res = run_experiment(args, splits["train"], splits["val"], splits["test"], device,
                             log=lambda m: print(m, flush=True))
        state = res.pop("_state")
        preds = res.pop("_preds")
        if a.save_ckpt:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save(state, out[:-5] + ".pt")
    res["wall_sec"] = time.time() - t0
    if a.save_preds:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out[:-5] + ".preds.npz", seq_len=a.seq_len, **preds)
    farm = os.path.basename(table_files(a.root)[0])[:-len("_15min.csv")]
    rec = dict(farm=farm, model=a.model, head=a.head, label=label, pred_len=a.pred_len, features=a.features, seed=a.seed,
               n_vars=n_vars,
               columns=splits["train"].cols, n_windows={f: len(s) for f, s in splits.items()},
               hparams={k: v for k, v in hp.items() if not k.startswith("_")}, result=res,
               env=dict(torch=torch.__version__, host=platform.node(),
                        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(out + ".tmp", out)
    te = res["test"]
    print("%s H=%d %s s%d | test RMSE %.2f MW MAE %.2f MW R2 %.4f | val RMSE %.2f | %.0fs" % (
        label, a.pred_len, a.features, a.seed, te["rmse"], te["mae"], te["r2"], res["val"]["rmse"],
        res["wall_sec"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
