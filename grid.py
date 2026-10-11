"""Run a grid of run.py jobs on the local GPUs (server), skipping finished results.

  python grid.py --grid lrscan --gpus 0 1 --slots 3     # 1. learning rate per model on validation (72 jobs)
  python grid.py --grid main --gpus 0 1 --slots 3       # 2. baselines + PAF head, 3 seeds, saves predictions
  python grid.py --grid ablation --gpus 0 1 --slots 3   # 3. head ablations and input ablations
  python grid.py --grid main --list                     # show the job list and what is done
  python grid.py --grid lrscan --show-lr                # learning rates chosen by the pre-registered rule
  python grid.py --grid lrscan --farm kelmarsh ...      # another farm (plan revision 3): data datasets/<farm>/,
                                                        # results results/<farm>/<grid>/, logs logs/<farm>/<grid>/
(`baselines`, the M2 grid, ran under the old protocol; its results are kept in results/baselines.)
On the revision-3 farms the main grid has the farm+cluster inputs only, and there is no ablation grid.
Safety (same rules as periodmix/grid.py):
  * single instance per grid (lock file logs/grid_<grid>.lock);
  * children get PR_SET_PDEATHSIG, and SIGTERM/SIGINT on grid.py terminates them (no orphans);
  * a failed job is retried up to --retries times after the others (logs kept as .try<k>.log);
  * the job log is logs/<grid>/<job>.log; failures are listed at the end and in logs/<grid>_failed.txt.
"""
import argparse
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wpf import farms  # noqa: E402

SEEDS = [2021, 2022, 2023]
TRAINED = ["DLinear", "PatchTST", "iTransformer", "TimeMixer", "LSTM", "GRU", "TCN", "Transformer"]
HS = [4, 8, 16]
LR_GRID = [1e-2, 1e-3, 1e-4]
ABL_BACKBONES = ["LSTM", "GRU", "TCN", "Transformer", "iTransformer"]  # channel-mixing backbones
ABL_HEADS = ["aux", "paf_noavail", "paf_noaux", "phys_only"]


def J(model, H, feat, seed, head="none", lr=None, tag=None, epochs=None, preds=True):
    return dict(model=model, H=H, feat=feat, seed=seed, head=head, lr=lr, tag=tag, epochs=epochs, preds=preds)


def lr_tag(lr):
    return "lrscan/lr%.0e" % lr


def select_lr(model, strict=True, farm="sdwpf"):
    """Pre-registered rule: the LR_GRID value with the lowest mean validation RMSE over the three horizons
    (farm+cluster, seed 2021, head none) in the farm's lrscan results. Ties -> the smaller lr."""
    import json
    scores = {}
    for lr in LR_GRID:
        v = []
        for H in HS:
            f = os.path.join(farms.results_dir(farm), lr_tag(lr), "farm+cluster", model, str(H), "s2021.json")
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    v.append(json.load(fh)["result"]["val"]["rmse"])
        if len(v) == len(HS):
            scores[lr] = sum(v) / len(v)
    if len(scores) < len(LR_GRID):
        if strict:
            sys.exit("lrscan incomplete for %s on %s (%d of %d learning rates): run grid.py --grid lrscan --farm %s first"
                     % (model, farm, len(scores), len(LR_GRID), farm))
        return None, scores
    best = min(sorted(scores), key=lambda k: (scores[k], k))
    return best, scores


def build(name, farm="sdwpf"):
    out = []
    if farm != "sdwpf" and name == "ablation":
        sys.exit("the ablation grid is SDWPF-only (plan revision 3)")
    if name == "smoke":  # laptop smoke test of the runner itself (1 epoch)
        for m, head in (("DLinear", "none"), ("GRU", "none"), ("GRU", "paf"), ("DLinear", "paf"), ("GRU", "ens2")):
            out.append(J(m, 4, "farm+cluster", 2021, head=head, epochs=1, lr=1e-3))
        out.append(J("Persistence", 4, "farm+cluster", 2021))
    elif name == "lrscan":  # learning-rate choice on validation, one seed
        for lr in LR_GRID:
            for H in HS:
                for m in TRAINED:
                    out.append(J(m, H, "farm+cluster", 2021, lr=lr, tag=lr_tag(lr), preds=False))
    elif name in ("main", "ablation"):
        lrs = {m: select_lr(m, farm=farm)[0] for m in TRAINED}
        feats = ["farm+cluster", "target"] if name == "main" and farm == "sdwpf" else ["farm+cluster"]
        for feat in feats:
            for H in HS:
                if name == "main":
                    out.append(J("Persistence", H, feat, 2021))
                for seed in SEEDS:
                    if name == "main":
                        for m in TRAINED:
                            out.append(J(m, H, feat, seed, lr=lrs[m]))
                            if feat == "farm+cluster":
                                out.append(J(m, H, feat, seed, head="paf", lr=lrs[m]))
                    else:
                        for m in ABL_BACKBONES:
                            for head in ABL_HEADS:
                                out.append(J(m, H, feat, seed, head=head, lr=lrs[m], preds=False))
        if name == "ablation":  # input ablation: farm features only, and + ERA5 (not available in real time)
            for feat in ("farm", "farm+cluster+era5"):
                for H in HS:
                    for seed in SEEDS:
                        for m in ("LSTM", "TCN"):
                            for head in ("none", "paf"):
                                out.append(J(m, H, feat, seed, head=head, lr=lrs[m], preds=False))
    elif name == "control":  # ensemble control (plan revision 2): +ens2 on all 8 backbones, results go to main
        lrs = {m: select_lr(m, farm=farm)[0] for m in TRAINED}
        for H in HS:
            for seed in SEEDS:
                for m in TRAINED:
                    out.append(J(m, H, "farm+cluster", seed, head="ens2", lr=lrs[m], tag="main"))
    else:
        raise KeyError(name)
    return out


GRIDS = ["smoke", "lrscan", "main", "ablation", "control"]


def jobs_of(name, farm="sdwpf"):
    out = build(name, farm)
    for j in out:
        tag = j["tag"] or name
        label = j["model"] if j["head"] == "none" else "%s+%s" % (j["model"], j["head"])
        j["tag"] = tag
        j["farm"] = farm
        j["name"] = "%s_%s_%s_H%d_s%d" % (tag.replace("/", "-"), j["feat"].replace("+", "-"), label, j["H"], j["seed"])
        j["out"] = os.path.join(farms.results_dir(farm), tag, j["feat"], label, str(j["H"]), "s%d.json" % j["seed"])
    return out


def cmd_of(j, grid):
    c = [sys.executable, os.path.join(HERE, "run.py"), "--model", j["model"], "--pred_len", str(j["H"]),
         "--features", j["feat"], "--seed", str(j["seed"]), "--tag", j["tag"],
         "--root", farms.data_dir(j["farm"]), "--results", farms.results_dir(j["farm"])]
    if j["head"] != "none":
        c += ["--head", j["head"]]
    if j["lr"] and j["model"] != "Persistence":
        c += ["--lr", repr(j["lr"])]
    if j["epochs"]:
        c += ["--epochs", str(j["epochs"])]
    if j["preds"]:
        c += ["--save_preds"]
    return c


def _child_setup():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001  (not Linux)
        pass


class Stop(Exception):
    pass


def acquire_lock(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a+")
    try:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass
    except OSError:
        sys.exit("another grid.py is already running this grid (%s)" % path)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, choices=GRIDS)
    ap.add_argument("--farm", default="sdwpf", choices=sorted(farms.FARMS))
    ap.add_argument("--gpus", nargs="*", default=["0"])
    ap.add_argument("--slots", type=int, default=3, help="concurrent jobs per GPU")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show-lr", dest="show_lr", action="store_true")
    a = ap.parse_args()
    if a.show_lr:
        import json
        sel = {}
        for m in TRAINED:
            best, sc = select_lr(m, strict=False, farm=a.farm)
            sel[m] = dict(chosen=best, mean_val_rmse={"%g" % k: v for k, v in sorted(sc.items())})
            print("%-13s chosen %-7s  mean val RMSE: %s" % (m, best, ", ".join("%g=%.3f" % kv for kv in sorted(sc.items()))))
        os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
        with open(os.path.join(HERE, "reports", farms.report_name(a.farm, "lr_selected.json")), "w",
                  encoding="utf-8") as f:
            json.dump(sel, f, indent=1)
        return 0

    jobs = jobs_of(a.grid, a.farm)
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    print("grid %s (farm %s): %d jobs, %d done, %d to run" % (a.grid, a.farm, len(jobs), len(jobs) - len(todo),
                                                            len(todo)), flush=True)
    if a.list:
        for j in jobs:
            print("done" if os.path.exists(j["out"]) else "todo", j["name"])
        return 0
    gname = farms.report_name(a.farm, a.grid)  # grid name for lock and failure list
    lock = acquire_lock(os.path.join(HERE, "logs", "grid_%s.lock" % gname))  # noqa: F841
    logdir = os.path.join(HERE, "logs", a.grid) if a.farm == "sdwpf" else os.path.join(HERE, "logs", a.farm, a.grid)
    os.makedirs(logdir, exist_ok=True)

    def on_signal(sig, _frame):
        raise Stop(sig)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    # persistence first (instant), then the slow models, so the long runs do not end up last
    queue = sorted(todo, key=lambda j: (j["model"] != "Persistence", j["model"] not in ("PatchTST", "TimeMixer")))
    tries = {j["name"]: 0 for j in queue}
    running = {}  # popen -> (job, gpu, logfile)
    slots = [g for g in a.gpus for _ in range(a.slots)]
    failed, t0, n_ok = [], time.time(), 0
    try:
        while queue or running:
            busy = [v[1] for v in running.values()]
            free = list(slots)
            for g in busy:
                free.remove(g)
            while queue and free:
                j = queue.pop(0)
                g = free.pop(0)
                tries[j["name"]] += 1
                suffix = "" if tries[j["name"]] == 1 else ".try%d" % tries[j["name"]]
                lf = open(os.path.join(logdir, j["name"] + suffix + ".log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CUDA_DEVICE_ORDER="PCI_BUS_ID")
                p = subprocess.Popen(cmd_of(j, a.grid), stdout=lf, stderr=subprocess.STDOUT, env=env,
                                     cwd=HERE, preexec_fn=_child_setup if os.name == "posix" else None)
                running[p] = (j, g, lf)
            time.sleep(1.0)
            for p in [p for p in running if p.poll() is not None]:
                j, g, lf = running.pop(p)
                lf.close()
                if p.returncode == 0 and os.path.exists(j["out"]):
                    n_ok += 1
                    print("[%5.0fm] ok   %-45s gpu%s (%d left, %d running)" % (
                        (time.time() - t0) / 60, j["name"], g, len(queue), len(running)), flush=True)
                elif tries[j["name"]] <= a.retries:
                    print("[%5.0fm] FAIL %-45s rc=%s -> requeued" % ((time.time() - t0) / 60, j["name"],
                                                                    p.returncode), flush=True)
                    queue.append(j)
                else:
                    print("[%5.0fm] FAIL %-45s rc=%s -> given up" % ((time.time() - t0) / 60, j["name"],
                                                                    p.returncode), flush=True)
                    failed.append(j["name"])
    except Stop:
        print("stopping: terminating %d running jobs" % len(running), flush=True)
        for p in running:
            p.terminate()
        for p in running:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        return 1
    with open(os.path.join(HERE, "logs", "%s_failed.txt" % gname), "w") as f:
        f.write("\n".join(failed) + ("\n" if failed else ""))
    print("finished: %d ok, %d failed%s, %.0f min" % (n_ok, len(failed), (": " + ", ".join(failed)) if failed else "",
                                                     (time.time() - t0) / 60), flush=True)
    print("done", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
