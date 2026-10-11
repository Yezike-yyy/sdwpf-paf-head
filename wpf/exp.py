"""Training / evaluation loop (TSLib Exp_Long_Term_Forecast with features='MS': all channels in, loss on `OT`).

Adam + MSE on the scaled target, TSLib lradj schedules (same epoch indexing as TSLib: the first two epochs
run at the initial lr under type1) and EarlyStopping on validation MSE of the target; the best weights are kept
in memory. With an output head (wpf/physics.py) the loss adds aux_weight x the MSE of the auxiliary channels.
Forecasts are mapped back to MW and clipped at 0 MW (no negative output) before the metrics; no upper clip,
because the imputed farm total slightly exceeds the 201 MW nameplate at times (max 203.5).
"""
import copy
import importlib
import math
import time

import numpy as np
import torch

from wpf.baselines import REGISTRY


def adjust_learning_rate(opt, epoch, args):
    """Called after 1-based epoch `epoch` finished (TSLib: adjust(epoch0 + 1) after 0-based epoch0)."""
    if args.lradj == "type1":
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif args.lradj == "cosine":
        lr = args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))
    else:
        return
    for g in opt.param_groups:
        g["lr"] = lr


def build_model(args):
    if args.model in REGISTRY:
        net = REGISTRY[args.model](args).float()
    else:
        net = importlib.import_module("models." + args.model).Model(args).float()
    head = getattr(args, "head", "none")
    if head == "none":
        return net
    from wpf.physics import HEAD_VARIANTS, SECOND_PATH, PAFHead
    _, use_phys, use_avail, phys_only, _ = HEAD_VARIANTS[head]
    c = args._head_ctx
    return PAFHead(net, args.pred_len, c["cols"], c["mean"], c["std"], c["curve"], c["capacity"],
                   use_phys=use_phys, use_avail=use_avail, phys_only=phys_only,
                   second=SECOND_PATH.get(head, "phys"))


class Resident(object):
    """One split on the device; batches are gathered by window index."""

    def __init__(self, split, device):
        self.s = split
        self.X = torch.as_tensor(split.data, device=device)
        self.M = torch.as_tensor(split.stamp, device=device)
        self.starts = torch.as_tensor(split.starts, device=device)
        L, ll, H = split.seq_len, split.label_len, split.pred_len
        self.ox = torch.arange(L, device=device)
        self.oy = torch.arange(L - ll, L + H, device=device)
        self.n = len(split)
        self.device = device

    def batch(self, idx):
        st = self.starts[idx]
        tx, ty = st[:, None] + self.ox[None], st[:, None] + self.oy[None]
        return self.X[tx], self.X[ty], self.M[tx], self.M[ty]

    def iterate(self, bs, shuffle, gen=None):
        order = torch.randperm(self.n, device=self.device, generator=gen) if shuffle else \
            torch.arange(self.n, device=self.device)
        for i in range(0, self.n, bs):
            yield self.batch(order[i:i + bs])


def forward_all(model, args, bx, by, bxm, bym):
    """All output channels over the horizon, [B, H, C] (target OT is the last channel)."""
    dec = torch.cat([by[:, :args.label_len], torch.zeros_like(by[:, -args.pred_len:])], dim=1)
    return model(bx, bxm, dec, bym)[:, -args.pred_len:, :]


def forward(model, args, bx, by, bxm, bym):
    return forward_all(model, args, bx, by, bxm, bym)[..., -1]


@torch.no_grad()
def predict(model, args, res):
    """Scaled target forecasts and truths, [N, H] numpy."""
    model.eval()
    P, Y = [], []
    for bx, by, bxm, bym in res.iterate(args.batch_size, False):
        P.append(forward(model, args, bx, by, bxm, bym).float().cpu().numpy())
        Y.append(by[:, -args.pred_len:, -1].float().cpu().numpy())
    model.train()
    return np.concatenate(P), np.concatenate(Y)


def persistence(split):
    """Last observed target value repeated over the horizon (scaled units)."""
    L, H = split.seq_len, split.pred_len
    last = split.data[split.starts + L - 1, -1]
    Y = split.data[split.starts[:, None] + L + np.arange(H)[None], -1]
    return np.repeat(last[:, None], H, axis=1), Y


def lag_average(P, starts, k):
    """Cross-origin (time-lagged) average: window s, step h <- mean of P[s', h+j] over origins s' = s - j
    (j = 0..k) that exist as stride-1 predecessors (starts[s'] == starts[s] - j)."""
    if k == 0:
        return P
    pos = {int(t): i for i, t in enumerate(starts)}
    N, H = P.shape
    out, cnt = P.copy(), np.ones_like(P)
    for j in range(1, min(k, H - 1) + 1):
        prev = np.array([pos.get(int(t) - j, -1) for t in starts])
        m = prev >= 0
        out[m, :H - j] += P[prev[m], j:]
        cnt[m, :H - j] += 1
    return out / cnt


def metrics(split, P, Y):
    """MW metrics after inverse scaling and clipping at 0; plus scaled MSE. N-metrics are % of capacity."""
    cap = float(split.meta["capacity_mw"])
    p = np.clip(split.to_mw(P.astype(np.float64)), 0.0, None)
    y = split.to_mw(Y.astype(np.float64))
    e = p - y
    out = dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
               r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
               mse_scaled=float(((P - Y).astype(np.float64) ** 2).mean()))
    out["nmae_pct"] = 100 * out["mae"] / cap
    out["nrmse_pct"] = 100 * out["rmse"] / cap
    out["rmse_by_step"] = [float(v) for v in np.sqrt((e ** 2).mean(0))]
    return out


def run_experiment(args, tr_split, va_split, te_split, device, log=print, lags=(1, 2, 4, 8)):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))
    tr, va, te = Resident(tr_split, device), Resident(va_split, device), Resident(te_split, device)
    model = build_model(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    aux_w = float(getattr(args, "aux_weight", 0.0))
    aux_idx = model.aux_targets() if hasattr(model, "aux_targets") else []
    if aux_w > 0 and not aux_idx:
        raise ValueError("aux_weight > 0 needs an output head with auxiliary channels")
    best, best_state, best_ep, bad = np.inf, None, 0, 0
    t0 = time.time()
    epochs_run = 0
    for ep in range(1, args.train_epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for bx, by, bxm, bym in tr.iterate(args.batch_size, True, gen):
            opt.zero_grad()
            out = forward_all(model, args, bx, by, bxm, bym)
            yt = by[:, -args.pred_len:, :]
            loss = torch.mean((out[..., -1] - yt[..., -1]) ** 2)
            if aux_w > 0:
                loss = loss + aux_w * torch.mean((out[..., aux_idx] - yt[..., aux_idx]) ** 2)
            loss.backward()
            opt.step()
            tl += float(loss)
            nb += 1
        P, Y = predict(model, args, va)
        vl = float(((P - Y) ** 2).mean())
        epochs_run = ep
        log("epoch %d train %.4f val %.4f (%.0fs)" % (ep, tl / nb, vl, time.time() - t0))
        if not np.isfinite(vl):
            raise FloatingPointError("validation loss is %s at epoch %d" % (vl, ep))
        if vl <= best:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
        adjust_learning_rate(opt, ep, args)
    model.load_state_dict(best_state)
    res = dict(best_epoch=best_ep, epochs_run=epochs_run, train_sec=time.time() - t0,
               n_params=sum(p.numel() for p in model.parameters()))
    if hasattr(model, "gate") and model.use_phys:
        res["gate_direct_weight"] = [float(v) for v in torch.sigmoid(model.gate.detach()).cpu()] \
            if not model.phys_only else [0.0] * args.pred_len
    preds = {}
    for name, rs, sp in (("val", va, va_split), ("test", te, te_split)):
        P, Y = predict(model, args, rs)
        res[name] = metrics(sp, P, Y)
        res[name + "_lag"] = {str(k): metrics(sp, lag_average(P, sp.starts, k), Y) for k in lags}
        preds[name + "_pred"] = np.clip(sp.to_mw(P.astype(np.float64)), 0.0, None).astype(np.float32)
        preds[name + "_starts"] = sp.starts.astype(np.int64)
    res["_state"] = best_state
    res["_preds"] = preds
    return res
