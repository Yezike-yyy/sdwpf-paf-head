"""Physics-informed, availability-aware fusion head (working name: PAF head), backbone-agnostic.

Every backbone here runs in TSLib "MS" layout and already emits a forecast for every input channel. The head
reads three of them -- farm power (OT, the target), farm-mean wind speed (Wspd) and the share of unavailable
turbines (Unav: not producing although Wspd > 2.5 m/s, i.e. zero power or feathered blades; low-wind standby is
not counted, see prepare_sdwpf.py) -- and builds a second, physics-based forecast of the farm power:

    p_phys = C * (1 - s_hat) * g(v_hat)

    C      installed capacity (201 MW)
    v_hat  forecast farm-mean wind speed (m/s)          = backbone output on the Wspd channel
    s_hat  forecast share of unavailable turbines in [0,1] = backbone output on the Unav channel
    g(.)   farm power curve per unit of available capacity, fitted on the valid TRAINING bins only:
           mean of OT / (C * (1 - Unav)) in 0.5 m/s wind-speed bins (the conditional mean, which is what an
           MSE-trained forecast should track), made non-decreasing (weighted isotonic regression), and
           evaluated by piecewise-linear interpolation (differentiable in v_hat). On SDWPF (training bins) it saturates at
           about 0.73 of the available capacity above ~10 m/s: even at high wind the farm rarely delivers its
           nameplate power (power limitation without feathering, wake losses), so g is an *effective* curve.

The final target forecast is a per-horizon-step convex combination of the direct and the physics forecast,

    y_hat_h = sigma(lambda_h) * y_direct_h + (1 - sigma(lambda_h)) * p_phys_h    (lambda_h learnable)

and training adds auxiliary MSE terms on the Wspd and Unav channels (multi-task), so that v_hat and s_hat are
actual forecasts of those quantities. Ablation switches: no physics path (aux only), no availability term
(s_hat := 0), no auxiliary loss, physics path only (sigma(lambda) := 0).

Caveats (for the paper): g is applied to a point forecast of v, so g(E v) != E g(v) (Jensen gap), and
(1 - s_hat) g(v_hat) is a product of two forecasts; the learnable gate absorbs part of that bias, so the physics
path is a physically structured second forecast, not an exact physical model. Without the auxiliary loss
(paf_noaux) v_hat and s_hat of channel-mixing backbones are free latent outputs.

Prior work with related ideas (see 文献对比_M2.md): UniWind (arXiv 2607.01670; day-ahead, NWP, latent states),
Zheng et al. 2025 Energies (Cp baseline + BiLSTM residual), direct vs indirect forecasting (Yakoub et al. 2023).
"""
import numpy as np
import torch
import torch.nn as nn

CURVE_BIN = 0.5      # m/s
CURVE_MIN_COUNT = 20  # bins with fewer training samples are dropped before the isotonic fit
CURVE_MIN_AVAIL = 0.2  # ignore bins where fewer than 20 % of turbines are available (ratio is unstable)
G_MAX = 1.05          # imputed farm totals can slightly exceed the nameplate capacity


def isotonic_increasing(y, w):
    """Weighted pool-adjacent-violators fit of a non-decreasing sequence."""
    blocks = []  # [value, weight, count]
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return np.array(out)


class FarmPowerCurve(object):
    """g(v): farm power per unit of available capacity as a function of farm-mean wind speed."""

    def __init__(self, knots, values):
        self.knots = np.asarray(knots, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)

    @classmethod
    def fit(cls, wspd, stop, power, capacity):
        """wspd (m/s), stop (share in [0,1]), power (MW): 1-D arrays of the valid TRAINING bins."""
        wspd, stop, power = (np.asarray(a, dtype=np.float64) for a in (wspd, stop, power))
        avail = 1.0 - stop
        ok = np.isfinite(wspd) & np.isfinite(avail) & np.isfinite(power) & (avail >= CURVE_MIN_AVAIL)
        ratio = power[ok] / (capacity * avail[ok])
        b = np.floor(wspd[ok] / CURVE_BIN).astype(int)
        knots, med, cnt = [], [], []
        for k in np.unique(b):
            m = b == k
            if m.sum() < CURVE_MIN_COUNT:
                continue
            knots.append((k + 0.5) * CURVE_BIN)
            med.append(np.mean(ratio[m]))
            cnt.append(m.sum())
        if len(knots) < 3:
            raise ValueError("power curve: too few populated wind-speed bins")
        g = np.clip(isotonic_increasing(np.array(med), np.array(cnt, dtype=np.float64)), 0.0, G_MAX)
        # anchor at 0 m/s so very low forecasts map to ~0 power
        knots = np.concatenate([[0.0], knots])
        g = np.concatenate([[0.0], g])
        return cls(knots, g)

    def __call__(self, v):
        return np.interp(v, self.knots, self.values)

    def to_dict(self):
        return dict(knots=[round(float(x), 4) for x in self.knots], values=[round(float(x), 6) for x in self.values])


def fit_curve_from_table(t):
    """Power curve from the valid TRAINING bins of a wpf.data.load_table() dict (needs Wspd, Unav, OT)."""
    cols = t["cols"]
    n_tr = t["b2"][0]
    rows = np.nonzero(t["valid"][:n_tr])[0]
    raw = t["data"][rows].astype(np.float64) * t["std"] + t["mean"]
    return FarmPowerCurve.fit(raw[:, cols.index("Wspd")], raw[:, cols.index("Unav")], raw[:, cols.index("OT")],
                              float(t["meta"]["capacity_mw"]))


def interp_torch(x, xp, fp):
    """Piecewise-linear interpolation, constant beyond the end knots; differentiable in x."""
    x = x.clamp(xp[0], xp[-1])
    idx = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    w = (x - x0) / (x1 - x0)
    return f0 + w * (f1 - f0)


class PAFHead(nn.Module):
    """Wraps a backbone with the physics-informed availability-aware fusion described above.

    Channel statistics are those of the global StandardScaler (wpf/data.py), so the head can move between
    scaled and physical units. forward() returns the backbone output with the OT channel replaced by the fused
    forecast; the last call's components are kept in self.last (for the auxiliary loss and for analysis).
    """

    def __init__(self, backbone, pred_len, cols, mean, std, curve, capacity,
                 use_phys=True, use_avail=True, phys_only=False, gate_init=0.0, second="phys"):
        super().__init__()
        self.backbone = backbone
        self.pred_len = pred_len
        self.i_y, self.i_v, self.i_s = cols.index("OT"), cols.index("Wspd"), cols.index("Unav")
        self.use_phys, self.use_avail, self.phys_only = use_phys, use_avail, phys_only
        self.capacity = float(capacity)
        for name, i in (("y", self.i_y), ("v", self.i_v), ("s", self.i_s)):
            self.register_buffer("mu_" + name, torch.tensor(float(mean[i])))
            self.register_buffer("sd_" + name, torch.tensor(float(std[i])))
        self.register_buffer("knots", torch.tensor(curve.knots, dtype=torch.float32))
        self.register_buffer("gvals", torch.tensor(curve.values, dtype=torch.float32))
        self.gate = nn.Parameter(torch.full((pred_len,), float(gate_init)))
        # control "ens2": the second forecast is a learned linear read-out of all backbone output channels
        # (same inputs as the physics path, same gate and aux loss, but no physics)
        self.second = second
        self.readout = nn.Linear(len(cols), 1) if second == "linear" else None
        self.last = {}

    def physics(self, out):
        """Physics forecast of OT in scaled units from the backbone's Wspd / Unav channel forecasts."""
        v = out[..., self.i_v] * self.sd_v + self.mu_v
        s = (out[..., self.i_s] * self.sd_s + self.mu_s).clamp(0.0, 1.0)
        avail = (1.0 - s) if self.use_avail else torch.ones_like(s)
        p_mw = self.capacity * avail * interp_torch(v, self.knots, self.gvals)
        return (p_mw - self.mu_y) / self.sd_y

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out = self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        direct = out[..., self.i_y]
        self.last = {"direct": direct}
        if not self.use_phys:
            return out
        phys = self.physics(out) if self.second == "phys" else self.readout(out).squeeze(-1)
        lam = torch.zeros_like(self.gate) if self.phys_only else torch.sigmoid(self.gate)
        fused = lam * direct + (1.0 - lam) * phys
        self.last.update(phys=phys, gate=lam)
        cols = [fused.unsqueeze(-1) if c == self.i_y else out[..., c:c + 1] for c in range(out.shape[-1])]
        return torch.cat(cols, dim=-1)

    def aux_targets(self):
        """Channel indices supervised by the auxiliary loss."""
        return [self.i_v, self.i_s]


HEAD_VARIANTS = {
    # name: (wrap with PAFHead, use_phys, use_avail, phys_only, aux loss weight)
    "none": (False, False, False, False, 0.0),
    "aux": (True, False, False, False, 1.0),
    "paf": (True, True, True, False, 1.0),
    "paf_noavail": (True, True, False, False, 1.0),
    "paf_noaux": (True, True, True, False, 0.0),
    "phys_only": (True, True, True, True, 1.0),
    "ens2": (True, True, False, False, 1.0),  # control: learned linear second head instead of physics
}
SECOND_PATH = {"ens2": "linear"}  # default "phys"
