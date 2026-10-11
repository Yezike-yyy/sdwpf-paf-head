"""wpf/physics.py: isotonic fit, power curve on real training bins, interpolation, PAF head wiring (CPU).

Run: python tests/test_physics.py
"""
import os
import sys

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from wpf.data import load_table  # noqa: E402
from wpf.physics import PAFHead, fit_curve_from_table, interp_torch, isotonic_increasing  # noqa: E402

ROOT = os.path.join(HERE, "dataset")


def test_isotonic():
    y = np.array([1.0, 3.0, 2.0, 4.0, 3.5, 5.0])
    w = np.ones_like(y)
    z = isotonic_increasing(y, w)
    assert np.all(np.diff(z) >= -1e-12)
    assert np.allclose(z, [1.0, 2.5, 2.5, 3.75, 3.75, 5.0])
    assert np.allclose(isotonic_increasing(np.arange(5.0), np.ones(5)), np.arange(5.0))
    print("ok isotonic regression")


def train_curve():
    t = load_table(ROOT, "farm+cluster")
    return t, fit_curve_from_table(t)


def test_curve_on_training_data():
    t, curve = train_curve()
    g = curve(np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0]))
    print("power curve g(v) at 2,4,6,8,10,12,15 m/s:", np.round(g, 3))
    assert np.all(np.diff(curve.values) >= -1e-12)
    # effective farm curve: ~0 below cut-in, saturating well above half of the available capacity
    assert 0.0 <= g[0] < 0.05 and 0.5 < g[-1] <= 1.05 and g[3] > 3 * g[1], "implausible farm power curve"
    # fitted from training rows only: changing validation/test values must not change it
    t2 = dict(t)
    t2["data"] = t["data"].copy()
    t2["data"][t["b2"][0]:] += 5.0
    assert np.array_equal(fit_curve_from_table(t2).values, curve.values)
    print("ok power curve (monotone, plausible shape, %d knots up to %.1f m/s)" % (len(curve.knots),
                                                                                  curve.knots[-1]))


def test_interp_torch():
    xp = torch.tensor([0.0, 1.0, 3.0, 6.0])
    fp = torch.tensor([0.0, 0.2, 0.8, 1.0])
    x = torch.tensor([-1.0, 0.5, 2.0, 5.0, 9.0], requires_grad=True)
    y = interp_torch(x, xp, fp)
    assert torch.allclose(y, torch.tensor(np.interp(x.detach().numpy(), xp.numpy(), fp.numpy()),
                                          dtype=torch.float32))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([0.0, 0.2, 0.3, 0.2 / 3, 0.0]), atol=1e-6)
    print("ok differentiable interpolation (values = np.interp, slopes per segment)")


class Echo(torch.nn.Module):
    """Fake backbone: returns the last H steps of a fixed tensor (B, H, C)."""

    def __init__(self, out):
        super().__init__()
        self.out = out
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args):
        return self.out + 0.0 * self.w


def test_head_wiring():
    t, curve = train_curve()
    cols, mean, std = t["cols"], t["mean"], t["std"]
    B, H, C = 2, 4, len(cols)
    out = torch.randn(B, H, C)
    head = PAFHead(Echo(out), H, cols, mean, std, curve, 201.0)
    y = head(None, None, None, None)
    iy, iv, i_s = cols.index("OT"), cols.index("Wspd"), cols.index("Unav")
    # untouched channels pass through, OT is the 50/50 mix at gate 0
    keep = [c for c in range(C) if c != iy]
    assert torch.allclose(y[..., keep], out[..., keep])
    v = out[..., iv] * std[iv] + mean[iv]
    s = (out[..., i_s] * std[i_s] + mean[i_s]).clamp(0, 1)
    p = 201.0 * (1 - s) * torch.tensor(curve(v.numpy()), dtype=torch.float32)
    phys = (p - mean[iy]) / std[iy]
    assert torch.allclose(y[..., iy], 0.5 * out[..., iy] + 0.5 * phys, atol=1e-4)
    # phys_only: gate is 0 -> pure physics forecast; no-availability: s := 0
    h2 = PAFHead(Echo(out), H, cols, mean, std, curve, 201.0, phys_only=True)
    assert torch.allclose(h2(None, None, None, None)[..., iy], phys, atol=1e-4)
    h3 = PAFHead(Echo(out), H, cols, mean, std, curve, 201.0, use_avail=False)
    p3 = 201.0 * torch.tensor(curve(v.numpy()), dtype=torch.float32)
    assert torch.allclose(h3(None, None, None, None)[..., iy], 0.5 * out[..., iy] + 0.5 * (p3 - mean[iy]) / std[iy],
                          atol=1e-4)
    # gradients reach the gate
    y[..., iy].sum().backward()
    assert head.gate.grad is not None and torch.isfinite(head.gate.grad).all()
    print("ok PAF head (pass-through, 50/50 fusion at init, phys-only, no-availability, gate gradient)")


if __name__ == "__main__":
    test_isotonic()
    test_curve_on_training_data()
    test_interp_torch()
    test_head_wiring()
    print("OK test_physics")
