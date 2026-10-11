"""Training-protocol checks added after code review rev-001 (CPU, seconds).

Run: python tests/test_protocol.py
"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE)

import torch  # noqa: E402

from wpf.exp import adjust_learning_rate, build_model  # noqa: E402
from wpf.hparams import get_hparams  # noqa: E402


def test_lr_schedule_matches_tslib():
    """TSLib: for epoch0 in range(E): train; adjust(epoch0 + 1)  ->  lr of epoch0 = lr0 * 0.5 ** max(0, epoch0 - 1)."""
    args = SimpleNamespace(lradj="type1", learning_rate=1e-3, train_epochs=6)
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=args.learning_rate)
    used = []
    for ep in range(1, 6):  # our loop: 1-based epoch, adjust(ep) after it
        used.append(opt.param_groups[0]["lr"])
        adjust_learning_rate(opt, ep, args)
    tslib = [1e-3 * 0.5 ** max(0, e - 1) for e in range(5)]
    assert all(abs(a - b) < 1e-15 for a, b in zip(used, tslib)), (used, tslib)
    print("ok lr schedule = TSLib type1:", ["%.2e" % v for v in used])


def test_timemixer_batch_independence():
    """Each sample's forecast must not depend on the other samples in the batch (time marks paired correctly)."""
    torch.manual_seed(0)
    n_vars, L, H = 5, 96, 4
    hp = get_hparams("TimeMixer", H, n_vars, seq_len=L)
    model = build_model(SimpleNamespace(**hp)).eval()
    B = 3
    x = torch.randn(B, L, n_vars)
    xm = torch.randn(B, L, 5)  # timeF marks for freq "t"
    dec = torch.zeros(B, H, n_vars)
    dm = torch.randn(B, H, 5)
    with torch.no_grad():
        full = model(x, xm, dec, dm)[:, -H:, -1]
        one = torch.cat([model(x[i:i + 1], xm[i:i + 1], dec[i:i + 1], dm[i:i + 1])[:, -H:, -1] for i in range(B)])
    assert torch.allclose(full, one, atol=1e-5), (full - one).abs().max()
    print("ok TimeMixer: batched forecast = per-sample forecast (max diff %.1e)" % (full - one).abs().max())


if __name__ == "__main__":
    test_lr_schedule_matches_tslib()
    test_timemixer_batch_independence()
    print("OK test_protocol")
