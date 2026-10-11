"""Classic deep baselines with the TSLib model interface: forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, H, C].

All are direct multi-step (one linear head emits the H steps) and see every input channel.
  LSTM / GRU  : 2-layer recurrent encoder, last hidden state -> linear head
  TCN         : Bai et al. (2018) stack of dilated causal residual blocks, last step -> linear head
  Transformer : encoder-only (sinusoidal positions), flattened tokens -> linear head
"""
import math

import torch
import torch.nn as nn


class _Head(nn.Module):
    def __init__(self, d_in, H, C):
        super().__init__()
        self.H, self.C = H, C
        self.lin = nn.Linear(d_in, H * C)

    def forward(self, h):
        return self.lin(h).view(h.shape[0], self.H, self.C)


class RNNModel(nn.Module):
    cell = None

    def __init__(self, cfg):
        super().__init__()
        self.rnn = self.cell(cfg.enc_in, cfg.d_model, num_layers=cfg.e_layers, batch_first=True,
                             dropout=cfg.dropout if cfg.e_layers > 1 else 0.0)
        self.head = _Head(cfg.d_model, cfg.pred_len, cfg.c_out)

    def forward(self, x, *_):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class LSTM(RNNModel):
    cell = nn.LSTM


class GRU(RNNModel):
    cell = nn.GRU


class _TBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dil, drop):
        super().__init__()
        self.pad = (k - 1) * dil
        self.c1 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_in, c_out, k, dilation=dil))
        self.c2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(c_out, c_out, k, dilation=dil))
        self.drop = nn.Dropout(drop)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        y = self.drop(torch.relu(self.c2(nn.functional.pad(y, (self.pad, 0)))))
        return torch.relu(y + self.down(x))


class TCN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        k = 3
        # receptive field 1 + 2(k-1)(2^n - 1) >= seq_len
        n = max(1, math.ceil(math.log2((cfg.seq_len - 1) / (2 * (k - 1)) + 1)))
        ch = [cfg.enc_in] + [cfg.d_model] * n
        self.net = nn.Sequential(*[_TBlock(ch[i], ch[i + 1], k, 2 ** i, cfg.dropout) for i in range(n)])
        self.head = _Head(cfg.d_model, cfg.pred_len, cfg.c_out)

    def forward(self, x, *_):
        return self.head(self.net(x.transpose(1, 2))[:, :, -1])


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(cfg.enc_in, d)
        pe = torch.zeros(cfg.seq_len, d)
        pos = torch.arange(cfg.seq_len).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe[None])
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.e_layers)
        self.head = _Head(d * cfg.seq_len, cfg.pred_len, cfg.c_out)

    def forward(self, x, *_):
        h = self.enc(self.inp(x) + self.pe)
        return self.head(h.flatten(1))


REGISTRY = {"LSTM": LSTM, "GRU": GRU, "TCN": TCN, "Transformer": Transformer}
