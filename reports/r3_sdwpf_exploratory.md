# Plan revision 3 statistics on SDWPF (EXPLORATORY: discovery data, not in the fourth/fifth family)

Gain = 100 (1 - RMSE_A / RMSE_B), backbone mean over LSTM, GRU, TCN, Transformer, DLinear, PatchTST, iTransformer; RMSE from seed-averaged squared errors. CI = joint 7-day block bootstrap (2000 reps). p = registered two-sided group t-test: the statistic is recomputed within each of q = 8 contiguous groups of the test windows (H2: groups with >= 50 flat-high window-steps). Calibration of the test: reports/r3_placebo_sdwpf.md.

## H1: backbone + PAF vs backbone (G > 0)

| farm | H | windows | G % [95% CI] | groups G>0 | t | p | p Holm |
|---|---|---|---|---|---|---|---|
| sdwpf | 4 | 12921 | +0.72 [-0.07, +1.41] | 6/8 | 1.30 | 0.2334 | - |
| sdwpf | 8 | 12881 | +1.22 [+0.66, +1.77] | 7/8 | 3.28 | 0.0134 | - |
| sdwpf | 16 | 12801 | +1.45 [+0.90, +2.01] | 7/8 | 4.78 | 0.0020 | - |

## H2: regime complementarity, theta = gain(PAF vs ens2 | flat-high) - gain(PAF vs ens2 | steep) > 0

| farm | H | flat-high % | gain flat-high % | gain steep % | theta [95% CI] | testable | groups used | groups theta>0 | t | p | p Holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sdwpf | 4 | 12.6 | +2.33 | -0.55 | +2.88 [+1.17, +4.26] | yes | 8 | 6 | 1.35 | 0.2179 | - |
| sdwpf | 8 | 12.6 | +2.50 | -0.49 | +2.99 [+1.29, +4.35] | yes | 8 | 7 | 1.41 | 0.2004 | - |
| sdwpf | 16 | 12.6 | +2.61 | -2.32 | +4.93 [+3.57, +6.07] | yes | 8 | 8 | 6.41 | 0.0004 | - |

## Descriptive: backbone-mean gains by region (%)

### sdwpf, H = 4

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 43.9 | +5.45 [+3.41, +7.90] | +3.63 [+2.43, +4.89] | -2.29 [-4.82, -0.56] |
| ex post steep | 43.5 | -0.55 [-1.07, +0.03] | -0.23 [-0.74, +0.23] | +0.32 [-0.28, +0.84] |
| ex post flat-high | 12.6 | +2.33 [+0.68, +3.68] | +1.46 [-0.70, +3.30] | -0.89 [-2.17, +0.44] |
| ex ante flat-low | 43.9 | +0.97 [+0.08, +2.09] | +0.47 [-0.26, +1.26] | -0.57 [-1.41, +0.19] |
| ex ante steep | 43.5 | +0.45 [-0.13, +1.09] | -0.15 [-0.67, +0.42] | -0.61 [-1.36, +0.03] |
| ex ante flat-high | 12.6 | +2.26 [+0.64, +3.69] | +2.41 [+0.25, +4.29] | +0.15 [-1.27, +1.64] |
| all | 100.0 | +1.07 [+0.36, +1.74] | +0.72 [-0.07, +1.41] | -0.36 [-1.11, +0.34] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 15.395 | 15.508 | 15.542 | -0.74 | +0.22 | -0.96 |
| GRU | 3 | 15.174 | 15.161 | 15.265 | +0.09 | +0.68 | -0.60 |
| TCN | 3 | 15.291 | 15.163 | 15.367 | +0.83 | +1.33 | -0.50 |
| Transformer | 3 | 15.655 | 15.169 | 15.319 | +3.11 | +0.98 | +2.15 |
| DLinear | 3 | 16.003 | 16.025 | 16.035 | -0.14 | +0.06 | -0.20 |
| PatchTST | 3 | 16.407 | 16.179 | 16.644 | +1.39 | +2.80 | -1.44 |
| iTransformer | 3 | 16.142 | 16.063 | 16.299 | +0.49 | +1.45 | -0.97 |
| TimeMixer (separate) | 2 | 17.342 | 15.576 | 16.036 | +10.19 | +2.87 | +7.53 |

Proposed (val-selected) GRU+paf vs strongest baseline (val-selected) GRU: RMSE 15.161 vs 15.174, gain +0.09% [-0.71, +0.99], p (group t) 0.8446 (descriptive).
Diverged runs (val RMSE > 29.721): {'TimeMixer+ens2 s2021': 50.635}

### sdwpf, H = 8

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 43.9 | +0.83 [-0.89, +2.78] | +3.11 [+2.37, +4.06] | +2.21 [+0.24, +4.05] |
| ex post steep | 43.5 | -0.49 [-0.97, +0.05] | -0.21 [-0.60, +0.20] | +0.27 [-0.20, +0.69] |
| ex post flat-high | 12.6 | +2.50 [+0.86, +3.78] | +2.69 [+1.13, +4.10] | +0.15 [-0.72, +1.25] |
| ex ante flat-low | 43.8 | -0.36 [-1.01, +0.30] | +1.01 [+0.34, +1.92] | +1.33 [+0.67, +2.21] |
| ex ante steep | 43.5 | +0.25 [-0.28, +0.87] | +0.15 [-0.21, +0.54] | -0.10 [-0.73, +0.43] |
| ex ante flat-high | 12.6 | +2.14 [+0.30, +3.57] | +3.58 [+2.10, +4.91] | +1.47 [+0.50, +2.76] |
| all | 100.0 | +0.68 [-0.11, +1.44] | +1.22 [+0.66, +1.77] | +0.54 [-0.01, +1.08] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 21.234 | 21.314 | 21.243 | -0.38 | -0.34 | -0.04 |
| GRU | 3 | 20.837 | 20.604 | 20.643 | +1.12 | +0.19 | +0.93 |
| TCN | 3 | 20.870 | 20.684 | 20.719 | +0.89 | +0.16 | +0.72 |
| Transformer | 3 | 21.492 | 21.170 | 21.215 | +1.50 | +0.21 | +1.29 |
| DLinear | 3 | 21.803 | 21.554 | 21.661 | +1.14 | +0.49 | +0.65 |
| PatchTST | 3 | 22.259 | 21.797 | 22.144 | +2.07 | +1.57 | +0.52 |
| iTransformer | 3 | 21.979 | 21.499 | 22.037 | +2.19 | +2.44 | -0.26 |
| TimeMixer (separate) | 3 | 22.878 | 22.522 | 21.737 | +1.56 | -3.61 | +4.99 |

Proposed (val-selected) GRU+paf vs strongest baseline (val-selected) TCN: RMSE 20.604 vs 20.870, gain +1.28% [-0.54, +2.97], p (group t) 0.2466 (descriptive).
Diverged runs (val RMSE > 40.084): none

### sdwpf, H = 16

| region | share % | PAF vs ens2 | PAF vs backbone | ens2 vs backbone |
|---|---|---|---|---|
| ex post flat-low | 43.9 | +1.79 [+0.02, +3.60] | +2.68 [+1.56, +3.66] | +0.64 [-1.11, +2.12] |
| ex post steep | 43.5 | -2.32 [-2.97, -1.46] | -0.17 [-0.67, +0.42] | +2.06 [+1.31, +2.67] |
| ex post flat-high | 12.6 | +2.61 [+1.10, +3.82] | +2.60 [+1.24, +3.76] | -0.02 [-1.00, +1.04] |
| ex ante flat-low | 43.8 | -0.77 [-1.41, -0.07] | +1.23 [+0.54, +2.01] | +1.98 [+1.39, +2.60] |
| ex ante steep | 43.5 | -0.47 [-1.19, +0.34] | -0.01 [-0.58, +0.54] | +0.46 [-0.38, +1.22] |
| ex ante flat-high | 12.7 | +2.91 [+1.07, +4.29] | +4.98 [+3.54, +6.47] | +2.13 [+0.63, +3.88] |
| all | 100.0 | +0.28 [-0.61, +1.18] | +1.45 [+0.90, +2.01] | +1.17 [+0.48, +1.79] |

| backbone | seeds | RMSE bb | RMSE +PAF | RMSE +ens2 | PAF vs bb % | PAF vs ens2 % | ens2 vs bb % |
|---|---|---|---|---|---|---|---|
| LSTM | 3 | 28.341 | 28.258 | 28.082 | +0.30 | -0.62 | +0.91 |
| GRU | 3 | 27.673 | 27.353 | 27.393 | +1.16 | +0.15 | +1.01 |
| TCN | 3 | 28.553 | 27.976 | 28.079 | +2.02 | +0.37 | +1.66 |
| Transformer | 3 | 28.949 | 28.259 | 28.707 | +2.38 | +1.56 | +0.84 |
| DLinear | 3 | 29.037 | 28.562 | 28.764 | +1.64 | +0.70 | +0.94 |
| PatchTST | 3 | 29.642 | 29.198 | 29.290 | +1.50 | +0.31 | +1.19 |
| iTransformer | 3 | 29.225 | 28.891 | 28.740 | +1.14 | -0.53 | +1.66 |
| TimeMixer (separate) | 2 | 30.011 | 28.583 | 28.779 | +4.76 | +0.68 | +4.10 |

Proposed (val-selected) TCN+paf vs strongest baseline (val-selected) TCN: RMSE 27.976 vs 28.553, gain +2.02% [+1.08, +2.97], p (group t) 0.0089 (descriptive).
Diverged runs (val RMSE > 52.893): {'TimeMixer s2021': 74.735}

