# M3 pre-registered tests (grid main, 3 seeds, test segment)

Gain = 100 (1 - RMSE_A / RMSE_B) of seed-averaged forecasts; CI = 7-day block bootstrap; p = Ibragimov-Mueller group t (q = 8), Holm within family; daily / window DM = sensitivity; ACF = decay time (h) of the validation loss differential to 0.1.

## Primary family: proposed vs strongest baseline

| H | A | B | RMSE A | RMSE B | gain % [95% CI] | MAE gain % | groups A better | p (IM) | p Holm | p daily DM | p window DM | val ACF h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | farm+cluster/GRU+paf | farm+cluster/GRU | 15.16 | 15.17 | +0.09 [-0.71, +0.99] | +1.09 | 4/8 | 0.8446 | 0.8446 | 0.8114 | 0.8295 | 0.8 |
| 8 | farm+cluster/GRU+paf | farm+cluster/TCN | 20.60 | 20.87 | +1.28 [-0.54, +2.97] | +3.99 | 5/8 | 0.2466 | 0.4932 | 0.1748 | 0.1131 | 1.2 |
| 16 | farm+cluster/TCN+paf | farm+cluster/TCN | 27.98 | 28.55 | +2.02 [+1.08, +2.97] | +2.07 | 7/8 | 0.0089 | 0.0268 | 0.0003 | 0.0001 | 2.0 |

## Secondary family: backbone + PAF vs backbone

| H | A | B | RMSE A | RMSE B | gain % [95% CI] | MAE gain % | groups A better | p (IM) | p Holm | p daily DM | p window DM | val ACF h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | farm+cluster/LSTM+paf | farm+cluster/LSTM | 15.51 | 15.39 | -0.74 [-1.44, -0.02] | +0.54 | 2/8 | 0.0507 | 0.7602 | 0.0519 | 0.0285 | 0.5 |
| 4 | farm+cluster/GRU+paf | farm+cluster/GRU | 15.16 | 15.17 | +0.09 [-0.71, +0.99] | +1.09 | 4/8 | 0.8446 | 1.0000 | 0.8114 | 0.8295 | 0.8 |
| 4 | farm+cluster/TCN+paf | farm+cluster/TCN | 15.16 | 15.29 | +0.83 [-0.07, +1.77] | +1.33 | 5/8 | 0.1831 | 1.0000 | 0.0924 | 0.0530 | 0.8 |
| 4 | farm+cluster/Transformer+paf | farm+cluster/Transformer | 15.17 | 15.66 | +3.11 [+2.28, +3.95] | +4.20 | 8/8 | 0.0049 | 0.0957 | 0.0000 | 0.0000 | 0.8 |
| 4 | farm+cluster/DLinear+paf | farm+cluster/DLinear | 16.03 | 16.00 | -0.14 [-1.55, +1.03] | -4.94 | 4/8 | 0.8685 | 1.0000 | 0.8460 | 0.8503 | 0.5 |
| 4 | farm+cluster/PatchTST+paf | farm+cluster/PatchTST | 16.18 | 16.41 | +1.39 [-1.36, +3.62] | +2.75 | 6/8 | 0.4018 | 1.0000 | 0.2872 | 0.2072 | 0.8 |
| 4 | farm+cluster/iTransformer+paf | farm+cluster/iTransformer | 16.06 | 16.14 | +0.49 [-1.16, +1.74] | -0.16 | 5/8 | 0.6075 | 1.0000 | 0.5203 | 0.5081 | 0.8 |
| 4 | farm+cluster/TimeMixer+paf | farm+cluster/TimeMixer | 15.60 | 16.97 | +8.06 [+6.38, +9.73] | +16.31 | 8/8 | 0.0002 | 0.0057 | 0.0000 | 0.0000 | 0.5 |
| 8 | farm+cluster/LSTM+paf | farm+cluster/LSTM | 21.31 | 21.23 | -0.38 [-1.27, +0.42] | +0.56 | 3/8 | 0.3076 | 1.0000 | 0.3772 | 0.4369 | 1.2 |
| 8 | farm+cluster/GRU+paf | farm+cluster/GRU | 20.60 | 20.84 | +1.12 [+0.14, +2.07] | +0.83 | 5/8 | 0.0659 | 0.7888 | 0.0533 | 0.0494 | 1.5 |
| 8 | farm+cluster/TCN+paf | farm+cluster/TCN | 20.68 | 20.87 | +0.89 [+0.11, +1.62] | +3.37 | 7/8 | 0.0695 | 0.7888 | 0.0311 | 0.0275 | 1.0 |
| 8 | farm+cluster/Transformer+paf | farm+cluster/Transformer | 21.17 | 21.49 | +1.50 [+0.29, +2.74] | +1.11 | 6/8 | 0.0528 | 0.7602 | 0.0293 | 0.0410 | 1.2 |
| 8 | farm+cluster/DLinear+paf | farm+cluster/DLinear | 21.55 | 21.80 | +1.14 [+0.03, +2.13] | -0.18 | 6/8 | 0.1414 | 1.0000 | 0.0472 | 0.0765 | 0.5 |
| 8 | farm+cluster/PatchTST+paf | farm+cluster/PatchTST | 21.80 | 22.26 | +2.07 [+0.50, +3.43] | +4.89 | 6/8 | 0.0475 | 0.7602 | 0.0113 | 0.0104 | 1.2 |
| 8 | farm+cluster/iTransformer+paf | farm+cluster/iTransformer | 21.50 | 21.98 | +2.19 [+0.99, +3.45] | +3.34 | 6/8 | 0.0076 | 0.1363 | 0.0012 | 0.0019 | 1.5 |
| 8 | farm+cluster/TimeMixer+paf | farm+cluster/TimeMixer | 22.52 | 22.88 | +1.56 [+0.11, +2.84] | +4.23 | 6/8 | 0.0683 | 0.7888 | 0.0292 | 0.0568 | 1.8 |
| 16 | farm+cluster/LSTM+paf | farm+cluster/LSTM | 28.26 | 28.34 | +0.30 [-0.79, +1.30] | -2.79 | 4/8 | 0.5475 | 1.0000 | 0.5956 | 0.6398 | 3.5 |
| 16 | farm+cluster/GRU+paf | farm+cluster/GRU | 27.35 | 27.67 | +1.16 [-0.23, +2.49] | +4.60 | 6/8 | 0.0657 | 0.7888 | 0.1405 | 0.0584 | 2.8 |
| 16 | farm+cluster/TCN+paf | farm+cluster/TCN | 27.98 | 28.55 | +2.02 [+1.08, +2.97] | +2.07 | 7/8 | 0.0089 | 0.1517 | 0.0003 | 0.0001 | 2.0 |
| 16 | farm+cluster/Transformer+paf | farm+cluster/Transformer | 28.26 | 28.95 | +2.38 [+1.31, +3.53] | -0.55 | 7/8 | 0.0048 | 0.0957 | 0.0004 | 0.0003 | 2.5 |
| 16 | farm+cluster/DLinear+paf | farm+cluster/DLinear | 28.56 | 29.04 | +1.64 [+0.68, +2.61] | +2.07 | 8/8 | 0.0010 | 0.0222 | 0.0005 | 0.0031 | 1.8 |
| 16 | farm+cluster/PatchTST+paf | farm+cluster/PatchTST | 29.20 | 29.64 | +1.50 [+0.21, +2.68] | +5.46 | 7/8 | 0.0028 | 0.0584 | 0.0262 | 0.0514 | 2.2 |
| 16 | farm+cluster/iTransformer+paf | farm+cluster/iTransformer | 28.89 | 29.22 | +1.14 [-0.08, +2.36] | +2.94 | 6/8 | 0.0510 | 0.7602 | 0.1001 | 0.0873 | 2.5 |
| 16 | farm+cluster/TimeMixer+paf | farm+cluster/TimeMixer | 28.60 | 44.19 | +35.29 [+32.57, +37.84] | +33.90 | 8/8 | 0.0009 | 0.0204 | 0.0000 | 0.0000 | 3.8 |

## Third family (revision 2): backbone + PAF vs backbone + ens2 control

| H | A | B | RMSE A | RMSE B | gain % [95% CI] | MAE gain % | groups A better | p (IM) | p Holm | p daily DM | p window DM | val ACF h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | farm+cluster/LSTM+paf | farm+cluster/LSTM+ens2 | 15.51 | 15.54 | +0.22 [-0.67, +1.03] | +1.52 | 5/8 | 0.5868 | 1.0000 | 0.6179 | 0.5584 | 0.5 |
| 4 | farm+cluster/GRU+paf | farm+cluster/GRU+ens2 | 15.16 | 15.26 | +0.68 [+0.15, +1.24] | +1.59 | 7/8 | 0.0310 | 0.5844 | 0.0100 | 0.0182 | 0.8 |
| 4 | farm+cluster/TCN+paf | farm+cluster/TCN+ens2 | 15.16 | 15.37 | +1.33 [+0.46, +2.24] | +4.08 | 8/8 | 0.0229 | 0.4816 | 0.0059 | 0.0015 | 1.0 |
| 4 | farm+cluster/Transformer+paf | farm+cluster/Transformer+ens2 | 15.17 | 15.32 | +0.98 [+0.49, +1.45] | -0.00 | 6/8 | 0.0292 | 0.5844 | 0.0016 | 0.0008 | 0.8 |
| 4 | farm+cluster/DLinear+paf | farm+cluster/DLinear+ens2 | 16.03 | 16.04 | +0.06 [-1.66, +1.30] | -6.04 | 5/8 | 0.9452 | 1.0000 | 0.9072 | 0.9170 | 0.8 |
| 4 | farm+cluster/PatchTST+paf | farm+cluster/PatchTST+ens2 | 16.18 | 16.64 | +2.80 [+0.69, +5.03] | +7.25 | 6/8 | 0.0840 | 1.0000 | 0.0124 | 0.0039 | 0.8 |
| 4 | farm+cluster/iTransformer+paf | farm+cluster/iTransformer+ens2 | 16.06 | 16.30 | +1.45 [-0.12, +3.30] | +4.54 | 6/8 | 0.1107 | 1.0000 | 0.0884 | 0.0624 | 0.8 |
| 4 | farm+cluster/TimeMixer+paf | farm+cluster/TimeMixer+ens2 | 15.60 | 33.23 | +53.07 [+48.71, +57.62] | +57.01 | 8/8 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 4.2 |
| 8 | farm+cluster/LSTM+paf | farm+cluster/LSTM+ens2 | 21.31 | 21.24 | -0.34 [-1.93, +1.19] | -0.36 | 4/8 | 0.7561 | 1.0000 | 0.6750 | 0.5919 | 1.5 |
| 8 | farm+cluster/GRU+paf | farm+cluster/GRU+ens2 | 20.60 | 20.64 | +0.19 [-0.74, +1.00] | +0.70 | 6/8 | 0.7362 | 1.0000 | 0.6927 | 0.5970 | 1.8 |
| 8 | farm+cluster/TCN+paf | farm+cluster/TCN+ens2 | 20.68 | 20.72 | +0.16 [-0.72, +1.02] | +0.88 | 3/8 | 0.7413 | 1.0000 | 0.7014 | 0.6828 | 1.0 |
| 8 | farm+cluster/Transformer+paf | farm+cluster/Transformer+ens2 | 21.17 | 21.21 | +0.21 [-0.88, +1.27] | +0.15 | 5/8 | 0.7277 | 1.0000 | 0.7069 | 0.6692 | 1.0 |
| 8 | farm+cluster/DLinear+paf | farm+cluster/DLinear+ens2 | 21.55 | 21.66 | +0.49 [-1.56, +2.17] | -4.18 | 6/8 | 0.6798 | 1.0000 | 0.5659 | 0.5094 | 1.2 |
| 8 | farm+cluster/PatchTST+paf | farm+cluster/PatchTST+ens2 | 21.80 | 22.14 | +1.57 [-0.14, +3.32] | +2.15 | 6/8 | 0.1169 | 1.0000 | 0.0782 | 0.0432 | 1.5 |
| 8 | farm+cluster/iTransformer+paf | farm+cluster/iTransformer+ens2 | 21.50 | 22.04 | +2.44 [+0.94, +4.16] | +5.56 | 8/8 | 0.0012 | 0.0278 | 0.0051 | 0.0010 | 1.2 |
| 8 | farm+cluster/TimeMixer+paf | farm+cluster/TimeMixer+ens2 | 22.52 | 21.74 | -3.61 [-5.79, -1.81] | +0.77 | 1/8 | 0.0203 | 0.4466 | 0.0004 | 0.0003 | 1.2 |
| 16 | farm+cluster/LSTM+paf | farm+cluster/LSTM+ens2 | 28.26 | 28.08 | -0.62 [-2.00, +0.57] | -3.99 | 2/8 | 0.4094 | 1.0000 | 0.3093 | 0.3003 | 2.8 |
| 16 | farm+cluster/GRU+paf | farm+cluster/GRU+ens2 | 27.35 | 27.39 | +0.15 [-0.62, +0.95] | +0.29 | 4/8 | 0.6335 | 1.0000 | 0.7510 | 0.6968 | 4.2 |
| 16 | farm+cluster/TCN+paf | farm+cluster/TCN+ens2 | 27.98 | 28.08 | +0.37 [-0.75, +1.44] | -0.12 | 5/8 | 0.4654 | 1.0000 | 0.4571 | 0.5290 | 5.2 |
| 16 | farm+cluster/Transformer+paf | farm+cluster/Transformer+ens2 | 28.26 | 28.71 | +1.56 [+0.17, +2.90] | +3.72 | 7/8 | 0.0298 | 0.5844 | 0.0266 | 0.0167 | 3.5 |
| 16 | farm+cluster/DLinear+paf | farm+cluster/DLinear+ens2 | 28.56 | 28.76 | +0.70 [-1.24, +2.57] | -0.19 | 6/8 | 0.5924 | 1.0000 | 0.4918 | 0.3601 | 2.2 |
| 16 | farm+cluster/PatchTST+paf | farm+cluster/PatchTST+ens2 | 29.20 | 29.29 | +0.31 [-0.99, +1.74] | +4.08 | 4/8 | 0.5726 | 1.0000 | 0.7064 | 0.6137 | 2.5 |
| 16 | farm+cluster/iTransformer+paf | farm+cluster/iTransformer+ens2 | 28.89 | 28.74 | -0.53 [-2.01, +0.94] | +7.33 | 5/8 | 0.4500 | 1.0000 | 0.4369 | 0.4499 | 2.5 |
| 16 | farm+cluster/TimeMixer+paf | farm+cluster/TimeMixer+ens2 | 28.60 | 28.59 | -0.03 [-1.71, +1.67] | +6.68 | 4/8 | 0.9742 | 1.0000 | 0.9256 | 0.9736 | 2.2 |

