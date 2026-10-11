"""Hyperparameters per model: architecture fixed, one shared training protocol, learning rate chosen on validation.

Architecture: TSLib models follow the TSLib / official ETTm1 scripts (15-min data, like ours); classic baselines
use common settings for direct multi-step RNN/TCN/Transformer forecasters. The same for every horizon and
feature set; no per-horizon tuning.

Training protocol (identical for all models, revised 2026-10-09 after code review rev-001): Adam, MSE on the
scaled target, TSLib lradj "type1" (lr halves after every epoch from the second on), at most 20 epochs, early
stopping with patience 5 on the validation MSE. The learning rate of each model is picked from LR_GRID by the
`lrscan` grid (farm+cluster inputs, seed 2021, all three horizons; lowest mean validation RMSE) and then fixed
for every run of that model (grid.py reads the choice from results/lrscan, see select_lr()). The values below
are the defaults used when no scan result is given (they were the M2 settings).
"""
BASE = dict(
    task_name="long_term_forecast", features="MS", target="OT", freq="t", embed="timeF",
    seq_len=96, label_len=48, pred_len=16,
    d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, moving_avg=25, factor=1, dropout=0.1,
    activation="gelu", top_k=5, num_class=0, channel_independence=1, decomp_method="moving_avg", use_norm=1,
    down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
    train_epochs=20, batch_size=32, patience=5, learning_rate=1e-4, lradj="type1",
)

LR_GRID = [1e-2, 1e-3, 1e-4]

MODELS = {
    "DLinear": {},
    "PatchTST": dict(e_layers=1, n_heads=2, factor=3),
    "iTransformer": dict(d_model=128, d_ff=128, e_layers=2),
    # channel-independent as in the official ETTm1 script (OT forecast uses only the OT history)
    "TimeMixer": dict(d_model=16, d_ff=32, e_layers=2, down_sampling_layers=3, down_sampling_method="avg",
                      down_sampling_window=2, label_len=0, learning_rate=0.01, batch_size=16),
    "LSTM": dict(d_model=128, e_layers=2, learning_rate=1e-3),
    "GRU": dict(d_model=128, e_layers=2, learning_rate=1e-3),
    "TCN": dict(d_model=64, learning_rate=1e-3),
    "Transformer": dict(d_model=64, n_heads=4, d_ff=128, e_layers=2, learning_rate=1e-3),
}

# models whose target forecast can use other input channels (others are channel-independent for OT)
CHANNEL_MIXING = ["iTransformer", "LSTM", "GRU", "TCN", "Transformer"]


def get_hparams(model, pred_len, n_vars, **over):
    hp = dict(BASE)
    hp.update(MODELS[model])
    hp.update(over)
    hp["model"], hp["pred_len"] = model, pred_len
    hp["enc_in"] = hp["dec_in"] = hp["c_out"] = n_vars
    return hp
