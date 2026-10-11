# Gated fusion output head (PAF) for ultra-short-term wind farm power forecasting

Code for the paper *Gated fusion of direct and power-curve-based forecasts for ultra-short-term wind farm power
forecasting: a case study on a 201 MW farm* (Zekai Ye, Linxin Xu).

The PAF head (`wpf/physics.py`) is attached to a multivariate forecasting backbone. It supervises the backbone's
wind-speed and unavailable-share forecasts, maps them to a second power forecast through a farm power curve
estimated on the training segment, and fuses this forecast with the direct power forecast through one learned
weight per lead time. `ens2` is the control head with a learned linear second path.

All results refer to the 201 MW SDWPF farm. The two UK farms (Penmanshiel, Kelmarsh) were used only for the scope
check reported in the supplementary information.

## Requirements

Python 3.9, PyTorch >= 2.6 (experiments: PyTorch 2.11, CUDA 12.8, NVIDIA RTX 5080), see `requirements.txt`.

## Data

* SDWPF (Zhou et al., *Scientific Data* 11:649, 2024; CC BY 4.0): download `sdwpf_2001_2112_full.csv` and
  `sdwpf_turb_location_elevation.csv` from https://doi.org/10.6084/m9.figshare.24798654 into `../data_raw/sdwpf/`, then
  `python prepare_sdwpf.py --raw ../data_raw/sdwpf` -> `dataset/sdwpf_15min.csv` (70,176 15-min bins) and
  `dataset/sdwpf_meta.json`.
* Penmanshiel / Kelmarsh (Zenodo 10.5281/zenodo.16807304 and 10.5281/zenodo.16807551; CC BY 4.0): 2021 and 2022
  SCADA zips and the static file into `../data_raw/<farm>/`, then `python prepare_cubico.py extract --farm <farm>` and
  `python prepare_cubico.py build --farm <farm>` -> `datasets/<farm>/`.

## Running

```
python run.py --model TCN --pred_len 16 --features farm+cluster --seed 2021              # backbone
python run.py --model TCN --head paf --pred_len 16 --features farm+cluster --save_preds  # backbone + PAF
python grid.py --grid lrscan --gpus 0 1   # learning rate per backbone (validation)
python grid.py --grid main --gpus 0 1     # 8 backbones x {no head, PAF} x 2 input sets x 3 horizons x 3 seeds
python grid.py --grid control --gpus 0 1  # ens2 control head
python grid.py --grid ablation --gpus 0 1
python collect.py --grid main             # tables -> reports/main.md, reports/main.csv
python tools/m3_tests.py                  # pre-specified tests -> reports/m3_tests.md
python tools/seed_ensemble.py             # seed-averaging control -> reports/seed_ensemble.md
python tools/make_figures.py              # figures (and source data) -> figures/
python grid.py --grid main --farm penmanshiel ...; python tools/r3_tests.py   # two-farm scope check
```

`grid.py` runs jobs in parallel on the listed GPUs (`--slots` per GPU) and is written for Linux. `tests/` contains
unit tests (`python -m pytest tests`; five of them need the prepared SDWPF dataset); `python grid.py --grid smoke --gpus 0 --slots 2` is a short end-to-end check.

## Layout

* `wpf/`: data windows, training protocol (`exp.py`, `hparams.py`), classic backbones (`baselines.py`), PAF and
  ens2 heads (`physics.py`).
* `models/`, `layers/`, `utils/`: copied from the Time-Series-Library (MIT licence, `LICENSE_TSLib`); one marked
  correction in `models/TimeMixer.py` (time-feature pairing with channel independence).
* `tools/`: statistical tests, controls and figures. `reports/`: the result tables and test reports used in the
  paper.

## Licence

MIT (see `LICENSE`); the Time-Series-Library files keep their own MIT licence.
