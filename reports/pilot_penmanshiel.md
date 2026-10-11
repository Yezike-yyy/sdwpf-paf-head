# Data pilot: Penmanshiel (raw 10-min SCADA, 2021-2022)

Source: Zenodo 10.5281/zenodo.16807304 (v3), CC BY 4.0. Profile only: nothing filtered or cleaned; threshold statistics use the training segment (before 2022-05-27 00:00) only.

## 1. Inventory

- zips: Penmanshiel_SCADA_2021_WT01-10_4460.zip, Penmanshiel_SCADA_2021_WT11-15_4461.zip, Penmanshiel_SCADA_2022_WT01-10_4462.zip, Penmanshiel_SCADA_2022_WT11-15_4463.zip
- turbine files: 28; other files in the zips: Status_Penmanshiel_01_2021-01-01_-_2022-01-01_1042.csv, Status_Penmanshiel_02_2021-01-01_-_2022-01-01_1043.csv, Status_Penmanshiel_04_2021-01-01_-_2022-01-01_1044.csv, Status_Penmanshiel_05_2021-01-01_-_2022-01-01_1045.csv, Status_Penmanshiel_06_2021-01-01_-_2022-01-01_1046.csv, Status_Penmanshiel_07_2021-01-01_-_2022-01-01_1047.csv, Status_Penmanshiel_08_2021-01-01_-_2022-01-01_1048.csv, Status_Penmanshiel_09_2021-01-01_-_2022-01-01_1049.csv, Status_Penmanshiel_10_2021-01-01_-_2022-01-01_1050.csv, Status_Penmanshiel_11_2021-01-01_-_2022-01-01_1051.csv, Status_Penmanshiel_12_2021-01-01_-_2022-01-01_1052.csv, Status_Penmanshiel_13_2021-01-01_-_2022-01-01_1053.csv, Status_Penmanshiel_14_2021-01-01_-_2022-01-01_1054.csv, Status_Penmanshiel_15_2021-01-01_-_2022-01-01_1056.csv, Status_Penmanshiel_01_2022-01-01_-_2023-01-01_1042.csv, Status_Penmanshiel_02_2022-01-01_-_2023-01-01_1043.csv, Status_Penmanshiel_04_2022-01-01_-_2023-01-01_1044.csv, Status_Penmanshiel_05_2022-01-01_-_2023-01-01_1045.csv, Status_Penmanshiel_06_2022-01-01_-_2023-01-01_1046.csv, Status_Penmanshiel_07_2022-01-01_-_2023-01-01_1047.csv, Status_Penmanshiel_08_2022-01-01_-_2023-01-01_1048.csv, Status_Penmanshiel_09_2022-01-01_-_2023-01-01_1049.csv, Status_Penmanshiel_10_2022-01-01_-_2023-01-01_1050.csv, Status_Penmanshiel_11_2022-01-01_-_2023-01-01_1051.csv, Status_Penmanshiel_12_2022-01-01_-_2023-01-01_1052.csv, Status_Penmanshiel_13_2022-01-01_-_2023-01-01_1053.csv, Status_Penmanshiel_14_2022-01-01_-_2023-01-01_1054.csv, Status_Penmanshiel_15_2022-01-01_-_2023-01-01_1056.csv
- turbines in the data: 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 (static file: 14 turbines)
- time stamps 2021-01-01 00:00:00 .. 2022-12-31 23:50:00; duplicates (turbine, stamp): 0; stamps off the 10-min grid: 0
- step between consecutive records (min: count): {10.0: 1471666}

Records per turbine and year (expected 52,560 = 365 x 144):

| turbine | 2021 | 2022 |
|---|---|---|
| 1 | 52560 | 52560 |
| 2 | 52560 | 52560 |
| 4 | 52560 | 52560 |
| 5 | 52560 | 52560 |
| 6 | 52560 | 52560 |
| 7 | 52560 | 52560 |
| 8 | 52560 | 52560 |
| 9 | 52560 | 52560 |
| 10 | 52560 | 52560 |
| 11 | 52560 | 52560 |
| 12 | 52560 | 52560 |
| 13 | 52560 | 52560 |
| 14 | 52560 | 52560 |
| 15 | 52560 | 52560 |

## 2. Missing values (% of records, NaN)

| turbine | Wspd | Wdir | Ndir | Patv | Pab1 | Pab2 | Pab3 | Etmp | Itmp | Prtv | Avail | Setp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.15 | 0.15 | 0.15 | 0.15 | 1.11 | 1.11 | 1.11 | 1.11 | 1.11 | 0.15 | 0.01 | 1.11 |
| 2 | 0.37 | 0.15 | 0.15 | 0.15 | 1.11 | 1.11 | 1.11 | 1.62 | 1.62 | 0.15 | 0.02 | 1.11 |
| 4 | 0.40 | 0.40 | 0.40 | 0.40 | 1.29 | 1.29 | 1.29 | 1.79 | 1.79 | 0.40 | 0.10 | 1.29 |
| 5 | 0.63 | 0.27 | 0.27 | 0.27 | 1.23 | 1.23 | 1.23 | 1.74 | 1.74 | 0.27 | 0.01 | 1.23 |
| 6 | 0.29 | 0.22 | 0.22 | 0.22 | 1.19 | 1.19 | 1.19 | 1.19 | 1.19 | 0.22 | 0.04 | 1.19 |
| 7 | 0.99 | 0.99 | 0.99 | 0.99 | 1.30 | 1.30 | 1.30 | 1.30 | 1.30 | 0.99 | 0.70 | 1.30 |
| 8 | 0.24 | 0.18 | 0.18 | 0.18 | 1.14 | 1.14 | 1.14 | 1.14 | 1.14 | 0.18 | 0.04 | 1.14 |
| 9 | 0.18 | 0.18 | 0.18 | 0.18 | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 | 0.18 | 0.04 | 1.15 |
| 10 | 0.26 | 0.26 | 0.26 | 0.26 | 1.17 | 1.17 | 1.17 | 1.17 | 1.17 | 0.26 | 0.09 | 1.17 |
| 11 | 4.69 | 4.54 | 4.54 | 4.54 | 4.90 | 4.90 | 4.90 | 4.90 | 4.90 | 4.54 | 0.03 | 4.90 |
| 12 | 0.37 | 0.21 | 0.21 | 0.21 | 1.09 | 1.09 | 1.09 | 1.09 | 1.09 | 0.21 | 0.04 | 1.09 |
| 13 | 1.12 | 0.23 | 0.23 | 0.23 | 1.11 | 1.11 | 1.11 | 1.11 | 1.11 | 0.23 | 0.04 | 1.11 |
| 14 | 0.37 | 0.37 | 0.37 | 0.37 | 1.24 | 1.24 | 1.24 | 1.24 | 1.24 | 0.37 | 0.04 | 1.24 |
| 15 | 0.35 | 0.29 | 0.29 | 0.29 | 1.12 | 1.12 | 1.12 | 1.11 | 1.11 | 0.29 | 0.09 | 1.11 |

Months in which some turbine has power in < 90 % of its records (% present):

| month | 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022-11 | 100 | 100 | 100 | 100 | 100 | 84 | 100 | 100 | 100 | 10 | 100 | 100 | 100 | 100 |

## 3. Value ranges (all records)

| signal | min | p0.1 | p99.9 | max |
|---|---|---|---|---|
| Power kW | -18.22 | -9.68 | 2063.29 | 2087.57 |
| Wind speed m/s | 0.00 | 0.46 | 24.62 | 39.30 |
| max blade pitch deg | -170.12 | 0.00 | 92.50 | 164.37 |
| Nacelle ambient temp C | -4.18 | -1.97 | 27.91 | 34.27 |
| Nacelle temp C | 1.20 | 3.00 | 34.65 | 40.31 |
| Reactive power kvar | -794.66 | -306.81 | 278.48 | 468.39 |
| Wind dir - nacelle pos deg | -180.00 | -169.97 | 169.65 | 179.99 |
| Setpoint kW | 205.00 | 206.00 | 2060.00 | 2060.00 |
| Time-based avail. | 0.00 | 0.00 | 1.00 | 1.00 |

- power < 0 (idling consumption): 19.46 % of records; power > 1.02 x rated (2050 kW): 0.000 %
- |wind direction - nacelle position|: median 5.2 deg (small = 'Wind direction' is an absolute direction derived from the nacelle position)
- blade angle outside [-10, 100] deg: 545 records (max > 100: 10, min < -10: 537), by turbine {1: 6, 2: 4, 4: 11, 5: 12, 6: 5, 7: 469, 9: 15, 10: 3, 11: 6, 13: 5, 14: 6, 15: 3}; median power -0.980573 kW, share with power <= 0: 1.0
- feather position by turbine (mode of max blade pitch, stopped with wind, all records): {1: 92, 2: 90, 4: 90, 5: 92, 6: 92, 7: 92, 8: 90, 9: 90, 10: 90, 11: 90, 12: 91, 13: 89, 14: 90, 15: 91} deg
- turbine power setpoint, median by power class: {'0-500 kW': 1083, '500-1000 kW': 1550, '1000-1500 kW': 1991, '1500-1906 kW': 2050, '1906-2255 kW': 2050} (it follows the power, so it is not a curtailment flag; not used)

## 4. Unav threshold statistics (training segment only)

- feather position (mode of max blade pitch, stopped with wind > 5 m/s): 90 deg (5598 records)
- max blade pitch of producing records (power > 5 % rated): p50 0.2, p99 16.6, p99.9 56.8, p99.99 79.8 deg
- first 0.5 m/s bin in which >= 50 % / >= 95 % of the non-feathered records produce: 3.5 / 4.0 m/s
- **proposed thresholds (SDWPF-reproducing rule): wind 4.0 m/s, pitch 89.0 deg**

Share of producing records by wind bin (non-feathered records):

| wind bin m/s | n | share_producing |
|---|---|---|
| 0.0-0.5 | 1074.000 | 0.000 |
| 0.5-1.0 | 6918.000 | 0.000 |
| 1.0-1.5 | 13977.000 | 0.000 |
| 1.5-2.0 | 23163.000 | 0.000 |
| 2.0-2.5 | 37392.000 | 0.001 |
| 2.5-3.0 | 47996.000 | 0.008 |
| 3.0-3.5 | 42589.000 | 0.189 |
| 3.5-4.0 | 54602.000 | 0.879 |
| 4.0-4.5 | 65340.000 | 0.984 |
| 4.5-5.0 | 63587.000 | 0.997 |
| 5.0-5.5 | 59239.000 | 1.000 |
| 5.5-6.0 | 56349.000 | 1.000 |

Max blade pitch histogram (5-deg bins, training records):

```
{
 "producing": {
  "-5..0": 468,
  "0..5": 595542,
  "5..10": 45968,
  "10..15": 25590,
  "15..20": 8446,
  "20..25": 455,
  "25..30": 237,
  "30..35": 236,
  "35..40": 228,
  "40..45": 237,
  "45..50": 224,
  "50..55": 305,
  "55..60": 163,
  "60..65": 140,
  "65..70": 154,
  "70..75": 139,
  "75..80": 87,
  "80..85": 58,
  "85..90": 8
 },
 "stopped_with_wind": {
  "-175..-170": 4,
  "-140..-135": 1,
  "-40..-35": 1,
  "0..5": 12,
  "10..15": 4,
  "20..25": 1,
  "25..30": 3,
  "30..35": 1,
  "35..40": 1,
  "40..45": 3,
  "45..50": 12,
  "50..55": 1,
  "55..60": 1,
  "60..65": 1,
  "65..70": 11,
  "70..75": 30,
  "75..80": 13,
  "80..85": 169,
  "85..90": 2682,
  "90..95": 9959,
  "95..100": 109,
  "100..105": 2,
  "105..110": 2
 }
}
```

## 5. Candidate Unav rules (training segment): share flagged and agreement with Greenbyte availability

Unav share: all training records. Agreement: records with wind > 4 m/s; Greenbyte 'Time-based System Avail.' < 0.5 is taken as unavailable (it does not count curtailment).

| rule | unav_share_pct | agree_with_greenbyte_pct | rule_unav_but_gb_available_pct | rule_avail_but_gb_unavailable_pct |
|---|---|---|---|---|
| 2.5 m/s, 89 deg | 11.02 | 99.64 | 0.18 | 0.17 |
| 4.0 m/s, 89 deg | 1.55 | 99.64 | 0.18 | 0.17 |

## 6. 15-min grid preview (descriptive)

- 70080 bins 2021-01-01 00:00:00 .. 2022-12-31 23:45:00; splits: {'train': ['2021-01-01 00:00:00', '2022-05-26 23:45:00'], 'val': ['2022-05-27 00:00:00', '2022-08-07 23:45:00'], 'test': ['2022-08-08 00:00:00', '2022-12-31 23:45:00']}
- bins where >= half of the turbines report power (SDWPF rule): 99.86 %
- bins where every turbine reports power at every record: 94.67 % (relevant when a cluster is a single turbine: a cluster feature is then missing whenever that turbine is)
- gaps >= 1 h (bins failing the half rule): 4; longest: [('2021-07-13 08:00:00', '2021-07-13 14:00:00', 25), ('2022-07-05 08:00:00', '2022-07-05 13:15:00', 22), ('2022-09-20 09:15:00', '2022-09-20 14:30:00', 22), ('2021-07-12 08:30:00', '2021-07-12 12:30:00', 17), ('2022-08-17 23:15:00', '2022-08-17 23:45:00', 3)]
- build preview (proposed thresholds, all SDWPF rules incl. 'every feature present'): 860 unusable bins; clusters {'0': [1, 4], '1': [2, 5, 6], '2': [8], '3': [7], '4': [9, 12, 13], '5': [10, 11, 14, 15]}; bins lost by column (beyond the reporting rule): {'Etmp': 565, 'Itmp': 565, 'Pitch': 566, 'Unav': 566, 'P_c0': 21, 'P_c1': 21, 'P_c2': 25, 'P_c3': 608, 'P_c5': 16, 'Ws_c0': 21, 'Ws_c1': 21, 'Ws_c2': 63, 'Ws_c3': 608, 'Ws_c5': 16}; gaps >= 1 h: 20; longest: [('2022-11-25 15:45:00', '2022-11-30 10:30:00', 460), ('2021-04-30 12:00:00', '2021-05-01 07:15:00', 78), ('2021-07-13 08:00:00', '2021-07-13 15:00:00', 29), ('2021-04-26 00:00:00', '2021-04-26 06:30:00', 27), ('2022-09-21 08:00:00', '2022-09-21 13:30:00', 23)]
- causes of the longest unusable stretches (bins below the reporting rule; columns missing in at least half of the bins):
  - 2022-11-25 15:45:00 .. 2022-11-30 10:30:00 (460 bins): low reporting 0.0 %; Etmp, Itmp, Pitch, Unav, P_c3, Ws_c3
  - 2021-04-30 12:00:00 .. 2021-05-01 07:15:00 (78 bins): low reporting 0.0 %; P_c3, Ws_c3
  - 2021-07-13 08:00:00 .. 2021-07-13 15:00:00 (29 bins): low reporting 86.2 %; no single column
  - 2021-04-26 00:00:00 .. 2021-04-26 06:30:00 (27 bins): low reporting 0.0 %; Etmp, Itmp, Pitch, Unav
  - 2022-09-21 08:00:00 .. 2022-09-21 13:30:00 (23 bins): low reporting 8.7 %; P_c0, P_c1, P_c3, Ws_c0, Ws_c1, Ws_c3

| split | bins | half reporting % | all reporting % | usable (build rules) % |
|---|---|---|---|---|
| train | 49056.00 | 99.91 | 98.58 | 99.45 |
| val | 7008.00 | 99.69 | 99.46 | 99.16 |
| test | 14016.00 | 99.76 | 78.58 | 96.21 |

