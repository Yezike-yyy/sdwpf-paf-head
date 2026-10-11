# Data pilot: Kelmarsh (raw 10-min SCADA, 2021-2022)

Source: Zenodo 10.5281/zenodo.16807551 (v4), CC BY 4.0. Profile only: nothing filtered or cleaned; threshold statistics use the training segment (before 2022-05-27 00:00) only.

## 1. Inventory

- zips: Kelmarsh_SCADA_2021_4456.zip, Kelmarsh_SCADA_2022_4457.zip
- turbine files: 12; other files in the zips: Status_Kelmarsh_1_2021-01-01_-_2022-01-01_228.csv, Status_Kelmarsh_2_2021-01-01_-_2022-01-01_229.csv, Status_Kelmarsh_3_2021-01-01_-_2022-01-01_230.csv, Status_Kelmarsh_4_2021-01-01_-_2022-01-01_231.csv, Status_Kelmarsh_5_2021-01-01_-_2022-01-01_232.csv, Status_Kelmarsh_6_2021-01-01_-_2022-01-01_233.csv, Status_Kelmarsh_1_2022-01-01_-_2023-01-01_228.csv, Status_Kelmarsh_2_2022-01-01_-_2023-01-01_229.csv, Status_Kelmarsh_3_2022-01-01_-_2023-01-01_230.csv, Status_Kelmarsh_4_2022-01-01_-_2023-01-01_231.csv, Status_Kelmarsh_5_2022-01-01_-_2023-01-01_232.csv, Status_Kelmarsh_6_2022-01-01_-_2023-01-01_233.csv
- turbines in the data: 1, 2, 3, 4, 5, 6 (static file: 6 turbines)
- time stamps 2021-01-01 00:00:00 .. 2022-12-31 23:50:00; duplicates (turbine, stamp): 0; stamps off the 10-min grid: 0
- step between consecutive records (min: count): {10.0: 630714}

Records per turbine and year (expected 52,560 = 365 x 144):

| turbine | 2021 | 2022 |
|---|---|---|
| 1 | 52560 | 52560 |
| 2 | 52560 | 52560 |
| 3 | 52560 | 52560 |
| 4 | 52560 | 52560 |
| 5 | 52560 | 52560 |
| 6 | 52560 | 52560 |

## 2. Missing values (% of records, NaN)

| turbine | Wspd | Wdir | Ndir | Patv | Pab1 | Pab2 | Pab3 | Etmp | Itmp | Prtv | Avail | Setp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.01 | 1.01 | 1.01 | 1.01 | 3.28 | 3.28 | 3.28 | 3.28 | 3.28 | 1.01 | 0.03 | 3.28 |
| 2 | 0.56 | 0.56 | 0.56 | 0.56 | 2.83 | 2.83 | 2.83 | 2.83 | 2.83 | 0.56 | 0.04 | 2.83 |
| 3 | 1.05 | 1.05 | 1.05 | 1.05 | 3.31 | 3.31 | 3.31 | 3.31 | 3.31 | 1.05 | 0.04 | 3.31 |
| 4 | 0.72 | 0.72 | 0.72 | 0.72 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 0.72 | 0.04 | 2.99 |
| 5 | 1.15 | 1.15 | 1.15 | 1.15 | 3.42 | 3.42 | 3.42 | 3.42 | 3.42 | 1.15 | 0.04 | 3.42 |
| 6 | 0.76 | 0.74 | 0.74 | 0.74 | 3.01 | 3.01 | 3.01 | 3.01 | 3.01 | 0.74 | 0.04 | 3.01 |

Months in which some turbine has power in < 90 % of its records (% present):

| month | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 2021-08 | 90 | 90 | 90 | 90 | 90 | 90 |
| 2021-11 | 100 | 100 | 89 | 100 | 87 | 99 |

## 3. Value ranges (all records)

| signal | min | p0.1 | p99.9 | max |
|---|---|---|---|---|
| Power kW | -17.79 | -10.28 | 2064.97 | 2082.69 |
| Wind speed m/s | 0.00 | 0.46 | 17.19 | 24.15 |
| max blade pitch deg | -169.75 | 0.00 | 105.12 | 211.72 |
| Nacelle ambient temp C | -4.80 | -2.80 | 33.54 | 39.33 |
| Nacelle temp C | -0.60 | 1.44 | 38.90 | 42.70 |
| Reactive power kvar | -747.25 | -646.17 | 0.93 | 5.05 |
| Wind dir - nacelle pos deg | -179.97 | -165.13 | 165.99 | 179.99 |
| Setpoint kW | 205.00 | 412.00 | 2060.00 | 2060.00 |
| Time-based avail. | 0.00 | 0.00 | 1.00 | 1.00 |

- power < 0 (idling consumption): 15.41 % of records; power > 1.02 x rated (2050 kW): 0.000 %
- |wind direction - nacelle position|: median 5.5 deg (small = 'Wind direction' is an absolute direction derived from the nacelle position)
- blade angle outside [-10, 100] deg: 1041 records (max > 100: 1014, min < -10: 27), by turbine {1: 16, 2: 6, 5: 6, 6: 1013}; median power -0.824258 kW, share with power <= 0: 1.0
- feather position by turbine (mode of max blade pitch, stopped with wind, all records): {1: 90, 2: 92, 3: 90, 4: 91, 5: 92, 6: 92} deg
- turbine power setpoint, median by power class: {'0-500 kW': 1092, '500-1000 kW': 1538, '1000-1500 kW': 1961, '1500-1906 kW': 2060, '1906-2255 kW': 2060} (it follows the power, so it is not a curtailment flag; not used)

## 4. Unav threshold statistics (training segment only)

- feather position (mode of max blade pitch, stopped with wind > 5 m/s): 92 deg (4589 records)
- max blade pitch of producing records (power > 5 % rated): p50 0.0, p99 11.7, p99.9 34.9, p99.99 61.3 deg
- first 0.5 m/s bin in which >= 50 % / >= 95 % of the non-feathered records produce: 2.5 / 3.5 m/s
- **proposed thresholds (SDWPF-reproducing rule): wind 3.5 m/s, pitch 91.0 deg**

Share of producing records by wind bin (non-feathered records):

| wind bin m/s | n | share_producing |
|---|---|---|
| 0.0-0.5 | 443.000 | 0.000 |
| 0.5-1.0 | 2813.000 | 0.000 |
| 1.0-1.5 | 5348.000 | 0.001 |
| 1.5-2.0 | 9066.000 | 0.011 |
| 2.0-2.5 | 16366.000 | 0.123 |
| 2.5-3.0 | 22549.000 | 0.518 |
| 3.0-3.5 | 23268.000 | 0.947 |
| 3.5-4.0 | 24933.000 | 0.998 |
| 4.0-4.5 | 27660.000 | 1.000 |
| 4.5-5.0 | 30075.000 | 1.000 |
| 5.0-5.5 | 30271.000 | 1.000 |
| 5.5-6.0 | 29427.000 | 1.000 |

Max blade pitch histogram (5-deg bins, training records):

```
{
 "producing": {
  "0..5": 281284,
  "5..10": 15371,
  "10..15": 4188,
  "15..20": 677,
  "20..25": 114,
  "25..30": 65,
  "30..35": 50,
  "35..40": 47,
  "40..45": 42,
  "45..50": 46,
  "50..55": 95,
  "55..60": 35,
  "60..65": 18,
  "65..70": 8,
  "70..75": 6,
  "75..80": 3,
  "80..85": 2
 },
 "stopped_with_wind": {
  "-170..-165": 1,
  "65..70": 2,
  "70..75": 5,
  "75..80": 14,
  "80..85": 25,
  "85..90": 1791,
  "90..95": 10234,
  "105..110": 594
 }
}
```

## 5. Candidate Unav rules (training segment): share flagged and agreement with Greenbyte availability

Unav share: all training records. Agreement: records with wind > 4 m/s; Greenbyte 'Time-based System Avail.' < 0.5 is taken as unavailable (it does not count curtailment).

| rule | unav_share_pct | agree_with_greenbyte_pct | rule_unav_but_gb_available_pct | rule_avail_but_gb_unavailable_pct |
|---|---|---|---|---|
| 2.5 m/s, 89 deg | 7.14 | 99.96 | 0.01 | 0.03 |
| 2.5 m/s, 91 deg | 7.14 | 99.96 | 0.01 | 0.03 |
| 3.5 m/s, 89 deg | 3.77 | 99.96 | 0.01 | 0.03 |
| 3.5 m/s, 91 deg | 3.77 | 99.96 | 0.01 | 0.03 |

## 6. 15-min grid preview (descriptive)

- 70080 bins 2021-01-01 00:00:00 .. 2022-12-31 23:45:00; splits: {'train': ['2021-01-01 00:00:00', '2022-05-26 23:45:00'], 'val': ['2022-05-27 00:00:00', '2022-08-07 23:45:00'], 'test': ['2022-08-08 00:00:00', '2022-12-31 23:45:00']}
- bins where >= half of the turbines report power (SDWPF rule): 99.46 %
- bins where every turbine reports power at every record: 97.54 % (relevant when a cluster is a single turbine: a cluster feature is then missing whenever that turbine is)
- gaps >= 1 h (bins failing the half rule): 5; longest: [('2021-08-09 08:15:00', '2021-08-12 11:15:00', 301), ('2021-02-02 09:15:00', '2021-02-02 15:30:00', 26), ('2022-06-07 10:00:00', '2022-06-07 15:00:00', 21), ('2022-09-01 07:45:00', '2022-09-01 12:00:00', 18), ('2021-01-03 22:30:00', '2021-01-03 23:15:00', 4)]
- build preview (proposed thresholds, all SDWPF rules incl. 'every feature present'): 3291 unusable bins; clusters {'0': [2], '1': [1], '2': [3], '3': [4], '4': [5], '5': [6]}; bins lost by column (beyond the reporting rule): {'Etmp': 1584, 'Itmp': 1584, 'Pitch': 1583, 'Unav': 1583, 'P_c0': 15, 'P_c1': 330, 'P_c2': 350, 'P_c3': 123, 'P_c4': 419, 'P_c5': 139, 'Ws_c0': 15, 'Ws_c1': 332, 'Ws_c2': 350, 'Ws_c3': 123, 'Ws_c4': 419, 'Ws_c5': 155}; gaps >= 1 h: 34; longest: [('2021-06-16 08:30:00', '2021-07-01 10:15:00', 1448), ('2021-11-11 09:45:00', '2021-11-18 13:30:00', 688), ('2021-08-09 08:15:00', '2021-08-12 11:15:00', 301), ('2022-09-08 08:15:00', '2022-09-10 08:45:00', 195), ('2022-03-10 10:15:00', '2022-03-11 11:30:00', 102)]
- causes of the longest unusable stretches (bins below the reporting rule; columns missing in at least half of the bins):
  - 2021-06-16 08:30:00 .. 2021-07-01 10:15:00 (1448 bins): low reporting 0.1 %; Etmp, Itmp, Pitch, Unav
  - 2021-11-11 09:45:00 .. 2021-11-18 13:30:00 (688 bins): low reporting 0.0 %; P_c4, Ws_c4
  - 2021-08-09 08:15:00 .. 2021-08-12 11:15:00 (301 bins): low reporting 100.0 %; no single column
  - 2022-09-08 08:15:00 .. 2022-09-10 08:45:00 (195 bins): low reporting 0.0 %; P_c1, Ws_c1
  - 2022-03-10 10:15:00 .. 2022-03-11 11:30:00 (102 bins): low reporting 0.0 %; P_c3, Ws_c3

| split | bins | half reporting % | all reporting % | usable (build rules) % |
|---|---|---|---|---|
| train | 49056.00 | 99.32 | 97.47 | 94.30 |
| val | 7008.00 | 99.66 | 97.70 | 97.47 |
| test | 14016.00 | 99.86 | 97.70 | 97.73 |

