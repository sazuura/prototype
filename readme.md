# Momentum-Based Forex Scalping dengan Machine Learning

> Repositori kode untuk penelitian skripsi:  
> **"Penerapan Machine Learning untuk Deteksi Sinyal Momentum pada Scalping Forex M15"**

---

## Gambaran Umum

Penelitian ini membangun sistem prediksi sinyal trading forex berbasis _momentum candle_ pada timeframe M15 (15 menit) menggunakan algoritma **Random Forest**. Sistem dievaluasi menggunakan metodologi _Rolling Walk-Forward Validation_ untuk memastikan tidak ada kebocoran data (_data leakage_) dan hasil yang dapat digeneralisasi ke kondisi pasar nyata.

**Pasangan mata uang (pairs):** EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD, NZDUSD  
**Periode data:** Januari 2020 – Desember 2025  
**Sumber data:** MetaTrader 5 (MT5)

---

## Hasil Utama

| Metrik        | Nilai                                |
| ------------- | ------------------------------------ |
| Net Profit    | +$189.27 (+18.93%) dari modal $1,000 |
| CAGR          | 3.60% per tahun                      |
| Win Rate      | 57.14% (breakeven: 50%)              |
| Sharpe Ratio  | **2.36** (> 2.0 = sangat baik)       |
| Max Drawdown  | 10.61%                               |
| Profit Factor | 1.30                                 |
| Total Trades  | 133 trades (2021–2025)               |
| Model Terbaik | Random Forest (Rolling Yearly)       |
| AUC           | ~0.544 konsisten di semua window     |

---

## Alur Pipeline

```
1data.py  →  2feat.py  →  3dataset.py  →  4train.py  →  5backtest.py
   ↓              ↓              ↓              ↓               ↓
Ambil data    Feature &      Gabung &       Training &     Simulasi
dari MT5      Labeling       Filter        Prediksi        Eksekusi
```

Setelah pipeline utama selesai, jalankan script visualisasi:

```
6imp_feat.py  →  Feature Importance
7shap.py      →  SHAP Analysis
8matrix.py    →  Confusion Matrix & ROC Curve
9threshold.py →  Threshold Sensitivity Analysis
```

Script eksperimen perbandingan (opsional, sudah selesai dijalankan):

```
99compareModels.py      →  Anchored Walk-Forward semua model
99compareModelsFair.py  →  Final experiment (Anchored vs Rolling x semua model)
```

---

## Deskripsi Script

### Pipeline Utama

| Script         | Fungsi                                                                 | Input                                  | Output                                                |
| -------------- | ---------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| `1data.py`     | Mengambil data OHLCV + spread dari MT5 via API                         | MT5 connection                         | `data/{PAIR}/raw.csv`                                 |
| `2feat.py`     | Membangun 46 fitur teknikal + labeling momentum (LONG/SHORT)           | `raw.csv`                              | `data/{PAIR}/labeled.csv`                             |
| `3dataset.py`  | Menggabungkan semua pair, filter kandidat momentum, drop leakage       | `labeled.csv`                          | `data/dataset_ml.csv`                                 |
| `4train.py`    | Rolling Walk-Forward training Random Forest, simpan model per tahun    | `dataset_ml.csv`                       | `data/models/model_YYYY.pkl`, `train_predictions.csv` |
| `5backtest.py` | Simulasi eksekusi trade dengan lot sizing, filter sesi, dan vol regime | `train_predictions.csv`, `labeled.csv` | `backtest_trades.csv`, `equity_curve.csv`             |

### Visualisasi

| Script          | Output                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| `6imp_feat.py`  | `feat_importance_aggregate.png`, `feat_importance_per_year.png`                                         |
| `7shap.py`      | `shap_summary.png`, `shap_bar.png`, `shap_dependence.png`                                               |
| `8matrix.py`    | `confusion_matrix_all.png`, `confusion_matrix_signal.png`, `roc_curve.png`, `classification_report.txt` |
| `9threshold.py` | `threshold_analysis.png`, `threshold_detail.csv`                                                        |

---

## Struktur Folder

```
prototype/
├── 1data.py                 # [1] Akuisisi data MT5
├── 2feat.py                 # [2] Feature engineering & labeling
├── 3dataset.py              # [3] Dataset preparation
├── 4train.py                # [4] Training & prediksi
├── 5backtest.py             # [5] Backtesting
├── 6imp_feat.py             # [VIZ] Feature importance
├── 7shap.py                 # [VIZ] SHAP analysis
├── 8matrix.py               # [VIZ] Confusion matrix & ROC
├── 9threshold.py            # [VIZ] Threshold analysis
├── 99compareModels.py       # [EXP] Anchored walk-forward comparison
├── 99compareModelsFair.py   # [EXP] Final model comparison experiment
├── data/
│   ├── EURUSD/raw.csv       # Data mentah per pair
│   ├── EURUSD/labeled.csv   # Data dengan fitur + label
│   ├── dataset_ml.csv       # Dataset gabungan untuk ML
│   ├── models/              # Model terlatih (pkl)
│   ├── train_predictions.csv
│   ├── train_summary.csv
│   ├── backtest_trades.csv
│   ├── backtest_report.csv
│   └── equity_curve.csv
├── output/                  # Gambar & laporan visualisasi
├── hasil compare.txt        # Output eksperimen perbandingan model
└── output codingan.txt      # Log output pipeline lengkap
```

---

## Cara Menjalankan

### 1. Prasyarat

```bash
pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib joblib MetaTrader5
```

> **Catatan:** `MetaTrader5` hanya tersedia di Windows. Script `1data.py` memerlukan MT5 terinstal dan akun aktif.

### 2. Akuisisi Data (perlu MT5)

```bash
python 1data.py
```

Mengambil ~160.000 bar M15 per pair dari MT5. Hasil disimpan di `data/{PAIR}/raw.csv`.

### 3. Feature Engineering & Labeling

```bash
python 2feat.py
```

Membangun 46 fitur dan melabeli setiap candle momentum sebagai LONG (1) atau SHORT (-1). Labeling menggunakan simulasi forward-looking: jika harga mencapai TP (2×ATR) dalam 40 bar ke depan → label 1, sebaliknya → label 0.

### 4. Persiapan Dataset

```bash
python 3dataset.py
```

Menggabungkan semua pair, memfilter hanya kandidat momentum, dan membuang kolom yang berpotensi menyebabkan _data leakage_ (harga absolut, EMA absolut).

### 5. Training

```bash
python 4train.py
```

Menggunakan **Rolling Walk-Forward Validation** (train 365 hari → test 365 hari, geser maju). Model dilatih untuk setiap window test dan disimpan secara terpisah. Threshold optimal dicari secara otomatis berdasarkan _expectancy_ tertinggi dengan minimal 10 trade.

### 6. Backtesting

```bash
python 5backtest.py
```

Mensimulasikan eksekusi nyata dengan:

- Risk 1% per trade
- Lot sizing sesuai tipe pair (direct vs USD-base)
- Filter sesi London–NY (07:00–17:00 UTC)
- Filter volatilitas regime minimum
- Spread dinamis dihitung per bar

### 7. Visualisasi (jalankan semua setelah step 5)

```bash
python 6imp_feat.py
python 7shap.py       # install shap dulu: pip install shap
python 8matrix.py
python 9threshold.py
```

---

## Metodologi

### Definisi Momentum Candle

Sebuah candle dikategorikan sebagai _momentum candle_ jika memenuhi tiga kriteria:

```
body ≥ 0.8 × ATR(14)          → candle cukup signifikan
wick atas ≤ 60% × body        → tidak ada penolakan kuat
wick bawah ≤ 60% × body       → arah pergerakan dominan
body ≥ 45% × candle range     → body mendominasi range
volume ratio > 0.7             → ada konfirmasi volume
```

### Fitur ML (46 total)

Semua fitur dinormalisasi terhadap ATR untuk membuat skala konsisten antar pair:

| Kategori       | Fitur                                                                                 |
| -------------- | ------------------------------------------------------------------------------------- |
| Candle anatomy | `body_atr`, `upper/lower_wick_atr`, `body_range_ratio`, `upper/lower_wick_body`       |
| Trend          | `micro/macro_trend_atr`, `dist_to_ema20/50_atr`, `ema20/50/200_slope`, `above_ema200` |
| Momentum       | `rsi`, `rsi_slope`, `rsi_dist50`, `macd_hist_atr`, `macd_hist_slope`                  |
| Volatilitas    | `atr`, `atr_slow`, `vol_regime`, `atr_std`, `vol_z_atr`, `spread_atr`                 |
| Konteks        | `pos_in_range`, `range100`, `dist_swing_high/low`, `vol_z`                            |
| Konsistensi    | `mom3`, `mom5`, `mom10`, `ret3/5/10_atr`, `candle_dir`                                |
| Volume         | `vol_ratio`, `tick_volume`, `vol_mean`                                                |
| Waktu          | `hour_sin`, `hour_cos`, `day_of_week`, `direction`                                    |

### Validasi: Rolling Walk-Forward

```
Window 1: Train 2020 → Test 2021
Window 2: Train 2021 → Test 2022
Window 3: Train 2022 → Test 2023
Window 4: Train 2023 → Test 2024
Window 5: Train 2024 → Test 2025
```

Model dilatih ulang setiap tahun menggunakan data 1 tahun sebelumnya, mensimulasikan kondisi deployment nyata di mana model harus menyesuaikan diri dengan perubahan kondisi pasar.

### Pemilihan Threshold

Threshold tidak dikunci di angka tetap, melainkan dicari secara otomatis per window berdasarkan **expectancy tertinggi** dengan minimal 10 trade:

```
Expectancy = (WR × RR) − (1 − WR) × 1
           = (WR × 1.0) − (1 − WR)
```

Threshold optimal yang ditemukan berkisar antara 0.56–0.62.

---

## Temuan Utama

**1. Fitur waktu dan likuiditas lebih prediktif dari indikator teknikal**

Top-5 fitur konsisten di semua window: `hour_sin`, `spread_atr`, `hour_cos`, `atr`, `vol_z_atr`. RSI, MACD, dan EMA tidak masuk top 10. Ini menunjukkan bahwa _kapan_ dan _dalam kondisi apa_ momentum terjadi lebih penting dari konfigurasi indikator teknikal.

**2. Anchored Walk-Forward vs Rolling: tidak ada pemenang mutlak**

Pada timeline Yearly, Rolling sedikit lebih baik (4 dari 7 model). Pada Monthly, Anchored lebih unggul (5 dari 7 model). Ini menunjukkan momentum M15 memiliki persistensi jangka menengah namun bukan jangka panjang.

**3. Model sederhana bersaing dengan model kompleks**

Random Forest mengungguli XGBoost dan LightGBM dalam eksperimen final. Di data noisy seperti forex M15, model yang lebih stabil dan kurang agresif dalam mencari pola memberikan generalisasi lebih baik.

**4. USDJPY memiliki karakteristik berbeda**

Satu-satunya pair yang merugi (-$39.99, WR 46.15%) dibanding semua pair lain yang profit. Kemungkinan disebabkan oleh pengaruh kebijakan Bank of Japan (BOJ) yang menciptakan dinamika berbeda dari mekanisme forex konvensional.

**5. High-precision, low-recall by design**

Model sangat selektif: dari ~24.000 bar test per tahun, hanya 10–52 yang diambil sebagai sinyal. Recall rendah di confusion matrix bukan kelemahan — ini trade-off yang disengaja untuk memastikan setiap sinyal yang diambil memiliki probabilitas tinggi.

---

## Parameter Kunci

| Parameter      | Nilai           | Keterangan                    |
| -------------- | --------------- | ----------------------------- |
| Timeframe      | M15             | 15 menit per bar              |
| RR Ratio       | 1:1             | Risk-Reward 1 banding 1       |
| SL Multiplier  | 2× ATR          | Stop Loss = 2× ATR dari entry |
| Max Hold       | 40 bar (10 jam) | Batas waktu posisi            |
| Risk per trade | 1%              | Dari balance saat itu         |
| Session filter | 07:00–17:00 UTC | London + New York             |
| Vol regime min | 0.8             | ATR/ATR_slow minimum          |
| Random seed    | 42              | Untuk reprodusibilitas        |

---

## Dependensi

```
pandas >= 1.5
numpy >= 1.23
scikit-learn >= 1.2
lightgbm >= 3.3
xgboost >= 1.7
shap >= 0.41
matplotlib >= 3.6
joblib >= 1.2
MetaTrader5 >= 5.0 (Windows only)
```

---

## Catatan Penting

- Script `1data.py` hanya dapat dijalankan di **Windows** dengan MetaTrader 5 terinstall dan terhubung ke akun broker.
- Data mentah (`raw.csv`) tidak disertakan dalam repositori karena ukurannya (~500MB total) dan keterbatasan lisensi broker.
- Semua angka yang tertera adalah hasil **out-of-sample** (model tidak pernah melihat data test saat training).
- Hasil backtest bersifat historis dan **tidak menjamin performa di masa depan**.

---

## Lisensi

Repositori ini dibuat untuk keperluan penelitian akademik.  
© 2025 — Universitas [nama universitas]
