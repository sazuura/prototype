"""
4train_baseline_logreg.py
=========================
Baseline model untuk skripsi — pembanding utama Random Forest.

Metodologi : Rolling Walk-Forward  (identik dengan 4train.py)
Timeline   : Yearly (train 365 hari → test 365 hari)
Model      : Logistic Regression (baseline)

Output:
  data/models_baseline/model_YYYY.pkl   ← model + threshold per tahun test
  data/baseline_predictions.csv         ← prediksi di setiap window test
  data/baseline_summary.csv             ← ringkasan performa per tahun
"""

import pandas as pd
import numpy as np
import joblib, os, warnings, gc
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import roc_auc_score, precision_score, recall_score

os.makedirs("data/models_baseline", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PARAMETER TERKUNCI  (sama persis dengan 4train.py)
# ══════════════════════════════════════════════════════════════
DATA_PATH    = "data/dataset_ml.csv"
RR_RATIO     = 1.0
MIN_TRADES   = 10
RANDOM_STATE = 42
TRAIN_DAYS   = 365
TEST_DAYS    = 365

# Logistic Regression perlu feature scaling → pakai Pipeline
MODEL_PARAMS = {
    "C"            : 0.1,          # regularisasi kuat → hindari overfit
    "penalty"      : "l2",
    "solver"       : "lbfgs",
    "class_weight" : "balanced",   # sama dengan RF: handle imbalance
    "max_iter"     : 1000,
    "random_state" : RANDOM_STATE,
}

# ══════════════════════════════════════════════════════════════
print("=" * 65)
print(" BASELINE TRAINING — Rolling Yearly | Logistic Regression")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

dates    = pd.to_datetime(df["time"].dt.date)
min_date = dates.min()
max_date = dates.max()

feature_cols = [c for c in df.columns
                if c not in ["label","time","pair","date","ret_future","price_diff"]]

X_all = df[feature_cols].values.astype(np.float32)
y_all = df["label"].values.astype(np.int8)

print(f"\nDataset : {len(X_all):,} baris | {len(feature_cols)} fitur")
print(f"WR raw  : {y_all.mean():.2%}")
print(f"Rentang : {min_date.date()} → {max_date.date()}")
print(f"Model   : LogisticRegression {MODEL_PARAMS}")
print()

# ── Buat rolling windows (identik 4train.py) ──────────────────
windows = []
cursor  = min_date + pd.Timedelta(days=TRAIN_DAYS)
while cursor + pd.Timedelta(days=TEST_DAYS) <= max_date + pd.Timedelta(days=1):
    windows.append((
        cursor - pd.Timedelta(days=TRAIN_DAYS),
        cursor - pd.Timedelta(days=1),
        cursor,
        cursor + pd.Timedelta(days=TEST_DAYS - 1),
    ))
    cursor += pd.Timedelta(days=TEST_DAYS)

print(f"Total windows : {len(windows)}")
print(f"Train window  : {TRAIN_DAYS} hari")
print(f"Test window   : {TEST_DAYS} hari\n")
print("-" * 65)

# ── Fungsi threshold optimizer (identik 4train.py) ────────────
def find_threshold(y_true, probs, rr=RR_RATIO, min_n=MIN_TRADES):
    best_th, best_exp, best_wr, best_n = 0.50, -999, 0, 0
    for th in np.arange(0.50, 0.85, 0.01):
        mask = probs >= th
        n    = mask.sum()
        if n < min_n: break
        wr  = y_true[mask].mean()
        exp = (wr * rr) - ((1 - wr) * 1)
        if exp > best_exp:
            best_exp, best_th, best_wr, best_n = exp, th, wr, n
    return round(best_th, 2), round(best_exp, 4), round(best_wr, 4), int(best_n)

# ── Training loop ─────────────────────────────────────────────
all_preds = []
summaries = []

for win_idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
    tr_mask = (dates >= tr_s) & (dates <= tr_e)
    te_mask = (dates >= te_s) & (dates <= te_e)

    X_tr = X_all[tr_mask]; y_tr = y_all[tr_mask]
    X_te = X_all[te_mask]; y_te = y_all[te_mask]

    if len(X_tr) < 100 or len(X_te) < 20:   continue
    if y_tr.sum() < 5 or (len(y_tr) - y_tr.sum()) < 5: continue
    if len(np.unique(y_te)) < 2:             continue

    test_year = te_s.year

    # ── Train: StandardScaler + LogisticRegression dalam Pipeline ──
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**MODEL_PARAMS)),
    ])
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_te)[:, 1]

    # Cari threshold optimal
    th, exp, wr, n_trades = find_threshold(y_te, probs)
    y_pred = (probs >= th).astype(int)

    # Metrik
    try:   auc = round(roc_auc_score(y_te, probs), 4)
    except: auc = np.nan

    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)

    print(f"Window {win_idx+1} | Test: {te_s.date()} → {te_e.date()}")
    print(f"  Train: {len(X_tr):,} baris | Test: {len(X_te):,} baris")
    print(f"  Threshold: {th} | Trades: {n_trades} | WR: {wr:.2%} | "
          f"Expect: {exp:+.4f} | AUC: {auc:.3f}")

    # Koefisien top-5 (pengganti feature importance di LR)
    coef = model.named_steps["clf"].coef_[0]
    top5 = sorted(zip(feature_cols, np.abs(coef)),
                  key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top-5 koefisien: {[f[0] for f in top5]}")
    print()

    # Simpan model (Pipeline sudah include scaler)
    model_path = f"data/models_baseline/model_{test_year}.pkl"
    joblib.dump({
        "model"       : model,
        "feature_cols": feature_cols,
        "threshold"   : th,
        "test_year"   : test_year,
        "tr_start"    : str(tr_s.date()),
        "tr_end"      : str(tr_e.date()),
        "te_start"    : str(te_s.date()),
        "te_end"      : str(te_e.date()),
        "model_params": MODEL_PARAMS,
    }, model_path)

    # Simpan prediksi untuk backtest
    te_df              = df[te_mask].copy()
    te_df["prob"]      = probs
    te_df["pred"]      = y_pred
    te_df["threshold"] = th
    te_df["test_year"] = test_year
    all_preds.append(te_df)

    summaries.append({
        "test_year"  : test_year,
        "te_start"   : te_s.date(),
        "te_end"     : te_e.date(),
        "n_train"    : len(X_tr),
        "n_test"     : len(X_te),
        "n_trades"   : n_trades,
        "threshold"  : th,
        "WR"         : wr,
        "expectancy" : exp,
        "AUC"        : auc,
        "precision"  : round(prec, 4),
        "recall"     : round(rec, 4),
    })
    gc.collect()

# ── Simpan output ─────────────────────────────────────────────
pred_df = pd.concat(all_preds, ignore_index=True)
pred_df.to_csv("data/baseline_predictions.csv", index=False)

sum_df = pd.DataFrame(summaries)
sum_df.to_csv("data/baseline_summary.csv", index=False)

print("=" * 65)
print(" RINGKASAN BASELINE (Logistic Regression)")
print("=" * 65)
print(f"\n{'Year':<6} {'Trades':>8} {'WR':>8} {'Expect':>9} "
      f"{'AUC':>7} {'Threshold':>10}")
print("-" * 55)
for _, r in sum_df.iterrows():
    print(f"{int(r['test_year']):<6} {int(r['n_trades']):>8} "
          f"{r['WR']:>8.2%} {r['expectancy']:>+9.4f} "
          f"{r['AUC']:>7.3f} {r['threshold']:>10.2f}")
print("-" * 55)
print(f"{'Rata-rata':<6} {sum_df['n_trades'].mean():>8.1f} "
      f"{sum_df['WR'].mean():>8.2%} {sum_df['expectancy'].mean():>+9.4f} "
      f"{sum_df['AUC'].mean():>7.3f}")

print(f"\nModel tersimpan : data/models_baseline/model_YYYY.pkl")
print(f"Prediksi        : data/baseline_predictions.csv")
print(f"Ringkasan       : data/baseline_summary.csv")
print("\nJalankan berikutnya: python 5backtest.py (arahkan ke baseline_predictions.csv)")