"""
final_experiment.py
===================
Eksperimen definitif untuk skripsi.
Kondisi dikunci agar hasil reproducible dan comparable.

Yang dibandingkan:
  - 2 metodologi : Anchored vs Rolling
  - 2 timeline   : Yearly vs Monthly  
  - 7 model      : set yang sama persis untuk keduanya

Semua parameter TIDAK BOLEH diubah setelah eksperimen ini dijalankan.
Ini adalah angka yang masuk ke skripsi.

Output:
  data/final_detail.csv
  data/final_summary.csv
"""

import pandas as pd
import numpy as np
import warnings, time, os, gc
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

try:    import lightgbm as lgb;  HAS_LGBM = True
except: HAS_LGBM = False; print("WARN: LightGBM tidak ada")

try:    import xgboost as xgb;   HAS_XGB  = True
except: HAS_XGB  = False; print("WARN: XGBoost tidak ada")

os.makedirs("data", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PARAMETER TERKUNCI — JANGAN DIUBAH
# ══════════════════════════════════════════════════════════════
DATA_PATH      = "data/dataset_ml.csv"
RR_RATIO       = 1.0
MIN_TRADES     = 10     # lebih ketat: min 10 trade agar statistik valid
RANDOM_STATE   = 42
MAX_TRAIN_ROWS = 25_000

# Timeline: (test_days, train_days_untuk_rolling)
TIMELINES = {
    "Yearly" : {"test_days": 365, "train_days": 365, "min_train": 200, "min_test": 20},
    "Monthly": {"test_days": 30,  "train_days": 90,  "min_train": 50,  "min_test": 5 },
}
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("  FINAL EXPERIMENT — Skripsi")
print("  Anchored vs Rolling | Yearly vs Monthly | 7 Model")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────
df         = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])
df         = df.sort_values("time").reset_index(drop=True)
dates      = pd.to_datetime(df["time"].dt.date)
min_date   = dates.min()
max_date   = dates.max()

feature_cols = [c for c in df.columns
                if c not in ["label","time","pair","date","ret_future","price_diff"]]

X_all = df[feature_cols].values.astype(np.float32)
y_all = df["label"].values.astype(np.int8)
del df; gc.collect()

print(f"\nDataset  : {len(X_all):,} baris | {len(feature_cols)} fitur")
print(f"WR raw   : {y_all.mean():.2%}  (breakeven: {1/(1+RR_RATIO):.2%})")
print(f"Rentang  : {min_date.date()} → {max_date.date()}")
print(f"Seed     : {RANDOM_STATE} (reproducible)")
print(f"MIN_TRADES: {MIN_TRADES} per window\n")


# ── Model — SET SAMA untuk Anchored DAN Rolling ───────────────
def get_models():
    """7 model identik, hyperparameter terkunci."""
    M = {}
    M["LogisticReg"] = Pipeline([("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced",
                                   max_iter=500, random_state=RANDOM_STATE))])
    M["LDA"] = Pipeline([("sc", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())])
    M["RandomForest"] = RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=15,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    M["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=15,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1)
    M["MLP"] = Pipeline([("sc", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                              early_stopping=True, validation_fraction=0.2,
                              random_state=RANDOM_STATE))])
    if HAS_LGBM:
        M["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=31, min_child_samples=20,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=1, verbose=-1)
    if HAS_XGB:
        M["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            min_child_weight=8,
            scale_pos_weight=(1-y_all.mean())/y_all.mean(),
            eval_metric="auc", random_state=RANDOM_STATE,
            n_jobs=1, verbosity=0)
    return M


# ── Windows ───────────────────────────────────────────────────
def make_anchored_windows(test_days):
    windows, cursor = [], min_date + pd.Timedelta(days=test_days)
    while cursor + pd.Timedelta(days=test_days) <= max_date + pd.Timedelta(days=1):
        windows.append((min_date, cursor - pd.Timedelta(days=1),
                        cursor,   cursor + pd.Timedelta(days=test_days-1)))
        cursor += pd.Timedelta(days=test_days)
    return windows

def make_rolling_windows(test_days, train_days):
    windows, cursor = [], min_date + pd.Timedelta(days=train_days)
    while cursor + pd.Timedelta(days=test_days) <= max_date + pd.Timedelta(days=1):
        windows.append((cursor - pd.Timedelta(days=train_days),
                        cursor - pd.Timedelta(days=1),
                        cursor, cursor + pd.Timedelta(days=test_days-1)))
        cursor += pd.Timedelta(days=test_days)
    return windows


# ── Threshold ─────────────────────────────────────────────────
def best_threshold(y_true, probs):
    best_th, best_exp, best_wr, best_n = 0.50, -999, 0, 0
    for th in np.arange(0.50, 0.85, 0.01):
        mask = probs >= th
        n    = mask.sum()
        if n < MIN_TRADES: break
        wr  = y_true[mask].mean()
        exp = (wr * RR_RATIO) - ((1-wr) * 1)
        if exp > best_exp:
            best_exp, best_th, best_wr, best_n = exp, th, wr, n
    return round(best_th,2), round(best_exp,4), round(best_wr,4), int(best_n)


# ── Runner ────────────────────────────────────────────────────
def run(method, tl_name, windows, min_train, min_test):
    models   = get_models()
    all_rows = []

    print(f"\n{'─'*65}")
    print(f"  {method.upper()} | {tl_name.upper()} | {len(windows)} windows")
    print(f"{'─'*65}")

    for model_name, model in models.items():
        t0, win_rows = time.time(), []

        for (tr_s, tr_e, te_s, te_e) in windows:
            tr_mask = (dates >= tr_s) & (dates <= tr_e)
            te_mask = (dates >= te_s) & (dates <= te_e)

            X_tr = X_all[tr_mask]; y_tr = y_all[tr_mask]
            X_te = X_all[te_mask]; y_te = y_all[te_mask]

            if method == "Anchored" and len(X_tr) > MAX_TRAIN_ROWS:
                X_tr = X_tr[-MAX_TRAIN_ROWS:]
                y_tr = y_tr[-MAX_TRAIN_ROWS:]

            if len(X_tr) < min_train or len(X_te) < min_test:  continue
            if y_tr.sum() < 2 or (len(y_tr)-y_tr.sum()) < 2:  continue
            if len(np.unique(y_te)) < 2:                        continue

            try:
                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:,1]
            except: continue

            th, exp, wr, n = best_threshold(y_te, probs)
            try:    auc = round(roc_auc_score(y_te, probs), 4)
            except: auc = np.nan

            win_rows.append({
                "method": method, "timeline": tl_name, "model": model_name,
                "te_start": te_s.date(), "te_end": te_e.date(),
                "n_train": len(X_tr), "n_test": len(X_te),
                "trades": n, "WR": wr, "expectancy": exp,
                "threshold": th, "AUC": auc,
            })

        gc.collect()
        if not win_rows: continue

        df_w    = pd.DataFrame(win_rows)
        avg_exp = df_w["expectancy"].mean()
        avg_wr  = df_w["WR"].mean()
        avg_auc = df_w["AUC"].mean()
        avg_tr  = df_w["trades"].mean()
        n_pos   = (df_w["expectancy"] > 0).sum()
        print(f"  [{model_name:<12}] "
              f"Expect:{avg_exp:+7.3f} | WR:{avg_wr:6.2%} | "
              f"AUC:{avg_auc:.3f} | Tr/win:{avg_tr:6.1f} | "
              f"+:{n_pos:3d}/{len(df_w)} | {time.time()-t0:.1f}s")
        all_rows.extend(win_rows)

    return all_rows


# ── Main ──────────────────────────────────────────────────────
all_results = []

for tl_name, cfg in TIMELINES.items():
    for method in ["Anchored", "Rolling"]:
        if method == "Anchored":
            windows = make_anchored_windows(cfg["test_days"])
        else:
            windows = make_rolling_windows(cfg["test_days"], cfg["train_days"])
        rows = run(method, tl_name, windows, cfg["min_train"], cfg["min_test"])
        all_results.extend(rows)

# ── Simpan & Ranking ──────────────────────────────────────────
detail = pd.DataFrame(all_results)
detail.to_csv("data/final_detail.csv", index=False)

summary = (detail
    .groupby(["method","timeline","model"])
    .agg(
        avg_exp    = ("expectancy","mean"),
        avg_WR     = ("WR",        "mean"),
        avg_AUC    = ("AUC",       "mean"),
        avg_trades = ("trades",    "mean"),
        n_windows  = ("expectancy","count"),
        n_positive = ("expectancy", lambda x: (x > 0).sum()),
    ).reset_index()
)
summary["consistency"] = summary["n_positive"] / summary["n_windows"]
summary["score"] = (
    summary["avg_exp"]           * 0.40 +
    (summary["avg_AUC"] - 0.5)   * 0.30 +
    summary["consistency"]        * 0.30
)
summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
summary.to_csv("data/final_summary.csv", index=False)

# ── Print hasil ───────────────────────────────────────────────
print(f"\n{'='*65}")
print("  HASIL FINAL — HEAD-TO-HEAD")
print(f"{'='*65}")

print(f"\n  {'Timeline':<10} {'Model':<14} {'Anchored':>10} {'Rolling':>10} Pemenang")
print("  " + "─"*55)

for tl in ["Yearly","Monthly"]:
    models_in_tl = summary[summary["timeline"]==tl]["model"].unique()
    for mdl in models_in_tl:
        a = summary[(summary["method"]=="Anchored") &
                    (summary["timeline"]==tl) &
                    (summary["model"]==mdl)]
        r = summary[(summary["method"]=="Rolling") &
                    (summary["timeline"]==tl) &
                    (summary["model"]==mdl)]
        if a.empty or r.empty: continue
        a_exp = a.iloc[0]["avg_exp"]
        r_exp = r.iloc[0]["avg_exp"]
        win   = "Anchored" if a_exp > r_exp else "Rolling "
        print(f"  {tl:<10} {mdl:<14} {a_exp:>+10.3f} {r_exp:>+10.3f} {win}")

print(f"\n{'='*65}")
print("  RANKING OVERALL TOP 10")
print(f"{'='*65}")
print(f"  {'#':<4} {'Method':<10} {'Timeline':<10} {'Model':<14} "
      f"{'Expect':>8} {'WR':>7} {'Tr/win':>7} {'Consist':>8} {'Score':>7}")
print("  " + "─"*70)
for i, r in summary.head(10).iterrows():
    tag = "  ◄ TERBAIK" if i == 0 else ""
    print(f"  {i+1:<4} {r['method']:<10} {r['timeline']:<10} {r['model']:<14} "
          f"{r['avg_exp']:>+8.3f} {r['avg_WR']:>7.2%} "
          f"{r['avg_trades']:>7.1f} {r['consistency']:>8.2%} "
          f"{r['score']:>7.3f}{tag}")

best = summary.iloc[0]
print(f"\n{'='*65}")
print(f"  KESIMPULAN UNTUK SKRIPSI:")
print(f"    Metodologi terbaik : {best['method']}")
print(f"    Timeline terbaik   : {best['timeline']}")
print(f"    Model terbaik      : {best['model']}")
print(f"    Expectancy         : {best['avg_exp']:+.4f} per trade")
print(f"    Win Rate           : {best['avg_WR']:.2%}")
print(f"    Trades per window  : {best['avg_trades']:.1f}")
print(f"    Konsistensi        : {best['consistency']:.2%} periode positif")
print(f"{'='*65}")
print(f"\nFile: data/final_detail.csv | data/final_summary.csv")
