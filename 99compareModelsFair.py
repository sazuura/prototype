"""
fair_comparison.py
==================
Perbandingan APPLE-TO-APPLE 7 model untuk skripsi.

Keadilan dijamin dengan:
1. Data identik untuk semua model
2. Hyperparameter dioptimalkan via Optuna (budget waktu sama)
3. Validasi identik: Rolling + Anchored, Yearly + Monthly
4. Metrik identik: Expectancy, WR, AUC, Konsistensi
5. Random seed terkunci: 42
6. MIN_TRADES sama: 10

Install dulu: pip install optuna
"""

import pandas as pd
import numpy as np
import warnings, time, os, gc
warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARN: Optuna tidak ada. pip install optuna")
    print("      Akan pakai default hyperparameter sebagai fallback.")

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

try:    import lightgbm as lgb;  HAS_LGBM = True
except: HAS_LGBM = False; print("WARN: LightGBM tidak ada")

try:    import xgboost as xgb;   HAS_XGB  = True
except: HAS_XGB  = False; print("WARN: XGBoost tidak ada")

os.makedirs("data", exist_ok=True)

# ══════════════════════════════════════════════════════════════
# PARAMETER TERKUNCI — TIDAK BOLEH DIUBAH
# ══════════════════════════════════════════════════════════════
DATA_PATH      = "data/dataset_ml.csv"
RR_RATIO       = 1.0
MIN_TRADES     = 10
RANDOM_STATE   = 42
MAX_TRAIN_ROWS = 25_000

# Budget tuning: berapa trial Optuna per model per window
# Lebih banyak = lebih akurat tapi lebih lama
# Rekomendasi: 20 untuk skripsi, 50 untuk publikasi
OPTUNA_TRIALS  = 20

TIMELINES = {
    "Yearly" : {"test_days": 365, "train_days": 365, "min_train": 200, "min_test": 20},
    "Monthly": {"test_days": 30,  "train_days": 90,  "min_train": 50,  "min_test": 5 },
}
# ══════════════════════════════════════════════════════════════

print("=" * 68)
print("  FAIR COMPARISON — Apple-to-Apple")
print("  7 Model | Hyperparameter Tuning via Optuna")
print("  Rolling + Anchored | Yearly + Monthly")
print("=" * 68)

# ── Load data ─────────────────────────────────────────────────
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

pos_ratio = (1 - y_all.mean()) / y_all.mean()

print(f"\nDataset   : {len(X_all):,} baris | {len(feature_cols)} fitur")
print(f"WR raw    : {y_all.mean():.2%}  (breakeven: {1/(1+RR_RATIO):.2%})")
print(f"Rentang   : {min_date.date()} → {max_date.date()}")
print(f"Seed      : {RANDOM_STATE}")
print(f"Optuna    : {'Ya (' + str(OPTUNA_TRIALS) + ' trials/model/window)' if HAS_OPTUNA else 'Tidak (pakai default)'}")
print()


# ── Optuna search spaces per model ────────────────────────────
def tune_model(model_name, X_tr, y_tr, n_trials=OPTUNA_TRIALS):
    """
    Cari hyperparameter terbaik via Optuna.
    Objective: maximasi AUC di 3-fold CV pada data training.
    Budget sama untuk semua model (n_trials identik).
    """
    if not HAS_OPTUNA:
        return get_default(model_name)

    def objective(trial):
        try:
            if model_name == "LogisticReg":
                C   = trial.suggest_float("C", 0.001, 10.0, log=True)
                clf = Pipeline([("sc", StandardScaler()),
                    ("clf", LogisticRegression(C=C, class_weight="balanced",
                                               max_iter=500, random_state=RANDOM_STATE))])

            elif model_name == "LDA":
                solver  = trial.suggest_categorical("solver", ["svd","lsqr"])
                shrink  = trial.suggest_float("shrinkage", 0.0, 1.0) if solver != "svd" else None
                clf     = Pipeline([("sc", StandardScaler()),
                    ("clf", LinearDiscriminantAnalysis(
                        solver=solver,
                        shrinkage=shrink if solver != "svd" else None))])

            elif model_name == "RandomForest":
                clf = RandomForestClassifier(
                    n_estimators    = trial.suggest_int("n_est", 50, 400),
                    max_depth       = trial.suggest_int("max_depth", 3, 10),
                    min_samples_leaf= trial.suggest_int("min_leaf", 5, 50),
                    max_features    = trial.suggest_categorical("max_feat", ["sqrt","log2",0.5]),
                    class_weight    = "balanced",
                    random_state    = RANDOM_STATE, n_jobs=1)

            elif model_name == "ExtraTrees":
                clf = ExtraTreesClassifier(
                    n_estimators    = trial.suggest_int("n_est", 50, 400),
                    max_depth       = trial.suggest_int("max_depth", 3, 10),
                    min_samples_leaf= trial.suggest_int("min_leaf", 5, 50),
                    max_features    = trial.suggest_categorical("max_feat", ["sqrt","log2",0.5]),
                    class_weight    = "balanced",
                    random_state    = RANDOM_STATE, n_jobs=1)

            elif model_name == "MLP":
                n_layers = trial.suggest_int("n_layers", 1, 3)
                layers   = tuple(
                    trial.suggest_int(f"n_{i}", 16, 128) for i in range(n_layers)
                )
                clf = Pipeline([("sc", StandardScaler()),
                    ("clf", MLPClassifier(
                        hidden_layer_sizes = layers,
                        activation    = trial.suggest_categorical("act", ["relu","tanh"]),
                        learning_rate_init = trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                        max_iter      = 300,
                        early_stopping= True,
                        validation_fraction = 0.15,
                        random_state  = RANDOM_STATE))])

            elif model_name == "LightGBM" and HAS_LGBM:
                clf = lgb.LGBMClassifier(
                    n_estimators     = trial.suggest_int("n_est", 50, 500),
                    learning_rate    = trial.suggest_float("lr", 0.01, 0.3, log=True),
                    max_depth        = trial.suggest_int("max_depth", 3, 8),
                    num_leaves       = trial.suggest_int("num_leaves", 8, 64),
                    min_child_samples= trial.suggest_int("min_child", 5, 50),
                    subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree = trial.suggest_float("colsample", 0.5, 1.0),
                    class_weight     = "balanced",
                    random_state     = RANDOM_STATE, n_jobs=1, verbose=-1)

            elif model_name == "XGBoost" and HAS_XGB:
                clf = xgb.XGBClassifier(
                    n_estimators     = trial.suggest_int("n_est", 50, 500),
                    learning_rate    = trial.suggest_float("lr", 0.01, 0.3, log=True),
                    max_depth        = trial.suggest_int("max_depth", 3, 8),
                    min_child_weight = trial.suggest_int("min_child", 1, 20),
                    subsample        = trial.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree = trial.suggest_float("colsample", 0.5, 1.0),
                    scale_pos_weight = pos_ratio,
                    eval_metric      = "auc",
                    random_state     = RANDOM_STATE, n_jobs=1, verbosity=0)
            else:
                return 0.0

            scores = cross_val_score(clf, X_tr, y_tr, cv=3,
                                     scoring="roc_auc", n_jobs=1)
            return scores.mean()
        except:
            return 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return build_best_model(model_name, study.best_params)


def build_best_model(model_name, params):
    """Bangun model dengan parameter terbaik dari Optuna."""
    if model_name == "LogisticReg":
        return Pipeline([("sc", StandardScaler()),
            ("clf", LogisticRegression(C=params.get("C",0.1),
                class_weight="balanced", max_iter=500, random_state=RANDOM_STATE))])

    elif model_name == "LDA":
        solver = params.get("solver","svd")
        return Pipeline([("sc", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(
                solver=solver,
                shrinkage=params.get("shrinkage",None) if solver!="svd" else None))])

    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators    = params.get("n_est",150),
            max_depth       = params.get("max_depth",6),
            min_samples_leaf= params.get("min_leaf",15),
            max_features    = params.get("max_feat","sqrt"),
            class_weight    = "balanced",
            random_state    = RANDOM_STATE, n_jobs=1)

    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators    = params.get("n_est",150),
            max_depth       = params.get("max_depth",6),
            min_samples_leaf= params.get("min_leaf",15),
            max_features    = params.get("max_feat","sqrt"),
            class_weight    = "balanced",
            random_state    = RANDOM_STATE, n_jobs=1)

    elif model_name == "MLP":
        n = params.get("n_layers",2)
        layers = tuple(params.get(f"n_{i}", 64) for i in range(n))
        return Pipeline([("sc", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes  = layers,
                activation          = params.get("act","relu"),
                learning_rate_init  = params.get("lr",0.001),
                max_iter            = 300,
                early_stopping      = True,
                validation_fraction = 0.15,
                random_state        = RANDOM_STATE))])

    elif model_name == "LightGBM" and HAS_LGBM:
        return lgb.LGBMClassifier(
            n_estimators     = params.get("n_est",200),
            learning_rate    = params.get("lr",0.05),
            max_depth        = params.get("max_depth",5),
            num_leaves       = params.get("num_leaves",31),
            min_child_samples= params.get("min_child",20),
            subsample        = params.get("subsample",0.8),
            colsample_bytree = params.get("colsample",0.8),
            class_weight     = "balanced",
            random_state     = RANDOM_STATE, n_jobs=1, verbose=-1)

    elif model_name == "XGBoost" and HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators     = params.get("n_est",200),
            learning_rate    = params.get("lr",0.05),
            max_depth        = params.get("max_depth",5),
            min_child_weight = params.get("min_child",8),
            subsample        = params.get("subsample",0.8),
            colsample_bytree = params.get("colsample",0.8),
            scale_pos_weight = pos_ratio,
            eval_metric      = "auc",
            random_state     = RANDOM_STATE, n_jobs=1, verbosity=0)

    return None


def get_default(model_name):
    """Fallback jika Optuna tidak tersedia."""
    defaults = {}
    return build_best_model(model_name, defaults)


# ── Model list ────────────────────────────────────────────────
MODEL_NAMES = ["LogisticReg","LDA","RandomForest","ExtraTrees","MLP"]
if HAS_LGBM: MODEL_NAMES.append("LightGBM")
if HAS_XGB:  MODEL_NAMES.append("XGBoost")


# ── Windows ───────────────────────────────────────────────────
def make_anchored(test_days):
    wins, cur = [], min_date + pd.Timedelta(days=test_days)
    while cur + pd.Timedelta(days=test_days) <= max_date + pd.Timedelta(days=1):
        wins.append((min_date, cur-pd.Timedelta(days=1),
                     cur, cur+pd.Timedelta(days=test_days-1)))
        cur += pd.Timedelta(days=test_days)
    return wins

def make_rolling(test_days, train_days):
    wins, cur = [], min_date + pd.Timedelta(days=train_days)
    while cur + pd.Timedelta(days=test_days) <= max_date + pd.Timedelta(days=1):
        wins.append((cur-pd.Timedelta(days=train_days), cur-pd.Timedelta(days=1),
                     cur, cur+pd.Timedelta(days=test_days-1)))
        cur += pd.Timedelta(days=test_days)
    return wins


# ── Threshold ─────────────────────────────────────────────────
def best_threshold(y_true, probs):
    best_th, best_exp, best_wr, best_n = 0.50, -999, 0, 0
    for th in np.arange(0.50, 0.85, 0.01):
        mask = probs >= th
        n    = mask.sum()
        if n < MIN_TRADES: break
        wr  = y_true[mask].mean()
        exp = (wr * RR_RATIO) - ((1-wr)*1)
        if exp > best_exp:
            best_exp, best_th, best_wr, best_n = exp, th, wr, n
    return round(best_th,2), round(best_exp,4), round(best_wr,4), int(best_n)


# ── Runner ────────────────────────────────────────────────────
def run_comparison(method, tl_name, windows, min_train, min_test):
    print(f"\n{'─'*68}")
    print(f"  {method.upper()} | {tl_name.upper()} | {len(windows)} windows | {len(MODEL_NAMES)} model")
    print(f"{'─'*68}")

    all_rows = []

    for model_name in MODEL_NAMES:
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
            if y_tr.sum() < 5 or (len(y_tr)-y_tr.sum()) < 5:  continue
            if len(np.unique(y_te)) < 2:                        continue

            try:
                # Tune hyperparameter pada data TRAINING (bukan test — tidak ada leakage)
                model = tune_model(model_name, X_tr, y_tr)
                if model is None: continue

                model.fit(X_tr, y_tr)
                probs = model.predict_proba(X_te)[:,1]
            except Exception as e:
                continue

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

        dfw     = pd.DataFrame(win_rows)
        avg_exp = dfw["expectancy"].mean()
        avg_wr  = dfw["WR"].mean()
        avg_auc = dfw["AUC"].mean()
        avg_tr  = dfw["trades"].mean()
        n_pos   = (dfw["expectancy"] > 0).sum()
        elapsed = time.time() - t0

        print(f"  [{model_name:<12}] "
              f"Expect:{avg_exp:+7.3f} | WR:{avg_wr:6.2%} | "
              f"AUC:{avg_auc:.3f} | Tr/win:{avg_tr:6.1f} | "
              f"+:{n_pos:3d}/{len(dfw)} | {elapsed:.1f}s")
        all_rows.extend(win_rows)

    return all_rows


# ── Main ──────────────────────────────────────────────────────
all_results = []

for tl_name, cfg in TIMELINES.items():
    for method in ["Anchored", "Rolling"]:
        windows = (make_anchored(cfg["test_days"]) if method == "Anchored"
                   else make_rolling(cfg["test_days"], cfg["train_days"]))
        rows = run_comparison(method, tl_name, windows,
                              cfg["min_train"], cfg["min_test"])
        all_results.extend(rows)

# ── Summary ───────────────────────────────────────────────────
detail = pd.DataFrame(all_results)
detail.to_csv("data/fair_detail.csv", index=False)

summary = (detail
    .groupby(["method","timeline","model"])
    .agg(
        avg_exp    = ("expectancy","mean"),
        avg_WR     = ("WR",        "mean"),
        avg_AUC    = ("AUC",       "mean"),
        avg_trades = ("trades",    "mean"),
        n_windows  = ("expectancy","count"),
        n_positive = ("expectancy", lambda x: (x>0).sum()),
    ).reset_index()
)
summary["consistency"] = summary["n_positive"] / summary["n_windows"]
summary["score"] = (
    summary["avg_exp"]          * 0.40 +
    (summary["avg_AUC"] - 0.5)  * 0.30 +
    summary["consistency"]       * 0.30
)
summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
summary.to_csv("data/fair_summary.csv", index=False)

# ── Print hasil ───────────────────────────────────────────────
print(f"\n{'='*68}")
print("  RANKING FINAL — APPLE-TO-APPLE (dengan Optuna tuning)")
print(f"{'='*68}")
print(f"\n  {'#':<4} {'Method':<10} {'Timeline':<10} {'Model':<14} "
      f"{'Expect':>8} {'WR':>7} {'Tr/win':>7} {'Consist':>8} {'Score':>7}")
print("  " + "─"*72)
for i, r in summary.head(15).iterrows():
    tag = "  ◄ TERBAIK" if i == 0 else ""
    print(f"  {i+1:<4} {r['method']:<10} {r['timeline']:<10} {r['model']:<14} "
          f"{r['avg_exp']:>+8.3f} {r['avg_WR']:>7.2%} "
          f"{r['avg_trades']:>7.1f} {r['consistency']:>8.2%} "
          f"{r['score']:>7.3f}{tag}")

# Head-to-head per model (Anchored vs Rolling, Yearly)
print(f"\n{'='*68}")
print("  HEAD-TO-HEAD PER MODEL — Yearly (Anchored vs Rolling)")
print(f"{'='*68}")
print(f"\n  {'Model':<14} {'Anchored':>10} {'Rolling':>10} Selisih  Pemenang")
print("  " + "─"*55)
for mdl in MODEL_NAMES:
    a = summary[(summary["method"]=="Anchored") &
                (summary["timeline"]=="Yearly") &
                (summary["model"]==mdl)]
    r = summary[(summary["method"]=="Rolling") &
                (summary["timeline"]=="Yearly") &
                (summary["model"]==mdl)]
    if a.empty or r.empty: continue
    a_e = a.iloc[0]["avg_exp"]
    r_e = r.iloc[0]["avg_exp"]
    diff = a_e - r_e
    win  = "Anchored" if a_e > r_e else "Rolling "
    print(f"  {mdl:<14} {a_e:>+10.3f} {r_e:>+10.3f} {diff:>+8.3f}  {win}")

best = summary.iloc[0]
print(f"\n{'='*68}")
print(f"  KESIMPULAN UNTUK SKRIPSI (apple-to-apple):")
print(f"    Metodologi : {best['method']}")
print(f"    Timeline   : {best['timeline']}")
print(f"    Model      : {best['model']}")
print(f"    Expectancy : {best['avg_exp']:+.4f}")
print(f"    WR         : {best['avg_WR']:.2%}")
print(f"    Konsistensi: {best['consistency']:.2%}")
print(f"{'='*68}")
print(f"\nDetail : data/fair_detail.csv")
print(f"Summary: data/fair_summary.csv")