"""
viz_threshold.py
================
Analisis sensitivitas threshold — menunjukkan trade-off antara:
  - Jumlah trade vs kualitas sinyal
  - Win Rate vs volume
  - Expectancy vs jumlah trade
  - Profit Factor per threshold

Penting untuk skripsi: menjelaskan MENGAPA threshold tertentu dipilih.

Output:
  output/threshold_analysis.png
  output/threshold_detail.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)

PRED_PATH  = "data/train_predictions.csv"
RR_RATIO   = 1.0
THRESHOLDS = np.arange(0.50, 0.86, 0.01)

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})

# ── Load prediksi ─────────────────────────────────────────────
if not os.path.exists(PRED_PATH):
    print(f"{PRED_PATH} tidak ditemukan. Jalankan dulu 5train.py")
    exit()

df    = pd.read_csv(PRED_PATH, parse_dates=["time"])
years = sorted(df["test_year"].unique())
print(f"Data: {len(df):,} baris | Tahun: {years}")

# ── Hitung metrik per threshold ───────────────────────────────
def compute_metrics(df_sub, thresholds, rr=RR_RATIO):
    rows = []
    y_true = df_sub["label"].values
    probs  = df_sub["prob"].values

    for th in thresholds:
        mask   = probs >= th
        n      = mask.sum()
        if n == 0:
            rows.append({"threshold": th, "trades": 0, "WR": np.nan,
                         "expectancy": np.nan, "profit_factor": np.nan,
                         "precision": np.nan})
            continue

        wr  = y_true[mask].mean()
        exp = (wr * rr) - ((1 - wr) * 1)

        wins   = (y_true[mask] == 1).sum()
        losses = (y_true[mask] == 0).sum()
        pf     = (wins * rr) / losses if losses > 0 else np.inf

        # Precision = dari yang diprediksi TP, berapa yang benar TP
        precision = wr  # sama dengan WR dalam konteks ini

        rows.append({
            "threshold"    : round(th, 2),
            "trades"       : int(n),
            "WR"           : round(wr * 100, 2),
            "expectancy"   : round(exp, 4),
            "profit_factor": round(pf, 3) if not np.isinf(pf) else 99.0,
            "precision"    : round(precision * 100, 2),
        })
    return pd.DataFrame(rows)

# Agregat semua tahun
df_agg = compute_metrics(df, THRESHOLDS)

# Per tahun
df_per_year = {}
for yr in years:
    df_per_year[yr] = compute_metrics(df[df["test_year"]==yr], THRESHOLDS)

# ── Plot utama: 4 panel ───────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

colors_year = ["#185FA5","#1D9E75","#D85A30","#7F77DD","#BA7517"]

# Panel 1: Jumlah Trade vs Threshold
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_agg["threshold"], df_agg["trades"], "k-", lw=2.5,
         label="Agregat", zorder=5)
for i, (yr, df_yr) in enumerate(df_per_year.items()):
    ax1.plot(df_yr["threshold"], df_yr["trades"],
             color=colors_year[i % len(colors_year)], lw=1.2,
             alpha=0.7, linestyle="--", label=str(yr))
ax1.set_xlabel("Threshold", fontsize=10)
ax1.set_ylabel("Jumlah Trade", fontsize=10)
ax1.set_title("Jumlah Trade vs Threshold\n(semakin tinggi threshold → semakin sedikit trade)",
              fontsize=10, fontweight="bold")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_xlim([0.50, 0.85])

# Panel 2: Win Rate vs Threshold
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhline(50, color="red", linestyle=":", lw=1.5,
            label="Breakeven (50%)")
ax2.plot(df_agg["threshold"], df_agg["WR"], "k-", lw=2.5,
         label="Agregat", zorder=5)
for i, (yr, df_yr) in enumerate(df_per_year.items()):
    ax2.plot(df_yr["threshold"], df_yr["WR"],
             color=colors_year[i % len(colors_year)], lw=1.2,
             alpha=0.7, linestyle="--", label=str(yr))
ax2.set_xlabel("Threshold", fontsize=10)
ax2.set_ylabel("Win Rate (%)", fontsize=10)
ax2.set_title("Win Rate vs Threshold\n(di atas garis merah = profitable)",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, loc="lower right")
ax2.set_xlim([0.50, 0.85])

# Panel 3: Expectancy vs Threshold
ax3 = fig.add_subplot(gs[1, 0])
ax3.axhline(0, color="red", linestyle=":", lw=1.5, label="Breakeven (0)")

# Isi area positif
valid = df_agg.dropna(subset=["expectancy"])
ax3.fill_between(valid["threshold"], valid["expectancy"], 0,
                 where=valid["expectancy"] > 0,
                 alpha=0.15, color="green", label="Zona profit")
ax3.fill_between(valid["threshold"], valid["expectancy"], 0,
                 where=valid["expectancy"] < 0,
                 alpha=0.15, color="red", label="Zona rugi")
ax3.plot(valid["threshold"], valid["expectancy"], "k-", lw=2.5,
         label="Agregat", zorder=5)
for i, (yr, df_yr) in enumerate(df_per_year.items()):
    v = df_yr.dropna(subset=["expectancy"])
    ax3.plot(v["threshold"], v["expectancy"],
             color=colors_year[i % len(colors_year)], lw=1.2,
             alpha=0.7, linestyle="--", label=str(yr))

# Tandai threshold optimal (expectancy tertinggi dengan trades >= 10)
valid_min = df_agg[(df_agg["trades"] >= 10) & df_agg["expectancy"].notna()]
if not valid_min.empty:
    best_th  = valid_min.loc[valid_min["expectancy"].idxmax(), "threshold"]
    best_exp = valid_min.loc[valid_min["expectancy"].idxmax(), "expectancy"]
    ax3.axvline(best_th, color="green", linestyle="-.", lw=2,
                label=f"Optimal th={best_th:.2f}")
    ax3.scatter([best_th], [best_exp], color="green", s=80, zorder=6)

ax3.set_xlabel("Threshold", fontsize=10)
ax3.set_ylabel("Expectancy", fontsize=10)
ax3.set_title("Expectancy vs Threshold\n(optimal = expectancy max dengan trades ≥ 10)",
              fontsize=10, fontweight="bold")
ax3.legend(fontsize=8, loc="lower right")
ax3.set_xlim([0.50, 0.85])

# Panel 4: Profit Factor vs Threshold
ax4 = fig.add_subplot(gs[1, 1])
ax4.axhline(1.0, color="red", linestyle=":", lw=1.5, label="Breakeven (PF=1)")
ax4.axhline(1.5, color="orange", linestyle=":", lw=1.2, alpha=0.7,
            label="Target baik (PF=1.5)")

valid_pf = df_agg.dropna(subset=["profit_factor"])
valid_pf = valid_pf[valid_pf["profit_factor"] < 10]  # clip outlier
ax4.plot(valid_pf["threshold"], valid_pf["profit_factor"], "k-", lw=2.5,
         label="Agregat", zorder=5)
for i, (yr, df_yr) in enumerate(df_per_year.items()):
    v = df_yr.dropna(subset=["profit_factor"])
    v = v[v["profit_factor"] < 10]
    ax4.plot(v["threshold"], v["profit_factor"],
             color=colors_year[i % len(colors_year)], lw=1.2,
             alpha=0.7, linestyle="--", label=str(yr))

ax4.set_xlabel("Threshold", fontsize=10)
ax4.set_ylabel("Profit Factor", fontsize=10)
ax4.set_title("Profit Factor vs Threshold\n(PF > 1 = profitable, PF > 1.5 = baik)",
              fontsize=10, fontweight="bold")
ax4.legend(fontsize=8, loc="upper left")
ax4.set_xlim([0.50, 0.85])
ax4.set_ylim([0, 5])

plt.suptitle("Analisis Sensitivitas Threshold — Random Forest Rolling Yearly\n"
             "Trade-off antara kualitas sinyal dan volume trade",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("output/threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/threshold_analysis.png")

# ── Simpan detail CSV ─────────────────────────────────────────
df_agg["year"] = "Agregat"
all_rows = [df_agg]
for yr, df_yr in df_per_year.items():
    df_yr["year"] = str(yr)
    all_rows.append(df_yr)

detail = pd.concat(all_rows, ignore_index=True)
detail.to_csv("output/threshold_detail.csv", index=False)
print("Tersimpan: output/threshold_detail.csv")

# ── Print tabel ringkasan ─────────────────────────────────────
print("\nRingkasan threshold (agregat):")
print(f"{'Threshold':>10} {'Trades':>8} {'WR':>8} {'Expect':>10} {'PF':>8}")
print("-" * 50)
for _, r in df_agg.iterrows():
    if pd.isna(r["expectancy"]): continue
    marker = " ◄ optimal" if (not valid_min.empty and
                               r["threshold"] == best_th) else ""
    print(f"  {r['threshold']:>8.2f} {r['trades']:>8.0f} "
          f"{r['WR']:>8.1f}% {r['expectancy']:>+10.4f} "
          f"{min(r['profit_factor'],9.99):>8.3f}{marker}")