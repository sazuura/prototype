"""
viz_feature_importance.py
=========================
Visualisasi Feature Importance dari model Random Forest
yang sudah dilatih di 5train.py.

Output:
  output/feat_importance_per_year.png
  output/feat_importance_aggregate.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, os, glob

os.makedirs("output", exist_ok=True)

MODEL_DIR    = "data/models"
TOP_N        = 15
FONT         = "DejaVu Sans"

plt.rcParams.update({
    "font.family"      : FONT,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})

# ── Load semua model ──────────────────────────────────────────
model_files = sorted(glob.glob(f"{MODEL_DIR}/model_*.pkl"))
if not model_files:
    print(f"Tidak ada model di {MODEL_DIR}. Jalankan dulu 5train.py")
    exit()

fi_per_year = {}
all_fi      = {}

for path in model_files:
    bundle = joblib.load(path)
    year   = bundle["test_year"]
    model  = bundle["model"]
    feats  = bundle["feature_cols"]

    fi = dict(zip(feats, model.feature_importances_))
    fi_per_year[year] = fi
    for f, v in fi.items():
        all_fi[f] = all_fi.get(f, 0) + v

# Normalisasi agregat
total = sum(all_fi.values())
all_fi = {k: v/total for k, v in all_fi.items()}

# ── Plot 1: Feature importance AGREGAT ───────────────────────
agg_df = pd.DataFrame.from_dict(all_fi, orient="index", columns=["importance"])
agg_df = agg_df.sort_values("importance", ascending=True).tail(TOP_N)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#185FA5" if i >= len(agg_df)-5 else "#B5D4F4"
          for i in range(len(agg_df))]
bars = ax.barh(agg_df.index, agg_df["importance"], color=colors, height=0.65)

for bar, val in zip(bars, agg_df["importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_xlabel("Importance (rata-rata semua window)", fontsize=11)
ax.set_title(f"Top {TOP_N} Feature Importance — Random Forest\n"
             f"Rolling Yearly Walk-Forward ({min(fi_per_year)} – {max(fi_per_year)})",
             fontsize=12, fontweight="bold", pad=12)
ax.axvline(agg_df["importance"].median(), color="gray",
           linestyle=":", alpha=0.6, label="Median")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("output/feat_importance_aggregate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/feat_importance_aggregate.png")

# ── Plot 2: Feature importance PER TAHUN (heatmap) ───────────
years      = sorted(fi_per_year.keys())
# Ambil top 15 fitur dari agregat
top_feats  = agg_df.index.tolist()[::-1]  # urutan descending

matrix = np.array([[fi_per_year[y].get(f, 0) for y in years]
                   for f in top_feats])

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(matrix, aspect="auto", cmap="Blues")

ax.set_xticks(range(len(years)))
ax.set_xticklabels([str(y) for y in years], fontsize=11)
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats, fontsize=10)

for i in range(len(top_feats)):
    for j in range(len(years)):
        val = matrix[i, j]
        color = "white" if val > matrix.max() * 0.6 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=8, color=color)

plt.colorbar(im, ax=ax, label="Importance", shrink=0.8)
ax.set_title(f"Feature Importance per Tahun (Heatmap)\n"
             f"Top {TOP_N} fitur — Random Forest Rolling Yearly",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Tahun Test", fontsize=11)
plt.tight_layout()
plt.savefig("output/feat_importance_per_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/feat_importance_per_year.png")

# ── Print ringkasan ───────────────────────────────────────────
print("\nTop 10 fitur (agregat):")
print("-" * 35)
for feat, imp in sorted(all_fi.items(), key=lambda x: x[1], reverse=True)[:10]:
    bar = "█" * int(imp * 300)
    print(f"  {feat:<22} {imp:.4f}  {bar}")