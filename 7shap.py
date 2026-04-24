"""
viz_shap.py
===========
SHAP (SHapley Additive exPlanations) analysis.
Menjelaskan MENGAPA model membuat keputusan tertentu.

Install dulu: pip install shap

Output:
  output/shap_summary.png        ← beeswarm plot semua fitur
  output/shap_bar.png            ← mean absolute SHAP per fitur
  output/shap_dependence_TOP.png ← dependence plot fitur terpenting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, glob, warnings
warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    print("SHAP belum terinstall. Jalankan: pip install shap")
    exit()

os.makedirs("output", exist_ok=True)

MODEL_DIR = "data/models"
DATA_PATH = "data/dataset_ml.csv"
SAMPLE_N  = 2000  # subsample untuk kecepatan

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ── Load model terbaru ────────────────────────────────────────
model_files = sorted(glob.glob(f"{MODEL_DIR}/model_*.pkl"))
if not model_files:
    print(f"Tidak ada model di {MODEL_DIR}. Jalankan dulu 5train.py")
    exit()

# Pakai model tahun terakhir sebagai representatif
latest = joblib.load(model_files[-1])
model  = latest["model"]
feats  = latest["feature_cols"]
year   = latest["test_year"]
print(f"Menggunakan model tahun {year}")

# ── Load data test untuk tahun tersebut ──────────────────────
df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])

te_start = pd.to_datetime(latest["te_start"])
te_end   = pd.to_datetime(latest["te_end"])
df_te    = df[(df["time"] >= te_start) & (df["time"] <= te_end)]

X_te = df_te[feats].values.astype(np.float32)
y_te = df_te["label"].values

# Subsample jika terlalu besar
if len(X_te) > SAMPLE_N:
    idx  = np.random.choice(len(X_te), SAMPLE_N, replace=False)
    idx  = np.sort(idx)
    X_te = X_te[idx]
    y_te = y_te[idx]

print(f"Data test  : {len(X_te):,} baris")
print(f"Menghitung SHAP values... (bisa 1-2 menit)")

# ── Hitung SHAP ───────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te)

# Logika perbaikan:
# Cek apakah shap_values itu list (versi lama) atau ndarray 3D (versi baru)
if isinstance(shap_values, list):
    # Untuk Random Forest versi lama (list of arrays)
    sv = shap_values[1]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # Untuk SHAP versi baru yang mengembalikan (samples, features, classes)
    # Kita ambil index [:, :, 1] untuk Class 1 (Positive/Profit)
    sv = shap_values[:, :, 1]
else:
    # Jika sudah 2D (misal XGBoost atau model regresi)
    sv = shap_values

print(f"Selesai menghitung SHAP. Shape akhir sv: {sv.shape}")

# ── Plot 1: SHAP Summary (Beeswarm) ──────────────────────────
plt.figure(figsize=(9, 7))
shap.summary_plot(sv, X_te, feature_names=feats, show=False,
                  max_display=15, plot_type="dot", color_bar=True)
plt.title(f"SHAP Summary Plot — Random Forest (Test {year})\n"
          "Merah = nilai fitur tinggi | Biru = nilai fitur rendah",
          fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("output/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/shap_summary.png")

# ── Plot 2: SHAP Bar (mean |SHAP|) ───────────────────────────
plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X_te, feature_names=feats, show=False,
                  max_display=15, plot_type="bar")
plt.title(f"SHAP Feature Importance — Random Forest (Test {year})\n"
          "Rata-rata |SHAP value| per fitur",
          fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("output/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/shap_bar.png")

# ── Plot 3: SHAP Dependence plot fitur terpenting ────────────
mean_abs = np.abs(sv).mean(axis=0).flatten()
top_feat_idx  = int(np.argsort(mean_abs)[::-1].flatten()[0])
top_feat_name = feats[top_feat_idx]
second_idx    = int(np.argsort(mean_abs)[::-1].flatten()[1])
second_name   = feats[second_idx]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
shap.dependence_plot(top_feat_idx, sv, X_te, feature_names=feats,
                     interaction_index=second_idx, ax=axes[0], show=False)
axes[0].set_title(f"SHAP Dependence: {top_feat_name}\n"
                  f"(warna = {second_name})", fontsize=10, fontweight="bold")

shap.dependence_plot(second_idx, sv, X_te, feature_names=feats,
                     interaction_index=top_feat_idx, ax=axes[1], show=False)
axes[1].set_title(f"SHAP Dependence: {second_name}\n"
                  f"(warna = {top_feat_name})", fontsize=10, fontweight="bold")

plt.suptitle(f"SHAP Dependence Plot — Test {year}", fontsize=12,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("output/shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/shap_dependence.png")

# ── Print ringkasan ───────────────────────────────────────────
print("\nTop 10 fitur berdasarkan mean |SHAP|:")
print("-" * 40)
for i in np.argsort(mean_abs)[::-1][:10]:
    bar = "█" * int(mean_abs[i] / mean_abs.max() * 20)
    print(f"  {feats[i]:<22} {mean_abs[i]:.5f}  {bar}")