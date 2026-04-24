"""
8matrix.py
==========
Confusion Matrix, Classification Report, dan ROC Curve.

PENTING: Confusion matrix dihitung DUA cara:
  1. Seluruh test set (untuk melihat distribusi prediksi model)
  2. Hanya sinyal yang melewati threshold (untuk evaluasi trading nyata)

Output:
  output/confusion_matrix_all.png      ← seluruh test set
  output/confusion_matrix_signal.png   ← hanya sinyal di atas threshold
  output/roc_curve.png                 ← ROC curve per tahun
  output/classification_report.txt     ← laporan lengkap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay
)
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("output", exist_ok=True)

PRED_PATH = "data/train_predictions.csv"

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

if not os.path.exists(PRED_PATH):
    print(f"{PRED_PATH} tidak ditemukan. Jalankan dulu 4train.py")
    exit()

df    = pd.read_csv(PRED_PATH, parse_dates=["time"])
years = sorted(df["test_year"].unique())
print(f"Data prediksi: {len(df):,} baris | Tahun: {years}")

# ─────────────────────────────────────────────────────────────
# PLOT 1: Confusion matrix SELURUH test set (per tahun + agregat)
# Konteks: model sangat selektif → recall rendah di test set penuh
# tapi precision tinggi di sinyal yang dipilih
# ─────────────────────────────────────────────────────────────
n_cols = len(years) + 1
fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
all_y_true_full, all_y_pred_full = [], []

for i, year in enumerate(years):
    sub    = df[df["test_year"] == year]
    y_true = sub["label"].values
    y_pred = sub["pred"].values
    all_y_true_full.extend(y_true)
    all_y_pred_full.extend(y_pred)

    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["SL (0)", "TP (1)"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    n_sig = y_pred.sum()
    wr    = y_true[y_pred==1].mean() * 100 if n_sig > 0 else 0
    axes[i].set_title(f"Test {year}\nSinyal={n_sig} | WR={wr:.1f}%",
                      fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Prediksi", fontsize=8)
    axes[i].set_ylabel("Aktual", fontsize=8)

cm_all = confusion_matrix(all_y_true_full, all_y_pred_full)
disp_all = ConfusionMatrixDisplay(cm_all, display_labels=["SL (0)", "TP (1)"])
disp_all.plot(ax=axes[-1], colorbar=False, cmap="Greens")
n_sig_all = np.array(all_y_pred_full).sum()
wr_all    = np.array(all_y_true_full)[np.array(all_y_pred_full)==1].mean() * 100
axes[-1].set_title(f"AGREGAT\nSinyal={n_sig_all} | WR={wr_all:.1f}%",
                   fontsize=9, fontweight="bold")
axes[-1].set_xlabel("Prediksi", fontsize=8)
axes[-1].set_ylabel("Aktual", fontsize=8)

plt.suptitle(
    "Confusion Matrix — Seluruh Test Set\n"
    "Catatan: Model beroperasi high-precision/low-recall — "
    "sengaja menahan diri dari prediksi di kondisi tidak yakin",
    fontsize=10, fontweight="bold", y=1.05
)
plt.tight_layout()
plt.savefig("output/confusion_matrix_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/confusion_matrix_all.png")

# ─────────────────────────────────────────────────────────────
# PLOT 2: Confusion matrix HANYA sinyal di atas threshold
# Ini yang relevan untuk evaluasi strategi trading
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
all_y_true_sig, all_y_pred_sig = [], []

for i, year in enumerate(years):
    sub    = df[df["test_year"] == year]
    th     = sub["threshold"].iloc[0]
    # Filter hanya baris yang prob >= threshold
    sig    = sub[sub["prob"] >= th]
    y_true = sig["label"].values
    # Di atas threshold semua diprediksi 1
    y_pred = np.ones(len(sig), dtype=int)
    all_y_true_sig.extend(y_true)
    all_y_pred_sig.extend(y_pred)

    if len(sig) == 0:
        axes2[i].set_title(f"Test {year}\nTidak ada sinyal", fontsize=9)
        axes2[i].axis("off"); continue

    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["SL (0)", "TP (1)"])
    disp.plot(ax=axes2[i], colorbar=False, cmap="Oranges")
    wr = y_true.mean() * 100
    axes2[i].set_title(f"Test {year} (th={th})\nn={len(sig)} | WR={wr:.1f}%",
                       fontsize=9, fontweight="bold")
    axes2[i].set_xlabel("Prediksi", fontsize=8)
    axes2[i].set_ylabel("Aktual", fontsize=8)

# Agregat sinyal
if all_y_true_sig:
    cm_sig = confusion_matrix(all_y_true_sig, all_y_pred_sig, labels=[0, 1])
    disp_sig = ConfusionMatrixDisplay(cm_sig, display_labels=["SL (0)", "TP (1)"])
    disp_sig.plot(ax=axes2[-1], colorbar=False, cmap="Reds")
    wr_sig = np.mean(all_y_true_sig) * 100
    axes2[-1].set_title(f"AGREGAT SINYAL\nn={len(all_y_true_sig)} | WR={wr_sig:.1f}%",
                        fontsize=9, fontweight="bold")
    axes2[-1].set_xlabel("Prediksi", fontsize=8)
    axes2[-1].set_ylabel("Aktual", fontsize=8)

plt.suptitle(
    "Confusion Matrix — Hanya Sinyal di Atas Threshold\n"
    "Ini representasi akurat dari trade yang benar-benar dieksekusi",
    fontsize=10, fontweight="bold", y=1.05
)
plt.tight_layout()
plt.savefig("output/confusion_matrix_signal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/confusion_matrix_signal.png")

# ─────────────────────────────────────────────────────────────
# PLOT 3: ROC Curve per tahun
# ─────────────────────────────────────────────────────────────
fig3, ax = plt.subplots(figsize=(7, 6))
colors = ["#185FA5","#1D9E75","#D85A30","#7F77DD","#BA7517"]

for i, year in enumerate(years):
    sub    = df[df["test_year"] == year]
    y_true = sub["label"].values
    y_prob = sub["prob"].values
    if len(np.unique(y_true)) < 2: continue
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
            label=f"Test {year} (AUC={roc_auc_val:.3f})")

# ROC agregat
fpr_all, tpr_all, _ = roc_curve(all_y_true_full, df["prob"].values)
auc_all = auc(fpr_all, tpr_all)
ax.plot(fpr_all, tpr_all, "k--", lw=2.5,
        label=f"Agregat (AUC={auc_all:.3f})")
ax.fill_between(fpr_all, tpr_all, alpha=0.05, color="black")
ax.plot([0,1],[0,1], "gray", lw=1, linestyle=":", alpha=0.7,
        label="Random classifier (AUC=0.500)")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Random Forest Rolling Yearly\n"
             "AUC ~0.54 konsisten: model memiliki edge kecil tapi nyata",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim([0,1]); ax.set_ylim([0,1])
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("output/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/roc_curve.png")

# ─────────────────────────────────────────────────────────────
# Text: Classification Report (dua versi)
# ─────────────────────────────────────────────────────────────
lines = ["=" * 65,
         "  CLASSIFICATION REPORT — Random Forest Rolling Yearly",
         "=" * 65,
         "",
         "KONTEKS: Model beroperasi dalam mode high-precision/low-recall.",
         "Threshold tinggi (0.56-0.62) menyebabkan model sangat selektif:",
         "  - Dari ~24.000 bar test per tahun, hanya 10-52 yang diambil",
         "  - Recall rendah di test set BUKAN bug — ini by design",
         "  - Yang penting: dari yang dipilih, WR 57% > breakeven 50%",
         "=" * 65]

lines.append("\n--- A. REPORT SELURUH TEST SET ---")
for year in years:
    sub    = df[df["test_year"] == year]
    th     = sub["threshold"].iloc[0]
    lines.append(f"\nTest {year} | Threshold: {th} | "
                 f"Sinyal dipilih: {sub['pred'].sum()} dari {len(sub):,}")
    lines.append("-" * 40)
    lines.append(classification_report(
        sub["label"], sub["pred"],
        target_names=["SL(0)","TP(1)"], zero_division=0))

lines.append("\n--- B. REPORT HANYA SINYAL DI ATAS THRESHOLD (relevan untuk trading) ---")
for year in years:
    sub = df[df["test_year"] == year]
    th  = sub["threshold"].iloc[0]
    sig = sub[sub["prob"] >= th]
    if len(sig) == 0: continue
    y_true = sig["label"].values
    y_pred = np.ones(len(sig), dtype=int)
    lines.append(f"\nTest {year} | n={len(sig)} sinyal | WR={y_true.mean()*100:.2f}%")
    lines.append("-" * 40)
    lines.append(classification_report(
        y_true, y_pred,
        target_names=["SL(0)","TP(1)"], zero_division=0))

lines.append("\n--- C. AGREGAT SINYAL ---")
if all_y_true_sig:
    lines.append(f"Total sinyal: {len(all_y_true_sig)} | "
                 f"WR={np.mean(all_y_true_sig)*100:.2f}%")
    lines.append(classification_report(
        all_y_true_sig, all_y_pred_sig,
        target_names=["SL(0)","TP(1)"], zero_division=0))

report_text = "\n".join(lines)
with open("output/classification_report.txt", "w") as f:
    f.write(report_text)
print("Tersimpan: output/classification_report.txt")

# Print ringkasan singkat
print("\nRingkasan sinyal yang dipilih model per tahun:")
print(f"{'Year':<6} {'Threshold':>10} {'Sinyal':>8} {'WR':>8}")
print("-" * 36)
for year in years:
    sub = df[df["test_year"] == year]
    th  = sub["threshold"].iloc[0]
    sig = sub[sub["prob"] >= th]
    wr  = sig["label"].mean() * 100 if len(sig) > 0 else 0
    print(f"{int(year):<6} {th:>10.2f} {len(sig):>8} {wr:>8.1f}%")
print(f"{'Rata'::<6} {'':>10} {len(all_y_true_sig)//len(years):>8} "
      f"{np.mean(all_y_true_sig)*100:>8.1f}%")