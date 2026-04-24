"""
viz_confusion_matrix.py
=======================
Confusion Matrix, Classification Report, dan ROC Curve
dari hasil prediksi 5train.py.

Output:
  output/confusion_matrix.png     ← matrix per tahun + agregat
  output/roc_curve.png            ← ROC curve per tahun
  output/classification_report.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── Load prediksi ─────────────────────────────────────────────
if not os.path.exists(PRED_PATH):
    print(f"{PRED_PATH} tidak ditemukan. Jalankan dulu 5train.py")
    exit()

df    = pd.read_csv(PRED_PATH, parse_dates=["time"])
years = sorted(df["test_year"].unique())
print(f"Data prediksi: {len(df):,} baris | Tahun: {years}")

# ── Plot 1: Confusion Matrix per tahun + agregat ──────────────
n_cols = len(years) + 1
fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))

all_y_true, all_y_pred = [], []

for i, year in enumerate(years):
    sub    = df[df["test_year"] == year]
    y_true = sub["label"].values
    y_pred = sub["pred"].values

    all_y_true.extend(y_true)
    all_y_pred.extend(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["SL (0)", "TP (1)"])
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(f"Test {year}\n"
                      f"WR={y_true[y_pred==1].mean()*100:.1f}% "
                      f"(n={y_pred.sum()})",
                      fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Prediksi", fontsize=9)
    axes[i].set_ylabel("Aktual", fontsize=9)

# Agregat
cm_all = confusion_matrix(all_y_true, all_y_pred)
disp_all = ConfusionMatrixDisplay(cm_all, display_labels=["SL (0)", "TP (1)"])
disp_all.plot(ax=axes[-1], colorbar=False, cmap="Greens")
wr_all = np.array(all_y_true)[np.array(all_y_pred)==1].mean()
axes[-1].set_title(f"AGREGAT\n"
                   f"WR={wr_all*100:.1f}% "
                   f"(n={np.array(all_y_pred).sum()})",
                   fontsize=10, fontweight="bold")
axes[-1].set_xlabel("Prediksi", fontsize=9)
axes[-1].set_ylabel("Aktual", fontsize=9)

plt.suptitle("Confusion Matrix — Random Forest Rolling Yearly\n"
             "TP = hit profit target | SL = hit stop loss",
             fontsize=12, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/confusion_matrix.png")

# ── Plot 2: ROC Curve per tahun ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#185FA5","#1D9E75","#D85A30","#7F77DD","#BA7517"]

all_fpr, all_tpr, all_probs_y = [], [], []

for i, year in enumerate(years):
    sub    = df[df["test_year"] == year]
    y_true = sub["label"].values
    y_prob = sub["prob"].values

    if len(np.unique(y_true)) < 2: continue

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    color       = colors[i % len(colors)]
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{year} (AUC = {roc_auc:.3f})")
    all_fpr.extend(fpr); all_tpr.extend(tpr)

# Agregat ROC
y_all_true = df["label"].values
y_all_prob = df["prob"].values
fpr_all, tpr_all, _ = roc_curve(y_all_true, y_all_prob)
auc_all = auc(fpr_all, tpr_all)
ax.plot(fpr_all, tpr_all, "k--", lw=2.5,
        label=f"Agregat (AUC = {auc_all:.3f})")

ax.plot([0,1],[0,1], "gray", lw=1, linestyle=":", alpha=0.7,
        label="Random classifier")
ax.fill_between(fpr_all, tpr_all, alpha=0.05, color="black")

ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curve — Random Forest Rolling Yearly\n"
             "Semakin ke kiri-atas = semakin baik",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("output/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Tersimpan: output/roc_curve.png")

# ── Text: Classification Report ───────────────────────────────
report_lines = ["=" * 60]
report_lines.append("  CLASSIFICATION REPORT — Random Forest Rolling Yearly")
report_lines.append("=" * 60)

for year in years:
    sub    = df[df["test_year"] == year]
    y_true = sub["label"].values
    y_pred = sub["pred"].values
    th     = sub["threshold"].iloc[0]
    report_lines.append(f"\nTest Year: {year} | Threshold: {th}")
    report_lines.append("-" * 40)
    report_lines.append(classification_report(
        y_true, y_pred,
        target_names=["SL (0)", "TP (1)"],
        zero_division=0
    ))

report_lines.append("=" * 60)
report_lines.append("AGREGAT (semua tahun)")
report_lines.append("-" * 40)
report_lines.append(classification_report(
    all_y_true, all_y_pred,
    target_names=["SL (0)", "TP (1)"],
    zero_division=0
))

report_text = "\n".join(report_lines)
with open("output/classification_report.txt", "w") as f:
    f.write(report_text)
print("Tersimpan: output/classification_report.txt")
print("\n" + report_text)