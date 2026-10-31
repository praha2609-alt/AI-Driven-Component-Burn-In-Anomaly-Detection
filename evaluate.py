"""
File 4: Runs everything and produces the results for your paper.
  1. Full 168 h comparison table (precision, recall, F1, ROC-AUC, confusion matrix)
  2. Early-detection test (only first N hours of data)
  3. Cost trade-off (escapes vs overkill vs burn-in time)
Run:  python evaluate.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from features import load_data, build_features, get_labels
from detectors import run_all

# ---- Cost assumptions (arbitrary units, state them in the paper) ----
COST_ESCAPE = 100.0     # a weak unit that ships and fails in the field
COST_OVERKILL = 1.0     # a good unit wrongly scrapped
COST_PER_UNIT_HOUR = 0.02  # chamber time per unit per hour

CUTOFFS = [12, 24, 36, 48, 72, 96, 120, 168]

df = load_data()
y = get_labels(df)
n_units = len(y)

# ---------- 1. Full burn-in comparison ----------
X = build_features(df, 168)
rows = []
for name, (score, flag) in run_all(X).items():
    tn, fp, fn, tp = confusion_matrix(y, flag).ravel()
    rows.append({
        "method": name,
        "caught (TP)": tp, "missed (FN)": fn,
        "false alarms (FP)": fp, "good kept (TN)": tn,
        "precision": round(precision_score(y, flag, zero_division=0), 3),
        "recall": round(recall_score(y, flag), 3),
        "F1": round(f1_score(y, flag), 3),
        "ROC-AUC": round(roc_auc_score(y, score), 3),
    })
full = pd.DataFrame(rows)
full.to_csv("results_full_burnin.csv", index=False)
print("\n=== Full 168 h burn-in ===")
print(full.to_string(index=False))

# ---------- 2 + 3. Early detection and cost ----------
records = []
for c in CUTOFFS:
    Xc = build_features(df, c)
    for name, (score, flag) in run_all(Xc).items():
        tn, fp, fn, tp = confusion_matrix(y, flag).ravel()
        cost = fn * COST_ESCAPE + fp * COST_OVERKILL + n_units * c * COST_PER_UNIT_HOUR
        records.append({"hours": c, "method": name,
                        "recall": recall_score(y, flag),
                        "false_alarms": fp,
                        "ROC-AUC": roc_auc_score(y, score),
                        "total_cost": cost})
early = pd.DataFrame(records)
early.to_csv("results_early_detection.csv", index=False)
print("\n=== Early detection (recall by hours of data) ===")
print(early.pivot(index="hours", columns="method", values="recall").round(2).to_string())

# ---------- Plots ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for name, g in early.groupby("method"):
    ax[0].plot(g.hours, g.recall * 100, marker="o", label=name)
    ax[1].plot(g.hours, g["ROC-AUC"], marker="o", label=name)
    ax[2].plot(g.hours, g.total_cost, marker="o", label=name)
ax[0].set(title="Weak units caught vs burn-in hours", xlabel="Burn-in hours used", ylabel="Recall (%)")
ax[1].set(title="Separation quality (ROC-AUC)", xlabel="Burn-in hours used", ylabel="ROC-AUC")
ax[2].set(title="Total cost vs burn-in hours", xlabel="Burn-in hours used", ylabel="Cost (arbitrary units)")
for a in ax:
    a.grid(alpha=0.3)
ax[0].legend()
plt.tight_layout()
plt.savefig("early_detection_and_cost.png", dpi=130)
print("\nSaved: results_full_burnin.csv, results_early_detection.csv, early_detection_and_cost.png")
