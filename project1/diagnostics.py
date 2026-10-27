"""CPU-only diagnostics on saved OOF predictions (no model training).

1. Residual analysis of the champion blend (v3 round-2 OOF): per-fold R2,
   age-binned bias/MAE, variance shrinkage, worst rows.
2. Cross-fitted linear recalibration check (honest estimate, no retraining).
3. Outlier audit: which rows iForest drops, whether they overlap with
   high-residual rows, and how iForest / mahalanobis / LOF masks compare.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold

from project1.src.data import data
from project1.src.outliers import get_outliers
from project1.src.config import config

RUNS = Path(__file__).resolve().parent / "runs"

# %% champion OOF blend: v3 round-2 weights

X, y, _, _ = data()
y = y.astype(float)

oof_members = pd.read_csv(RUNS / "ensemble_v3_oof_r2.csv", index_col=0).loc[X.index]
result = json.loads((RUNS / "ensemble_v3_result.json").read_text())
w = result["r2_weights"]
pred = sum(w[n] * oof_members[n] for n in oof_members.columns if w.get(n, 0) > 0)
resid = y - pred

print(f"champion blend OOF R2: {r2_score(y, pred):.4f}")
print(f"prediction std {pred.std():.2f} vs target std {y.std():.2f} "
      f"(shrinkage ratio {pred.std() / y.std():.3f})")

# %% fold reconstruction (same seed/protocol as the experiments)

splits = list(
    RepeatedKFold(
        n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
    ).split(X)
)
fold_of = pd.Series(0, index=X.index)
for k, (_, val) in enumerate(splits):
    fold_of.iloc[val] = k

print("\n=== per-fold R2 ===")
per_fold = {k: r2_score(y[fold_of == k], pred[fold_of == k]) for k in sorted(fold_of.unique())}
for k, s in per_fold.items():
    print(f"  fold {k}: {s:.4f}")
print(f"  mean {np.mean(list(per_fold.values())):.4f}")

# %% age-binned residuals

print("\n=== age-binned residuals ===")
bins = [42, 55, 65, 75, 85, 98]
lab = ["42-54", "55-64", "65-74", "75-84", "85-97"]
grp = pd.cut(y, bins=bins, labels=lab, right=False)
tbl = pd.DataFrame(
    {
        "n": grp.value_counts().reindex(lab),
        "bias": pd.Series(resid, index=y.index).groupby(grp).mean().reindex(lab),
        "mae": pd.Series((y - pred).abs(), index=y.index).groupby(grp).mean().reindex(lab),
        "pred_std": pred.groupby(grp).std().reindex(lab),
        "y_std": y.groupby(grp).std().reindex(lab),
    }
).round(2)
print(tbl.to_string())

slope = np.polyfit(pred, y, 1)[0]
corr = np.corrcoef(pred, y)[0, 1]
print(f"\nglobal regression slope y~pred: {slope:.3f} (<1 = shrinkage), corr {corr:.3f}")

# %% cross-fitted recalibration

print("\n=== linear recalibration y ~ a + b*pred (cross-fitted by fold) ===")
cal_raw, cal_fit = [], []
for k in sorted(fold_of.unique()):
    te = fold_of == k
    tr = ~te
    a, b = np.polyfit(pred[tr], y[tr], 1)
    cal_raw.append(r2_score(y[te], pred[te]))
    cal_fit.append(r2_score(y[te], a + b * pred[te]))
print(f"raw mean {np.mean(cal_raw):.4f} -> calibrated mean {np.mean(cal_fit):.4f} "
      f"(delta {np.mean(cal_fit) - np.mean(cal_raw):+.4f})")

# %% outlier audit

print("\n=== outlier audit ===")
X_raw = pd.read_csv(Path(__file__).resolve().parent / "data" / "X_train.csv", index_col=0)
med = X_raw.median()
mad = 1.4826 * (X_raw - med).abs().median().replace(0, np.nan)
rz = (X_raw - med).abs() / mad
ext_cells = (rz > 20).sum(axis=1)

iso_full = get_outliers(X, method="isolation_forest", contamination=0.01, n_estimators=200)
print(f"iForest full-train flags: {iso_full.sum()} of {len(X)}")

flagged = X.index[iso_full]
audit = pd.DataFrame(
    {
        "y": y[flagged],
        "pred": pred[flagged],
        "resid": resid[flagged],
        "ext_cells_z20": ext_cells[flagged],
    }
).round(2)
print("\nflagged rows (iForest):")
print(audit.sort_values("resid", key=np.abs, ascending=False).to_string())

# overlap with high-residual rows
worst = resid.abs().sort_values(ascending=False).head(30).index
print(f"\nof the 30 highest-|residual| rows, {iso_full[worst].sum()} are iForest-flagged")
print(f"mean |resid| of flagged rows: {resid[flagged].abs().mean():.2f} "
      f"vs unflagged {resid[~iso_full].abs().mean():.2f}")

# alternative methods
maha = get_outliers(X, method="mahalanobis", alpha=0.001, variance=0.95)
lof = get_outliers(X, method="lof", contamination=0.01, n_neighbors=20, variance=0.8)
print(f"\nmahalanobis flags: {maha.sum()}, LOF flags: {lof.sum()}")
print(f"overlap iForest & mahalanobis: {(iso_full & maha).sum()}, iForest & LOF: {(iso_full & lof).sum()}")
print(f"mean |resid| mahalanobis-flagged: {resid[maha].abs().mean():.2f}")
print(f"mean |resid| LOF-flagged: {resid[lof].abs().mean():.2f}")

# %% worst rows table

print("\n=== 15 worst predictions ===")
worst15 = resid.abs().sort_values(ascending=False).head(15).index
print(
    pd.DataFrame(
        {
            "y": y[worst15],
            "pred": pred[worst15].round(1),
            "resid": resid[worst15].round(1),
            "fold": fold_of[worst15],
            "ext_cells_z20": ext_cells[worst15],
            "iforest_flag": iso_full[worst15].astype(int),
        }
    ).to_string()
)
