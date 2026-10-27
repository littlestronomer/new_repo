"""Ensemble v2: wider TabPFN member pool + pseudo-labeling experiment.

Phase A (pure): OOF predictions for 8 TabPFN members (different selectors,
filters, imputers, ensemble sizes, softmax temperatures), simplex blend.

Phase B (pseudo): per fold, a teacher (the v1 champion blend) predicts the
unlabeled X_test; the pseudo-labeled test rows are appended to the fold's
training data and every member is refit. No label leakage: teachers and
members only ever see fold-train labels.
"""

from dotenv import load_dotenv

load_dotenv()  # loads TABPFN_TOKEN etc. from .env

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline

from project1.src.data import data
from project1.src.filter import get_filter
from project1.src.scalers import get_scaler
from project1.src.imputation import get_imputer
from project1.src.selectors import get_selector
from project1.src.outliers import get_outliers
from project1.src.models import get_model
from project1.src.config import config

# %% members: (name, model params, filter params, imputer params, selector params)

W10 = {"method": "winsorize", "z_threshold": 10}
KNN = {"method": "knn", "n_neighbors": 15}
DEDUP = {"method": "unsupervised", "min_partner": 0.0, "max_correlation": 0.99}
MEDIAN = {"method": "median"}

MEMBERS = [
    ("tabpfn_lasso", {"method": "tabpfn", "n_estimators": 8}, W10, KNN,
     {"method": "lasso", "alpha": 0.1}),
    ("tabpfn_tree", {"method": "tabpfn", "n_estimators": 8}, W10, KNN,
     {"method": "tree", "threshold": "median"}),
    ("tabpfn_mi200", {"method": "tabpfn", "n_estimators": 8}, W10, KNN,
     {"method": "mutual_info", "k": 200}),
    ("tabpfn_all", {"method": "tabpfn", "n_estimators": 8}, W10, KNN,
     {"method": "passthrough"}),
    ("tabpfn_lasso_ne16", {"method": "tabpfn", "n_estimators": 16}, W10, KNN,
     {"method": "lasso", "alpha": 0.1}),
    ("tabpfn_lasso_t07", {"method": "tabpfn", "n_estimators": 8,
                          "softmax_temperature": 0.7}, W10, KNN,
     {"method": "lasso", "alpha": 0.1}),
    ("tabpfn_dedup_tree", {"method": "tabpfn", "n_estimators": 8}, DEDUP, KNN,
     {"method": "tree", "threshold": "median"}),
    ("tabpfn_lasso_medimp", {"method": "tabpfn", "n_estimators": 8}, W10, MEDIAN,
     {"method": "lasso", "alpha": 0.1}),
]

# v1 champion teacher (weights from runs/ensemble_result.json)
TEACHER_WEIGHTS = {"tabpfn_lasso": 0.571, "tabpfn_tree": 0.429}
TEACHER_MEMBERS = {n: m for n, m, *_ in MEMBERS if n in TEACHER_WEIGHTS}

OUTLIER_SETTING = {"method": "isolation_forest", "contamination": 0.01, "n_estimators": 200}

# %% data and folds

X, y, X_test, _ = data()
y = np.asarray(y, dtype=float)

splits = list(
    RepeatedKFold(
        n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
    ).split(X)
)
mask_outliers = [get_outliers(X.iloc[tr], **OUTLIER_SETTING) for tr, _ in splits]
names = [m[0] for m in MEMBERS]


def make_pipeline(filt, imputer, selector):
    return Pipeline(
        [
            ("filter", get_filter(**filt)),
            ("scaler", get_scaler(method="robust")),
            ("imputer", get_imputer(**imputer)),
            ("selector", get_selector(**selector)),
        ]
    )


oof_pure = pd.DataFrame(np.nan, index=X.index, columns=names)
oof_pseudo = pd.DataFrame(np.nan, index=X.index, columns=names)

# %% CV loop

for k, ((tr, val), mask) in enumerate(zip(splits, mask_outliers)):
    X_tr, y_tr = X.iloc[tr][~mask], y[tr][~mask]
    X_val = X.iloc[val]
    print(f"--- fold {k + 1}/{len(splits)} ---")

    # teacher predicts unlabeled test for the pseudo phase
    pseudo_y = np.zeros(len(X_test))
    for name, w in TEACHER_WEIGHTS.items():
        *_, filt, imp, sel = next(m for m in MEMBERS if m[0] == name)
        prep = make_pipeline(filt, imp, sel)
        Z_tr = prep.fit_transform(X_tr, y_tr)
        model = get_model(n_features=Z_tr.shape[1], **TEACHER_MEMBERS[name])
        model.fit(Z_tr, y_tr)
        pseudo_y += w * model.predict(prep.transform(X_test))
    X_aug = pd.concat([X_tr, X_test], ignore_index=True)
    y_aug = np.concatenate([y_tr, pseudo_y])

    # members: pure and pseudo
    for name, model_params, filt, imp, sel in MEMBERS:
        prep = make_pipeline(filt, imp, sel)

        Z_tr = prep.fit_transform(X_tr, y_tr)
        m = get_model(n_features=Z_tr.shape[1], **model_params)
        m.fit(Z_tr, y_tr)
        oof_pure.loc[X_val.index, name] = m.predict(prep.transform(X_val))

        Z_aug = prep.fit_transform(X_aug, y_aug)
        m = get_model(n_features=Z_aug.shape[1], **model_params)
        m.fit(Z_aug, y_aug)
        oof_pseudo.loc[X_val.index, name] = m.predict(prep.transform(X_val))

        print(f"  {name} done")

# %% scores and blends


def blend(df):
    def neg_r2(w):
        return -r2_score(y, df[names].to_numpy() @ w)

    res = minimize(
        neg_r2,
        x0=np.full(len(names), 1 / len(names)),
        bounds=[(0, 1)] * len(names),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
    )
    return res.x, -res.fun


print("\n=== OOF member scores (pure | pseudo) ===")
for n in names:
    print(f"  {n}: {r2_score(y, oof_pure[n]):.4f} | {r2_score(y, oof_pseudo[n]):.4f}")

w_pure, r2_pure = blend(oof_pure)
w_pseudo, r2_pseudo = blend(oof_pseudo)

print(f"\npure blend R2:   {r2_pure:.4f}")
print("  weights:", {n: round(w, 3) for n, w in zip(names, w_pure) if w > 0.001})
print(f"pseudo blend R2: {r2_pseudo:.4f}")
print("  weights:", {n: round(w, 3) for n, w in zip(names, w_pseudo) if w > 0.001})

# %% persist

runs_dir = Path(__file__).resolve().parent / "runs"
out = {
    "member_oof_pure": {n: r2_score(y, oof_pure[n]) for n in names},
    "member_oof_pseudo": {n: r2_score(y, oof_pseudo[n]) for n in names},
    "pure_blend_r2": r2_pure,
    "pseudo_blend_r2": r2_pseudo,
    "pure_weights": dict(zip(names, w_pure.round(4).tolist())),
    "pseudo_weights": dict(zip(names, w_pseudo.round(4).tolist())),
}
(runs_dir / "ensemble_v2_result.json").write_text(json.dumps(out, indent=2))
oof_pure.to_csv(runs_dir / "ensemble_v2_oof_pure.csv")
oof_pseudo.to_csv(runs_dir / "ensemble_v2_oof_pseudo.csv")
print(f"\nsaved: {runs_dir / 'ensemble_v2_result.json'}")
