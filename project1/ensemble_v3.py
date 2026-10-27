"""Ensemble v3: iterated (round-2) pseudo-labeling.

Round 0: teacher (v2 pseudo-weighted members, fit on fold-train) labels X_test.
Round 1: members refit on fold-train + pseudo0 labels -> OOF_r1 (= v2 pseudo).
Round 2: the same weighted members refit on fold-train + pseudo0 become the
new teacher -> pseudo1 labels; members refit on fold-train + pseudo1 ->
OOF_r2. Compare blends. No label leakage: only fold-train labels are ever
used.
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

W10 = {"method": "winsorize", "z_threshold": 10}
KNN = {"method": "knn", "n_neighbors": 15}
DEDUP = {"method": "unsupervised", "min_partner": 0.0, "max_correlation": 0.99}

# (name, model params, filter params, selector params)
MEMBERS = [
    ("tabpfn_lasso", {"method": "tabpfn", "n_estimators": 8}, W10,
     {"method": "lasso", "alpha": 0.1}),
    ("tabpfn_tree", {"method": "tabpfn", "n_estimators": 8}, W10,
     {"method": "tree", "threshold": "median"}),
    ("tabpfn_mi200", {"method": "tabpfn", "n_estimators": 8}, W10,
     {"method": "mutual_info", "k": 200}),
    ("tabpfn_all", {"method": "tabpfn", "n_estimators": 8}, W10,
     {"method": "passthrough"}),
    ("tabpfn_lasso_ne16", {"method": "tabpfn", "n_estimators": 16}, W10,
     {"method": "lasso", "alpha": 0.1}),
    ("tabpfn_dedup_tree", {"method": "tabpfn", "n_estimators": 8}, DEDUP,
     {"method": "tree", "threshold": "median"}),
]

# fixed teacher weights (v2 pseudo weights, renormalized over the 4 members)
TEACHER_WEIGHTS = {
    "tabpfn_lasso": 0.105,
    "tabpfn_mi200": 0.24,
    "tabpfn_lasso_ne16": 0.269,
    "tabpfn_dedup_tree": 0.386,
}

OUTLIER_SETTING = {"method": "isolation_forest", "contamination": 0.01, "n_estimators": 200}

X, y, X_test, _ = data()
y = np.asarray(y, dtype=float)

splits = list(
    RepeatedKFold(
        n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
    ).split(X)
)
mask_outliers = [get_outliers(X.iloc[tr], **OUTLIER_SETTING) for tr, _ in splits]
names = [m[0] for m in MEMBERS]


def make_pipeline(filt, sel):
    return Pipeline(
        [
            ("filter", get_filter(**filt)),
            ("scaler", get_scaler(method="robust")),
            ("imputer", get_imputer(**KNN)),
            ("selector", get_selector(**sel)),
        ]
    )


oof_r1 = pd.DataFrame(np.nan, index=X.index, columns=names)
oof_r2 = pd.DataFrame(np.nan, index=X.index, columns=names)


def fit_predict(name, model_params, filt, sel, X_fit, y_fit, X_pred):
    prep = make_pipeline(filt, sel)
    Z_fit = prep.fit_transform(X_fit, y_fit)
    model = get_model(n_features=Z_fit.shape[1], **model_params)
    model.fit(Z_fit, y_fit)
    return model.predict(prep.transform(X_pred))


for k, ((tr, val), mask) in enumerate(zip(splits, mask_outliers)):
    X_tr, y_tr = X.iloc[tr][~mask], y[tr][~mask]
    X_val = X.iloc[val]
    print(f"--- fold {k + 1}/{len(splits)} ---")

    # round 0: teacher labels the test set
    pseudo0 = np.zeros(len(X_test))
    for name, w in TEACHER_WEIGHTS.items():
        model_params, filt, sel = next((m[1], m[2], m[3]) for m in MEMBERS if m[0] == name)
        pseudo0 += w * fit_predict(name, model_params, filt, sel, X_tr, y_tr, X_test)
    X_aug0 = pd.concat([X_tr, X_test], ignore_index=True)
    y_aug0 = np.concatenate([y_tr, pseudo0])

    # round 1: members on fold-train + pseudo0 (also the round-2 teacher models)
    preds_aug0, preds_val1 = {}, {}
    for name, model_params, filt, sel in MEMBERS:
        preds_aug0[name] = fit_predict(name, model_params, filt, sel, X_tr, y_tr, X_aug0)
        preds_val1[name] = fit_predict(name, model_params, filt, sel, X_aug0, y_aug0, X_val)
        oof_r1.loc[X_val.index, name] = preds_val1[name]
        print(f"  r1 {name} done")

    # round 2: round-1 teacher (weighted members evaluated on the augmented set)
    pseudo1 = np.zeros(len(X_test))
    for name, w in TEACHER_WEIGHTS.items():
        pseudo1 += w * preds_aug0[name][len(X_tr):]  # test slice of augmented preds
    X_aug1 = pd.concat([X_tr, X_test], ignore_index=True)
    y_aug1 = np.concatenate([y_tr, pseudo1])

    for name, model_params, filt, sel in MEMBERS:
        oof_r2.loc[X_val.index, name] = fit_predict(
            name, model_params, filt, sel, X_aug1, y_aug1, X_val
        )
        print(f"  r2 {name} done")


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


print("\n=== OOF member scores (round1 | round2) ===")
for n in names:
    print(f"  {n}: {r2_score(y, oof_r1[n]):.4f} | {r2_score(y, oof_r2[n]):.4f}")

w1, r2_1 = blend(oof_r1)
w2, r2_2 = blend(oof_r2)
print(f"\nround-1 blend R2: {r2_1:.4f}")
print("  weights:", {n: round(w, 3) for n, w in zip(names, w1) if w > 0.001})
print(f"round-2 blend R2: {r2_2:.4f}")
print("  weights:", {n: round(w, 3) for n, w in zip(names, w2) if w > 0.001})

runs_dir = Path(__file__).resolve().parent / "runs"
out = {
    "member_oof_r1": {n: r2_score(y, oof_r1[n]) for n in names},
    "member_oof_r2": {n: r2_score(y, oof_r2[n]) for n in names},
    "r1_blend_r2": r2_1,
    "r2_blend_r2": r2_2,
    "r1_weights": dict(zip(names, w1.round(4).tolist())),
    "r2_weights": dict(zip(names, w2.round(4).tolist())),
}
(runs_dir / "ensemble_v3_result.json").write_text(json.dumps(out, indent=2))
oof_r1.to_csv(runs_dir / "ensemble_v3_oof_r1.csv")
oof_r2.to_csv(runs_dir / "ensemble_v3_oof_r2.csv")
print(f"\nsaved: {runs_dir / 'ensemble_v3_result.json'}")
