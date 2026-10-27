"""OOF ensemble: TabPFN + gradient boosting members, simplex-weighted blend.

Members differ in model and feature set (lasso(0.1) -> ~260 features vs
tree(median) -> ~414 features). All preprocessing is fit per fold on the
training split only (identical protocol to the runner). Blend weights are
optimized on out-of-fold predictions to maximize R^2.
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

# %% members: (name, model params, selector params)

MEMBERS = [
    ("tabpfn_lasso", {"method": "tabpfn", "n_estimators": 8},
     {"method": "lasso", "alpha": 0.1}),
    ("tabpfn_tree", {"method": "tabpfn", "n_estimators": 8},
     {"method": "tree", "threshold": "median"}),
    ("catboost_tree", {"method": "catboost", "iterations": 650, "depth": 6,
                       "learning_rate": 0.035, "l2_leaf_reg": 3, "thread_count": 8},
     {"method": "tree", "threshold": "median"}),
    ("catboost_lasso", {"method": "catboost", "iterations": 650, "depth": 4,
                        "learning_rate": 0.09, "l2_leaf_reg": 10, "thread_count": 8},
     {"method": "lasso", "alpha": 0.1}),
    ("lightgbm_tree", {"method": "lightgbm", "n_estimators": 650, "num_leaves": 15,
                       "learning_rate": 0.035, "min_child_samples": 30, "n_jobs": 8},
     {"method": "tree", "threshold": "median"}),
    ("lightgbm_lasso", {"method": "lightgbm", "n_estimators": 650, "num_leaves": 31,
                        "learning_rate": 0.09, "min_child_samples": 10, "n_jobs": 8},
     {"method": "lasso", "alpha": 0.1}),
    ("xgboost_tree", {"method": "xgboost", "n_estimators": 650, "max_depth": 5,
                      "learning_rate": 0.035, "min_child_weight": 1, "n_jobs": 8},
     {"method": "tree", "threshold": "median"}),
    ("xgboost_lasso", {"method": "xgboost", "n_estimators": 650, "max_depth": 3,
                       "learning_rate": 0.09, "min_child_weight": 5, "n_jobs": 8},
     {"method": "lasso", "alpha": 0.1}),
]

OUTLIER_SETTING = {"method": "isolation_forest", "contamination": 0.01, "n_estimators": 200}

# %% data and folds

X, y, _, _ = data()
y = np.asarray(y, dtype=float)

splits = list(
    RepeatedKFold(
        n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
    ).split(X)
)
mask_outliers = [get_outliers(X.iloc[tr], **OUTLIER_SETTING) for tr, _ in splits]

# %% per-fold preprocessing, OOF predictions per member

oof = pd.DataFrame(index=X.index, columns=[m[0] for m in MEMBERS], dtype=float)

for (tr, val), mask in zip(splits, mask_outliers):
    X_tr, y_tr = X.iloc[tr][~mask], y[tr][~mask]
    X_val = X.iloc[val]

    for name, model_params, selector_params in MEMBERS:
        prep = Pipeline(
            [
                ("filter", get_filter(method="winsorize", z_threshold=10)),
                ("scaler", get_scaler(method="robust")),
                ("imputer", get_imputer(method="knn", n_neighbors=15)),
                ("selector", get_selector(**selector_params)),
            ]
        )
        Z_tr = prep.fit_transform(X_tr, y_tr)
        Z_val = prep.transform(X_val)

        model = get_model(n_features=Z_tr.shape[1], **model_params)
        model.fit(Z_tr, y_tr)
        oof.loc[X_val.index, name] = model.predict(Z_val)
        print(f"fold done: {name}")

# %% member scores and blend

names = [m[0] for m in MEMBERS]
scores = {n: r2_score(y, oof[n]) for n in names}
print("\n=== OOF member scores ===")
for n, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"  {n}: {s:.4f}")


def neg_r2(w):
    return -r2_score(y, oof[names].to_numpy() @ w)


cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
res = minimize(
    neg_r2,
    x0=np.full(len(names), 1 / len(names)),
    bounds=[(0, 1)] * len(names),
    constraints=[cons],
    method="SLSQP",
)
w = res.x
blend_r2 = r2_score(y, oof[names].to_numpy() @ w)
simple_r2 = r2_score(y, oof[names].mean(axis=1))

print(f"\nsimple average R2: {simple_r2:.4f}")
print(f"weighted blend R2: {blend_r2:.4f}")
print("\n=== blend weights ===")
for n, wi in sorted(zip(names, w), key=lambda kv: -kv[1]):
    print(f"  {n}: {wi:.3f}")

# %% persist

out = {
    "member_oof_r2": scores,
    "blend_r2": blend_r2,
    "simple_avg_r2": simple_r2,
    "weights": dict(zip(names, w.round(4).tolist())),
    "outlier_setting": OUTLIER_SETTING,
}
runs_dir = Path(__file__).resolve().parent / "runs"
(runs_dir / "ensemble_result.json").write_text(json.dumps(out, indent=2))
oof.to_csv(runs_dir / "ensemble_oof.csv", index=True)
print(f"\nsaved: {runs_dir / 'ensemble_result.json'}")
