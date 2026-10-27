"""Generate a submission: pseudo-labeled TabPFN ensemble (ensemble_v2 recipe).

Teacher: blend of TabPFN members fit on the full (outlier-cleaned) training
set predicts the unlabeled X_test. Each member is then refit on
train + pseudo-labeled test and predicts X_test again. Outputs the weighted
blend to project1/runs/submission.csv (format: id,y).

Weights: runs/ensemble_v2_result.json -> pseudo_weights (CV OOF 0.6740).
"""

from dotenv import load_dotenv

load_dotenv()  # loads TABPFN_TOKEN etc. from .env

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from project1.src.data import data
from project1.src.filter import get_filter
from project1.src.scalers import get_scaler
from project1.src.imputation import get_imputer
from project1.src.selectors import get_selector
from project1.src.outliers import get_outliers
from project1.src.models import get_model

OUTLIER_SETTING = {"method": "isolation_forest", "contamination": 0.01, "n_estimators": 200}

W10 = {"method": "winsorize", "z_threshold": 10}
KNN = {"method": "knn", "n_neighbors": 15}
DEDUP = {"method": "unsupervised", "min_partner": 0.0, "max_correlation": 0.99}

# (name, model params, filter params, selector params) — imputer is KNN for all
MEMBERS = {
    "tabpfn_lasso": ({"method": "tabpfn", "n_estimators": 8}, W10,
                     {"method": "lasso", "alpha": 0.1}),
    "tabpfn_mi200": ({"method": "tabpfn", "n_estimators": 8}, W10,
                     {"method": "mutual_info", "k": 200}),
    "tabpfn_lasso_ne16": ({"method": "tabpfn", "n_estimators": 16}, W10,
                          {"method": "lasso", "alpha": 0.1}),
    "tabpfn_dedup_tree": ({"method": "tabpfn", "n_estimators": 8}, DEDUP,
                          {"method": "tree", "threshold": "median"}),
}

# %% load data, drop outlier rows, read blend weights

X, y, X_test, test_ids = data()

mask = get_outliers(X, **OUTLIER_SETTING)
X, y = X[~mask], y[~mask]
print(f"train after outlier removal: {X.shape} (dropped {mask.sum()})")

result = json.loads(
    (Path(__file__).resolve().parent / "runs" / "ensemble_v2_result.json").read_text()
)
weights = {n: result["pseudo_weights"][n] for n in MEMBERS}
total = sum(weights.values())
weights = {n: w / total for n, w in weights.items()}  # renormalize over members
print("weights:", {n: round(w, 3) for n, w in weights.items()})


def make_pipeline(filt, sel):
    return Pipeline(
        [
            ("filter", get_filter(**filt)),
            ("scaler", get_scaler(method="robust")),
            ("imputer", get_imputer(**KNN)),
            ("selector", get_selector(**sel)),
        ]
    )


# %% teacher pass: members fit on train only, predict unlabeled test

pseudo_y = np.zeros(len(X_test))
preps = {}
for name, (model_params, filt, sel) in MEMBERS.items():
    prep = make_pipeline(filt, sel)
    Z_tr = prep.fit_transform(X, np.asarray(y, dtype=float))
    model = get_model(n_features=Z_tr.shape[1], **model_params)
    model.fit(Z_tr, np.asarray(y, dtype=float))
    pseudo_y += weights[name] * model.predict(prep.transform(X_test))
    preps[name] = prep
print(f"teacher predictions: mean {pseudo_y.mean():.1f}, std {pseudo_y.std():.1f}")

# %% student pass: members refit on train + pseudo-labeled test

X_aug = pd.concat([X, X_test], ignore_index=True)
y_aug = np.concatenate([np.asarray(y, dtype=float), pseudo_y])

y_pred = np.zeros(len(X_test))
for name, (model_params, filt, sel) in MEMBERS.items():
    prep_aug = make_pipeline(filt, sel)
    Z_aug = prep_aug.fit_transform(X_aug, y_aug)
    model = get_model(n_features=Z_aug.shape[1], **model_params)
    model.fit(Z_aug, y_aug)
    y_pred += weights[name] * model.predict(prep_aug.transform(X_test))
    print(f"{name} student done")

# %% write submission

submission = pd.DataFrame({"id": test_ids, "y": y_pred})
out = Path(__file__).resolve().parent / "runs" / "submission.csv"
submission.to_csv(out, index=False)
print(f"\nwrote {out} ({len(submission)} rows)")
print(f"prediction stats: mean {y_pred.mean():.1f}, std {y_pred.std():.1f}")
