# %%
import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import r2_score

from sklearn.model_selection import ParameterGrid, RepeatedKFold
from sklearn.pipeline import Pipeline


from itertools import product

from project1.src.outliers import get_outliers
from project1.src.imputation import get_imputer
from project1.src.scalers import get_scaler
from project1.src.selectors import get_selector
from project1.src.models import get_model
from project1.src.config import config

# %% load and prepare data

# load data
X_test_raw = pd.read_csv("project1/data/X_test.csv", header=0)
X_train_raw = pd.read_csv("project1/data/X_train.csv", header=0)
y_train_raw = pd.read_csv("project1/data/y_train.csv", header=0)

# assert alingment and shape
assert X_train_raw.iloc[:, 0].equals(y_train_raw.iloc[:, 0])
assert X_train_raw.shape[1] == X_test_raw.shape[1]

# store test ids for submission
test_ids = X_test_raw.iloc[:, 0].astype(int).to_numpy()

# drop index
X_test = X_test_raw.iloc[:, 1:]
X_train = X_train_raw.iloc[:, 1:]
y_train = y_train_raw.iloc[:, 1]

# create cv sets
cv = RepeatedKFold(
    n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed
)
splits = list(cv.split(X_train))

# %% grids
outlier_options = [
    {"method": ["passthrough"]},
    {"method": ["mahalanobis"], "alpha": [1e-4, 1e-2], "variance": [0.8]},
]

imputer_options = [
    {"method": ["median"]},
    {"method": ["knn"], "n_neighbors": [5]},
]

scaler_options = [
    {"method": ["standard", "robust"]},
]

selector_options = [
    {"method": ["kbest"], "k": [50, 100, 150, 200, 250, 300]},
    {"method": ["lasso"], "alpha": [0.3, 1.0, 2.0]},
]

model_options = [
    {"method": ["gpr"], "kernel": ["rbf"]},
    {"method": ["gpr"], "kernel": ["matern"], "nu": [1.5, 2.5]},
    {
        "method": ["krr"],
        "kernel": ["rbf"],
        "alpha": [0.1, 1],
        "gamma": [None, 3e-3, 1e-2, 3e-2],
    },
    {"method": ["svr"], "kernel": ["rbf"], "C": [10, 50, 100], "epsilon": [0.01, 0.1]},
]

# create grids
outliers = list(ParameterGrid(outlier_options))
scalers = list(ParameterGrid(scaler_options))
imputers = list(ParameterGrid(imputer_options))
selectors = list(ParameterGrid(selector_options))
models = list(ParameterGrid(model_options))

# calculate fits
n_fits = (
    len(outliers)
    * len(scalers)
    * len(imputers)
    * len(selectors)
    * len(models)
    * len(splits)
)
print("Model fits:", n_fits)

# %% grid search

# init loop
out_file = "project1/runs/" + datetime.now().strftime("run_%Y%m%d_%H%M.csv")
pbar = tqdm(total=n_fits, desc="model fits")
rows = []

# outliers loop (does not depend on model)
for o in outliers:
    mask_outliers = [get_outliers(X_train.iloc[tr], **o) for tr, _ in splits]
    n_outliers = np.mean([m.sum() for m in mask_outliers])

    # scaler, imputer, selectors loop
    for scaler, imputer, selector in product(scalers, imputers, selectors):
        folds = []

        # cv loop
        for (tr, val), mask in zip(splits, mask_outliers):
            # drop outliers in train
            X_tr, y_tr = X_train.iloc[tr][~mask], y_train.iloc[tr][~mask]

            # build pipeline
            prep = Pipeline(
                [
                    ("scaler", get_scaler(**scaler)),
                    ("imputer", get_imputer(**imputer)),
                    ("selector", get_selector(**selector)),
                ]
            )

            # transform
            Z_tr = prep.fit_transform(X_tr, y_tr)
            Z_val = prep.transform(X_train.iloc[val])
            folds.append((Z_tr, y_tr, Z_val, y_train.iloc[val]))

        # modle loop
        for m in models:
            # fit model
            scores = [
                r2_score(y_val, get_model(**m).fit(Z_tr, y_tr).predict(Z_val))
                for Z_tr, y_tr, Z_val, y_val in folds
            ]

            # update progress
            pbar.update(len(splits))

            # store results
            rows.append(
                {
                    "outliers": o,
                    "outliers_detected": n_outliers,
                    "scaler": scaler,
                    "imputer": imputer,
                    "selector": selector,
                    "model": m,
                    "r2_mean": np.mean(scores),
                    "r2_std": np.std(scores),
                }
            )

            # print best score
            pbar.set_postfix(best=round(max(r["r2_mean"] for r in rows), 3))

        pd.json_normalize(rows).to_csv(out_file, index=False)

pbar.close()


# %% inspect restuls

results = pd.json_normalize(rows).sort_values("r2_mean", ascending=False)
results.head(10)
