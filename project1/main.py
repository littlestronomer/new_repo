# %%
import pandas as pd
from sklearn.model_selection import ParameterGrid
from project1.src.runner import run_experiment
from project1.src.data import data

import warnings

warnings.filterwarnings("ignore", message=".*k2__noise_level.*lower bound.*")
warnings.filterwarnings("ignore", message=".*k1__k2__alpha.*upper bound.*")

# %% load data

X_train, y_train, X_test, test_ids = data()

# %% grids

outlier_options = [
    {"method": ["passthrough"]},
    {"method": ["isolation_forest"], "contamination": [0.01]},
]

imputer_options = [
    {"method": ["median"]},
    {"method": ["knn"], "n_neighbors": [15]},
]

scaler_options = [
    {"method": ["robust"]},
]

filter_options = [
    {"method": ["passthrough"]},
]

selector_options = [
    {"method": ["passthrough"]},
    {"method": ["tree"], "threshold": ["median"]},
]

model_options = [
    # CatBoost
    {
        "method": ["catboost"],
        "iterations": [350, 650],
        "depth": [4, 6, 8],
        "learning_rate": [0.035, 0.09],
        "l2_leaf_reg": [3, 10],
    },

    # LightGBM
    {
        "method": ["lightgbm"],
        "n_estimators": [350, 650],
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.035, 0.09],
        "min_child_samples": [10, 30],
    },

    # XGBoost
    {
        "method": ["xgboost"],
        "n_estimators": [350, 650],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.035, 0.09],
        "min_child_weight": [1, 5],
    },
]

# create grids
outliers = list(ParameterGrid(outlier_options))
scalers = list(ParameterGrid(scaler_options))
imputers = list(ParameterGrid(imputer_options))
filters = list(ParameterGrid(filter_options))
selectors = list(ParameterGrid(selector_options))
models = list(ParameterGrid(model_options))


# %% run experiment
results = run_experiment(
    X_train=X_train,
    y_train=y_train,
    outliers=outliers,
    scalers=scalers,
    imputers=imputers,
    filters=filters,
    selectors=selectors,
    models=models,
)

# %% inspect restuls
print(results.head(10))

