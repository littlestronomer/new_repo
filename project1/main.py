# %%
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import ParameterGrid
from project1.src.runner import run_experiment
from project1.src.data import data

import warnings

load_dotenv()  # loads TABPFN_TOKEN etc. from .env

warnings.filterwarnings("ignore", message=".*k2__noise_level.*lower bound.*")
warnings.filterwarnings("ignore", message=".*k1__k2__alpha.*upper bound.*")

# %% load data

X_train, y_train, X_test, test_ids = data()

# %% grids

outlier_options = [
    {"method": ["isolation_forest"], "contamination": [0.01]},
]

imputer_options = [
    {"method": ["knn"], "n_neighbors": [15]},
]

scaler_options = [
    {"method": ["robust"]},
]

filter_options = [
    {"method": ["winsorize"], "z_threshold": [10]},
    {"method": ["unsupervised"], "min_partner": [0.0], "max_correlation": [0.99]},
]

selector_options = [
    {"method": ["passthrough"]},
    {"method": ["tree"], "threshold": ["median"]},
    {"method": ["kbest"], "k": [50, 100, 200]},
    {"method": ["mutual_info"], "k": [50, 100, 200]},
    {"method": ["lasso"], "alpha": [0.01, 0.03, 0.1]},
    {"method": ["rfe"], "n_features_to_select": [50, 100], "step": [0.1]},
]

model_options = [
    # TabPFN (HuggingFace tabular foundation model)
    {
        "method": ["tabpfn"],
        "n_estimators": [8],
    },
]

# boosting baselines (re-enable for the full benchmark)
gbm_options = [
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

# model_options += gbm_options  # enable for the full benchmark

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

# %% TabPFN with native NaN handling (no explicit imputation / selection)
native_models = list(ParameterGrid([
    {"method": ["tabpfn"], "n_estimators": [4, 8]},
]))
results_native = run_experiment(
    X_train=X_train,
    y_train=y_train,
    outliers=outliers,
    scalers=scalers,
    imputers=[{"method": "passthrough"}],
    filters=filters,
    selectors=[{"method": "passthrough"}],
    models=native_models,
)

# %% inspect restuls
print(results.head(10))
print(results_native.head(10))

