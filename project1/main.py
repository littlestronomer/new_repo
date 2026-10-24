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
    {"method": ["mahalanobis"], "alpha": [0.01], "variance": [0.8]},
]


imputer_options = [
    {"method": ["median"]},
    {"method": ["knn"], "n_neighbors": [30]},
]

scaler_options = [{"method": ["robust"]}]

filter_options = [
    {"method": ["passthrough"]},
    {"method": ["unsupervised"], "min_partner": [0.3, 0.5], "max_correlation": [0.99]},
]

selector_options = [
    {"method": ["lasso"], "alpha": [0.03, 0.1]},
    {"method": ["kbest"], "k": [180, 190, 200]},
]

model_options = [
    {"method": ["gpr"], "kernel": ["matern"], "nu": [2.5]},
    {"method": ["gpr"], "kernel": ["rq"]},
    {
        "method": ["krr"],
        "kernel": ["rbf"],
        "alpha": [0.003, 0.01, 0.03],
        "gamma_mult": [2, 2.5, 3, 3.5],
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

