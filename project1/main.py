# %%
import pandas as pd
from sklearn.model_selection import ParameterGrid
from project1.src.runner import run_experiment

import warnings

warnings.filterwarnings("ignore", message=".*k2__noise_level.*lower bound.*")
warnings.filterwarnings("ignore", message=".*k1__k2__alpha.*upper bound.*")

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

# get non-unique columns
non_constant = X_train.nunique() > 1

# get almost uniform columns
uniform_like = (X_train.kurt() < -0.7) & (X_train.skew().abs() < 0.1)

# drop columns
X_train = X_train.loc[:, non_constant & ~uniform_like]
X_test = X_test.loc[:, non_constant & ~uniform_like]

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

